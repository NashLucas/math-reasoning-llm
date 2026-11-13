#!/usr/bin/env python3
"""Root CLI entrypoint and TRL GRPOTrainer wrapper for Phase 6 GRPO Post-Training.

Provides the canonical high-level interface for Group Relative Policy Optimization (GRPO)
training, combining:
    1. Base Model & SFT Adapter Anchor: DeepSeekMath-7B-Base + Round 2 Targeted SFT adapter.
    2. Trainable PEFT Configuration: Low-Rank Adaptation (LoRA, r=16, alpha=32).
    3. Multi-Component Composite Rewards: Rule-based verification for accuracy, canonical
       format (\\n#### <number>), reasoning steps, and length penalties.
    4. Single-Model PEFT Reference Sharing: Disables active LoRA adapter to query reference
       log-probabilities on the same base model in memory, bounding VRAM <= 16.0 GB on Colab G4.
    5. Schulman k3 KL Divergence Constraint: Enforces non-negative drift tracking with beta=0.04.
    6. Strict Test-Set Wall Enforcement: Protects the 1,319 GSM8K test problems from contamination.

Usage:
    # Full GPU execution (Colab G4 / A100):
    python grpo_trainer.py --config configs/grpo_args.json

    # Fast CPU smoke test / dry run:
    python grpo_trainer.py --smoke_test --sample_mode --max_steps 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_grpo import (  # noqa: E402
    load_grpo_checkpoint,
    load_grpo_prompts,
    run_grpo_training,
    save_grpo_checkpoint,
    verify_test_set_wall,
)
from src.data.grpo_dataset import TestSetWallError  # noqa: E402
from src.rewards.grpo_rewards import (  # noqa: E402
    CompositeGRPOReward,
    accuracy_reward,
    format_reward,
    load_reward_config,
    reasoning_structure_reward,
)
from src.rewards.reference_model import ReferenceModelManager  # noqa: E402
from src.run_info import save_run_info  # noqa: E402
from src.training.grpo_engine import (  # noqa: E402
    GRPOTrainerEngine,
    load_grpo_engine_config,
    verify_zero_advantage_invariance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grpo_trainer")


# -------------------------------------------------------------------------
# TRL GRPOTrainer-Compatible Reward Functions
# -------------------------------------------------------------------------

def accuracy_reward_func(
    prompts: Sequence[str],
    completions: Sequence[str],
    answer: Optional[Sequence[str]] = None,
    gold_answer: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """TRL GRPOTrainer-compatible reward function for answer accuracy.

    Returns:
        List of floats: +1.0 for exact numerical match, -1.0 for mismatch.
    """
    targets = gold_answer if gold_answer is not None else (answer or [""] * len(completions))
    return accuracy_reward(completions, targets)


def format_reward_func(
    prompts: Sequence[str],
    completions: Sequence[str],
    **kwargs: Any,
) -> List[float]:
    """TRL GRPOTrainer-compatible reward function for canonical format compliance.

    Returns:
        List of floats: +0.5 for \\n#### <number> format, -0.5 for violation.
    """
    return format_reward(completions)


def reasoning_structure_reward_func(
    prompts: Sequence[str],
    completions: Sequence[str],
    **kwargs: Any,
) -> List[float]:
    """TRL GRPOTrainer-compatible reward function for step derivation presence.

    Returns:
        List of floats: +0.2 for multi-step reasoning, -0.2 for empty or single token answers.
    """
    return reasoning_structure_reward(completions)


def composite_reward_func(
    prompts: Sequence[str],
    completions: Sequence[str],
    answer: Optional[Sequence[str]] = None,
    gold_answer: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """Combined composite reward function wrapping all weighted signals."""
    calc = CompositeGRPOReward()
    targets = gold_answer if gold_answer is not None else (answer or [""] * len(completions))
    breakdowns = calc.score_batch(completions, targets, compute_advantages=False)
    return [b.total_reward for b in breakdowns]


# -------------------------------------------------------------------------
# Adapter Resolution & Model Anchoring
# -------------------------------------------------------------------------

def resolve_starting_adapter(
    adapter_override: Optional[Union[str, Path]] = None,
    project_root: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve the starting Phase 5 SFT adapter directory for GRPO policy initialization.

    Priority:
        1. Explicit path passed via adapter_override
        2. outputs/sft_targeted/adapter_model (Phase 5 Round 2 targeted adapter)
        3. outputs/sft/adapter_model (Phase 4 Round 1 adapter)
    """
    root = project_root or PROJECT_ROOT

    if adapter_override:
        cand = Path(adapter_override)
        if not cand.is_absolute():
            cand = root / cand
        if cand.exists():
            return cand
        logger.warning(f"Specified starting adapter not found at {cand}")

    # Check Phase 5 Round 2 targeted adapter
    round2_adapter = root / "outputs" / "sft_targeted" / "adapter_model"
    if round2_adapter.exists():
        logger.info(f"Anchoring GRPO policy from Phase 5 Round 2 targeted adapter: {round2_adapter}")
        return round2_adapter

    # Check Phase 4 Round 1 adapter
    round1_adapter = root / "outputs" / "sft" / "adapter_model"
    if round1_adapter.exists():
        logger.info(f"Anchoring GRPO policy from Phase 4 Round 1 SFT adapter: {round1_adapter}")
        return round1_adapter

    logger.warning("No existing SFT adapter found; policy will be trained from base model weights.")
    return None


def build_peft_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Construct PEFT LoRA configuration dictionary for GRPOTrainer."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "bias": "none",
    }


def load_grpo_training_dataset(
    dataset_path: Optional[Union[str, Path]] = None,
    sample_mode: bool = False,
    limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load and validate GRPO training prompts with strict Test-Set Wall enforcement."""
    path = Path(dataset_path) if dataset_path else PROJECT_ROOT / "data" / "processed" / "grpo" / "grpo_train.parquet"
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    verify_test_set_wall(path)
    raw_prompts = load_grpo_prompts(path, sample_mode=sample_mode, limit=limit)

    dataset: List[Dict[str, str]] = []
    for item in raw_prompts:
        dataset.append({
            "prompt": str(item.get("prompt", "")),
            "answer": str(item.get("gold_answer", "")),
            "problem_id": str(item.get("problem_id", "")),
        })

    return dataset


# -------------------------------------------------------------------------
# High-Level GRPOTrainer Factory
# -------------------------------------------------------------------------

class MathGRPOTrainer:
    """High-level wrapper orchestrating GRPO training with TRL GRPOTrainer or engine fallback."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        base_model_name: str = "deepseek-ai/deepseek-math-7b-base",
        starting_adapter: Optional[Union[str, Path]] = None,
        use_mock: bool = False,
    ):
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_grpo_engine_config(config_path)
        else:
            self.config = load_grpo_engine_config()

        self.base_model_name = base_model_name
        self.starting_adapter = resolve_starting_adapter(starting_adapter)
        self.use_mock = use_mock
        self.peft_config = build_peft_config()

        self.reward_funcs = [
            accuracy_reward_func,
            format_reward_func,
            reasoning_structure_reward_func,
        ]

    def train(
        self,
        train_file: Optional[Union[str, Path]] = None,
        val_file: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        adapter_dir: Optional[Union[str, Path]] = None,
        reports_dir: Optional[Union[str, Path]] = "reports",
        max_steps: Optional[int] = None,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        sample_mode: bool = False,
        smoke_test: bool = False,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute GRPO training run."""
        return run_grpo_training(
            config_file=PROJECT_ROOT / "configs" / "grpo_args.json",
            train_file=train_file,
            val_file=val_file,
            output_dir=output_dir,
            adapter_dir=adapter_dir,
            reports_dir=reports_dir,
            max_steps=max_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            sample_mode=sample_mode,
            smoke_test=smoke_test,
            use_mock=self.use_mock,
            resume_from_checkpoint=resume_from_checkpoint,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for grpo_trainer.py."""
    parser = argparse.ArgumentParser(
        description="GRPO Trainer for Mathematical Reasoning LLM (Phase 6 Post-Training)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "grpo_args.json"),
        help="Path to GRPO training arguments JSON.",
    )
    parser.add_argument(
        "--base_model",
        "--model_name_or_path",
        dest="base_model",
        type=str,
        default="deepseek-ai/deepseek-math-7b-base",
        help="Pretrained base model identifier or path.",
    )
    parser.add_argument(
        "--sft_adapter",
        "--starting_adapter",
        dest="sft_adapter",
        type=str,
        default=None,
        help="Path to starting Phase 5 SFT adapter.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to training prompts parquet file.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        help="Path to validation prompts parquet file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Directory to export final trained adapter.",
    )
    parser.add_argument(
        "--reports_dir",
        "--report_dir",
        dest="reports_dir",
        type=str,
        default="reports",
        help="Directory to save evaluation reports and plots.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Total training steps to execute.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Validation evaluation step frequency.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Checkpoint save step frequency.",
    )
    parser.add_argument(
        "--sample_mode",
        action="store_true",
        help="Enable fallback to sample data when processed dataset is absent.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run fast 5-step CPU smoke test for pipeline verification.",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use mock rollout and reference engine for CI/verification.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory or 'latest'.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    logger.info("Initializing GRPO Trainer (Phase 6)...")

    trainer = MathGRPOTrainer(
        config_path=args.config,
        base_model_name=args.base_model,
        starting_adapter=args.sft_adapter,
        use_mock=args.use_mock or args.smoke_test,
    )

    results = trainer.train(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        adapter_dir=args.adapter_dir,
        reports_dir=args.reports_dir,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        sample_mode=args.sample_mode,
        smoke_test=args.smoke_test,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    print("=" * 72)
    print("GRPO Trainer Execution Complete")
    print(f"Status:             {results.get('status', 'success')}")
    print(f"Steps Executed:     {results.get('total_steps_executed', 0)}")
    print(f"Final Step:         {results.get('final_step', 0)}")
    print(f"Initial Reward:     {results.get('initial_reward', 0.0):.4f}")
    print(f"Final Reward:       {results.get('final_reward', 0.0):.4f}")
    print(f"Val Accuracy:       {results.get('final_val_accuracy', 0.0) * 100:.1f}%")
    print(f"Adapter Output:     {results.get('adapter_dir', 'N/A')}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
