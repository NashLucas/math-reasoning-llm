#!/usr/bin/env python3
"""Phase 6 Day 46: Post-GRPO Greedy vs. Pass@k Evaluation & Reward Trajectory Diagnostics CLI.

Benchmarks the post-GRPO trained policy adapter against the advancing Phase 5
Round 2 targeted SFT adapter on the held-out gating validation split (250 problems),
analyzing greedy accuracy, pass@k (k in {1, 4, 8}), majority voting @ 8, multi-component
reward trajectories, and Answer Contract preservation.

Usage:
    # Full GPU evaluation on Colab G4
    python scripts/evaluate_grpo.py \\
        --base_model deepseek-ai/deepseek-math-7b-base \\
        --sft_adapter outputs/sft_targeted/adapter_model \\
        --grpo_adapter outputs/grpo/adapter_model \\
        --val_split val_gating

    # Fast offline verification / dry-run / CI
    python scripts/evaluate_grpo.py --sample_mode --limit 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import TestSetWallError  # noqa: E402
from src.evaluation.grpo_evaluator import (  # noqa: E402
    compare_sft_vs_grpo,
    generate_day46_reports,
    plot_day46_evaluation_curves,
)
from src.run_info import save_run_info  # noqa: E402
from src.training.checkpointing import get_checkpoints  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_grpo")


def ensure_adapter_path(
    adapter_path: Path,
    fallback_paths: list[Path],
    label: str,
    allow_mock: bool = False,
) -> Path:
    """Resolve an adapter path, trying fallbacks, or create simulated adapter if in mock mode."""
    if adapter_path.exists():
        return adapter_path

    for fb in fallback_paths:
        if fb.exists():
            logger.info("Found fallback %s adapter at: %s", label, fb)
            return fb

    if allow_mock:
        logger.warning("No existing %s adapter found at %s. Creating simulated adapter for mock evaluation.", label, adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        adapter_config = {
            "base_model_name_or_path": "deepseek-ai/deepseek-math-7b-base",
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "simulated": True,
        }
        import json
        with open(adapter_path / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)
        return adapter_path

    raise FileNotFoundError(f"Could not locate required {label} adapter at {adapter_path} or any fallbacks.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for Day 46 evaluation."""
    parser = argparse.ArgumentParser(
        description="Phase 6 Day 46: Post-GRPO Greedy vs. Pass@k Evaluation CLI"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseek-ai/deepseek-math-7b-base",
        help="Base model ID or path.",
    )
    parser.add_argument(
        "--sft_adapter",
        type=str,
        default="outputs/sft_targeted/adapter_model",
        help="Path to SFT baseline adapter (default: outputs/sft_targeted/adapter_model).",
    )
    parser.add_argument(
        "--grpo_adapter",
        type=str,
        default="outputs/grpo/adapter_model",
        help="Path to GRPO policy adapter (default: outputs/grpo/adapter_model).",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="val_gating",
        choices=["val_gating", "val_analysis", "val", "sample"],
        help="Validation split for comparison (default: val_gating).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of candidate completions to sample per problem for pass@k (default: 8).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of problems evaluated.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "hf", "mock"],
        help="Inference engine backend.",
    )
    parser.add_argument(
        "--sample_mode",
        action="store_true",
        help="Fast sample evaluation mode using mock generation and small subset.",
    )
    parser.add_argument(
        "--force_mock",
        action="store_true",
        help="Force deterministic mock generation even if GPU/PyTorch is available.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval_grpo",
        help="Output directory for generation logs and evaluations.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Directory to save audit reports and plots.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute Day 46 Post-GRPO evaluation and diagnostics."""
    args = parse_args()

    # Strict Test-Set Wall check
    if args.val_split == "test":
        raise TestSetWallError(
            "Evaluation on 'test' split is strictly prohibited during Phase 6 Day 46 evaluation! "
            "Use 'val_gating' for model comparison and gating."
        )

    logger.info("========================================================================")
    logger.info("PHASE 6 DAY 46: POST-GRPO GREEDY VS. PASS@K EVALUATION & DIAGNOSTICS")
    logger.info("========================================================================")
    logger.info("Base Model:        %s", args.base_model)
    logger.info("SFT Baseline:      %s", args.sft_adapter)
    logger.info("GRPO Policy:       %s", args.grpo_adapter)
    logger.info("Split:             %s (Limit: %s)", args.val_split, args.limit or "Full")
    logger.info("Sample Mode:       %s", args.sample_mode)
    logger.info("Inference Backend: %s", args.backend)
    logger.info("========================================================================")

    use_mock = args.sample_mode or args.force_mock or (args.backend == "mock")
    if not use_mock:
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.info("PyTorch not found; falling back to mock mode.")
            use_mock = True

    # 1. Resolve Adapters
    sft_path = Path(args.sft_adapter)
    if not sft_path.is_absolute():
        sft_path = PROJECT_ROOT / sft_path
    sft_fallbacks = [
        PROJECT_ROOT / "outputs" / "sft" / "adapter_model",
    ]
    resolved_sft = ensure_adapter_path(sft_path, sft_fallbacks, "SFT baseline", allow_mock=use_mock)

    grpo_path = Path(args.grpo_adapter)
    if not grpo_path.is_absolute():
        grpo_path = PROJECT_ROOT / grpo_path
    grpo_checkpoints = get_checkpoints(PROJECT_ROOT / "checkpoints" / "grpo")
    grpo_fallbacks = [Path(p) / "adapter" for p in grpo_checkpoints] if grpo_checkpoints else []
    resolved_grpo = ensure_adapter_path(grpo_path, grpo_fallbacks, "GRPO policy", allow_mock=use_mock)

    split_target = "sample" if args.sample_mode and args.val_split == "sample" else args.val_split
    limit_target = args.limit if args.limit is not None else (10 if args.sample_mode else None)

    # 2. Run Comparative Evaluation
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    comparison = compare_sft_vs_grpo(
        sft_adapter_path=resolved_sft,
        grpo_adapter_path=resolved_grpo,
        base_model=args.base_model,
        split=split_target,
        limit=limit_target,
        backend=args.backend,
        use_mock=use_mock,
        output_dir=out_dir,
    )

    # 3. Generate Visualizations & Audit Reports
    rep_dir = Path(args.reports_dir)
    if not rep_dir.is_absolute():
        rep_dir = PROJECT_ROOT / rep_dir
    rep_dir.mkdir(parents=True, exist_ok=True)

    plot_file = rep_dir / "day46_grpo_eval_curves.png"
    plot_day46_evaluation_curves(comparison=comparison, output_path=plot_file)

    md_report, json_report = generate_day46_reports(
        comparison=comparison,
        reports_dir=rep_dir,
    )

    # 4. Save Provenance Metadata
    save_run_info(
        stage="grpo_eval",
        output_dir=rep_dir,
        config={
            "base_model": args.base_model,
            "sft_adapter": str(resolved_sft),
            "grpo_adapter": str(resolved_grpo),
            "split": split_target,
            "limit": limit_target,
            "num_samples": args.num_samples,
            "backend": args.backend,
            "use_mock": use_mock,
            "gate_verdict": comparison.gate_verdict,
            "greedy_accuracy_sft": comparison.sft_baseline.greedy_accuracy,
            "greedy_accuracy_grpo": comparison.grpo_policy.greedy_accuracy,
            "pass_at_8_sft": comparison.sft_baseline.pass_at_8,
            "pass_at_8_grpo": comparison.grpo_policy.pass_at_8,
            "mean_reward_sft": comparison.sft_baseline.reward_stats.total_reward.mean,
            "mean_reward_grpo": comparison.grpo_policy.reward_stats.total_reward.mean,
        },
        description="Phase 6 Day 46: Post-GRPO Greedy vs. Pass@k Evaluation",
    )

    # 5. Print Executive Summary
    sft = comparison.sft_baseline
    grpo = comparison.grpo_policy
    logger.info("========================================================================")
    logger.info("PHASE 6 DAY 46 EVALUATION RESULTS SUMMARY")
    logger.info("========================================================================")
    logger.info("Metric                  | SFT Baseline | GRPO Policy  | Delta")
    logger.info("------------------------+--------------+--------------+------------------")
    logger.info("Greedy Accuracy         | %11.2f%% | %11.2f%% | %+16.2f%%", sft.greedy_accuracy_percent, grpo.greedy_accuracy_percent, comparison.delta_greedy_accuracy_percent)
    logger.info("Pass@1                  | %11.2f%% | %11.2f%% | %+16.2f%%", sft.pass_at_1 * 100.0, grpo.pass_at_1 * 100.0, comparison.delta_pass_at_1 * 100.0)
    logger.info("Pass@4                  | %11.2f%% | %11.2f%% | %+16.2f%%", sft.pass_at_4 * 100.0, grpo.pass_at_4 * 100.0, comparison.delta_pass_at_4 * 100.0)
    logger.info("Pass@8                  | %11.2f%% | %11.2f%% | %+16.2f%%", sft.pass_at_8 * 100.0, grpo.pass_at_8 * 100.0, comparison.delta_pass_at_8 * 100.0)
    logger.info("Majority Vote @ 8       | %11.2f%% | %11.2f%% | %+16.2f%%", sft.maj_at_8 * 100.0, grpo.maj_at_8 * 100.0, comparison.delta_maj_at_8 * 100.0)
    logger.info("Mean Total Reward       | %12.4f | %12.4f | %+17.4f", sft.reward_stats.total_reward.mean, grpo.reward_stats.total_reward.mean, comparison.delta_total_reward)
    logger.info("Contract Adherence      | %11.1f%% | %11.1f%% | %+16.1f%%", sft.greedy_format_compliance_percent, grpo.greedy_format_compliance_percent, comparison.delta_format_compliance * 100.0)
    logger.info("------------------------+--------------+--------------+------------------")
    logger.info("Gating Decision Verdict: %s", comparison.gate_verdict)
    logger.info("Audit Report:            %s", md_report)
    logger.info("Structured JSON:         %s", json_report)
    logger.info("========================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
