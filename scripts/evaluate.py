"""Evaluation entrypoint CLI script for mathematical reasoning models.

Usage examples:
  # Dry run on sample problems with greedy decoding (development model):
  python scripts/evaluate.py --stage baseline --preset greedy --split sample --limit 20

  # Fast CI testing with deterministic mock:
  python scripts/evaluate.py --stage baseline --preset greedy --split sample --use_mock

  # Full validation baseline evaluation with self-consistency:
  python scripts/evaluate.py --stage baseline --preset self_consistency --split val

  # Final test evaluation (strictly protected by the Test-Set Wall):
  python scripts/evaluate.py --stage final --preset greedy --split test --final
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import MathEvaluator, TestSetWallError  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation on mathematical reasoning models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="baseline",
        help="Pipeline stage name (e.g. baseline, sft, targeted_sft, grpo, final).",
    )
    parser.add_argument(
        "--checkpoint",
        "--model_name_or_path",
        dest="checkpoint",
        type=str,
        default=None,
        help="Hugging Face model ID or checkpoint directory path.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to trained LoRA adapter weights, if applicable.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="greedy",
        choices=["greedy", "sampled", "self_consistency"],
        help="Decoding preset from generation.yaml.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "val_analysis", "val_gating", "sample", "train", "test"],
        help="Dataset split to evaluate. ('test' split strictly requires --final flag).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of problems to evaluate (useful for dry runs).",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        default=False,
        help="Authorization flag required to unlock evaluation on the 'test' split.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "hf"],
        help="Inference engine backend ('auto', 'vllm', or 'hf').",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="auto",
        choices=["auto", "few_shot", "zero_shot", "chat"],
        help="Prompt formatting strategy.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for model generation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory for evaluation results.",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        default=False,
        help="Use deterministic mock generation for fast testing without GPU.",
    )
    return parser.parse_args()


def print_results_summary(results: dict) -> None:
    """Print an aligned, readable summary table of the evaluation results."""
    metrics = results.get("metrics", {})
    sep = "=" * 64

    print(f"\n{sep}")
    print(f"  EVALUATION RESULTS SUMMARY: stage={results.get('stage')} | preset={results.get('preset')}")
    print(f"{sep}")
    print(f"  Model Checkpoint:        {results.get('checkpoint')}")
    if results.get("adapter_path"):
        print(f"  Adapter:                 {results.get('adapter_path')}")
    print(f"  Split:                   {results.get('split')} (Problems: {results.get('num_problems')})")
    print(f"  Generations File:        {results.get('generations_file')}")
    print(f"{'-' * 64}")

    if metrics.get("is_multi_sample"):
        print(f"  maj@k Accuracy:          {metrics.get('maj_at_k_percent')}% ({metrics.get('maj_at_k')})")
        print(f"  pass@k Rate:             {metrics.get('pass_at_k_percent')}% ({metrics.get('pass_at_k')})")
        print(f"  Sample-1 Accuracy:       {metrics.get('accuracy_percent')}% ({metrics.get('accuracy')})")
    else:
        print(f"  Accuracy (Greedy):       {metrics.get('accuracy_percent')}% ({metrics.get('accuracy')})")

    print(f"  Format Compliance:       {metrics.get('format_compliance_percent')}% ({metrics.get('canonical_hash_count')}/{metrics.get('total_completions')})")
    print(f"  Fallback Rate:           {metrics.get('fallback_percent')}% ({metrics.get('fallback_count')}/{metrics.get('total_completions')})")
    print(f"  Unparseable Rate:        {metrics.get('unparseable_percent')}% ({metrics.get('unparseable_count')}/{metrics.get('total_completions')})")
    print(f"  Truncation Rate:         {metrics.get('truncation_percent')}% ({metrics.get('truncated_count')}/{metrics.get('total_completions')})")
    print(f"  Avg Prompt Tokens:       {metrics.get('average_prompt_tokens')}")
    print(f"  Avg Completion Tokens:   {metrics.get('average_completion_tokens')}")
    print(f"  Avg Total Tokens:        {metrics.get('average_total_tokens')}")
    print(f"{sep}\n")


def main() -> int:
    args = parse_args()

    try:
        evaluator = MathEvaluator(
            stage=args.stage,
            checkpoint=args.checkpoint,
            adapter_path=args.adapter_path,
            preset=args.preset,
            split=args.split,
            limit=args.limit,
            final=args.final,
            prompt_format=args.prompt_format,
            backend=args.backend,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            use_mock=args.use_mock,
        )

        results = evaluator.evaluate()
        print_results_summary(results)
        return 0

    except TestSetWallError as e:
        logger.critical(f"Execution halted by Test-Set Wall: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
