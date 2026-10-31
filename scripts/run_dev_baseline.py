"""Execute validation baseline evaluation for the development model (Day 13).

Evaluates the development model (Qwen/Qwen2.5-1.5B-Instruct) across all 500 validation
problems on both 'greedy' and 'self_consistency' decoding presets.

Usage:
  # On GPU (Colab T4):
  python scripts/run_dev_baseline.py

  # On CPU / CI (fast mock execution):
  python scripts/run_dev_baseline.py --use_mock

  # Dry run on first 20 problems:
  python scripts/run_dev_baseline.py --limit 20
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

from scripts.evaluate import print_results_summary  # noqa: E402
from src.config import ConfigManager  # noqa: E402
from src.evaluation.evaluator import MathEvaluator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dev_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run development model baseline on GSM8K validation set (greedy & self-consistency).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Model ID or checkpoint path (defaults to dev_model: Qwen/Qwen2.5-1.5B-Instruct).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "val_analysis", "val_gating", "sample"],
        help="Validation split to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional problem limit (e.g. 20 for dry run, None for all 500).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "hf"],
        help="Inference engine backend.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        default=False,
        help="Use deterministic mock generation for CI and environments without GPU.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg_mgr = ConfigManager()
    model_name = args.model_name_or_path or cfg_mgr.base_config.get("models", {}).get(
        "dev_model", "Qwen/Qwen2.5-1.5B-Instruct"
    )

    stage = "baseline"
    logger.info("=" * 64)
    logger.info(f"Starting Day 13 Development Model Baseline: {model_name}")
    logger.info(f"Split: {args.split} (Limit: {args.limit}) | Backend: {args.backend}")
    logger.info("=" * 64)

    # 1. Run Greedy Decoding Evaluation
    logger.info("\n--- Phase 1/2: Greedy Decoding Evaluation ---")
    greedy_evaluator = MathEvaluator(
        stage=stage,
        checkpoint=model_name,
        preset="greedy",
        split=args.split,
        limit=args.limit,
        backend=args.backend,
        batch_size=args.batch_size,
        use_mock=args.use_mock,
    )
    greedy_results = greedy_evaluator.evaluate()
    print_results_summary(greedy_results)

    # 2. Run Self-Consistency Decoding Evaluation
    logger.info("\n--- Phase 2/2: Self-Consistency Decoding Evaluation ---")
    sc_evaluator = MathEvaluator(
        stage=stage,
        checkpoint=model_name,
        preset="self_consistency",
        split=args.split,
        limit=args.limit,
        backend=args.backend,
        batch_size=args.batch_size,
        use_mock=args.use_mock,
    )
    sc_results = sc_evaluator.evaluate()
    print_results_summary(sc_results)

    logger.info("=" * 64)
    logger.info("Development baseline evaluations successfully completed!")
    logger.info(f"Results stored in: outputs/{stage}/")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
