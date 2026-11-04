#!/usr/bin/env python3
"""Phase 5 Day 38: SFT Round 2 Category Shift Analysis CLI.

Reruns error classification on Round 2's validation analysis-half errors
(data/processed/val/val_analysis.parquet, 250 problems), compares error category
frequencies against Round 1, analyzes problem-level transitions (recoveries vs
regressions), and generates before-and-after movement audit reports.

Usage:
    # Full GPU execution (Colab G4/A100)
    python scripts/run_day38_shift_analysis.py \\
        --base_model deepseek-ai/deepseek-math-7b-base \\
        --round2_adapter outputs/sft_targeted/adapter_model \\
        --val_analysis_file data/processed/val/val_analysis.parquet

    # Fast offline verification / dry-run / CI
    python scripts/run_day38_shift_analysis.py --sample_mode --limit 10
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

from src.evaluation.category_shift import run_category_shift_analysis  # noqa: E402
from src.evaluation.evaluator import TestSetWallError  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("category_shift_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 Day 38: Category Shift Analysis comparing Round 1 vs Round 2 errors on val_analysis."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseek-ai/deepseek-math-7b-base",
        help="Base model identifier or path (default: deepseek-ai/deepseek-math-7b-base).",
    )
    parser.add_argument(
        "--round2_adapter",
        type=str,
        default=None,
        help="Path to Round 2 LoRA adapter directory (default: outputs/sft_targeted/adapter_model).",
    )
    parser.add_argument(
        "--round1_report",
        type=str,
        default="reports/error_analysis_sft.json",
        help="Path to Day 32 Round 1 error analysis JSON report (default: reports/error_analysis_sft.json).",
    )
    parser.add_argument(
        "--val_analysis_file",
        type=str,
        default=None,
        help="Path to validation analysis parquet file (default: data/processed/val/val_analysis.parquet).",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="deepseek-ai/deepseek-math-7b-instruct",
        help="Teacher model for error classification (default: deepseek-ai/deepseek-math-7b-instruct).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of problems evaluated (useful for quick verification).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "hf", "mock"],
        help="Inference engine backend (default: auto).",
    )
    parser.add_argument(
        "--sample_mode",
        "--mock",
        dest="use_mock",
        action="store_true",
        help="Run in mock/sample mode for testing without GPU/inference dependencies.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Directory to write reports and visualizations (default: reports).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Runtime configuration profile (default: auto-detected).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    use_mock = args.use_mock or (args.backend == "mock")

    logger.info("==================================================")
    logger.info("Phase 5 Day 38: SFT Round 2 Category Shift Analysis")
    logger.info("Base Model:          %s", args.base_model)
    logger.info("Round 2 Adapter:     %s", args.round2_adapter or "auto (outputs/sft_targeted/adapter_model)")
    logger.info("Round 1 Baseline:    %s", args.round1_report)
    logger.info("Analysis Split:      %s", args.val_analysis_file or "auto (data/processed/val/val_analysis.parquet)")
    logger.info("Teacher Model:       %s", args.teacher_model)
    logger.info("Problem Limit:       %s", args.limit or "Full (250 problems)")
    logger.info("Inference Backend:   %s", args.backend)
    logger.info("Mock Mode:           %s", use_mock)
    logger.info("==================================================")

    try:
        results = run_category_shift_analysis(
            model_name_or_path=args.base_model,
            round2_adapter=args.round2_adapter,
            round1_report=args.round1_report,
            val_analysis_file=args.val_analysis_file,
            teacher_model=args.teacher_model,
            limit=args.limit,
            backend=args.backend,
            use_mock=use_mock,
            reports_dir=args.reports_dir,
            profile=args.profile,
        )
    except TestSetWallError as err:
        logger.error("Test-Set Wall violation: %s", err)
        sys.exit(1)
    except Exception as err:
        logger.exception("Error executing category shift analysis: %s", err)
        sys.exit(1)

    summary = results["summary"]
    logger.info("\n" + "=" * 60)
    logger.info("CATEGORY SHIFT SUMMARY")
    logger.info("=" * 60)
    logger.info("Total Evaluated:        %d", summary["total_evaluated"])
    logger.info("Round 1 Accuracy:       %.2f%% (%d errors)", summary["round1_accuracy"], summary["round1_error_count"])
    logger.info("Round 2 Accuracy:       %.2f%% (%d errors)", summary["round2_accuracy"], summary["round2_error_count"])
    logger.info("Accuracy Delta:         %+.2f%%", summary["accuracy_delta"])
    logger.info("Net Errors Eliminated:  %+d", summary["net_errors_eliminated"])
    logger.info("-" * 60)
    tc = summary.get("transition_counts", {})
    logger.info("Transitions:            Fixed (Recoveries): %d | Regressed: %d",
                tc.get("fixed", 0), tc.get("regressed", 0))
    logger.info("                        Retained Same: %d | Shifted Category: %d",
                tc.get("retained_same", 0), tc.get("retained_shifted", 0))
    logger.info("                        Consistently Correct: %d",
                tc.get("consistently_correct", 0))
    logger.info("-" * 60)
    logger.info("Category Shift Table:")
    for m in summary["category_metrics"]:
        logger.info("  - %-22s: R1=%2d (%4.1f%%) -> R2=%2d (%4.1f%%) | Delta=%+2d | %s",
                    m["category"], m["round1_count"], m["round1_percentage"],
                    m["round2_count"], m["round2_percentage"], m["absolute_change"], m["status"])
    logger.info("=" * 60)
    logger.info("Reports written:")
    logger.info("  - JSON: %s", results["json_report"])
    logger.info("  - Updated Main MD: %s", results["main_markdown_report"])
    logger.info("  - Standalone MD: %s", results["standalone_markdown_report"])
    if results.get("plot_path"):
        logger.info("  - Plot: %s", results["plot_path"])
    logger.info("==================================================")


if __name__ == "__main__":
    main()
