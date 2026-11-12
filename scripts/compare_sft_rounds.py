#!/usr/bin/env python3
"""Phase 5 Day 37: Compare SFT Rounds (Gating Evaluation CLI).

Evaluates Round-1 (Curated SFT) and Round-2 (Targeted SFT) adapters on the
gating validation half (data/processed/val/val_gating.parquet, 250 problems)
using greedy decoding, applies the decision rule, analyzes error mitigation
shifts, and stages the advancing adapter for Phase 6 (GRPO).

Usage:
    # Full G4/A100 evaluation
    python scripts/compare_sft_rounds.py \\
        --base_model deepseek-ai/deepseek-math-7b-base \\
        --round1_adapter outputs/sft/adapter_model \\
        --round2_adapter outputs/sft_targeted/adapter_model \\
        --val_split val_gating

    # Fast offline verification / dry-run / CI
    python scripts/compare_sft_rounds.py --sample_mode --limit 10
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
from src.evaluation.round_comparator import run_round_comparison  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("compare_sft_rounds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 Day 37: Evaluate and compare Round-1 vs Round-2 SFT adapters on val_gating."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseek-ai/deepseek-math-7b-base",
        help="Base model identifier or path (default: deepseek-ai/deepseek-math-7b-base).",
    )
    parser.add_argument(
        "--round1_adapter",
        type=str,
        default=None,
        help="Path to Round 1 (curated) LoRA adapter directory (default: outputs/sft/adapter_model).",
    )
    parser.add_argument(
        "--round2_adapter",
        type=str,
        default=None,
        help="Path to Round 2 (targeted) LoRA adapter directory (default: outputs/sft_targeted/adapter_model).",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="val_gating",
        choices=["val_gating", "val", "val_analysis", "sample"],
        help="Validation split to evaluate against (default: val_gating, 250 problems).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of problems evaluated per round (useful for quick verification).",
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
        help="Directory to write comparison reports and visualizations (default: reports).",
    )
    parser.add_argument(
        "--staging_dir",
        type=str,
        default=None,
        help="Destination directory to stage advancing adapter (default: outputs/sft_targeted/adapter_model).",
    )
    parser.add_argument(
        "--tie_margin_pct",
        type=float,
        default=2.0,
        help="Accuracy percentage point difference threshold to classify a tie (default: 2.0).",
    )
    parser.add_argument(
        "--tie_margin_count",
        type=int,
        default=5,
        help="Problem count difference threshold to classify a tie (default: 5 problems).",
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

    # Enforce Test-Set Wall immediately
    if args.val_split == "test":
        logger.error("TestSetWallError: Evaluation on 'test' split is strictly prohibited in Day 37.")
        sys.exit(1)

    # If backend is mock, force use_mock
    use_mock = args.use_mock or (args.backend == "mock")

    # Auto-detect profile if not explicitly specified
    active_profile = args.profile or ("colab" if Path("/content").exists() else "local")

    logger.info("==================================================")
    logger.info("Phase 5 Day 37: SFT Round Comparison (Gating Split)")
    logger.info("Base Model:        %s", args.base_model)
    logger.info("Round 1 Adapter:   %s", args.round1_adapter or "auto (outputs/sft/adapter_model)")
    logger.info("Round 2 Adapter:   %s", args.round2_adapter or "auto (outputs/sft_targeted/adapter_model)")
    logger.info("Evaluation Split:  %s", args.val_split)
    logger.info("Problem Limit:     %s", args.limit or "Full (250 problems)")
    logger.info("Inference Backend: %s", args.backend)
    logger.info("Mock Mode:         %s", use_mock)
    logger.info("Profile:           %s", active_profile)
    logger.info("Tie Margin:        +/-%.1f%% (+/-%d problems)", args.tie_margin_pct, args.tie_margin_count)
    logger.info("==================================================")

    try:
        results = run_round_comparison(
            model_name_or_path=args.base_model,
            round1_adapter=args.round1_adapter,
            round2_adapter=args.round2_adapter,
            val_split=args.val_split,
            limit=args.limit,
            backend=args.backend,
            use_mock=use_mock,
            reports_dir=args.reports_dir,
            staging_dir=args.staging_dir,
            tie_margin_pct=args.tie_margin_pct,
            tie_margin_count=args.tie_margin_count,
            profile=active_profile,
        )
    except TestSetWallError as err:
        logger.error("Test-Set Wall violation: %s", err)
        sys.exit(1)
    except Exception as err:
        logger.exception("Error executing round comparison: %s", err)
        sys.exit(1)

    r1 = results["round1"]
    r2 = results["round2"]
    dec = results["decision"]
    pb = results["problem_breakdown"]

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Round 1 (Curated SFT):  Acc: %5.2f%% (%d/%d) | Fmt: %5.2f%% | Trunc: %5.2f%%",
                r1["accuracy_percent"], r1["correct_count"], r1["total_problems"],
                r1["format_compliance_percent"], r1["truncation_percent"])
    logger.info("Round 2 (Targeted SFT): Acc: %5.2f%% (%d/%d) | Fmt: %5.2f%% | Trunc: %5.2f%%",
                r2["accuracy_percent"], r2["correct_count"], r2["total_problems"],
                r2["format_compliance_percent"], r2["truncation_percent"])
    logger.info("Delta (R2 - R1):        Acc: %+5.2f%% (%+d problems)",
                dec["difference_accuracy_percent"], dec["difference_correct"])
    logger.info("-" * 60)
    logger.info("Problem Breakdown:      Both Correct: %d | Both Incorrect: %d",
                pb.get("both_correct_count", 0), pb.get("both_incorrect_count", 0))
    logger.info("                        R2 Recoveries (Fixed): %d | R1 Regressions: %d",
                pb.get("r2_recoveries_count", 0), pb.get("r1_regressions_count", 0))
    logger.info("                        Net Error Movement: %+d",
                pb.get("net_recovery", 0))
    logger.info("-" * 60)
    logger.info("Decision Verdict:       %s", dec["decision"].upper())
    logger.info("Advancing Adapter:      %s", dec["advancing_adapter_path"])
    logger.info("Advancing Stage:        %s", dec["advancing_stage"])
    logger.info("Rationale:              %s", dec["rationale"])
    logger.info("=" * 60)
    logger.info("Reports written:")
    logger.info("  - JSON: %s", results["json_report"])
    logger.info("  - Markdown: %s", results["md_report"])
    if results.get("plot_path"):
        logger.info("  - Plot: %s", results["plot_path"])
    logger.info("==================================================")


if __name__ == "__main__":
    main()
