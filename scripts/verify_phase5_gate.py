#!/usr/bin/env python3
"""Phase 5 Day 39: Phase 5 Gate Verifier & Wrap-Up CLI Runner (Release v0.6).

Verifies all Phase 5 Gate criteria:
1. SFT Round 2 Performance & Advancement (win / <=2.0% tie margin).
2. Canonical format compliance >= 98.0% ending in `#### <number>`.
3. Truncation rate <= 5.0% under max sequence budget (512 tokens).
4. Strict Test-Set Wall isolation (0 test leaks, test set isolated).
5. Targeted error mitigation & category shift (net error reduction >= 0).
6. Advancing adapter verification in outputs/sft_targeted/adapter_model.
7. Round-2 dataset freeze manifest with SHA-256 integrity.

Generates:
- reports/phase5_wrapup_report.md
- reports/phase5_summary.json

Usage:
    python scripts/verify_phase5_gate.py
    python scripts/verify_phase5_gate.py --reports_dir reports --output_report reports/phase5_wrapup_report.md
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

from src.evaluation.phase5_verifier import run_phase5_wrapup  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_phase5_gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Phase 5 Gate and generate release v0.6 wrap-up deliverables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reports_dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Path to directory containing round_comparison.json and category_shift_analysis.json",
    )
    parser.add_argument(
        "--adapter_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sft_targeted" / "adapter_model",
        help="Path to staged advancing LoRA adapter directory",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "sft_round2",
        help="Path to frozen Round-2 dataset directory",
    )
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits",
        help="Path to data/splits directory",
    )
    parser.add_argument(
        "--output_report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase5_wrapup_report.md",
        help="Path where markdown wrap-up report will be written",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase5_summary.json",
        help="Path where JSON summary will be written",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress banner and criteria printing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    comp_file = args.reports_dir / "round_comparison.json"
    shift_file = args.reports_dir / "category_shift_analysis.json"

    logger.info("Starting Phase 5 Gate verification...")
    logger.info("Round comparison report: %s", comp_file)
    logger.info("Category shift report: %s", shift_file)
    logger.info("Advancing adapter dir: %s", args.adapter_dir)

    try:
        summary = run_phase5_wrapup(
            comp_input=comp_file,
            shift_input=shift_file,
            splits_dir=args.splits_dir,
            adapter_path=args.adapter_dir,
            dataset_dir=args.dataset_dir,
            output_report_path=args.output_report,
            output_json_path=args.output_json,
            save_provenance=True,
        )
    except Exception as e:
        logger.error("Phase 5 Gate verification failed with exception: %s", e, exc_info=True)
        return 1

    if not args.quiet:
        print("\n" + "=" * 78)
        print("PHASE 5 GATE VERIFICATION SUMMARY (Release v0.6)")
        print("=" * 78)
        print(f"Timestamp:          {summary.timestamp}")
        print(f"Base Model:         {summary.base_model}")
        print(f"Advancing Adapter:  {summary.advancing_adapter}")
        print(f"Overall Verdict:    {summary.verdict}")
        print("-" * 78)
        for i, c in enumerate(summary.criteria, start=1):
            status_str = "PASS" if c.passed else "FAIL"
            print(f"[{status_str}] Criterion {i}: {c.name}")
            print(f"       Threshold: {c.threshold}")
            print(f"       Observed:  {c.observed}")
            if c.details:
                print(f"       Details:   {c.details}")
        print("=" * 78)
        print("Reports generated:")
        print(f"  Markdown: {args.output_report}")
        print(f"  JSON:     {args.output_json}")
        print("=" * 78 + "\n")

    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
