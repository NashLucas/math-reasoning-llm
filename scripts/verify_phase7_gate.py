#!/usr/bin/env python3
"""Phase 7 Day 50: Phase 7 Gate Verifier & Release v1.0 Wrap-Up CLI Runner.

Evaluates and audits all 7 Phase 7 milestones (Days 48–50):
1. GSM8K Test-Set Wall Unfreezing & Cryptographic Integrity Check (Day 48).
2. Unified Benchmark Infrastructure & Statistical Bootstrap Regimes (Day 48).
3. Problem Complexity & Reasoning Step Stratification (Day 49).
4. Error Taxonomy Breakdown & Relative Error Reduction (Day 49).
5. Empirical Post-Training Superiority & Statistical Significance (Day 49).
6. Production Serving Engine & Interactive Web Application (Day 50).
7. Release v1.0 Certification & Final Model Lineage Packaging (Day 50).

Generates:
- reports/phase7_gate_report.md
- reports/phase7_summary.json

Usage:
    python scripts/verify_phase7_gate.py
    python scripts/verify_phase7_gate.py --reports_dir reports --output_report reports/phase7_gate_report.md
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

from src.evaluation.phase7_verifier import run_phase7_wrapup  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_phase7_gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Phase 7 Gate and generate release v1.0 wrap-up deliverables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reports_dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Path to directory containing Phase 7 evaluation reports",
    )
    parser.add_argument(
        "--adapter_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "grpo" / "adapter_model",
        help="Path to advancing post-GRPO adapter directory",
    )
    parser.add_argument(
        "--output_report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase7_gate_report.md",
        help="Path where markdown wrap-up report will be written",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase7_summary.json",
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

    logger.info("Starting Phase 7 Gate verification...")
    logger.info("Reports directory:    %s", args.reports_dir)
    logger.info("Advancing adapter:    %s", args.adapter_dir)
    logger.info("Target Markdown:      %s", args.output_report)
    logger.info("Target JSON summary:  %s", args.output_json)

    try:
        summary = run_phase7_wrapup(
            reports_dir=args.reports_dir,
            output_report_path=args.output_report,
            output_summary_path=args.output_json,
            adapter_dir=args.adapter_dir,
        )
    except Exception as e:
        logger.error("Phase 7 Gate verification failed with exception: %s", e, exc_info=True)
        return 1

    if not args.quiet:
        print("\n" + "=" * 80)
        print("PHASE 7 GATE VERIFICATION SUMMARY (Release v1.0)")
        print("=" * 80)
        print(f"Timestamp:          {summary.timestamp}")
        print(f"Base Model:         {summary.base_model}")
        print(f"Advancing Adapter:  {summary.advancing_adapter}")
        print(f"Release Tag:        {summary.release_tag}")
        print(f"Overall Verdict:    {summary.verdict}")
        print("-" * 80)
        for i, c in enumerate(summary.criteria, start=1):
            status_str = "PASS" if c.passed else "FAIL"
            print(f"[{status_str}] Criterion {i}: {c.name}")
            print(f"       Threshold: {c.threshold}")
            print(f"       Observed:  {c.observed}")
            if c.details:
                print(f"       Details:   {c.details}")
        print("=" * 80)
        print("Deliverables generated:")
        print(f"  Markdown: {args.output_report}")
        print(f"  JSON:     {args.output_json}")
        print("=" * 80 + "\n")

    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
