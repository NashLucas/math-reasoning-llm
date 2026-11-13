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
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.phase7_verifier import run_phase7_wrapup  # noqa: E402
from src.run_info import save_run_info  # noqa: E402

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
        "--reports-dir",
        dest="reports_dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Path to directory containing Phase 7 evaluation reports",
    )
    parser.add_argument(
        "--adapter_dir",
        "--adapter-dir",
        dest="adapter_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "grpo" / "adapter_model",
        help="Path to advancing post-GRPO adapter directory",
    )
    parser.add_argument(
        "--output_report",
        "--output-report",
        dest="output_report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase7_gate_report.md",
        help="Path where markdown wrap-up report will be written",
    )
    parser.add_argument(
        "--output_json",
        "--output-json",
        dest="output_json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase7_summary.json",
        help="Path where JSON summary will be written",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="auto",
        choices=["local", "colab", "auto"],
        help="Execution environment profile",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress banner and criteria printing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve paths
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = PROJECT_ROOT / reports_dir

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.is_absolute():
        adapter_dir = PROJECT_ROOT / adapter_dir

    output_report = Path(args.output_report)
    if not output_report.is_absolute():
        output_report = PROJECT_ROOT / output_report

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json

    logger.info("========================================================================")
    logger.info("PHASE 7 DAY 50: PHASE 7 GATE VERIFICATION & RELEASE v1.0 WRAP-UP")
    logger.info("========================================================================")
    logger.info("Reports directory:    %s", reports_dir)
    logger.info("Advancing adapter:    %s", adapter_dir)
    logger.info("Target Markdown:      %s", output_report)
    logger.info("Target JSON summary:  %s", output_json)
    logger.info("Environment profile:  %s", args.profile)
    logger.info("========================================================================")

    try:
        summary = run_phase7_wrapup(
            reports_dir=reports_dir,
            output_report_path=output_report,
            output_summary_path=output_json,
            adapter_dir=adapter_dir,
        )
    except Exception as e:
        logger.error("Phase 7 Gate verification failed with exception: %s", e, exc_info=True)
        return 1

    # Record run provenance
    try:
        save_run_info(
            stage="phase7_wrapup",
            output_dir=reports_dir,
            config=summary.to_dict(),
            description="Phase 7 Gate Verification & Release v1.0 Wrap-Up",
        )
    except Exception as exc:
        logger.warning("Could not save run_info: %s", exc)

    # Mirror deliverables to Google Drive if in Colab environment
    drive_base = Path("/content/drive/MyDrive/math-reasoning-llm")
    if drive_base.exists() or args.profile == "colab":
        try:
            drive_reports = drive_base / "reports"
            drive_reports.mkdir(parents=True, exist_ok=True)
            if output_report.exists():
                shutil.copy2(output_report, drive_reports / output_report.name)
            if output_json.exists():
                shutil.copy2(output_json, drive_reports / output_json.name)
            run_info_file = reports_dir / "run_info.json"
            if run_info_file.exists():
                shutil.copy2(run_info_file, drive_reports / run_info_file.name)
            logger.info("Mirrored Phase 7 gate reports to Google Drive at: %s", drive_reports)
        except Exception as exc:
            logger.warning("Could not mirror reports to Drive: %s", exc)

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
        print(f"  Markdown: {output_report}")
        print(f"  JSON:     {output_json}")
        print("=" * 80 + "\n")

    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
