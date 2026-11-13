#!/usr/bin/env python3
"""Phase 6 Day 47: Phase 6 Gate Verifier & Release v0.7 Wrap-Up CLI.

Verifies all 7 Phase 6 Gate criteria:
1. GRPO Reward Architecture & Advantage Normalization (Day 40).
2. Intermediate-Difficulty Dataset & Test-Set Wall Isolation (Day 41).
3. Reference Policy Management & Schulman k3 Non-Negative KL Divergence (Day 42).
4. GRPO Rollout Generation & Policy Gradient Loss Engine (Day 43).
5. Local Smoke Test & VRAM Feasibility Budget (Day 44).
6. Full GRPO Post-Training Execution & Convergence Diagnostics (Day 45).
7. Post-GRPO Comparative Evaluation & Answer Contract Preservation (Day 46).

Emits:
- reports/phase6_gate_report.md
- reports/phase6_summary.json

Usage:
    python scripts/verify_phase6_gate.py
    python scripts/verify_phase6_gate.py --reports_dir reports --output_report reports/phase6_gate_report.md
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.phase6_verifier import run_phase6_wrapup  # noqa: E402
from src.run_info import save_run_info  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_phase6_gate")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Phase 6 Gate verification."""
    parser = argparse.ArgumentParser(
        description="Phase 6 Day 47: Phase 6 Gate Verifier & Release v0.7 Wrap-Up CLI"
    )
    parser.add_argument(
        "--reports_dir",
        "--reports-dir",
        dest="reports_dir",
        type=str,
        default="reports",
        help="Directory containing milestone audit reports (default: reports).",
    )
    parser.add_argument(
        "--dataset_dir",
        "--dataset-dir",
        dest="dataset_dir",
        type=str,
        default="data/processed/grpo",
        help="Directory containing GRPO dataset files and manifest.",
    )
    parser.add_argument(
        "--output_report",
        "--output-report",
        dest="output_report",
        type=str,
        default=None,
        help="Output path for Markdown gate report (default: reports/phase6_gate_report.md).",
    )
    parser.add_argument(
        "--summary_json",
        "--summary-json",
        dest="summary_json",
        type=str,
        default=None,
        help="Output path for summary JSON (default: reports/phase6_summary.json).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="local",
        choices=["local", "colab"],
        help="Execution profile (default: local).",
    )
    return parser.parse_args()


def main() -> int:
    """Execute Phase 6 Gate verification and report generation."""
    args = parse_args()

    rep_dir = Path(args.reports_dir)
    if not rep_dir.is_absolute():
        rep_dir = PROJECT_ROOT / rep_dir

    ds_dir = Path(args.dataset_dir)
    if not ds_dir.is_absolute():
        ds_dir = PROJECT_ROOT / ds_dir

    out_md = Path(args.output_report) if args.output_report else (rep_dir / "phase6_gate_report.md")
    out_json = Path(args.summary_json) if args.summary_json else (rep_dir / "phase6_summary.json")

    logger.info("========================================================================")
    logger.info("PHASE 6 DAY 47: PHASE 6 GATE VERIFICATION & RELEASE v0.7 WRAP-UP")
    logger.info("========================================================================")
    logger.info("Reports Directory: %s", rep_dir)
    logger.info("Dataset Directory: %s", ds_dir)
    logger.info("Output Markdown:   %s", out_md)
    logger.info("Output JSON:       %s", out_json)
    logger.info("========================================================================")

    summary = run_phase6_wrapup(
        reports_dir=rep_dir,
        dataset_dir=ds_dir,
        output_report=out_md,
        summary_json=out_json,
    )

    # Save provenance metadata
    save_run_info(
        stage="phase6_wrapup",
        output_dir=rep_dir,
        config={
            "verdict": summary.verdict,
            "all_passed": summary.all_passed,
            "release_tag": summary.release_tag,
            "advancing_adapter": summary.advancing_adapter,
            "report_file": str(out_md),
            "summary_json": str(out_json),
        },
        description="Phase 6 Gate Verification & Release v0.7 Wrap-Up",
    )

    # Mirror reports to Google Drive if in Colab environment
    drive_base = Path("/content/drive/MyDrive/math-reasoning-llm")
    if drive_base.exists() or getattr(args, "profile", None) == "colab":
        try:
            drive_reports = drive_base / "reports"
            drive_reports.mkdir(parents=True, exist_ok=True)
            if out_md.exists():
                shutil.copy2(out_md, drive_reports / out_md.name)
            if out_json.exists():
                shutil.copy2(out_json, drive_reports / out_json.name)
            run_info_file = rep_dir / "run_info.json"
            if run_info_file.exists():
                shutil.copy2(run_info_file, drive_reports / run_info_file.name)
            logger.info("Mirrored Phase 6 Gate reports to Google Drive at: %s", drive_reports)
        except Exception as exc:
            logger.warning("Could not mirror reports to Drive: %s", exc)

    # Print summary table
    logger.info("========================================================================")
    logger.info("PHASE 6 GATE CRITERIA VERIFICATION RESULTS")
    logger.info("========================================================================")
    for c in summary.criteria:
        status_str = "[PASS]" if c.passed else "[FAIL]"
        logger.info("%-45s | %s | %s", c.name, status_str, c.observed[:50])
    logger.info("------------------------------------------------------------------------")
    logger.info("Overall Phase 6 Gate Verdict: %s", summary.verdict)
    logger.info("Advancing Policy Adapter:     %s", summary.advancing_adapter)
    logger.info("Certified Release Tag:        %s", summary.release_tag)
    logger.info("Handoff Status:               %s", summary.phase7_handoff.get("status"))
    logger.info("Markdown Report:              %s", out_md)
    logger.info("Summary JSON:                 %s", out_json)
    logger.info("========================================================================")

    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
