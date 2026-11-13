"""Phase 7 Day 49: Full Benchmark Run & Comparative Diagnostics Runner CLI.

Executes comprehensive post-training benchmark diagnostics across the complete model lineup
on the quarantined GSM8K test set (1,319 problems).

Features:
- Enforces Test-Set Wall authorization via the mandatory '--final' flag.
- Problem complexity & reasoning step stratification (Short <=3 steps, Medium 4-5 steps, Complex >=6 steps).
- Error taxonomy classification and error shift matrix (Base 7B -> SFT R1 -> SFT R2 -> Post-GRPO Policy).
- Token efficiency & completion length economy analysis.
- 4-panel publication-grade diagnostic curves generation (reports/day49_final_benchmark_curves.png).
- Emits detailed audit report (Markdown & JSON) and logs execution provenance.

Usage examples:
  # Fast deterministic diagnostics run with mock generator:
  python scripts/run_day49_diagnostics.py --final --use_mock --limit 50

  # Full test benchmark diagnostics with deterministic mock (CI mode):
  python scripts/run_day49_diagnostics.py --final --use_mock

  # Full GPU benchmark run on Colab G4:
  python scripts/run_day49_diagnostics.py --final
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.comparative_diagnostics import (  # noqa: E402
    ComparativeDiagnosticsEngine,
    ComplexityTier,
    generate_day49_markdown_report,
    plot_day49_final_benchmark_curves,
)
from src.evaluation.evaluator import TestSetWallError  # noqa: E402
from src.run_info import save_run_info  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_day49_diagnostics")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 7 Day 49: Full Benchmark Run & Comparative Diagnostics CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--final",
        action="store_true",
        default=False,
        help="Authorization flag required to unfreeze and evaluate the test split.",
    )
    parser.add_argument(
        "--use_mock",
        "--use-mock",
        dest="use_mock",
        action="store_true",
        default=False,
        help="Run using high-fidelity deterministic mock generator (no GPU weights needed).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional problem limit (useful for dry runs and fast local verification).",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="outputs/final_eval",
        help="Destination directory for raw evaluation artifacts.",
    )
    parser.add_argument(
        "--report_path",
        "--report-path",
        dest="report_path",
        type=str,
        default="reports/day49_final_benchmark_report.md",
        help="Destination path for Markdown diagnostics report.",
    )
    parser.add_argument(
        "--summary_json",
        "--summary-json",
        dest="summary_json",
        type=str,
        default="reports/day49_final_benchmark_report.json",
        help="Destination path for JSON summary report.",
    )
    parser.add_argument(
        "--curves_path",
        "--curves-path",
        "--plot_path",
        "--plot-path",
        dest="curves_path",
        type=str,
        default="reports/day49_final_benchmark_curves.png",
        help="Destination path for 4-panel publication diagnostic curves plot.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="local",
        choices=["local", "colab"],
        help="Execution environment profile (default: local).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Hardware execution device target (default: auto).",
    )
    return parser.parse_args()


def main() -> int:
    """Main CLI entrypoint."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("PHASE 7 DAY 49: FULL BENCHMARK RUN & COMPARATIVE DIAGNOSTICS")
    logger.info("=" * 80)

    # 1. Enforce Test-Set Wall Authorization
    if not args.final:
        err_msg = (
            "TEST-SET WALL VIOLATION: Evaluation and comparative diagnostics on the "
            "quarantined GSM8K test set require explicit '--final' authorization flag. "
            "Access denied."
        )
        logger.error(err_msg)
        raise TestSetWallError(err_msg)

    # 2. Initialize Diagnostics Engine
    engine = ComparativeDiagnosticsEngine(
        output_dir=args.output_dir,
        use_mock=args.use_mock,
        seed=42,
    )

    # 3. Execute Full Diagnostics
    logger.info("Running full stratified benchmark diagnostics across model lineup...")
    summary = engine.run_full_diagnostics(limit=args.limit, final=args.final)

    # 4. Generate Reports and Visualizations
    rep_path = Path(args.report_path)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    generate_day49_markdown_report(summary, output_report_path=rep_path)

    json_path = Path(args.summary_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_day49_final_benchmark_curves(summary, output_image_path=args.curves_path)

    # 5. Save Provenance
    try:
        run_info = save_run_info(
            stage="final",
            output_dir=args.output_dir,
            config={
                "day": 49,
                "split": "test",
                "limit": args.limit,
                "use_mock": args.use_mock,
                "num_problems": summary.get("num_problems"),
            },
            description="final_comparative_diagnostics",
        )
        logger.info(f"Saved run provenance to: {run_info}")
    except Exception as exc:
        logger.warning(f"Failed to record run info: {exc}")

    # Mirror reports and curves to Google Drive if in Colab environment
    drive_base = Path("/content/drive/MyDrive/math-reasoning-llm")
    if drive_base.exists() or getattr(args, "profile", None) == "colab":
        try:
            drive_reports = drive_base / "reports"
            drive_reports.mkdir(parents=True, exist_ok=True)
            if rep_path.exists():
                shutil.copy2(rep_path, drive_reports / rep_path.name)
            if json_path.exists():
                shutil.copy2(json_path, drive_reports / json_path.name)
            curves_file = Path(args.curves_path)
            if curves_file.exists():
                shutil.copy2(curves_file, drive_reports / curves_file.name)
            run_info_file = Path(args.output_dir) / "run_info.json"
            if run_info_file.exists():
                shutil.copy2(run_info_file, drive_reports / run_info_file.name)
            logger.info("Mirrored Day 49 diagnostics reports to Google Drive at: %s", drive_reports)
        except Exception as exc:
            logger.warning("Could not mirror reports to Drive: %s", exc)

    # 6. Print Console Summary
    profiles = summary.get("profiles", {})
    print("\n" + "=" * 80)
    print("PHASE 7 DAY 49: FINAL BENCHMARK COMPARATIVE DIAGNOSTICS SUMMARY")
    print("=" * 80)
    print(f"Problems Evaluated: {summary.get('num_problems'):,} (Split: test)")
    print("-" * 80)
    print(f"{'Model':<28} | {'Greedy Acc':<11} | {'Short (<=3)':<12} | {'Med (4-5)':<11} | {'Complex (>=6)':<13}")
    print("-" * 80)

    for m_id, prof in profiles.items():
        name = prof.get("model_name", m_id)
        g_acc = prof.get("overall_greedy_acc", 0.0) * 100.0
        tp = prof.get("tier_performance", {})
        t1 = tp.get(ComplexityTier.TIER_1_SHORT.value, {}).get("greedy_accuracy", 0.0) * 100.0
        t2 = tp.get(ComplexityTier.TIER_2_MEDIUM.value, {}).get("greedy_accuracy", 0.0) * 100.0
        t3 = tp.get(ComplexityTier.TIER_3_COMPLEX.value, {}).get("greedy_accuracy", 0.0) * 100.0
        print(f"{name:<28} | {g_acc:>9.1f}% | {t1:>10.1f}% | {t2:>9.1f}% | {t3:>11.1f}%")

    print("-" * 80)
    reductions = summary.get("error_reductions", {})
    if "grpo_vs_base" in reductions:
        rvb = reductions["grpo_vs_base"]
        print(f"GRPO vs Base 7B Error Reduction:        {rvb.get('total_errors_eliminated', 0):+,} errors ({rvb.get('relative_error_reduction_pct', 0.0):.1f}% relative reduction)")
    if "grpo_vs_sft_targeted" in reductions:
        rvs = reductions["grpo_vs_sft_targeted"]
        print(f"GRPO vs Targeted SFT R2 Error Reduction: {rvs.get('total_errors_eliminated', 0):+,} errors ({rvs.get('relative_error_reduction_pct', 0.0):.1f}% relative reduction)")
    print("=" * 80 + "\n")

    logger.info(f"Markdown diagnostics report written to: {rep_path}")
    logger.info(f"Summary JSON written to:               {json_path}")
    logger.info(f"Curves image destination:              {args.curves_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
