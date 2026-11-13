"""Final benchmark evaluation CLI script for mathematical reasoning models.

Executes the comprehensive Phase 7 benchmark across the model lineup on the quarantined
GSM8K test set (1,319 problems).

Features:
- Enforces the Test-Set Wall: requires explicit '--final' flag to unfreeze and evaluate the test split.
- Formally certifies the unfreezing event with cryptographic SHA-256 verification against splits/manifest.json.
- Benchmarks the complete lineup: Base 7B, Instruct 7B, RL 7B, SFT Round 1, SFT Round 2, and Post-GRPO Policy.
- Evaluates across greedy decoding, temperature-sampled rollouts, pass@k, and majority voting @ 8.
- Calculates 95% bootstrap confidence intervals (B=1,000) and pairwise statistical significance deltas.
- Emits publication-quality audit reports (Markdown & JSON) and logs execution provenance.

Usage examples:
  # Fast deterministic verification on test split:
  python scripts/evaluate_final.py --final --use_mock --limit 50

  # Full test benchmark with deterministic mock (CI mode):
  python scripts/evaluate_final.py --final --use_mock

  # Production full GPU run (Colab / G4):
  python scripts/evaluate_final.py --final
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import TestSetWallError  # noqa: E402
from src.evaluation.final_evaluator import (  # noqa: E402
    DEFAULT_BENCHMARK_LINEUP,
    FinalBenchmarkEvaluator,
    verify_and_unfreeze_test_set,
)
from src.run_info import save_run_info  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_final")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 7: Final Benchmark Evaluation CLI on GSM8K Test Split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--final",
        action="store_true",
        default=False,
        help="Authorization flag required to unfreeze and evaluate the quarantined test set.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model IDs to evaluate ('all', 'lineup', or specific IDs like 'base_7b', 'grpo_policy').",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["greedy", "sampled", "self_consistency"],
        choices=["greedy", "sampled", "self_consistency"],
        help="Decoding regimes to benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of problems evaluated (useful for fast verification).",
    )
    parser.add_argument(
        "--bootstrap_resamples",
        "--bootstrap-resamples",
        dest="bootstrap_resamples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for 95% confidence intervals.",
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
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="outputs/final_eval",
        help="Directory to save raw evaluation records and summaries.",
    )
    parser.add_argument(
        "--report_path",
        "--report-path",
        dest="report_path",
        type=str,
        default="reports/day48_evaluation_infrastructure_report.md",
        help="Destination markdown report path.",
    )
    parser.add_argument(
        "--summary_json",
        "--summary-json",
        dest="summary_json",
        type=str,
        default="reports/day48_evaluation_infrastructure_report.json",
        help="Destination summary JSON path.",
    )
    parser.add_argument(
        "--unfreeze_manifest",
        "--unfreeze-manifest",
        dest="unfreeze_manifest",
        type=str,
        default="reports/day48_test_unfreeze_manifest.json",
        help="Destination test unfreeze audit manifest path.",
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
    logger.info("PHASE 7 DAY 48: FINAL BENCHMARK EVALUATION INFRASTRUCTURE & TEST UNFREEZING")
    logger.info("=" * 80)

    # 1. Enforce Test-Set Wall Authorization
    if not args.final:
        err_msg = (
            "TEST-SET WALL VIOLATION: Evaluation on the quarantined GSM8K test set strictly "
            "requires the explicit '--final' authorization flag. Test access denied."
        )
        logger.error(err_msg)
        raise TestSetWallError(err_msg)

    # 2. Formally unfreeze and verify test set
    logger.info("Executing Test-Set Wall unfreezing and cryptographic checksum audit...")
    unfreeze_manifest = verify_and_unfreeze_test_set(
        output_manifest_path=args.unfreeze_manifest,
        authorized_token="PHASE_7_BENCHMARK_UNFREEZE",
    )
    logger.info(
        f"Test set verified: {unfreeze_manifest.problem_count} problems. "
        f"IDs SHA-256: {unfreeze_manifest.ids_sha256[:16]}... (Match: {unfreeze_manifest.ids_match})"
    )

    # 3. Initialize Evaluator
    evaluator = FinalBenchmarkEvaluator(
        output_dir=args.output_dir,
        use_mock=args.use_mock,
        seed=42,
    )
    evaluator.unfreeze_manifest = unfreeze_manifest

    # 4. Run Benchmark Suite
    models_to_eval: List[str] = args.models
    if "all" in models_to_eval or "lineup" in models_to_eval:
        models_to_eval = [m.id for m in DEFAULT_BENCHMARK_LINEUP]

    logger.info(f"Running benchmark on models: {models_to_eval}")
    logger.info(f"Regimes: {args.regimes}")

    summary = evaluator.run_benchmark_suite(
        models=models_to_eval,
        regimes=args.regimes,
        limit=args.limit,
        final=args.final,
    )

    # 5. Emit Markdown Report and JSON Summary
    rep_path = Path(args.report_path)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.generate_markdown_report(summary, output_report_path=rep_path)

    json_path = Path(args.summary_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 6. Save Run Info Provenance
    try:
        run_info = save_run_info(
            stage="final",
            output_dir=args.output_dir,
            config={
                "models": models_to_eval,
                "regimes": args.regimes,
                "limit": args.limit,
                "use_mock": args.use_mock,
                "num_problems": summary.get("num_problems"),
                "unfreeze_status": unfreeze_manifest.authorization_status,
            },
            description="benchmark_evaluation",
        )
        logger.info(f"Saved run provenance to: {run_info}")
    except Exception as exc:
        logger.warning(f"Failed to record run info: {exc}")

    # 7. Mirror reports to Google Drive if in Colab environment
    drive_base = Path("/content/drive/MyDrive/math-reasoning-llm")
    if drive_base.exists() or getattr(args, "profile", None) == "colab":
        try:
            drive_reports = drive_base / "reports"
            drive_reports.mkdir(parents=True, exist_ok=True)
            if rep_path.exists():
                shutil.copy2(rep_path, drive_reports / rep_path.name)
            if json_path.exists():
                shutil.copy2(json_path, drive_reports / json_path.name)
            unfreeze_file = Path(args.unfreeze_manifest)
            if unfreeze_file.exists():
                shutil.copy2(unfreeze_file, drive_reports / unfreeze_file.name)
            run_info_file = Path(args.output_dir) / "run_info.json"
            if run_info_file.exists():
                shutil.copy2(run_info_file, drive_reports / run_info_file.name)
            logger.info("Mirrored Day 48 evaluation reports to Google Drive at: %s", drive_reports)
        except Exception as exc:
            logger.warning("Could not mirror reports to Drive: %s", exc)

    # 8. Print Console Summary
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK EVALUATION SUMMARY (GSM8K TEST SPLIT)")
    print("=" * 80)
    print(f"Problems Evaluated: {summary.get('num_problems'):,}")
    print(f"Unfreeze Status:    {unfreeze_manifest.authorization_status} ({unfreeze_manifest.dataset_path})")
    print("-" * 80)
    print(f"{'Model':<30} | {'Greedy Acc':<12} | {'pass@8':<10} | {'maj@8':<10} | {'Format':<10}")
    print("-" * 80)

    for m_id, m_data in summary.get("results", {}).items():
        g_acc = m_data.get("greedy", {}).get("accuracy", 0.0) * 100.0
        p8 = m_data.get("sampled", {}).get("pass_at_k", {}).get("pass_at_8", 0.0) * 100.0
        maj8 = m_data.get("self_consistency", {}).get("maj_at_8")
        maj8_str = f"{maj8 * 100.0:.1f}%" if maj8 is not None else "—"
        fmt = m_data.get("greedy", {}).get("format_compliance", 1.0) * 100.0
        print(f"{m_id:<30} | {g_acc:>10.1f}% | {p8:>8.1f}% | {maj8_str:>8} | {fmt:>8.1f}%")

    print("-" * 80)
    deltas = summary.get("statistical_deltas", {})
    if "grpo_vs_base_greedy" in deltas:
        d = deltas["grpo_vs_base_greedy"]
        print(f"GRPO vs Base Greedy Delta:      {d['delta']*100:+.2f}% (95% CI: [{d['ci_lower']*100:+.2f}%, {d['ci_upper']*100:+.2f}%], p={d['p_value']:.4f})")
    if "grpo_vs_sft2_greedy" in deltas:
        d = deltas["grpo_vs_sft2_greedy"]
        print(f"GRPO vs SFT Round 2 Greedy Delta: {d['delta']*100:+.2f}% (95% CI: [{d['ci_lower']*100:+.2f}%, {d['ci_upper']*100:+.2f}%], p={d['p_value']:.4f})")
    print("=" * 80 + "\n")

    logger.info(f"Markdown audit report written to: {rep_path}")
    logger.info(f"Summary JSON written to:          {json_path}")
    logger.info(f"Unfreeze manifest written to:      {args.unfreeze_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
