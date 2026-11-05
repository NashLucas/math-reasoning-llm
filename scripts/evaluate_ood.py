#!/usr/bin/env python3
"""Phase 8 Day 51: Out-of-Distribution Mathematical Generalization CLI Runner.

Evaluates the Post-GRPO Policy model on out-of-distribution reasoning archetypes:
1. Multi-Rate & Combined Work.
2. Linguistic & Structural Perturbations (SVAMP-style).
3. Compound Financial & Multi-Tier Percentages.
4. Algebraic Systems in Natural Language.
5. Geometric & Spatial Arithmetic.

Emits:
- reports/day51_ood_generalization_report.md
- reports/day51_ood_generalization.json

Usage:
    python scripts/evaluate_ood.py
    python scripts/evaluate_ood.py --output_dir reports --use_mock
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

from src.evaluation.ood_evaluator import run_ood_evaluation  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_ood")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate out-of-distribution mathematical reasoning generalizability."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Directory to save report outputs",
    )
    parser.add_argument(
        "--output_report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "day51_ood_generalization_report.md",
        help="Path where markdown report will be written",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "day51_ood_generalization.json",
        help="Path where JSON summary will be written",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="outputs/grpo/adapter_model",
        help="Path to post-GRPO policy adapter",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        default=True,
        help="Use deterministic mock reasoning engine",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress banner and results output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("========================================================================")
    logger.info("PHASE 8 DAY 51: OUT-OF-DISTRIBUTION REASONING GENERALIZATION BENCHMARK")
    logger.info("========================================================================")
    logger.info("Reports Directory:   %s", args.output_dir)
    logger.info("Advancing Adapter:   %s", args.adapter_path)
    logger.info("Target Markdown:     %s", args.output_report)
    logger.info("Target JSON Summary: %s", args.output_json)
    logger.info("Backend Mode:        %s", "Deterministic Mock" if args.use_mock else "CUDA GPU")
    logger.info("========================================================================")

    try:
        report = run_ood_evaluation(
            output_dir=args.output_dir,
            output_report_path=args.output_report,
            output_json_path=args.output_json,
            use_mock=args.use_mock,
            adapter_path=args.adapter_path,
        )
    except Exception as err:
        logger.error("OOD Evaluation failed with exception: %s", err, exc_info=True)
        return 1

    if not args.quiet:
        print("\n" + "=" * 80)
        print("PHASE 8 DAY 51: OOD MATHEMATICAL GENERALIZATION SUMMARY")
        print("=" * 80)
        print(f"Timestamp:              {report.timestamp}")
        print(f"Target Model:           {report.model_name}")
        print(f"Advancing Adapter:      {report.adapter_path}")
        print(f"Total OOD Problems:     {report.total_problems}")
        print(f"Overall Greedy Acc:     {report.overall_greedy_acc * 100:.1f}%")
        print(f"95% Bootstrap CI:       [{report.overall_greedy_ci.get('ci_lower', 0)*100:.1f}%, {report.overall_greedy_ci.get('ci_upper', 0)*100:.1f}%]")
        print(f"Answer Contract:        {report.overall_format_compliance * 100:.1f}%")
        print(f"Mean Serving Latency:   {report.overall_mean_latency_ms:.2f} ms")
        print(f"Overall Throughput:     {report.overall_throughput_tok_s:.1f} tok/s")
        print("-" * 80)
        print("Stratified Archetype Breakdown:")
        for cat, m in report.category_metrics.items():
            cat_name = cat.replace("_", " ").title()
            print(f"  - {cat_name:<32} Acc: {m.accuracy*100:>5.1f}% | Contract: {m.format_compliance*100:>5.1f}% | {m.mean_latency_ms:>5.1f} ms")
        print("=" * 80)
        print("Generated Artifacts:")
        print(f"  Markdown: {args.output_report}")
        print(f"  JSON:     {args.output_json}")
        print("=" * 80 + "\n")

    return 0 if (report.overall_greedy_acc >= 0.70 and report.overall_format_compliance >= 0.98) else 1


if __name__ == "__main__":
    sys.exit(main())
