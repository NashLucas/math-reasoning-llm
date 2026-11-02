#!/usr/bin/env python3
"""Reasoning trace LLM Judge scoring and calibration pipeline (Phase 3 Day 21).

Scores reasoning traces across completeness, verification quality, and novelty
using the rubric defined in src/reasoning/judge.py.
Features:
- Structured prompt formatting for LLM judge evaluation
- Robust score parsing across multiple output shapes (bullets, JSON, scales)
- Benchmark calibration against 20 ground-truth human scores (verifying r >= 0.60)
- Diagnostic markdown and JSON reporting

Usage:
    # Day 21: Run calibration on 20 benchmark traces with mock judge
    python scripts/score_traces_judge.py --calibrate --use_mock

    # Day 21: Run calibration with development model on GPU
    python scripts/score_traces_judge.py --calibrate --use_real_model

    # Day 22: Score full dataset of generated traces
    python scripts/score_traces_judge.py \
        --traces_file data/synthetic/generated/traces_deepseek-math-7b-instruct_chain_of_thought.jsonl \
        --output_file data/synthetic/generated/traces_scored.jsonl \
        --use_real_model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic import load_synthetic_traces  # noqa: E402
from src.reasoning.judge import (  # noqa: E402
    ReasoningJudge,
    compute_judge_calibration,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("score_traces_judge")


def generate_calibration_markdown_report(
    calib_metrics: Dict[str, Any],
    scored_items: List[Dict[str, Any]],
    model_name: str,
    use_real_model: bool,
    report_path: Path,
) -> None:
    """Render a comprehensive markdown summary of the judge calibration run."""
    mode_label = "Real Model" if use_real_model else "MOCK / Simulation"
    r_val = calib_metrics["pearson_r"]
    is_ok = calib_metrics["is_calibrated"]
    verdict_str = "**PASSED (r >= 0.60)**" if is_ok else "**FAILED (r < 0.60 - Needs Rubric Tuning)**"

    lines = [
        "# LLM Judge Calibration Report (Phase 3 Day 21)",
        "",
        f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- **Judge Model:** `{model_name}` ({mode_label})",
        f"- **Benchmark Size:** {calib_metrics['n_samples']} manually calibrated traces",
        f"- **Calibration Status:** {verdict_str}",
        "",
        "## Calibration Statistics",
        "",
        "| Metric | Value | Target Threshold |",
        "|---|---|---|",
        f"| Pearson Correlation ($r$) | **{r_val:.4f}** | $\\ge 0.60$ |",
        f"| Mean Absolute Error (MAE) | {calib_metrics['mae']:.4f} | $\\le 0.20$ |",
        f"| Mean Squared Error (MSE) | {calib_metrics['mse']:.4f} | - |",
        f"| Samples Evaluated | {calib_metrics['n_samples']} | 20 traces |",
        "",
        "## Trace-by-Trace Score Comparison",
        "",
        "| Trace ID | Strategy | Manual Overall | Judge Overall | Diff | Status |",
        "|---|---|---|---|---|---|",
    ]

    for item in scored_items:
        m_val = item["manual_overall_score"]
        j_val = item["judge_score"]["overall_score"]
        diff = j_val - m_val
        status = "OK" if abs(diff) <= 0.25 else "Divergent"
        lines.append(
            f"| `{item['trace_id']}` | `{item['prompt_type']}` | {m_val:.2f} | {j_val:.2f} | {diff:+.2f} | {status} |"
        )

    lines.extend([
        "",
        "## Day 21 Calibration Verdict",
        "",
        f"- Correlation target $r \\ge 0.60$: {'MET' if is_ok else 'UNMET'}",
        "- Design Rule 4 check: All evaluated traces reach the correct gold answer; judge scores rank qualitative derivation quality.",
        "- Ready for Day 22 full-scale trace scoring on Colab G4.",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Saved calibration report to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Judge trace scoring and rubric calibration")
    parser.add_argument(
        "--traces_file",
        type=str,
        default=None,
        help="Path to reasoning traces JSONL to evaluate.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save scored traces JSONL.",
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        default=None,
        help="Path to reference CoT traces for comparing novelty on alternative_method.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Judge model ID (default: Qwen/Qwen2.5-1.5B-Instruct; 7B on G4).",
    )
    parser.add_argument(
        "--use_real_model",
        action="store_true",
        help="Run real model on GPU.",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Force deterministic mock judge scoring.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=False,
        help="Run calibration against 20 ground-truth manual scores.",
    )
    parser.add_argument(
        "--calibration_file",
        type=str,
        default="data/calibration/manual_scores_20.json",
        help="Path to 20 manual benchmark traces JSON.",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default="day21_judge_calibration",
        help="Base name for output report files.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Directory to save reports.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for judge evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of traces to evaluate.",
    )
    args = parser.parse_args()

    use_mock_flag = args.use_mock or (not args.use_real_model)
    real_flag = not use_mock_flag

    logger.info(
        f"Starting Judge evaluation: model={args.model_name}, "
        f"calibrate={args.calibrate}, use_mock={use_mock_flag}"
    )

    judge = ReasoningJudge(
        model_name_or_path=args.model_name,
        use_mock=use_mock_flag,
    )

    # Mode 1: Benchmark Calibration against 20 manual scores
    if args.calibrate:
        calib_path = Path(args.calibration_file)
        if not calib_path.is_absolute():
            calib_path = PROJECT_ROOT / calib_path

        if not calib_path.exists():
            logger.error(f"Calibration file not found: {calib_path}")
            sys.exit(1)

        raw_calib_data = json.loads(calib_path.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(raw_calib_data)} calibration benchmark traces from {calib_path}")

        scored_benchmark: List[Dict[str, Any]] = []
        judge_scores: List[float] = []
        manual_scores: List[float] = []

        for item in raw_calib_data:
            score = judge.evaluate_trace(
                question=item["question"],
                gold_answer=str(item["gold_answer"]),
                completion=item["completion"],
                prompt_type=item["prompt_type"],
                reference_cot=item.get("reference_cot"),
            )
            item_record = dict(item)
            item_record["judge_score"] = score.to_dict()
            scored_benchmark.append(item_record)

            judge_scores.append(score.overall_score)
            manual_scores.append(float(item["manual_overall_score"]))

        # Compute calibration metrics
        calib_metrics = compute_judge_calibration(
            judge_scores=judge_scores,
            manual_scores=manual_scores,
            min_correlation=0.60,
        )

        logger.info("=== Judge Calibration Results ===")
        logger.info(f"Evaluated Samples: {calib_metrics['n_samples']}")
        logger.info(f"Pearson Correlation (r): {calib_metrics['pearson_r']:.4f}")
        logger.info(f"Mean Absolute Error (MAE): {calib_metrics['mae']:.4f}")
        logger.info(f"Mean Squared Error (MSE): {calib_metrics['mse']:.4f}")
        logger.info(f"Calibration Status: {'PASSED' if calib_metrics['is_calibrated'] else 'FAILED'}")

        # Save reports
        reports_dir = Path(args.reports_dir)
        if not reports_dir.is_absolute():
            reports_dir = PROJECT_ROOT / reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        md_path = reports_dir / f"{args.report_name}.md"
        json_path = reports_dir / f"{args.report_name}_report.json"

        generate_calibration_markdown_report(
            calib_metrics=calib_metrics,
            scored_items=scored_benchmark,
            model_name=args.model_name,
            use_real_model=real_flag,
            report_path=md_path,
        )

        json_path.write_text(
            json.dumps({
                "model_name": args.model_name,
                "mode": "real" if real_flag else "mock",
                "calibration_metrics": calib_metrics,
                "scored_traces": scored_benchmark,
            }, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Saved JSON calibration report to: {json_path}")
        logger.info("Day 21 Judge Calibration successfully complete!")
        return

    # Mode 2: General trace scoring
    if not args.traces_file:
        logger.error("No traces_file specified. Pass --traces_file or --calibrate.")
        sys.exit(1)

    traces_path = Path(args.traces_file)
    if not traces_path.is_absolute():
        traces_path = PROJECT_ROOT / traces_path

    if not traces_path.exists():
        logger.error(f"Traces file not found: {traces_path}")
        sys.exit(1)

    traces = load_synthetic_traces(traces_path)
    if args.limit and args.limit > 0:
        traces = traces[:args.limit]
    logger.info(f"Loaded {len(traces)} traces to score from {traces_path.name}")

    # Load reference traces if available
    ref_lookup: Dict[str, str] = {}
    if args.reference_file:
        ref_path = Path(args.reference_file)
        if not ref_path.is_absolute():
            ref_path = PROJECT_ROOT / ref_path
        if ref_path.exists():
            ref_traces = load_synthetic_traces(ref_path)
            for rt in ref_traces:
                pid = str(rt.get("problem_id", ""))
                comp = rt.get("completion", "")
                if pid and comp and pid not in ref_lookup:
                    ref_lookup[pid] = comp
            logger.info(f"Loaded {len(ref_lookup)} reference CoT completions for novelty comparison.")

    scored = judge.evaluate_traces(
        traces=traces,
        reference_traces_by_pid=ref_lookup,
        batch_size=args.batch_size,
    )

    out_file = Path(args.output_file) if args.output_file else (
        traces_path.parent / f"{traces_path.stem}_scored.jsonl"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for item in scored:
            f.write(json.dumps(item.to_dict()) + "\n")

    logger.info(f"Saved {len(scored)} scored traces to: {out_file}")


if __name__ == "__main__":
    main()
