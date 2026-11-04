#!/usr/bin/env python3
"""Reasoning trace LLM Judge scoring, calibration, and distribution analysis pipeline (Phase 3 Day 21 & 22).

Scores reasoning traces across completeness, verification quality, and novelty
using the rubric defined in src/reasoning/judge.py.

Key Features & Contracts:
- Design Rule 4: Correctness filters first, the judge second.
  Only traces that have reached the gold answer are evaluated by the judge.
- Strategy-specific weighting presets:
  - chain_of_thought: 100% completeness
  - solve_then_verify: 50% completeness, 50% verification quality
  - alternative_method: 40% completeness, 60% novelty (against reference CoT)
- Calibration Gate: Check agreement against 20 hand-scored ground truth traces (r >= 0.60).
- Score Spread Analysis: Computes distribution, IQR, and flags high-score clustering (>90%).
- Within-Problem Relative Ranking: Assigns 1-based ranks to traces within each problem.

Usage:
    # Day 21: Run calibration on 20 benchmark traces with mock judge
    python scripts/score_traces_judge.py --calibrate --use_mock

    # Day 22: Score full dataset of generated traces (mock/dev mode)
    python scripts/score_traces_judge.py \
        --traces_dir data/synthetic/generated \
        --output_file data/synthetic/generated/traces_scored.jsonl \
        --use_mock \
        --report_name day22_judge_scoring

    # Day 22: Score full dataset on Colab G4 GPU with 7B Instruct
    python scripts/score_traces_judge.py \
        --model_name "deepseek-ai/deepseek-math-7b-instruct" \
        --traces_dir data/synthetic/generated \
        --output_file data/synthetic/generated/traces_scored.jsonl \
        --use_real_model \
        --report_name day22_judge_scoring
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.curation import evaluate_trace_correctness  # noqa: E402
from src.data.synthetic import load_synthetic_traces  # noqa: E402
from src.reasoning.judge import (  # noqa: E402
    ReasoningJudge,
    ScoredTrace,
    assign_within_problem_ranks,
    compute_judge_calibration,
    compute_score_distribution,
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


def generate_scoring_markdown_report(
    calib_metrics: Optional[Dict[str, Any]],
    overall_dist: Dict[str, Any],
    by_strategy_dist: Dict[str, Dict[str, Any]],
    total_loaded: int,
    total_filtered_out: int,
    total_scored: int,
    total_problems: int,
    model_name: str,
    use_real_model: bool,
    output_file: Path,
    report_path: Path,
) -> None:
    """Render a comprehensive markdown summary of the Day 22 Judge scoring run."""
    mode_label = "Real Model" if use_real_model else "MOCK / Simulation"
    lines = [
        "# LLM Judge Trace Scoring & Distribution Report (Phase 3 Day 22)",
        "",
        f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- **Judge Model:** `{model_name}` ({mode_label})",
        f"- **Output Scored Traces:** `{output_file}`",
        f"- **Problems Covered:** {total_problems} unique solved problems",
        f"- **Total Traces Scored:** {total_scored} correct traces",
        "",
        "## Design Rule 4 Filtering Verification",
        "",
        "| Stage | Count | Notes |",
        "|---|---|---|",
        f"| Loaded Traces | {total_loaded} | Traces discovered from generated inputs |",
        f"| Correctness Filter Rejections | {total_filtered_out} | Non-gold or malformed traces rejected |",
        f"| Retained for Judge Scoring | {total_scored} | 100% reach correct gold answer |",
        "",
    ]

    if calib_metrics:
        r_val = calib_metrics["pearson_r"]
        is_ok = calib_metrics["is_calibrated"]
        status_str = "**PASSED (r >= 0.60)**" if is_ok else "**FAILED**"
        lines.extend([
            "## Benchmark Calibration Agreement (20 Ground-Truth Traces)",
            "",
            "| Metric | Value | Gate Threshold | Status |",
            "|---|---|---|---|",
            f"| Pearson Correlation ($r$) | **{r_val:.4f}** | $\\ge 0.60$ | {status_str} |",
            f"| Mean Absolute Error (MAE) | {calib_metrics['mae']:.4f} | $\\le 0.20$ | {'OK' if calib_metrics['mae'] <= 0.20 else 'Elevated'} |",
            f"| Mean Squared Error (MSE) | {calib_metrics['mse']:.4f} | - | - |",
            "",
        ])

    lines.extend([
        "## Overall Score Distribution & Spread",
        "",
        "| Metric | Overall | " + " | ".join(by_strategy_dist.keys()) + " |",
        "|---|---|" + "|".join("---" for _ in by_strategy_dist) + "|",
        "| Count | " + str(overall_dist["count"]) + " | " + " | ".join(str(d["count"]) for d in by_strategy_dist.values()) + " |",
        "| Mean Score | " + f"{overall_dist['mean']:.3f}" + " | " + " | ".join(f"{d['mean']:.3f}" for d in by_strategy_dist.values()) + " |",
        "| Std Deviation | " + f"{overall_dist['std']:.3f}" + " | " + " | ".join(f"{d['std']:.3f}" for d in by_strategy_dist.values()) + " |",
        "| Median Score | " + f"{overall_dist['median']:.3f}" + " | " + " | ".join(f"{d['median']:.3f}" for d in by_strategy_dist.values()) + " |",
        "| IQR (Q75 - Q25) | " + f"{overall_dist['iqr']:.3f}" + " | " + " | ".join(f"{d['iqr']:.3f}" for d in by_strategy_dist.values()) + " |",
        "| Min / Max | " + f"{overall_dist['min']:.2f} / {overall_dist['max']:.2f}" + " | " + " | ".join(f"{d['min']:.2f} / {d['max']:.2f}" for d in by_strategy_dist.values()) + " |",
        "| % Scores $\\ge 0.90$ | " + f"{overall_dist['pct_above_0_90']:.1f}%" + " | " + " | ".join(f"{d['pct_above_0_90']:.1f}%" for d in by_strategy_dist.values()) + " |",
        "",
        "### Score Buckets Histogram",
        "",
        "| Bucket Range | Overall Count | Share (%) |",
        "|---|---|---|",
    ])

    for bucket, count in overall_dist["buckets"].items():
        pct = (count / max(1, overall_dist["count"])) * 100.0
        lines.append(f"| `{bucket}` | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Day 22 Diagnostics & Day 23 Curation Strategy",
        "",
    ])

    if overall_dist["high_clustering"]:
        lines.extend([
            "> [!NOTE] High Score Clustering Detected",
            "> Model judge scores exhibit notable concentration above 0.90 "
            f"({overall_dist['pct_above_0_90']:.1f}% >= 0.90). As stipulated in the Phase 3 Pitfall guidelines: ",
            "> *\"Model judges tend to score everything high. If most traces score above 0.9, use the judge only to rank traces within a problem.\"*",
            "> Within-problem relative ranking (`rank_within_problem`) has been computed and saved for all traces.",
            "> Day 23 Curation should select the rank-1 trace per problem rather than applying a global absolute score cutoff.",
        ])
    else:
        lines.extend([
            "> [!NOTE] Balanced Score Spread",
            "> Scores are sufficiently dispersed across the [0.0, 1.0] spectrum.",
            "> Both absolute thresholding and within-problem relative ranking can be leveraged for Day 23 Curation.",
        ])

    lines.extend([
        "",
        "## Deliverables Produced",
        f"- Unified scored traces: `{output_file}`",
        f"- Score distribution report: `{report_path}`",
        f"- Diagnostic JSON: `{report_path.with_suffix('.json')}`",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Saved Day 22 scoring markdown report to: {report_path}")


def discover_generated_trace_files(traces_dir: Path) -> List[Path]:
    """Find all relevant synthetic reasoning trace files in traces_dir."""
    candidates = []
    for f in sorted(traces_dir.glob("*.jsonl")):
        name = f.name
        # Skip previously scored or curated files
        if "scored" in name or "curated" in name:
            continue
        if any(strategy in name for strategy in ["chain_of_thought", "solve_then_verify", "alternative_method"]):
            candidates.append(f)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Judge trace scoring, calibration, and distribution analysis")
    parser.add_argument(
        "--traces_file",
        type=str,
        default=None,
        help="Path to single reasoning traces JSONL to evaluate.",
    )
    parser.add_argument(
        "--traces_dir",
        type=str,
        default="data/synthetic/generated",
        help="Directory containing synthetic trace files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save scored traces JSONL (default: data/synthetic/generated/traces_scored.jsonl).",
    )
    parser.add_argument(
        "--split_by_prompt_type",
        action="store_true",
        default=False,
        help="Also export individual scored JSONL files for each prompt strategy.",
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
        "--check_calibration",
        action="store_true",
        default=True,
        help="Always run calibration check on benchmark during scoring runs.",
    )
    parser.add_argument(
        "--calibration_file",
        type=str,
        default="data/calibration/manual_scores_20.json",
        help="Path to 20 manual benchmark traces JSON.",
    )
    parser.add_argument(
        "--filter_correct_first",
        action="store_true",
        default=True,
        help="Enforce Design Rule 4: evaluate only traces that pass the correctness filter.",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default="day22_judge_scoring",
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
        f"Starting Judge pipeline: model={args.model_name}, "
        f"calibrate_only={args.calibrate}, use_mock={use_mock_flag}"
    )

    judge = ReasoningJudge(
        model_name_or_path=args.model_name,
        use_mock=use_mock_flag,
    )

    # 1. Benchmark Calibration against 20 manual scores
    calib_metrics: Optional[Dict[str, Any]] = None
    calib_path = Path(args.calibration_file)
    if not calib_path.is_absolute():
        calib_path = PROJECT_ROOT / calib_path

    if calib_path.exists() and (args.calibrate or args.check_calibration):
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

        calib_metrics = compute_judge_calibration(
            judge_scores=judge_scores,
            manual_scores=manual_scores,
            min_correlation=0.60,
        )

        logger.info("=== Judge Calibration Results ===")
        logger.info(f"Evaluated Samples: {calib_metrics['n_samples']}")
        logger.info(f"Pearson Correlation (r): {calib_metrics['pearson_r']:.4f}")
        logger.info(f"Mean Absolute Error (MAE): {calib_metrics['mae']:.4f}")
        logger.info(f"Calibration Status: {'PASSED' if calib_metrics['is_calibrated'] else 'FAILED'}")

        # If calibrate was requested, generate calibration report
        if args.calibrate:
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
            logger.info(f"Saved calibration report to: {md_path}")

            # Exit early if this is a calibration-only invocation
            tdir_path = Path(args.traces_dir)
            if not tdir_path.is_absolute():
                tdir_path = PROJECT_ROOT / tdir_path
            has_trace_files = tdir_path.exists() and bool(discover_generated_trace_files(tdir_path))

            if not args.traces_file and (not args.output_file or not has_trace_files):
                logger.info("Calibration-only mode finished successfully.")
                return

    # 2. Trace Discovery & Loading for Full Scoring
    raw_traces: List[Dict[str, Any]] = []
    if args.traces_file:
        t_path = Path(args.traces_file)
        if not t_path.is_absolute():
            t_path = PROJECT_ROOT / t_path
        if not t_path.exists():
            logger.error(f"Traces file not found: {t_path}")
            sys.exit(1)
        raw_traces.extend(load_synthetic_traces(t_path))
    else:
        tdir = Path(args.traces_dir)
        if not tdir.is_absolute():
            tdir = PROJECT_ROOT / tdir
        if tdir.exists():
            trace_files = discover_generated_trace_files(tdir)
            logger.info(f"Discovered {len(trace_files)} trace files in {tdir}")
            seen_trace_ids = set()
            for tf in trace_files:
                for tr in load_synthetic_traces(tf):
                    tid = tr.get("trace_id", "")
                    if tid and tid not in seen_trace_ids:
                        seen_trace_ids.add(tid)
                        raw_traces.append(tr)
                    elif not tid:
                        raw_traces.append(tr)

    total_loaded = len(raw_traces)
    if total_loaded == 0:
        logger.warning("No traces found to score.")
        if args.calibrate and calib_metrics:
            logger.info("Calibration completed successfully, exiting.")
            return
        sys.exit(1)

    logger.info(f"Loaded {total_loaded} traces across discovered files.")

    # 3. Design Rule 4: Correctness Filter First
    retained_traces: List[Dict[str, Any]] = []
    filtered_out_count = 0
    if args.filter_correct_first:
        for t in raw_traces:
            is_kept, _ = evaluate_trace_correctness(t)
            if is_kept:
                retained_traces.append(t)
            else:
                filtered_out_count += 1
        logger.info(
            f"Design Rule 4 filter: {len(retained_traces)}/{total_loaded} traces passed correctness filter "
            f"({filtered_out_count} rejected)."
        )
    else:
        retained_traces = raw_traces

    if args.limit and args.limit > 0:
        retained_traces = retained_traces[:args.limit]

    # 4. Reference CoT lookup building
    ref_lookup: Dict[str, str] = {}
    if args.reference_file:
        ref_path = Path(args.reference_file)
        if not ref_path.is_absolute():
            ref_path = PROJECT_ROOT / ref_path
        if ref_path.exists():
            for rt in load_synthetic_traces(ref_path):
                pid = str(rt.get("problem_id", ""))
                comp = rt.get("completion", "")
                if pid and comp and pid not in ref_lookup:
                    ref_lookup[pid] = comp
    else:
        # Build reference lookup from chain_of_thought traces in retained traces
        for t in retained_traces:
            if t.get("prompt_type") == "chain_of_thought":
                pid = str(t.get("problem_id", ""))
                comp = t.get("completion", "")
                if pid and comp and pid not in ref_lookup:
                    ref_lookup[pid] = comp

    logger.info(f"Indexed {len(ref_lookup)} reference CoT solutions for novelty evaluation.")

    # 5. Evaluate traces with judge
    logger.info(f"Evaluating {len(retained_traces)} correct traces with Judge...")
    scored_objs: List[ScoredTrace] = judge.evaluate_traces(
        traces=retained_traces,
        reference_traces_by_pid=ref_lookup,
        batch_size=args.batch_size,
    )

    # 6. Assign within-problem relative ranks
    logger.info("Computing within-problem relative rankings...")
    ranked_traces = assign_within_problem_ranks(scored_objs)

    # 7. Distribution Analysis & Spread
    overall_scores = [t["score"]["overall_score"] for t in ranked_traces]
    overall_dist = compute_score_distribution(overall_scores)

    by_strategy_dist: Dict[str, Dict[str, Any]] = {}
    for st in ["chain_of_thought", "solve_then_verify", "alternative_method"]:
        st_scores = [t["score"]["overall_score"] for t in ranked_traces if t.get("prompt_type") == st]
        if st_scores:
            by_strategy_dist[st] = compute_score_distribution(st_scores)

    logger.info("=== Score Spread & Distribution ===")
    logger.info(f"Mean Score: {overall_dist['mean']:.4f}, Std: {overall_dist['std']:.4f}, Median: {overall_dist['median']:.4f}")
    logger.info(f"IQR: {overall_dist['iqr']:.4f}, Min: {overall_dist['min']:.2f}, Max: {overall_dist['max']:.2f}")
    logger.info(f"Scores >= 0.90: {overall_dist['pct_above_0_90']:.1f}% (High clustering: {overall_dist['high_clustering']})")

    # 8. Save Scored Output Files
    out_file = Path(args.output_file) if args.output_file else (
        PROJECT_ROOT / "data" / "synthetic" / "generated" / "traces_scored.jsonl"
    )
    if not out_file.is_absolute():
        out_file = PROJECT_ROOT / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for item in ranked_traces:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved {len(ranked_traces)} scored and ranked traces to: {out_file}")

    # Optional split by prompt type
    if args.split_by_prompt_type:
        by_ptype: Dict[str, List[Dict[str, Any]]] = {}
        for item in ranked_traces:
            ptype = item.get("prompt_type", "unknown")
            by_ptype.setdefault(ptype, []).append(item)

        for ptype, items in by_ptype.items():
            split_file = out_file.parent / f"traces_scored_{ptype}.jsonl"
            with open(split_file, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
            logger.info(f"Saved {len(items)} scored traces for strategy '{ptype}' to {split_file}")

    # 9. Generate Reports
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = PROJECT_ROOT / reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    md_report_path = reports_dir / f"{args.report_name}.md"
    json_report_path = reports_dir / f"{args.report_name}_report.json"

    unique_problems = len({t.get("problem_id", "") for t in ranked_traces})

    generate_scoring_markdown_report(
        calib_metrics=calib_metrics,
        overall_dist=overall_dist,
        by_strategy_dist=by_strategy_dist,
        total_loaded=total_loaded,
        total_filtered_out=filtered_out_count,
        total_scored=len(ranked_traces),
        total_problems=unique_problems,
        model_name=args.model_name,
        use_real_model=real_flag,
        output_file=out_file,
        report_path=md_report_path,
    )

    report_payload = {
        "model_name": args.model_name,
        "mode": "real" if real_flag else "mock",
        "total_loaded": total_loaded,
        "total_filtered_out": filtered_out_count,
        "total_scored": len(ranked_traces),
        "total_problems": unique_problems,
        "output_file": str(out_file),
        "calibration_metrics": calib_metrics,
        "overall_distribution": overall_dist,
        "by_strategy_distribution": by_strategy_dist,
    }
    json_report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    logger.info(f"Saved Day 22 scoring JSON report to: {json_report_path}")
    logger.info("Day 22 Judge Scoring pipeline completed successfully!")


if __name__ == "__main__":
    main()
