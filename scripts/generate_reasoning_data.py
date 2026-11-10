#!/usr/bin/env python3
"""Synthetic reasoning data generation and multi-strategy sampling pipeline (Phase 3).

Generates reasoning traces across problems in the 3,000-problem training subset
using various prompt strategies (chain_of_thought, solve_then_verify, alternative_method).
Features:
- Resumable chunked generation skipping already-finished traces
- Left-padded batched inference for high throughput
- Deterministic mock generation mode for rapid development and testing
- Integration with CorrectnessFilter to discard incorrect or degenerate traces
- Support for filtering to solved CoT problems for Day 20 verification & alternative traces
- Individual prompt-type slicing for downstream judge evaluations (Day 21)

Usage:
    # Day 17: Development baseline on 100 problems
    python scripts/generate_reasoning_data.py --limit 100 --samples_per_problem 4

    # Day 18: 7B Instruct CoT sampling on G4
    python scripts/generate_reasoning_data.py \
        --model_name "deepseek-ai/deepseek-math-7b-instruct" \
        --prompt_types "chain_of_thought" \
        --samples_per_problem 4 \
        --use_real_model

    # Day 20: Solve-then-verify and alternative-method traces on solved problems
    python scripts/generate_reasoning_data.py \
        --prompt_types "solve_then_verify,alternative_method" \
        --samples_per_problem 1 \
        --only_solved \
        --report_name "day20_multi_strategy_traces"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.curation import (  # noqa: E402
    CorrectnessFilter,
    CurationFilterStats,
    filter_problems_by_pass_rate,
)
from src.data.synthetic import (  # noqa: E402
    TraceSampler,
    compute_trace_yield,
    load_synthetic_problems,
    load_synthetic_traces,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_markdown_report(
    yield_metrics: dict,
    model_name: str,
    prompt_types: list[str],
    samples_per_problem: int,
    output_jsonl: Path,
    report_path: Path,
    use_real_model: bool,
    filter_stats: Optional[CurationFilterStats] = None,
    report_title: Optional[str] = None,
    per_strategy_breakdown: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Generate Markdown summary of the reasoning data generation run."""
    mode_label = "Real Model" if use_real_model else "MOCK / Simulation"
    verdict = "PASSED" if use_real_model else "MOCK (Plumbing Verified; Real Model Run Pending)"
    title = report_title or "Reasoning Data Generation Report"

    lines = [
        f"# {title}",
        "",
        f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- **Teacher Model:** `{model_name}` ({mode_label})",
        f"- **Prompt Strategies:** {', '.join(f'`{p}`' for p in prompt_types)}",
        f"- **Samples per Problem:** {samples_per_problem}",
        f"- **Traces Output File:** [`{output_jsonl.name}`]({output_jsonl.as_posix()})",
        "",
    ]

    if not use_real_model:
        lines.extend([
            "> [!WARNING]",
            "> **Mock run.** The mock generator writes `#### <gold answer>` into every completion, so the",
            "> 100% accuracy and format compliance numbers below verify the pipeline plumbing only.",
            "> Run with `--use_real_model` on a Colab GPU to measure real yield.",
            "",
        ])

    lines.extend([
        "## Overall Yield & Quality Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Problems Sampled | {yield_metrics['total_problems']:,} |",
        f"| Total Traces Generated | {yield_metrics['total_traces']:,} |",
        f"| Format Compliance Rate | {yield_metrics['format_compliance_percent']:.1f}% ({yield_metrics['format_compliance_rate']:.4f}) |",
        f"| Correct Traces Count | {yield_metrics['correct_traces']:,} |",
        f"| Trace Accuracy | {yield_metrics['trace_accuracy_percent']:.1f}% ({yield_metrics['trace_accuracy']:.4f}) |",
        f"| Problems with $\\ge 1$ Correct Trace | {yield_metrics['problems_with_at_least_one_correct']:,} / {yield_metrics['total_problems']:,} ({yield_metrics['problem_coverage_rate'] * 100:.1f}%) |",
        "",
    ])

    if per_strategy_breakdown:
        lines.extend([
            "## Per-Strategy Breakdown",
            "",
            "| Strategy | Traces | Format Compliance | Correct Count | Accuracy (%) | Kept by Filter |",
            "|---|---|---|---|---|---|",
        ])
        for ptype, pmetrics in per_strategy_breakdown.items():
            lines.append(
                f"| `{ptype}` | {pmetrics['total_traces']:,} | "
                f"{pmetrics['format_compliance_percent']:.1f}% | "
                f"{pmetrics['correct_traces']:,} | "
                f"{pmetrics['trace_accuracy_percent']:.1f}% | "
                f"{pmetrics.get('kept_count', pmetrics['correct_traces']):,} |"
            )
        lines.append("")

    if filter_stats is not None:
        lines.extend([
            "## Curation Filter Results",
            "",
            "| Curation Metric | Value |",
            "|---|---|",
            f"| Evaluated Traces | {filter_stats.total_evaluated:,} |",
            f"| Kept (Correct & Clean) | {filter_stats.kept_count:,} ({filter_stats.retention_percent:.1f}%) |",
            f"| Rejected Count | {filter_stats.rejected_count:,} |",
            "",
            "### Rejection Breakdown",
            "",
        ])
        if filter_stats.rejection_breakdown:
            lines.extend([
                "| Reason | Count | Share (%) |",
                "|---|---|---|",
            ])
            rejected_total = filter_stats.rejected_count
            for reason, count in filter_stats.rejection_breakdown.items():
                share = (count / rejected_total * 100.0) if rejected_total > 0 else 0.0
                lines.append(f"| `{reason}` | {count:,} | {share:.1f}% |")
        else:
            lines.append("No traces rejected (100% retention).")

    lines.extend([
        "",
        "## Pass Rate Distribution",
        "",
        "| Bucket | Problem Count | Share (%) |",
        "|---|---|---|",
    ])

    total_probs = yield_metrics["total_problems"]
    for bucket, count in yield_metrics.get("pass_rate_distribution", {}).items():
        pct = (count / total_probs * 100.0) if total_probs > 0 else 0.0
        lines.append(f"| {bucket} | {count:,} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Status & Next Steps",
        "",
        f"**Status:** {verdict}",
        "- Records contain prompt, completion, parsed answer, extraction method, correctness flag, token counts, and timestamp.",
    ])
    if use_real_model:
        lines.append("- Traces verified and ready for Day 21 Judge scoring and SFT curation.")
    else:
        lines.append("- Mock run verified: run `--use_real_model` on Colab G4 for real generation.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Saved Markdown report to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning traces")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of problems to sample from the training subset (default: 100; 0 for all 3,000)",
    )
    parser.add_argument(
        "--samples_per_problem",
        type=int,
        default=4,
        help="Number of reasoning samples per problem (default: 4)",
    )
    parser.add_argument(
        "--prompt_types",
        type=str,
        default="chain_of_thought",
        help="Comma-separated prompt types (default: chain_of_thought)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Teacher model ID (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "hf"],
        help="Generation backend (default: auto)",
    )
    parser.add_argument(
        "--use_real_model",
        action="store_true",
        help="Run real model on GPU (e.g. on Google Colab T4/L4/G4)",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Force mock generation mode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save generated traces (default: profile's data/synthetic/generated on Drive or local)",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default=None,
        help="Base name for output report files (default: auto)",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Directory to save reports (default: reports)",
    )
    parser.add_argument(
        "--only_solved",
        action="store_true",
        help="Filter problems to those with at least one correct CoT trace (Phase 3 Day 20)",
    )
    parser.add_argument(
        "--pass_rates_file",
        type=str,
        default=None,
        help="Path to problem pass rates file for filtering solved problems",
    )
    parser.add_argument(
        "--split_by_prompt_type",
        action="store_true",
        default=True,
        help="Save individual JSONL files for each prompt type in addition to the combined file",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Problem chunk size for flushing traces to disk (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Prompts per batch call (default: from configs/generation.yaml)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling (default: 0.95)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output traces file and metadata instead of resuming",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    use_mock_flag = args.use_mock or (not args.use_real_model)
    real_flag = not use_mock_flag
    prompt_types_list = [p.strip() for p in args.prompt_types.split(",") if p.strip()]
    limit_val: Optional[int] = args.limit if args.limit and args.limit > 0 else None

    logger.info(
        f"Starting reasoning data generation: limit={limit_val}, "
        f"samples_per_prob={args.samples_per_problem}, prompt_types={prompt_types_list}, "
        f"model={args.model_name}, use_mock={use_mock_flag}, only_solved={args.only_solved}"
    )

    # 1. Load problems from synthetic subset
    problems = load_synthetic_problems(
        limit=limit_val,
        allow_sample_fallback=use_mock_flag,
    )
    logger.info(f"Loaded {len(problems)} initial problems from training subset.")

    # 1b. Filter to solved problems if requested (Day 20 verification & alternative traces)
    if args.only_solved:
        pass_rates_path = Path(args.pass_rates_file) if args.pass_rates_file else (
            PROJECT_ROOT / "data" / "synthetic" / "generated" / "problem_pass_rates.json"
        )
        if pass_rates_path.exists():
            problems = filter_problems_by_pass_rate(problems, pass_rates_path, min_correct=1)
            logger.info(
                f"Filtered to {len(problems)} solved problems (>=1 correct CoT trace) using {pass_rates_path}"
            )
        else:
            logger.warning(
                f"Pass rates file {pass_rates_path} not found; proceeding with loaded problems without pass-rate filtering."
            )

    if not problems:
        logger.error("No problems available for reasoning trace generation.")
        sys.exit(1)

    # 2. Initialize sampler
    sampler = TraceSampler(
        model_name_or_path=args.model_name,
        backend=args.backend,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        seed=args.seed,
        use_mock=use_mock_flag,
    )

    try:
        # 3. Sample traces
        out_path = sampler.sample_traces(
            problems=problems,
            prompt_types=prompt_types_list,
            samples_per_problem=args.samples_per_problem,
            temperature=args.temperature,
            top_p=args.top_p,
            overwrite=args.overwrite,
        )

        # 4. Compute overall yield metrics
        all_traces = load_synthetic_traces(out_path)
        yield_metrics = compute_trace_yield(all_traces)

        logger.info("=== Trace Yield & Quality Summary ===")
        logger.info(f"Total Traces: {yield_metrics['total_traces']}")
        logger.info(f"Format Compliance: {yield_metrics['format_compliance_percent']}%")
        logger.info(f"Correct Traces: {yield_metrics['correct_traces']}")
        logger.info(f"Trace Accuracy: {yield_metrics['trace_accuracy_percent']}%")
        logger.info(
            f"Coverage: {yield_metrics['problems_with_at_least_one_correct']}/{yield_metrics['total_problems']} "
            f"({yield_metrics['problem_coverage_rate'] * 100:.1f}%)"
        )

        # 5. Run Correctness Filter
        filter_engine = CorrectnessFilter()
        kept_traces, filter_stats = filter_engine.filter_traces(all_traces)
        logger.info(
            f"Curation Correctness Filter: Kept {filter_stats.kept_count}/{filter_stats.total_evaluated} "
            f"({filter_stats.retention_percent:.1f}%), Rejected {filter_stats.rejected_count}"
        )

        # 5b. Compute per-strategy breakdowns and save individual prompt slices if requested
        per_strategy: Dict[str, Dict[str, Any]] = {}
        if len(prompt_types_list) > 1:
            for ptype in prompt_types_list:
                ptype_traces = [t for t in all_traces if t.get("prompt_type") == ptype]
                ptype_yield = compute_trace_yield(ptype_traces)
                _, ptype_fstats = filter_engine.filter_traces(ptype_traces)
                ptype_yield["kept_count"] = ptype_fstats.kept_count
                per_strategy[ptype] = ptype_yield

                if args.split_by_prompt_type:
                    slice_file = sampler.output_dir / f"traces_{sampler.model_slug}_{ptype}.jsonl"
                    with open(slice_file, "w", encoding="utf-8") as f:
                        for tr in ptype_traces:
                            f.write(json.dumps(tr) + "\n")
                    logger.info(f"Saved {len(ptype_traces)} traces for '{ptype}' to {slice_file.name}")

        # 6. Save reports
        reports_dir = Path(args.reports_dir)
        if not reports_dir.is_absolute():
            reports_dir = PROJECT_ROOT / reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        if args.report_name:
            base_name = args.report_name
        elif args.limit and args.limit <= 100:
            base_name = "day17_synthetic_generation"
        elif len(prompt_types_list) == 1 and prompt_types_list[0] == "chain_of_thought":
            base_name = "day18_cot_sampling"
        elif set(prompt_types_list) == {"solve_then_verify", "alternative_method"}:
            base_name = "day20_multi_strategy_traces"
        else:
            base_name = "reasoning_data_generation"

        json_report = reports_dir / f"{base_name}_report.json"
        with open(json_report, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": args.model_name,
                "mode": "real" if real_flag else "mock",
                "verdict": "PASSED" if real_flag else "MOCK",
                "prompt_types": prompt_types_list,
                "samples_per_problem": args.samples_per_problem,
                "only_solved": args.only_solved,
                "output_file": str(out_path),
                "yield_metrics": yield_metrics,
                "curation_filter": filter_stats.to_dict(),
                "per_strategy_breakdown": per_strategy,
            }, f, indent=2)

        md_report = reports_dir / f"{base_name}.md"
        report_title_str = "Day 17 Reasoning Data Generation Report" if base_name == "day17_synthetic_generation" else f"{base_name.replace('_', ' ').title()}"
        generate_markdown_report(
            yield_metrics=yield_metrics,
            model_name=args.model_name,
            prompt_types=prompt_types_list,
            samples_per_problem=args.samples_per_problem,
            output_jsonl=out_path,
            report_path=md_report,
            use_real_model=real_flag,
            filter_stats=filter_stats,
            report_title=report_title_str,
            per_strategy_breakdown=per_strategy if per_strategy else None,
        )

        if not real_flag:
            logger.warning("Mock run finished: pipeline plumbing verified, but real model run is still required.")
        else:
            logger.info("Reasoning data generation run completed successfully.")

    finally:
        sampler.close()


if __name__ == "__main__":
    main()
