"""Script to generate synthetic reasoning traces (Phase 3 Day 17).

Samples reasoning traces from the 3,000-problem training subset across specified
prompt strategies, evaluating correctness and format compliance, and persisting
chunked, resumable JSONL records.

Supports:
- Real model generation on GPU via --use_real_model (e.g. Qwen/Qwen2.5-1.5B-Instruct or DeepSeek-Math-7B-Instruct)
- Mock simulation on CPU via --use_mock (default when --use_real_model is omitted; checks plumbing only)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.curation import CorrectnessFilter, CurationFilterStats  # noqa: E402
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
        "## Yield & Quality Metrics",
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
            "| Reason | Count |",
            "|---|---|",
        ])
        for reason, rcount in filter_stats.rejection_breakdown.items():
            lines.append(f"| `{reason}` | {rcount:,} |")
        lines.append("")

    lines.extend([
        "## Pass-Rate Distribution (Correct Samples per Problem)",
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
        "## Phase 3 Day 17 Status",
        "",
        f"**Status:** {verdict}",
        "- Records contain prompt, completion, parsed answer, extraction method, correctness flag, token counts, and timestamp.",
    ])
    if use_real_model:
        lines.append("- Ready for Day 18: Full 3,000-problem Chain-of-Thought generation with 7B Instruct.")
    else:
        lines.append("- Next: Run `--use_real_model` on 100 problems with `Qwen/Qwen2.5-1.5B-Instruct` on a cheap GPU.")

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
        f"model={args.model_name}, use_mock={use_mock_flag}"
    )

    # 1. Load problems from synthetic subset
    problems = load_synthetic_problems(
        limit=limit_val,
        allow_sample_fallback=use_mock_flag,
    )
    logger.info(f"Loaded {len(problems)} problems for synthetic trace generation.")

    # 2. Initialize sampler (output_dir=None resolves through ConfigManager to Drive under colab)
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
        )

        # 4. Compute yield metrics
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

        # 6. Save reports
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        base_name = args.report_name or ("day18_cot_sampling" if "chain_of_thought" in prompt_types_list and len(prompt_types_list) == 1 else "day17_synthetic_generation")
        json_report = reports_dir / f"{base_name}_report.json"
        with open(json_report, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": args.model_name,
                "mode": "real" if real_flag else "mock",
                "verdict": "PASSED" if real_flag else "MOCK",
                "prompt_types": prompt_types_list,
                "samples_per_problem": args.samples_per_problem,
                "output_file": str(out_path),
                "yield_metrics": yield_metrics,
                "curation_filter": filter_stats.to_dict(),
            }, f, indent=2)

        md_report = reports_dir / f"{base_name}.md"
        generate_markdown_report(
            yield_metrics=yield_metrics,
            model_name=args.model_name,
            prompt_types=prompt_types_list,
            samples_per_problem=args.samples_per_problem,
            output_jsonl=out_path,
            report_path=md_report,
            use_real_model=real_flag,
            filter_stats=filter_stats,
            report_title=f"{base_name.replace('_', ' ').title()}",
        )

        if not real_flag:
            logger.warning("Mock run finished: pipeline plumbing verified, but real model run is still required.")
        else:
            logger.info("Reasoning data generation run completed successfully.")

    finally:
        sampler.close()


if __name__ == "__main__":
    main()
