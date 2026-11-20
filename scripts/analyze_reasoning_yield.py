#!/usr/bin/env python3
"""Reasoning trace yield analysis and per-problem pass-rate calculation (Phase 3 Day 19).

Computes and saves:
1. Per-problem pass rates (correct samples out of 4) for Phase 6 difficulty filtering.
2. Overall trace yield, problem solve rate, unsolved count, and formatting failures.
3. Diagnostic markdown report and JSON report for experiment tracking.
4. Optional initial curated SFT dataset (verified correct traces) in data/synthetic/curated/.

Usage:
    # Run on default generated traces:
    python scripts/analyze_reasoning_yield.py

    # Run on specific trace file:
    python scripts/analyze_reasoning_yield.py \
        --traces_file data/synthetic/generated/traces_deepseek-math-7b-instruct_chain_of_thought.jsonl \
        --output_dir data/synthetic/generated/ \
        --save_curated

    # Run in mock/dev mode on sample traces:
    python scripts/analyze_reasoning_yield.py --mock
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager  # noqa: E402
from src.data.curation import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    CorrectnessFilter,
    analyze_reasoning_yield,
    save_problem_pass_rates,
)
from src.data.synthetic import (  # noqa: E402
    load_existing_traces_by_id,
    load_synthetic_problems,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("analyze_reasoning_yield")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reasoning trace yield analysis and per-problem pass rate aggregation (Phase 3 Day 19)."
    )
    parser.add_argument(
        "--traces_file",
        type=str,
        default=None,
        help="Path or glob pattern to reasoning traces JSONL. Defaults to 7B Instruct CoT traces or dev baseline.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/synthetic/generated",
        help="Directory to save pass rate metrics and report.",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default="day19_yield_analysis",
        help="Base name for markdown and JSON reports.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Directory to save markdown and JSON reports.",
    )
    parser.add_argument(
        "--pass_rates_name",
        type=str,
        default="problem_pass_rates.json",
        help="Filename for per-problem pass rates.",
    )
    parser.add_argument(
        "--save_curated",
        action="store_true",
        default=True,
        help="Save verified correct traces to data/synthetic/curated/.",
    )
    parser.add_argument(
        "--curated_dir",
        type=str,
        default="data/synthetic/curated",
        help="Directory to save curated SFT traces.",
    )
    parser.add_argument(
        "--curated_name",
        type=str,
        default="traces_kept_cot.jsonl",
        help="Filename for kept CoT traces.",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help="Minimum required completion tokens.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum allowed completion tokens.",
    )
    parser.add_argument(
        "--require_hash_format",
        action="store_true",
        default=False,
        help="Require strict '#### <number>' canonical format.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use mock generation if no trace files exist.",
    )
    return parser.parse_args()


def resolve_trace_files(
    traces_arg: Optional[str],
    project_root: Path,
    is_mock: bool = False,
) -> List[Path]:
    """Resolve trace file path(s), handling explicit paths, globs, and default locations (local & Drive)."""
    mgr = ConfigManager()
    drive_gen_dir = mgr.data_path("synthetic", "generated")
    local_gen_dir = project_root / "data" / "synthetic" / "generated"

    if traces_arg:
        raw_p = Path(traces_arg)
        search_paths = (
            [raw_p]
            if raw_p.is_absolute()
            else [
                drive_gen_dir / raw_p.name,
                project_root / raw_p,
                mgr.get_profile_dir("data_dir") / raw_p,
            ]
        )

        if "*" in traces_arg:
            for sp in search_paths:
                matched = [Path(p) for p in glob.glob(str(sp))]
                if matched:
                    return matched
        else:
            existing = [sp for sp in search_paths if sp.exists()]
            if existing:
                best = max(existing, key=lambda p: p.stat().st_size)
                return [best]
            logger.warning(f"Specified trace file does not exist in local or Drive paths: {search_paths[0]}")

    # Check candidate files in priority order across both Drive and local directories
    search_dirs = [drive_gen_dir, local_gen_dir] if drive_gen_dir != local_gen_dir else [local_gen_dir]
    for s_dir in search_dirs:
        candidates = [
            s_dir / "traces_deepseek-math-7b-instruct_chain_of_thought.jsonl",
            s_dir / "traces_Qwen2.5-1.5B-Instruct_chain_of_thought.jsonl",
        ]
        for c in candidates:
            if c.exists():
                return [c]

    # Look for any non-scored jsonl in data/synthetic/generated/ (Drive or local)
    for s_dir in search_dirs:
        if s_dir.exists():
            found = [
                f for f in s_dir.glob("traces_*.jsonl")
                if not f.name.startswith("traces_scored")
            ]
            if found:
                return sorted(found)

    return []


def main() -> None:
    args = parse_args()
    mgr = ConfigManager()
    project_root = mgr.project_root

    logger.info("=== Starting Reasoning Trace Yield Analysis (Phase 3 Day 19) ===")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve trace files
    trace_files = resolve_trace_files(args.traces_file, project_root, is_mock=args.mock)

    traces_map: Dict[str, Dict[str, Any]] = {}
    if trace_files:
        for tf in trace_files:
            logger.info(f"Loading traces from: {tf}")
            loaded = load_existing_traces_by_id(tf)
            logger.info(f"Loaded {len(loaded)} traces from {tf.name}")
            traces_map.update(loaded)
    else:
        if args.mock:
            logger.info("No traces file found. Generating mock traces for verification...")
            from src.data.synthetic import ReasoningPromptType, TraceSampler
            sample_probs = load_synthetic_problems(limit=10, allow_sample_fallback=True)
            mock_sampler = TraceSampler(
                model_name_or_path="mock-model",
                output_dir=out_dir,
                use_mock=True,
            )
            traces_path = mock_sampler.sample_traces(
                problems=sample_probs,
                prompt_types=[ReasoningPromptType.CHAIN_OF_THOUGHT],
                samples_per_problem=4,
            )
            traces_map = load_existing_traces_by_id(traces_path)
            trace_files = [traces_path]
        else:
            logger.error(
                "No reasoning trace files found in data/synthetic/generated/. "
                "Run scripts/generate_reasoning_data.py first or pass --mock."
            )
            sys.exit(1)

    all_traces = list(traces_map.values())
    logger.info(f"Total unique reasoning traces to analyze: {len(all_traces)}")

    # 2. Configure Correctness Filter
    c_filter = CorrectnessFilter(
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        require_format_compliance=args.require_hash_format,
        recheck_answer=True,
    )

    # 3. Execute Yield Analysis
    report, pass_rates, kept_traces = analyze_reasoning_yield(
        traces=all_traces,
        problems=None,
        correctness_filter=c_filter,
    )

    # 4. Save per-problem pass rates
    pass_rates_path = out_dir / args.pass_rates_name
    json_saved, parquet_saved = save_problem_pass_rates(pass_rates, pass_rates_path)
    logger.info(f"Problem pass rates saved to: {json_saved}")
    if parquet_saved:
        logger.info(f"Problem pass rates parquet saved to: {parquet_saved}")

    # Mirror pass rates between local workspace and Drive if out_dir is the default generated dir
    try:
        import shutil

        local_gen_dir = (project_root / "data" / "synthetic" / "generated").resolve()
        drive_gen_dir = mgr.data_path("synthetic", "generated").resolve()
        out_res = out_dir.resolve()
        mirror_dir: Optional[Path] = None
        if local_gen_dir != drive_gen_dir:
            if out_res == local_gen_dir:
                mirror_dir = drive_gen_dir
            elif out_res == drive_gen_dir:
                mirror_dir = local_gen_dir
        if mirror_dir is not None:
            mirror_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(json_saved, mirror_dir / json_saved.name)
            if parquet_saved and parquet_saved.exists():
                shutil.copy2(parquet_saved, mirror_dir / parquet_saved.name)
            logger.info(f"Mirrored pass rates to: {mirror_dir}")
    except Exception as exc:
        logger.debug(f"Could not mirror pass rates: {exc}")

    # 5. Save curated SFT traces if requested
    if args.save_curated and kept_traces:
        curated_dir = Path(args.curated_dir)
        if not curated_dir.is_absolute():
            curated_dir = project_root / curated_dir
        curated_dir.mkdir(parents=True, exist_ok=True)

        curated_file = curated_dir / args.curated_name
        with open(curated_file, "w", encoding="utf-8") as f:
            for item in kept_traces:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(kept_traces)} curated SFT traces to: {curated_file}")

    # 6. Save Markdown and JSON Reports
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = project_root / reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    md_report_path = reports_dir / f"{args.report_name}.md"
    json_report_path = reports_dir / f"{args.report_name}_report.json"

    md_content = report.to_markdown_summary()
    md_report_path.write_text(md_content, encoding="utf-8")
    logger.info(f"Markdown yield report saved to: {md_report_path}")

    report_payload = {
        "report": report.to_dict(),
        "input_trace_files": [str(tf) for tf in trace_files],
        "pass_rates_json": str(json_saved),
        "pass_rates_parquet": str(parquet_saved) if parquet_saved else None,
        "curated_traces_count": len(kept_traces),
    }
    json_report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    logger.info(f"JSON yield report saved to: {json_report_path}")

    try:
        import shutil

        profile_root = mgr.get_profile_dir("data_dir").parent
        drive_reports = profile_root / "reports"
        if (
            profile_root.resolve() != project_root.resolve()
            and reports_dir.resolve() == (project_root / "reports").resolve()
            and drive_reports.resolve() != reports_dir.resolve()
            and profile_root.exists()
        ):
            drive_reports.mkdir(parents=True, exist_ok=True)
            shutil.copy2(md_report_path, drive_reports / md_report_path.name)
            shutil.copy2(json_report_path, drive_reports / json_report_path.name)
            logger.info(f"Mirrored yield reports to Drive: {drive_reports}")
    except Exception as exc:
        logger.debug(f"Could not mirror yield reports to Drive: {exc}")

    # 7. Print summary to console
    logger.info("=== Reasoning Trace Yield Analysis Results ===")
    logger.info(f"Total Problems Evaluated: {report.total_problems:,}")
    logger.info(f"Total Traces Evaluated:   {report.total_traces:,}")
    logger.info(f"Correct Traces:           {report.correct_traces:,} ({report.trace_yield_percent:.2f}% yield)")
    logger.info(f"Problems Solved (>=1 ok): {report.solved_problems:,} ({report.problem_solve_percent:.2f}%)")
    logger.info(f"Problems Unsolved (0 ok): {report.unsolved_problems:,} ({report.unsolved_percent:.2f}%)")
    logger.info(f"Format Failures:          {report.format_failures:,} ({report.format_failure_rate * 100:.2f}%)")
    logger.info(f"Difficulty Breakdown:     {report.distribution_by_difficulty}")
    logger.info("Day 19 Yield Analysis successfully complete!")


if __name__ == "__main__":
    main()
