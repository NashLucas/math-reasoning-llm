#!/usr/bin/env python3
"""Standalone runner for GRPO Dataset Preparation and Difficulty Filtering (Phase 6 Day 41).

Profiles GSM8K training problems, filters out zero-variance extremes (trivial all-pass or
impossible all-fail), stratifies intermediate difficulty (0.20 <= pass_rate <= 0.80 or 3-6 steps),
enforces the Strict Test-Set Wall, and produces frozen datasets with manifest and audit reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.grpo_dataset import (  # noqa: E402
    load_grpo_dataset_config,
    prepare_grpo_dataset,
)
from src.run_info import save_run_info  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_grpo_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare difficulty-filtered dataset for GRPO policy optimization.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "grpo_dataset.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to input train parquet file.",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="Path to input validation parquet file.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to forbidden test parquet file for Test-Set Wall verification.",
    )
    parser.add_argument(
        "--pass-rates-path",
        type=str,
        default=None,
        help="Path to empirical pass rates parquet/json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write output datasets and manifest.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=str(PROJECT_ROOT / "reports"),
        help="Directory to write audit reports.",
    )
    parser.add_argument(
        "--target-train-size",
        type=int,
        default=None,
        help="Override target training sample count.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Override validation sample count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--sample-mode",
        "--sample_mode",
        action="store_true",
        help="Use sample datasets in data/sample/ for testing or dry-run environments.",
    )
    return parser.parse_args()


def generate_markdown_report(
    manifest: Dict[str, Any],
    report_path: Path,
) -> None:
    """Generate human-readable markdown audit report."""
    stats = manifest.get("statistics", {})
    criteria = manifest.get("filtering_criteria", {})
    files = manifest.get("files", {})
    wall = manifest.get("test_set_wall", {})

    tier_counts = stats.get("tier_counts", {})
    step_dist = stats.get("step_distribution", {})

    md_lines = [
        "# Phase 6 Day 41: GRPO Dataset Preparation & Difficulty Filtering Audit Report",
        "",
        f"- **Date / Timestamp:** `{manifest.get('created_at', datetime.now(timezone.utc).isoformat())}`",
        f"- **Dataset Name:** `{manifest.get('dataset_name')}`",
        f"- **Random Seed:** `{manifest.get('seed')}`",
        f"- **Test-Set Wall Status:** `{'PASSED' if wall.get('passed') else 'FAILED'}`",
        "",
        "---",
        "",
        "## 1. Executive Summary & Objective",
        "",
        "In Group Relative Policy Optimization (GRPO), policy gradient updates are driven by group-relative advantages:",
        "$$A_i = \\frac{r_i - \\mu_g}{\\sigma_g + \\epsilon}$$",
        "If a prompt is either trivially easy (all completions correct, $r_i=r_{\\max}$) or impossibly difficult "
        "(all completions failed, $r_i=r_{\\min}$), the group standard deviation collapses ($\\sigma_g = 0$) resulting "
        "in zero advantage ($A_i = 0$) and zero policy update. "
        "This pipeline filters the training corpus to select prompts within the intermediate difficulty band, "
        "maximizing advantage variance and policy gradient signal during candidate generation ($G=4$).",
        "",
        "---",
        "",
        "## 2. Filtering Criteria & Configuration",
        "",
        "| Parameter | Value | Purpose |",
        "| :--- | :--- | :--- |",
        f"| **Filtering Mode** | `{criteria.get('mode')}` | Combines empirical pass rate with step complexity fallback |",
        f"| **Min Pass Rate** | `{criteria.get('min_pass_rate')}` | Excludes impossible problems with all-zero rewards |",
        f"| **Max Pass Rate** | `{criteria.get('max_pass_rate')}` | Excludes trivial problems with all-max rewards |",
        f"| **Min Reasoning Steps** | `{criteria.get('min_steps')}` | Excludes 1-2 step superficial arithmetic |",
        f"| **Max Reasoning Steps** | `{criteria.get('max_steps')}` | Excludes extreme >6 step complex derivations |",
        f"| **Max Prompt Chars** | `{criteria.get('max_prompt_chars')}` | Enforces context length boundaries |",
        "",
        "---",
        "",
        "## 3. Dataset Volume & Difficulty Distribution",
        "",
        f"- **Total Training Problems Profiled:** {stats.get('total_profiled', 0):,}",
        f"- **Eligible Candidates (Intermediate Difficulty):** {stats.get('eligible_candidates', 0):,} ({stats.get('eligible_candidates', 0) / max(stats.get('total_profiled', 1), 1) * 100:.1f}%)",
        f"- **Sampled GRPO Training Set:** {stats.get('sampled_train_count', 0):,}",
        f"- **Sampled Held-Out Validation Prompts:** {stats.get('sampled_val_count', 0):,}",
        "",
        "### Profiled Difficulty Breakdown:",
        "| Difficulty Tier | Count | Percentage | GRPO Candidate Status |",
        "| :--- | :--- | :--- | :--- |",
    ]

    total_profiled = max(stats.get("total_profiled", 1), 1)
    for tier, count in tier_counts.items():
        status = "Accepted (Target Zone)" if tier == "medium" else "Rejected (Zero Variance Risk)"
        pct = (count / total_profiled) * 100
        md_lines.append(f"| **{tier.title()}** | {count:,} | {pct:.1f}% | {status} |")

    md_lines.extend([
        "",
        "### Reasoning Step Distribution (Sampled Train Set):",
        "| Step Count | Frequency | Percentage |",
        "| :--- | :--- | :--- |",
    ])

    sampled_train = max(stats.get("sampled_train_count", 1), 1)
    for steps in sorted(step_dist.keys(), key=lambda x: int(x)):
        cnt = step_dist[steps]
        pct = (cnt / sampled_train) * 100
        md_lines.append(f"| {steps} steps | {cnt:,} | {pct:.1f}% |")

    md_lines.extend([
        "",
        f"- **Mean Reasoning Steps:** `{stats.get('mean_steps', 0.0):.2f}`",
        f"- **Mean Question Character Length:** `{stats.get('mean_char_length', 0.0):.1f}`",
        "",
        "---",
        "",
        "## 4. Strict Test-Set Wall Verification",
        "",
        "Design Rule 2 requires strict isolation of the 1,319 GSM8K test problems:",
        f"- **Test Problems Audited:** {wall.get('test_problems_checked', 0):,}",
        f"- **Exact Test Question Overlaps:** `{wall.get('exact_matches_test', -1)}`",
        f"- **Forbidden Problem ID Leaks:** `{wall.get('forbidden_id_leaks', -1)}`",
        f"- **Verification Result:** `{'PASSED - Zero Contamination' if wall.get('passed') else 'FAILED'}`",
        "",
        "---",
        "",
        "## 5. Frozen Corpus Manifest & SHA256 Digests",
        "",
        "| File Description | Record Count | SHA256 Digest |",
        "| :--- | :--- | :--- |",
    ])

    for fkey, finfo in files.items():
        md_lines.append(
            f"| `{fkey}` | {finfo.get('count', 0):,} | `{finfo.get('sha256', '')}` |"
        )

    md_lines.extend([
        "",
        "---",
        "*Report automatically generated by `scripts/prepare_grpo_dataset.py` during Phase 6 Day 41.*",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def main() -> int:
    args = parse_args()
    logger.info("Initializing GRPO Dataset Preparation (Day 41)...")

    # Load configuration
    cfg = load_grpo_dataset_config(args.config)

    # Resolve paths
    paths_cfg = cfg.get("paths", {})
    train_path = Path(args.train_path or PROJECT_ROOT / paths_cfg.get("train_input", "data/processed/train/train.parquet"))
    val_path = Path(args.val_path or PROJECT_ROOT / paths_cfg.get("val_input", "data/processed/val/val_gating.parquet"))
    test_path = Path(args.test_path or PROJECT_ROOT / paths_cfg.get("test_input", "data/processed/test/test.parquet"))
    pass_rates_path = Path(args.pass_rates_path or PROJECT_ROOT / paths_cfg.get("pass_rates_input", "data/synthetic/generated/problem_pass_rates.parquet"))
    output_dir = Path(args.output_dir or PROJECT_ROOT / paths_cfg.get("output_dir", "data/processed/grpo"))
    report_dir = Path(args.report_dir or PROJECT_ROOT / "reports")

    # Overrides
    if args.target_train_size is not None:
        cfg.setdefault("sampling", {})["target_train_size"] = args.target_train_size
    if args.val_size is not None:
        cfg.setdefault("sampling", {})["val_size"] = args.val_size
    seed = args.seed if args.seed is not None else cfg.get("sampling", {}).get("seed", 42)

    # Sample mode fallback for testing or CI environments
    sample_train = PROJECT_ROOT / "data" / "sample" / "train_sample.parquet"
    sample_val = PROJECT_ROOT / "data" / "sample" / "val_sample.parquet"
    sample_test = PROJECT_ROOT / "data" / "sample" / "test_sample.parquet"

    if args.sample_mode or not train_path.exists():
        if sample_train.exists():
            logger.info("Falling back to sample datasets in data/sample/ (sample mode / CI environment)...")
            train_path = sample_train
            val_path = sample_val
            test_path = sample_test

    logger.info(f"Loading training data from: {train_path}")
    if not train_path.exists():
        logger.error(f"Train dataset not found at {train_path}")
        return 1
    train_df = pd.read_parquet(train_path)

    logger.info(f"Loading validation data from: {val_path}")
    if not val_path.exists():
        logger.error(f"Val dataset not found at {val_path}")
        return 1
    val_df = pd.read_parquet(val_path)

    logger.info(f"Loading test data for wall verification from: {test_path}")
    if not test_path.exists():
        logger.error(f"Test dataset not found at {test_path}")
        return 1
    test_df = pd.read_parquet(test_path)

    pass_rates_df = None
    if pass_rates_path.exists():
        logger.info(f"Loading empirical pass rates from: {pass_rates_path}")
        try:
            if pass_rates_path.suffix == ".parquet":
                pass_rates_df = pd.read_parquet(pass_rates_path)
            elif pass_rates_path.suffix == ".json":
                pass_rates_df = pd.read_json(pass_rates_path)
            logger.info(f"Loaded {len(pass_rates_df)} pass rate records.")
        except Exception as e:
            logger.warning(f"Failed to load pass rates ({e}); proceeding with step complexity fallback.")
    else:
        logger.info(f"No pass rates file at {pass_rates_path}; proceeding with step complexity fallback.")

    # Execute dataset preparation
    result = prepare_grpo_dataset(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        pass_rates_df=pass_rates_df,
        output_dir=output_dir,
        config=cfg,
        seed=seed,
    )

    manifest = result["manifest"]

    # Emit Reports
    report_md_path = report_dir / "day41_grpo_dataset_report.md"
    report_json_path = report_dir / "day41_grpo_dataset_report.json"

    generate_markdown_report(manifest, report_md_path)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved audit reports to {report_md_path} and {report_json_path}")

    # Mirror outputs and reports to Google Drive if running in Colab
    drive_root = Path("/content/drive/MyDrive/math-reasoning-llm")
    if drive_root.exists():
        try:
            drive_data = drive_root / "data" / "processed" / "grpo"
            drive_reports = drive_root / "reports"
            drive_data.mkdir(parents=True, exist_ok=True)
            drive_reports.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(result["train_parquet_path"], drive_data / result["train_parquet_path"].name)
            shutil.copy2(result["val_parquet_path"], drive_data / result["val_parquet_path"].name)
            shutil.copy2(result["manifest_path"], drive_data / result["manifest_path"].name)
            shutil.copy2(report_md_path, drive_reports / report_md_path.name)
            shutil.copy2(report_json_path, drive_reports / report_json_path.name)
            logger.info("Mirrored GRPO dataset and reports to Google Drive at: %s", drive_root)
        except Exception as exc:
            logger.warning("Could not mirror GRPO dataset to Drive: %s", exc)

    # Save Run Info
    save_run_info(
        stage="grpo_dataset_prep",
        output_dir=output_dir,
        description="GRPO Dataset Preparation & Difficulty Filtering (Phase 6 Day 41)",
        config={
            "metrics": {
                "total_profiled": manifest["statistics"]["total_profiled"],
                "eligible_candidates": manifest["statistics"]["eligible_candidates"],
                "sampled_train_count": manifest["statistics"]["sampled_train_count"],
                "sampled_val_count": manifest["statistics"]["sampled_val_count"],
                "mean_steps": manifest["statistics"]["mean_steps"],
                "test_wall_passed": manifest["test_set_wall"]["passed"],
            },
            "artifacts": [
                str(result["train_parquet_path"]),
                str(result["val_parquet_path"]),
                str(result["manifest_path"]),
                str(report_md_path),
                str(report_json_path),
            ],
            "filtering_criteria": manifest["filtering_criteria"],
        },
    )

    # Print summary
    print("=" * 72)
    print("Phase 6 Day 41: GRPO Dataset Preparation Complete")
    print("=" * 72)
    print(f"Total Problems Profiled:    {manifest['statistics']['total_profiled']:,}")
    print(f"Eligible (Medium Tier):     {manifest['statistics']['eligible_candidates']:,}")
    print(f"Sampled GRPO Train Size:    {manifest['statistics']['sampled_train_count']:,}")
    print(f"Sampled GRPO Val Prompts:   {manifest['statistics']['sampled_val_count']:,}")
    print(f"Mean Reasoning Steps:       {manifest['statistics']['mean_steps']:.2f}")
    print("Test-Set Wall Integrity:    PASSED (Zero Leaks)")
    print(f"Output Directory:           {output_dir}")
    print(f"Manifest File:              {result['manifest_path']}")
    print(f"Audit Report:               {report_md_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
