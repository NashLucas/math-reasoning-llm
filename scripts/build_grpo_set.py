#!/usr/bin/env python3
r"""Build and filter intermediate-difficulty GRPO problem set (Phase 6 Day 41).

As specified in Phase 6:
- Filters training problems to isolate the intermediate difficulty zone ($0.20 \le \text{pass\_rate} \le 0.80$ or 3–6 reasoning steps).
- Discards trivial problems (pass_rate > 0.80) and impossible problems (pass_rate < 0.20) to prevent zero-variance advantage collapse.
- Enforces Design Rule 2 (The Test-Set Wall): 100% strict quarantine of the 1,319 GSM8K test problems.
- Saves the frozen problem set to data/processed/grpo/ (`grpo_train.parquet`, `grpo_val.parquet`, `manifest.json`).
- Automatically mirrors to Google Drive under Colab environments.

Usage:
    python scripts/build_grpo_set.py
    python scripts/build_grpo_set.py --config configs/grpo_dataset.yaml --output_dir data/processed/grpo
    python scripts/build_grpo_set.py --sample_mode
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

from scripts.prepare_grpo_dataset import main as run_prepare_dataset  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_grpo_set")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build intermediate-difficulty problem set for GRPO policy training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "grpo_dataset.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--train_path",
        "--train-path",
        type=str,
        default=None,
        help="Path to input train parquet file.",
    )
    parser.add_argument(
        "--val_path",
        "--val-path",
        type=str,
        default=None,
        help="Path to input validation parquet file.",
    )
    parser.add_argument(
        "--test_path",
        "--test-path",
        type=str,
        default=None,
        help="Path to forbidden test parquet file for Test-Set Wall verification.",
    )
    parser.add_argument(
        "--pass_rates_path",
        "--pass-rates-path",
        type=str,
        default=None,
        help="Path to empirical pass rates parquet/json file.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write output datasets and manifest.",
    )
    parser.add_argument(
        "--report_dir",
        "--report-dir",
        type=str,
        default=str(PROJECT_ROOT / "reports"),
        help="Directory to write audit reports.",
    )
    parser.add_argument(
        "--target_train_size",
        "--target-train-size",
        type=int,
        default=None,
        help="Target number of intermediate-difficulty problems (default: 1000).",
    )
    parser.add_argument(
        "--val_size",
        "--val-size",
        type=int,
        default=None,
        help="Target number of validation problems (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--sample_mode",
        "--sample-mode",
        action="store_true",
        help="Use sample datasets in data/sample/ for dry-run/testing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("==================================================")
    logger.info("Phase 6 Day 41: Building GRPO Intermediate Set")
    logger.info("Config: %s", args.config)
    logger.info("==================================================")

    # Forward to dataset preparation pipeline
    sys.argv = [
        sys.argv[0],
        "--config", str(args.config),
        "--report-dir", str(args.report_dir),
    ]
    if args.train_path:
        sys.argv.extend(["--train-path", str(args.train_path)])
    if args.val_path:
        sys.argv.extend(["--val-path", str(args.val_path)])
    if args.test_path:
        sys.argv.extend(["--test-path", str(args.test_path)])
    if args.pass_rates_path:
        sys.argv.extend(["--pass-rates-path", str(args.pass_rates_path)])
    if args.output_dir:
        sys.argv.extend(["--output-dir", str(args.output_dir)])
    if args.target_train_size:
        sys.argv.extend(["--target-train-size", str(args.target_train_size)])
    if args.val_size:
        sys.argv.extend(["--val-size", str(args.val_size)])
    if args.seed:
        sys.argv.extend(["--seed", str(args.seed)])
    if args.sample_mode:
        sys.argv.append("--sample-mode")

    return run_prepare_dataset()


if __name__ == "__main__":
    sys.exit(main())
