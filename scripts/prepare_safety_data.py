"""Assemble the safety prompt set in data/safety/ (Day 15).

Downloads JailbreakBench JBB-Behaviors (harmful split) and samples an equal number of
harmful prompts per category, then samples benign math questions from the GSM8K train split.
Requires data/processed/train/train.parquet (run scripts/prepare_data.py first).

Usage:
  python scripts/prepare_safety_data.py
  python scripts/prepare_safety_data.py --n_harmful 50 --n_benign 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.safety_data import (  # noqa: E402
    HARMFUL_CONFIG,
    HARMFUL_SOURCE,
    HARMFUL_SPLIT,
    build_safety_prompt_set,
    save_safety_prompt_set,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("prepare_safety_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the safety prompt set (harmful red-teaming prompts + benign math questions).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_harmful", type=int, default=50, help="Harmful prompts (split evenly across categories).")
    parser.add_argument("--n_benign", type=int, default=50, help="Benign GSM8K train questions.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--train_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "train" / "train.parquet"),
        help="Processed GSM8K train split.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "safety"),
        help="Destination folder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import pandas as pd
    from datasets import load_dataset

    train_path = Path(args.train_path)
    if not train_path.exists():
        logger.error(f"Train split not found: {train_path}. Run scripts/prepare_data.py first.")
        return 1

    logger.info(f"Loading harmful behaviors from {HARMFUL_SOURCE} ({HARMFUL_CONFIG}/{HARMFUL_SPLIT})")
    harmful_rows = list(load_dataset(HARMFUL_SOURCE, HARMFUL_CONFIG, split=HARMFUL_SPLIT))

    logger.info(f"Loading benign questions from {train_path}")
    train_rows = pd.read_parquet(train_path).to_dict(orient="records")

    records = build_safety_prompt_set(
        harmful_rows=harmful_rows,
        train_rows=train_rows,
        n_harmful=args.n_harmful,
        n_benign=args.n_benign,
        seed=args.seed,
    )
    manifest = save_safety_prompt_set(records, args.output_dir, seed=args.seed)

    logger.info(f"Saved safety prompt set to {Path(args.output_dir).resolve()}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
