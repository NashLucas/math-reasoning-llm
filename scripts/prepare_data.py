"""CLI Script to prepare and split the GSM8K dataset.

Downloads raw data if not present, strips calculator annotations, extracts gold answers,
carves validation splits (analysis and gating halves), and saves processed Parquet datasets
and problem ID lists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import prepare_all_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and split GSM8K dataset")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for validation split carving (default: source.seed in configs/data.yaml)",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Deliberately (re)write the committed split lists in data/splits/ instead of verifying against them",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory (default: <profile data dir>/processed)",
    )
    args = parser.parse_args()

    print("Starting GSM8K data preparation...")
    saved_paths = prepare_all_data(seed=args.seed, output_dir=args.output_dir, freeze=args.freeze)

    print("\nData preparation completed successfully!")
    print(f"Total files generated: {len(saved_paths)}")
    for k, p in saved_paths.items():
        print(f"  {k:<20}: {p}")


if __name__ == "__main__":
    main()
