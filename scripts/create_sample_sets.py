"""CLI Script to extract and save small sample evaluation sets (20 problems per split)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sample_sets import create_and_save_sample_sets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample evaluation sets for GSM8K")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20,
        help="Number of problems to sample per split (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed (default: 42)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/sample",
        help="Output directory (default: data/sample)",
    )
    args = parser.parse_args()

    print(f"Creating sample sets ({args.sample_size} problems per split, seed={args.seed})...")
    saved = create_and_save_sample_sets(
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("\nSample sets successfully created:")
    for k, p in saved.items():
        print(f"  {k:<20}: {p}")


if __name__ == "__main__":
    main()
