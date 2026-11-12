#!/usr/bin/env python3
"""Freeze and validate the combined Round-2 SFT dataset (Phase 5 Day 35).

Combines Phase 3 curated reasoning traces with Phase 5 Day 34 error-targeted examples,
applies optional 2x targeted example weighting, asserts zero test contamination (Design Rule 2),
verifies 100% canonical format compliance (Design Rule 1), and freezes the dataset into
train_round2.jsonl, train_round2.parquet, and manifest.json with cryptographic SHA-256 hashes.

Usage:
    python scripts/freeze_round2_dataset.py \
        --curated_file data/synthetic/curated/traces_curated_sft.jsonl \
        --targeted_file data/targeted/targeted_examples.jsonl \
        --output_dir data/processed/sft_round2 \
        --duplicate_targeted
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.round2_dataset import (  # noqa: E402
    Round2DatasetManifest,
    freeze_round2_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("freeze_round2_dataset")


def resolve_path(path: Path, profile: Optional[str] = None) -> Path:
    """Resolve a file or directory path across local workspace, project root, and Colab Drive."""
    if path.exists():
        return path

    if not path.is_absolute() and (PROJECT_ROOT / path).exists():
        return PROJECT_ROOT / path

    try:
        from src.config import ConfigManager

        mgr = ConfigManager()
        active_profile = profile or ("colab" if Path("/content").exists() else "local")
        drive_root = mgr.get_profile_dir("outputs_dir", profile_name=active_profile).parent

        # 1. Check relative to drive root
        drive_candidate = drive_root / path
        if drive_candidate.exists():
            return drive_candidate

        # 2. Check under drive data_dir if path starts with data/
        if str(path).startswith("data"):
            rel_parts = path.parts[1:] if len(path.parts) > 1 else ()
            data_cand = mgr.get_profile_dir("data_dir", profile_name=active_profile).joinpath(*rel_parts)
            if data_cand.exists():
                return data_cand
    except Exception:
        pass

    return path


def prepare_sample_datasets(sample_dir: Path) -> Tuple[Path, Path]:
    """Create lightweight sample curated and targeted datasets for CI / testing."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_curated_path = sample_dir / "sample_curated.jsonl"
    sample_targeted_path = sample_dir / "sample_targeted.jsonl"

    sample_train = PROJECT_ROOT / "data" / "sample" / "train_sample.parquet"
    if sample_train.exists():
        df = pd.read_parquet(sample_train)
    else:
        df = pd.DataFrame([
            {
                "problem_id": f"sample_prob_{i}",
                "question": f"Sample math question {i}?",
                "cleaned_solution": f"Step 1: Solve {i}.\n#### {i * 10}",
                "gold_answer": str(i * 10),
            }
            for i in range(10)
        ])

    curated_records = []
    for idx, row in df.iloc[:5].iterrows():
        curated_records.append({
            "problem_id": row["problem_id"],
            "trace_id": f"{row['problem_id']}_curated_0",
            "question": row["question"],
            "completion": f"Let's think step by step.\n{row['cleaned_solution']}",
            "gold_answer": str(row["gold_answer"]),
            "prompt_type": "chain_of_thought",
        })

    with open(sample_curated_path, "w", encoding="utf-8") as f:
        for r in curated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    targeted_records = []
    for idx, row in df.iloc[5:10].iterrows():
        targeted_records.append({
            "problem_id": row["problem_id"],
            "question": row["question"],
            "solution": f"Step-by-step targeted derivation.\n{row['cleaned_solution']}",
            "gold_answer": str(row["gold_answer"]),
            "target_category": "arithmetic_slip" if idx % 2 == 0 else "wrong_setup",
            "similarity_score": 0.85,
            "source": "cleaned_gold_solution",
        })

    with open(sample_targeted_path, "w", encoding="utf-8") as f:
        for r in targeted_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return sample_curated_path, sample_targeted_path


def run_freeze_round2_pipeline(
    curated_file: Optional[Path] = None,
    targeted_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    duplicate_targeted: bool = True,
    seed: int = 42,
    sample_mode: bool = False,
    splits_dir: Optional[Path] = None,
    test_file: Optional[Path] = None,
    profile: Optional[str] = None,
) -> Round2DatasetManifest:
    """Execute complete Round-2 dataset creation, safety verification, and freezing."""
    out_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "data" / "processed" / "sft_round2")
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    raw_c = curated_file or (PROJECT_ROOT / "data" / "synthetic" / "curated" / "traces_curated_sft.jsonl")
    raw_t = targeted_file or (PROJECT_ROOT / "data" / "targeted" / "targeted_examples.jsonl")

    c_file = resolve_path(Path(raw_c), profile=profile)
    t_file = resolve_path(Path(raw_t), profile=profile)

    if splits_dir:
        splits_dir = resolve_path(Path(splits_dir), profile=profile)
    if test_file:
        test_file = resolve_path(Path(test_file), profile=profile)

    if sample_mode or not c_file.exists() or not t_file.exists():
        logger.info("Executing in sample/mock mode for Round-2 dataset freezing...")
        sample_scratch = out_dir / "sample_inputs"
        c_file, t_file = prepare_sample_datasets(sample_scratch)

    logger.info("==================================================")
    logger.info("Phase 5 Day 35: Freezing Round-2 SFT Dataset")
    logger.info("Curated Source:      %s", c_file)
    logger.info("Targeted Source:     %s", t_file)
    logger.info("Output Directory:    %s", out_dir)
    logger.info("Duplicate Targeted:  %s (2x representation)", duplicate_targeted)
    logger.info("Interleave Seed:     %d", seed)
    logger.info("Profile:             %s", profile or "auto")
    logger.info("==================================================")

    jsonl_path, parquet_path, manifest = freeze_round2_dataset(
        curated_file=c_file,
        targeted_file=t_file,
        output_dir=out_dir,
        duplicate_targeted=duplicate_targeted,
        seed=seed,
        splits_dir=splits_dir,
        test_file=test_file,
    )

    # Mirror artifacts to Google Drive if in Colab environment
    try:
        from src.config import ConfigManager

        mgr = ConfigManager()
        active_profile = profile or ("colab" if Path("/content").exists() else "local")
        if active_profile == "colab":
            drive_root = mgr.get_profile_dir("outputs_dir", profile_name="colab").parent
            drive_target_dir = drive_root / "data" / "processed" / "sft_round2"
            drive_target_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            for artifact_name in [
                "train_round2.jsonl",
                "train_round2.parquet",
                "manifest.json",
                "run_info.json",
            ]:
                src_f = out_dir / artifact_name
                if src_f.exists():
                    shutil.copy2(src_f, drive_target_dir / artifact_name)
            logger.info("Mirrored Round-2 dataset artifacts to Drive at: %s", drive_target_dir)
    except Exception as e:
        logger.warning("Could not mirror Round-2 dataset artifacts to Drive: %s", e)

    # Print summary dashboard
    print("\n" + "=" * 65)
    print("PHASE 5 DAY 35: ROUND-2 SFT DATASET FREEZE SUMMARY")
    print("=" * 65)
    print(f"Total Dataset Examples:     {manifest.total_examples:,}")
    print(f"Unique Problem IDs:         {manifest.unique_problems:,}")
    print(f"Curated Traces:             {manifest.curated_count:,}")
    print(f"Targeted Raw Count:         {manifest.targeted_raw_count:,}")
    print(f"Targeted Effective (2x):    {manifest.targeted_effective_count:,}")
    print(f"Duplicate Targeted:         {manifest.duplicate_targeted}")
    print(f"Test-Set Wall Leaks:        {manifest.forbidden_id_leaks} (Target: 0)")
    print(f"Exact Matches to Test:      {manifest.exact_matches_test} (Target: 0)")
    print(f"JSONL Checksum SHA-256:     {manifest.sha256_jsonl[:16]}...")
    print(f"Parquet Checksum SHA-256:   {manifest.sha256_parquet[:16]}...")
    print("\nCategory Breakdown in Targeted Data:")
    for cat, count in sorted(manifest.category_breakdown.items()):
        print(f"  - {cat:<25}: {count} examples")
    print("=" * 65 + "\n")

    return manifest


def main(args_list: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze combined Round-2 SFT dataset (Phase 5 Day 35)."
    )
    parser.add_argument(
        "--curated_file",
        type=str,
        default=None,
        help="Path to Phase 3 curated reasoning traces JSONL.",
    )
    parser.add_argument(
        "--targeted_file",
        type=str,
        default=None,
        help="Path to Phase 5 Day 34 error-targeted examples JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory where frozen dataset will be exported (default: data/processed/sft_round2).",
    )
    parser.add_argument(
        "--duplicate_targeted",
        action="store_true",
        default=True,
        help="Include targeted examples twice for 2x error mitigation representation (default: True).",
    )
    parser.add_argument(
        "--no_duplicate_targeted",
        action="store_false",
        dest="duplicate_targeted",
        help="Include targeted examples exactly once without duplication.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic interleaving/shuffling (default: 42).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Execution profile ('local' or 'colab').",
    )
    parser.add_argument(
        "--sample_mode",
        action="store_true",
        help="Run quickly on sample data for CI / development testing.",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=None,
        help="Path to splits directory containing forbidden IDs (default: data/splits).",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to test.parquet for contamination verification.",
    )

    args = parser.parse_args(args_list)

    curated_path = Path(args.curated_file) if args.curated_file else None
    targeted_path = Path(args.targeted_file) if args.targeted_file else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    splits_dir = Path(args.splits_dir) if args.splits_dir else None
    test_file = Path(args.test_file) if args.test_file else None

    try:
        run_freeze_round2_pipeline(
            curated_file=curated_path,
            targeted_file=targeted_path,
            output_dir=output_dir,
            duplicate_targeted=args.duplicate_targeted,
            seed=args.seed,
            sample_mode=args.sample_mode,
            splits_dir=splits_dir,
            test_file=test_file,
            profile=args.profile,
        )
        return 0
    except Exception as e:
        logger.exception("Failed to freeze Round-2 SFT dataset: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
