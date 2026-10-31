"""Deterministic generation of small sample evaluation sets (20 problems per split).

Used for fast smoke tests, dry runs, pipeline validation, and debugging without
loading full dataset splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.config import PROJECT_ROOT
from src.data.preprocessing import PROCESSED_DATA_DIR, prepare_all_data

# Sample sets are small and tracked in git, so they live in the repo rather than on Drive
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def extract_sample_splits(
    splits: Dict[str, pd.DataFrame],
    sample_size: int = 20,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Sample exactly `sample_size` problems from train, val, and test splits deterministically.

    For validation, draws evenly from `val_analysis` (sample_size // 2) and
    `val_gating` (sample_size // 2) to maintain representative balance.
    """
    train_df = splits["train"]
    test_df = splits["test"]

    # Sample train deterministically
    train_sample = train_df.sample(n=min(sample_size, len(train_df)), random_state=seed).copy()

    # Sample val evenly from analysis and gating halves if present
    if "val_analysis" in splits and "val_gating" in splits:
        half = sample_size // 2
        val_an_sample = splits["val_analysis"].sample(n=min(half, len(splits["val_analysis"])), random_state=seed)
        val_ga_sample = splits["val_gating"].sample(n=min(half, len(splits["val_gating"])), random_state=seed)
        val_sample = pd.concat([val_an_sample, val_ga_sample], ignore_index=True)
    else:
        val_df = splits["val"]
        val_sample = val_df.sample(n=min(sample_size, len(val_df)), random_state=seed).copy()

    # Sample test deterministically
    test_sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=seed).copy()

    return {
        "train": train_sample.reset_index(drop=True),
        "val": val_sample.reset_index(drop=True),
        "test": test_sample.reset_index(drop=True),
    }


def create_and_save_sample_sets(
    sample_size: int = 20,
    seed: int = 42,
    processed_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Load processed GSM8K splits, extract sample sets, and persist as Parquet and JSONL."""
    base_proc = Path(processed_dir) if processed_dir else PROCESSED_DATA_DIR
    target_out = Path(output_dir) if output_dir else DEFAULT_SAMPLE_DIR
    target_out.mkdir(parents=True, exist_ok=True)

    # Ensure processed data exists
    train_file = base_proc / "train" / "train.parquet"
    if not train_file.exists():
        prepare_all_data(seed=seed, output_dir=base_proc)

    # Load splits
    splits = {
        "train": pd.read_parquet(base_proc / "train" / "train.parquet"),
        "val": pd.read_parquet(base_proc / "val" / "val.parquet"),
        "val_analysis": pd.read_parquet(base_proc / "val" / "val_analysis.parquet"),
        "val_gating": pd.read_parquet(base_proc / "val" / "val_gating.parquet"),
        "test": pd.read_parquet(base_proc / "test" / "test.parquet"),
    }

    sample_splits = extract_sample_splits(splits, sample_size=sample_size, seed=seed)

    saved_paths: Dict[str, Path] = {}
    manifest: Dict[str, Any] = {
        "sample_size": sample_size,
        "seed": seed,
        "splits": {},
    }

    for split_name, df in sample_splits.items():
        parquet_path = target_out / f"{split_name}_sample.parquet"
        jsonl_path = target_out / f"{split_name}_sample.jsonl"

        df.to_parquet(parquet_path, index=False)
        df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

        saved_paths[f"{split_name}_parquet"] = parquet_path
        saved_paths[f"{split_name}_jsonl"] = jsonl_path

        manifest["splits"][split_name] = {
            "count": len(df),
            "parquet_file": str(parquet_path.name),
            "jsonl_file": str(jsonl_path.name),
            "problem_ids": df["problem_id"].tolist(),
        }

    manifest_path = target_out / "sample_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    saved_paths["manifest"] = manifest_path

    return saved_paths
