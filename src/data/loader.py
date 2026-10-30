"""GSM8K dataset loading, stable ID hashing, and raw Parquet persistence."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union

import pandas as pd

# Ensure project root is on sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_GSM8K_DIR = PROJECT_ROOT / "data" / "raw" / "gsm8k"

logger = logging.getLogger(__name__)


def generate_problem_id(question: str) -> str:
    """Generate a deterministic, stable problem ID from a SHA-256 hash of the question text."""
    normalized = " ".join(question.strip().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"gsm8k_{digest}"


def download_and_save_raw_gsm8k(
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Download openai/gsm8k (main config), assign stable problem IDs, and save as Parquet.

    Returns:
        Tuple of (train_parquet_path, test_parquet_path)
    """
    from datasets import load_dataset

    target_dir = Path(output_dir) if output_dir else RAW_GSM8K_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Loading openai/gsm8k (configuration: main) from Hugging Face Hub...")
    raw_dataset = load_dataset("openai/gsm8k", "main")

    # Process Train Split
    train_records: List[dict] = []
    for item in raw_dataset["train"]:
        q = item["question"].strip()
        ans = item["answer"].strip()
        train_records.append(
            {
                "problem_id": generate_problem_id(q),
                "question": q,
                "answer": ans,
                "split": "train",
            }
        )
    train_df = pd.DataFrame(train_records)

    # Process Test Split
    test_records: List[dict] = []
    for item in raw_dataset["test"]:
        q = item["question"].strip()
        ans = item["answer"].strip()
        test_records.append(
            {
                "problem_id": generate_problem_id(q),
                "question": q,
                "answer": ans,
                "split": "test",
            }
        )
    test_df = pd.DataFrame(test_records)

    # Validate Counts per Implementation Plan (7,473 train, 1,319 test)
    assert len(train_df) == 7473, f"Expected 7,473 train problems, found {len(train_df)}"
    assert len(test_df) == 1319, f"Expected 1,319 test problems, found {len(test_df)}"

    train_path = target_dir / "train.parquet"
    test_path = target_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Raw train split saved: {train_path} ({len(train_df)} problems)")
    print(f"Raw test split saved:  {test_path} ({len(test_df)} problems)")
    return train_path, test_path


def load_raw_gsm8k(
    split: str = "train", data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Load a raw GSM8K Parquet dataset split ('train' or 'test')."""
    base_dir = Path(data_dir) if data_dir else RAW_GSM8K_DIR
    target_file = base_dir / f"{split}.parquet"
    if not target_file.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at {target_file}. Run download_and_save_raw_gsm8k first."
        )
    return pd.read_parquet(target_file)


if __name__ == "__main__":
    train_p, test_p = download_and_save_raw_gsm8k()
    print("\nVerifying loaded raw datasets:")
    train_df = load_raw_gsm8k("train")
    test_df = load_raw_gsm8k("test")
    print(f"Train columns: {list(train_df.columns)}, Shape: {train_df.shape}")
    print(f"Test columns:  {list(test_df.columns)}, Shape: {test_df.shape}")

    print("\nInspecting first 5 problem IDs:")
    for _, row in train_df.head(5).iterrows():
        print(f"[{row['problem_id']}] Q: {row['question'][:60]}... -> Ans ending: {row['answer'][-20:]}")
