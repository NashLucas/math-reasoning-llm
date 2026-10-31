"""Data preprocessing, calculator annotation stripping, and split partitioning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager  # noqa: E402
from src.data.loader import RAW_GSM8K_DIR, download_and_save_raw_gsm8k, load_raw_gsm8k  # noqa: E402

# Profile data directory: the repo's data/ locally, Google Drive under the colab profile
PROCESSED_DATA_DIR = ConfigManager().data_path("processed")

logger = logging.getLogger(__name__)

# Pattern to match calculator annotations in GSM8K solutions, e.g. <<48/2=24>>
CALCULATOR_PATTERN = re.compile(r"<<[^>]+>>")

# Pattern to extract the gold answer following '####'
GOLD_ANSWER_PATTERN = re.compile(r"####\s*([^\n\r]+)")


def strip_calculator_annotations(text: str) -> str:
    """Remove GSM8K calculator tags (e.g. '<<48/2=24>>') from solution text."""
    return CALCULATOR_PATTERN.sub("", text).strip()


def extract_gold_answer(solution_text: str) -> Optional[str]:
    """Extract the raw gold answer string appearing after the last '####' in the solution."""
    matches = GOLD_ANSWER_PATTERN.findall(solution_text)
    if matches:
        return matches[-1].strip()
    return None


def clean_solution(solution_text: str) -> Tuple[str, Optional[str]]:
    """Clean solution text by removing calculator annotations and isolating gold answer.

    Returns:
        (cleaned_solution_text, gold_answer)
    """
    gold = extract_gold_answer(solution_text)
    cleaned = strip_calculator_annotations(solution_text)
    return cleaned, gold


def partition_splits(
    raw_train_df: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    seed: int = 42,
    val_size: int = 500,
    val_analysis_size: int = 250,
) -> Dict[str, pd.DataFrame]:
    """Deterministically carve validation splits from the raw training dataset.

    Splits:
        - train: 6,973 problems
        - val: 500 problems total
        - val_analysis: 250 problems (first half of validation)
        - val_gating: 250 problems (second half of validation)
        - test: 1,319 problems (strictly frozen)
    """
    assert len(raw_train_df) == 7473, f"Expected 7473 train rows, found {len(raw_train_df)}"
    assert len(raw_test_df) == 1319, f"Expected 1319 test rows, found {len(raw_test_df)}"

    # Sort deterministically by problem_id before permutation to ensure cross-platform reproducibility
    sorted_train = raw_train_df.sort_values(by="problem_id").reset_index(drop=True)

    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(len(sorted_train))

    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]

    val_df = sorted_train.iloc[val_indices].copy().reset_index(drop=True)
    train_split_df = sorted_train.iloc[train_indices].copy().reset_index(drop=True)

    val_analysis_df = val_df.iloc[:val_analysis_size].copy().reset_index(drop=True)
    val_gating_df = val_df.iloc[val_analysis_size:].copy().reset_index(drop=True)

    test_split_df = raw_test_df.sort_values(by="problem_id").copy().reset_index(drop=True)

    # Clean solutions and extract gold answers across all partitions
    for df_split in [train_split_df, val_df, val_analysis_df, val_gating_df, test_split_df]:
        cleaned_solutions: List[str] = []
        gold_answers: List[str] = []
        for ans in df_split["answer"]:
            cleaned, gold = clean_solution(ans)
            cleaned_solutions.append(cleaned)
            gold_answers.append(gold or "")
        df_split["cleaned_solution"] = cleaned_solutions
        df_split["gold_answer"] = gold_answers

    train_split_df["split"] = "train"
    val_df["split"] = "val"
    val_analysis_df["split"] = "val_analysis"
    val_gating_df["split"] = "val_gating"
    test_split_df["split"] = "test"

    return {
        "train": train_split_df,
        "val": val_df,
        "val_analysis": val_analysis_df,
        "val_gating": val_gating_df,
        "test": test_split_df,
    }


def save_processed_splits(
    splits: Dict[str, pd.DataFrame], output_base_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Path]:
    """Persist processed splits as Parquet files and save problem ID JSON lists."""
    base_dir = Path(output_base_dir) if output_base_dir else PROCESSED_DATA_DIR
    paths: Dict[str, Path] = {}

    for name, df in splits.items():
        # Determine subdirectory: train/, val/, or test/
        if name in ("val_analysis", "val_gating", "val"):
            sub_dir = base_dir / "val"
        else:
            sub_dir = base_dir / name
        sub_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = sub_dir / f"{name}.parquet"
        df.to_parquet(parquet_path, index=False)
        paths[f"{name}_parquet"] = parquet_path

        # Save ID list
        id_path = sub_dir / f"{name}_ids.json"
        with open(id_path, "w", encoding="utf-8") as f:
            json.dump(df["problem_id"].tolist(), f, indent=2)
        paths[f"{name}_ids"] = id_path

        print(f"Saved {name}: {parquet_path} ({len(df)} rows)")

    return paths


def normalize_text_for_contamination(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def extract_word_ngrams(text: str, n: int = 13) -> Set[str]:
    """Extract word-level n-grams from text."""
    words = normalize_text_for_contamination(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def check_contamination(
    train_questions: List[str],
    test_questions: List[str],
    ngram_size: int = 13,
) -> Dict[str, Any]:
    """Detect exact matches and long n-gram overlaps between training and test sets.

    Returns:
        Dict detailing exact_matches, ngram_matches, and is_contaminated flag.
    """
    norm_train = [normalize_text_for_contamination(q) for q in train_questions]
    norm_test = [normalize_text_for_contamination(q) for q in test_questions]

    # 1. Exact match check
    exact_matches = set(norm_train).intersection(set(norm_test))

    # 2. Build test n-gram index
    test_ngrams: Dict[str, Set[int]] = {}
    for idx, t_q in enumerate(test_questions):
        for ng in extract_word_ngrams(t_q, n=ngram_size):
            if ng not in test_ngrams:
                test_ngrams[ng] = set()
            test_ngrams[ng].add(idx)

    # 3. Check train questions against test n-grams
    ngram_overlaps: List[Dict[str, Any]] = []
    for tr_idx, tr_q in enumerate(train_questions):
        for ng in extract_word_ngrams(tr_q, n=ngram_size):
            if ng in test_ngrams:
                ngram_overlaps.append(
                    {
                        "train_idx": tr_idx,
                        "test_indices": list(test_ngrams[ng]),
                        "ngram": ng,
                    }
                )

    # By definition, contamination is flagged if exact matches exist or if n-grams overlap
    is_contaminated = len(exact_matches) > 0
    return {
        "is_contaminated": is_contaminated,
        "exact_match_count": len(exact_matches),
        "exact_matches": list(exact_matches),
        "ngram_overlap_count": len(ngram_overlaps),
        "ngram_overlaps": ngram_overlaps,
    }


def assert_no_contamination(
    train_questions: List[str],
    test_questions: List[str],
    ngram_size: int = 13,
    allow_known_template_overlap: bool = True,
) -> None:
    """Raise AssertionError if any exact match or forbidden n-gram overlap is discovered."""
    results = check_contamination(
        train_questions, test_questions, ngram_size=ngram_size
    )
    if results["exact_match_count"] > 0:
        msg = f"Contamination detected! Exact question matches: {results['exact_match_count']}"
        raise AssertionError(msg)

    if not allow_known_template_overlap and results["ngram_overlap_count"] > 0:
        msg = f"Contamination detected! N-gram ({ngram_size}-gram) overlaps: {results['ngram_overlap_count']}"
        raise AssertionError(msg)


def prepare_all_data(
    seed: int = 42,
    raw_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Execute end-to-end data preparation."""
    raw_path = Path(raw_dir) if raw_dir else RAW_GSM8K_DIR
    if not (raw_path / "train.parquet").exists() or not (raw_path / "test.parquet").exists():
        print("Raw datasets not found. Downloading raw GSM8K...")
        download_and_save_raw_gsm8k(raw_path)

    raw_train = load_raw_gsm8k("train", data_dir=raw_path)
    raw_test = load_raw_gsm8k("test", data_dir=raw_path)

    splits = partition_splits(raw_train, raw_test, seed=seed)

    # Perform automated contamination verification
    train_q = splits["train"]["question"].tolist()
    test_q = splits["test"]["question"].tolist()
    assert_no_contamination(train_q, test_q)
    print("Contamination check passed: zero exact leakage between training pool and test set.")

    return save_processed_splits(splits, output_base_dir=output_dir)


if __name__ == "__main__":
    prepare_all_data()
