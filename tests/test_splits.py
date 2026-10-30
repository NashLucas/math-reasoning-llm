"""Unit tests for dataset cleaning, validation split carving, and split isolation."""

import json
import tempfile
import unittest

import pandas as pd

from src.data.loader import load_raw_gsm8k
from src.data.preprocessing import (
    clean_solution,
    extract_gold_answer,
    partition_splits,
    prepare_all_data,
    strip_calculator_annotations,
)


class TestSplits(unittest.TestCase):
    def test_strip_calculator_annotations(self):
        raw = "Natalia sold 48/2 = <<48/2=24>>24 clips in May. // Total: <<48+24=72>>72"
        expected = "Natalia sold 48/2 = 24 clips in May. // Total: 72"
        self.assertEqual(strip_calculator_annotations(raw), expected)

    def test_extract_gold_answer(self):
        sol1 = "Step by step reasoning\n#### 42"
        self.assertEqual(extract_gold_answer(sol1), "42")

        sol2 = "Intermediate #### 10 followed by final\n#### 99"
        self.assertEqual(extract_gold_answer(sol2), "99")

        sol3 = "Reasoning with currency\n#### $1,250"
        self.assertEqual(extract_gold_answer(sol3), "$1,250")

    def test_clean_solution(self):
        raw = "She had 10*2 = <<10*2=20>>20 apples.\n#### 20"
        cleaned, gold = clean_solution(raw)
        self.assertEqual(gold, "20")
        self.assertNotIn("<<", cleaned)
        self.assertNotIn(">>", cleaned)
        self.assertTrue(cleaned.endswith("#### 20"))

    def test_split_counts_and_isolation(self):
        raw_train = load_raw_gsm8k("train")
        raw_test = load_raw_gsm8k("test")

        splits = partition_splits(raw_train, raw_test, seed=42)

        # 1. Verify counts
        self.assertEqual(len(splits["train"]), 6973)
        self.assertEqual(len(splits["val"]), 500)
        self.assertEqual(len(splits["val_analysis"]), 250)
        self.assertEqual(len(splits["val_gating"]), 250)
        self.assertEqual(len(splits["test"]), 1319)

        train_ids = set(splits["train"]["problem_id"])
        val_ids = set(splits["val"]["problem_id"])
        val_analysis_ids = set(splits["val_analysis"]["problem_id"])
        val_gating_ids = set(splits["val_gating"]["problem_id"])
        test_ids = set(splits["test"]["problem_id"])
        raw_train_ids = set(raw_train["problem_id"])

        # 2. Check no overlap between train, val, and test
        self.assertEqual(train_ids.intersection(val_ids), set())
        self.assertEqual(train_ids.intersection(test_ids), set())
        self.assertEqual(val_ids.intersection(test_ids), set())

        # 3. Check val split properties
        self.assertEqual(val_analysis_ids.intersection(val_gating_ids), set())
        self.assertEqual(val_analysis_ids.union(val_gating_ids), val_ids)
        self.assertEqual(train_ids.union(val_ids), raw_train_ids)

    def test_reproducibility_on_rerun(self):
        """Verify that running data preparation twice produces identical splits."""
        raw_train = load_raw_gsm8k("train")
        raw_test = load_raw_gsm8k("test")

        run1 = partition_splits(raw_train, raw_test, seed=42)
        run2 = partition_splits(raw_train, raw_test, seed=42)

        for split_name in ["train", "val", "val_analysis", "val_gating", "test"]:
            ids1 = run1[split_name]["problem_id"].tolist()
            ids2 = run2[split_name]["problem_id"].tolist()
            self.assertEqual(ids1, ids2, f"Split {split_name} problem IDs differed on rerun!")

    def test_prepare_all_data_execution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_paths = prepare_all_data(seed=42, output_dir=tmp_dir)

            # Check that files were created
            self.assertTrue(out_paths["train_parquet"].exists())
            self.assertTrue(out_paths["val_analysis_parquet"].exists())
            self.assertTrue(out_paths["val_gating_parquet"].exists())
            self.assertTrue(out_paths["test_parquet"].exists())

            # Read back and verify row count
            df_train = pd.read_parquet(out_paths["train_parquet"])
            self.assertEqual(len(df_train), 6973)

            with open(out_paths["val_analysis_ids"], "r", encoding="utf-8") as f:
                val_ids = json.load(f)
            self.assertEqual(len(val_ids), 250)


if __name__ == "__main__":
    unittest.main()
