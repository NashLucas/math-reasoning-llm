"""Unit tests for sample evaluation sets generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.sample_sets import create_and_save_sample_sets, extract_sample_splits


class TestSampleSets(unittest.TestCase):
    """Test suite for sample sets creation, validation, and serialization."""

    def setUp(self) -> None:
        """Create mock splits DataFrame for testing."""
        self.mock_splits = {
            "train": pd.DataFrame([
                {
                    "problem_id": f"train_{i}",
                    "question": f"Question train {i}",
                    "answer": f"Reasoning {i} #### {i}",
                    "cleaned_solution": f"Reasoning {i} #### {i}",
                    "gold_answer": str(i),
                    "split": "train",
                }
                for i in range(100)
            ]),
            "val_analysis": pd.DataFrame([
                {
                    "problem_id": f"val_an_{i}",
                    "question": f"Question val an {i}",
                    "answer": f"Reasoning {i} #### {i}",
                    "cleaned_solution": f"Reasoning {i} #### {i}",
                    "gold_answer": str(i),
                    "split": "val_analysis",
                }
                for i in range(25)
            ]),
            "val_gating": pd.DataFrame([
                {
                    "problem_id": f"val_ga_{i}",
                    "question": f"Question val ga {i}",
                    "answer": f"Reasoning {i} #### {i}",
                    "cleaned_solution": f"Reasoning {i} #### {i}",
                    "gold_answer": str(i),
                    "split": "val_gating",
                }
                for i in range(25)
            ]),
            "test": pd.DataFrame([
                {
                    "problem_id": f"test_{i}",
                    "question": f"Question test {i}",
                    "answer": f"Reasoning {i} #### {i}",
                    "cleaned_solution": f"Reasoning {i} #### {i}",
                    "gold_answer": str(i),
                    "split": "test",
                }
                for i in range(50)
            ]),
        }

    def test_extract_sample_splits_counts(self) -> None:
        """Verify that sample sets have exactly the requested count."""
        samples = extract_sample_splits(self.mock_splits, sample_size=20, seed=42)
        self.assertEqual(len(samples["train"]), 20)
        self.assertEqual(len(samples["val"]), 20)
        self.assertEqual(len(samples["test"]), 20)

    def test_extract_sample_splits_balance_in_validation(self) -> None:
        """Verify that validation sample contains equal parts analysis and gating."""
        samples = extract_sample_splits(self.mock_splits, sample_size=20, seed=42)
        val_df = samples["val"]
        analysis_ids = set(self.mock_splits["val_analysis"]["problem_id"])
        gating_ids = set(self.mock_splits["val_gating"]["problem_id"])

        an_count = sum(1 for pid in val_df["problem_id"] if pid in analysis_ids)
        ga_count = sum(1 for pid in val_df["problem_id"] if pid in gating_ids)

        self.assertEqual(an_count, 10)
        self.assertEqual(ga_count, 10)

    def test_sample_sets_no_overlap(self) -> None:
        """Verify zero problem ID overlap between train, val, and test samples."""
        samples = extract_sample_splits(self.mock_splits, sample_size=20, seed=42)
        train_ids = set(samples["train"]["problem_id"])
        val_ids = set(samples["val"]["problem_id"])
        test_ids = set(samples["test"]["problem_id"])

        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)

    def test_sample_sets_determinism(self) -> None:
        """Verify that same seed generates strictly identical problem sets."""
        s1 = extract_sample_splits(self.mock_splits, sample_size=20, seed=42)
        s2 = extract_sample_splits(self.mock_splits, sample_size=20, seed=42)

        for split in ["train", "val", "test"]:
            self.assertEqual(
                s1[split]["problem_id"].tolist(),
                s2[split]["problem_id"].tolist(),
            )

    def test_create_and_save_sample_sets_disk(self) -> None:
        """Verify that create_and_save_sample_sets creates valid Parquet, JSONL, and manifest files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            proc_dir = tmp_path / "processed"
            out_dir = tmp_path / "sample"

            # Setup processed dir structure
            (proc_dir / "train").mkdir(parents=True)
            (proc_dir / "val").mkdir(parents=True)
            (proc_dir / "test").mkdir(parents=True)

            self.mock_splits["train"].to_parquet(proc_dir / "train" / "train.parquet")
            val_full = pd.concat([self.mock_splits["val_analysis"], self.mock_splits["val_gating"]])
            val_full.to_parquet(proc_dir / "val" / "val.parquet")
            self.mock_splits["val_analysis"].to_parquet(proc_dir / "val" / "val_analysis.parquet")
            self.mock_splits["val_gating"].to_parquet(proc_dir / "val" / "val_gating.parquet")
            self.mock_splits["test"].to_parquet(proc_dir / "test" / "test.parquet")

            saved = create_and_save_sample_sets(
                sample_size=20,
                seed=42,
                processed_dir=proc_dir,
                output_dir=out_dir,
            )

            # Check that files exist on disk
            for key in ["train_parquet", "train_jsonl", "val_parquet", "val_jsonl", "test_parquet", "test_jsonl", "manifest"]:
                self.assertIn(key, saved)
                self.assertTrue(saved[key].exists())

            # Read back parquet and verify
            df_train = pd.read_parquet(saved["train_parquet"])
            self.assertEqual(len(df_train), 20)
            self.assertIn("problem_id", df_train.columns)
            self.assertIn("gold_answer", df_train.columns)

            # Read manifest
            with open(saved["manifest"], "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            self.assertEqual(manifest_data["sample_size"], 20)
            self.assertEqual(manifest_data["seed"], 42)
            self.assertEqual(len(manifest_data["splits"]["train"]["problem_ids"]), 20)


if __name__ == "__main__":
    unittest.main()
