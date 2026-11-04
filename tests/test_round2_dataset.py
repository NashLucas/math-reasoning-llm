"""Unit tests for Phase 5 Day 35: Round-2 SFT Dataset Preparation & Freezing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.freeze_round2_dataset import main as cli_main
from src.data.round2_dataset import (
    Round2DatasetManifest,
    TestSetWallError,
    build_round2_dataset,
    compute_file_sha256,
    freeze_round2_dataset,
    load_forbidden_ids,
    verify_round2_dataset_safety,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRound2Dataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Mock curated records
        self.mock_curated = [
            {
                "problem_id": f"train_prob_{i}",
                "trace_id": f"train_prob_{i}_curated_0",
                "question": f"Train question {i} has 3 baskets with 5 apples each. How many apples total?",
                "completion": f"Let's think step by step.\n3 * 5 = 15.\n#### {15 + i}",
                "gold_answer": str(15 + i),
                "prompt_type": "chain_of_thought",
            }
            for i in range(4)
        ]

        # Mock targeted records
        self.mock_targeted = [
            {
                "problem_id": f"train_prob_{i + 10}",
                "trace_id": f"train_prob_{i + 10}_targeted_arithmetic_slip",
                "question": f"Targeted problem {i}: A runner completes 10 laps of 400m. How many km?",
                "solution": f"Step 1: 10 * 400 = 4000m.\nStep 2: 4000 / 1000 = 4km.\n#### {4 + i}",
                "gold_answer": str(4 + i),
                "target_category": "arithmetic_slip",
                "similarity_score": 0.88,
                "source": "curated_trace",
            }
            for i in range(2)
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_round2_dataset_no_duplication(self):
        """Verify build with duplicate_targeted=False contains exact sum of records."""
        examples = build_round2_dataset(
            curated_records=self.mock_curated,
            targeted_records=self.mock_targeted,
            duplicate_targeted=False,
            seed=42,
        )
        self.assertEqual(len(examples), 6)  # 4 curated + 2 targeted
        self.assertTrue(all(not ex.is_duplicate for ex in examples))

        sources = [ex.source for ex in examples]
        self.assertEqual(sources.count("curated_sft"), 4)
        self.assertEqual(sources.count("targeted_error_mitigation"), 2)

    def test_build_round2_dataset_with_duplication(self):
        """Verify build with duplicate_targeted=True adds 2x representation for targeted examples."""
        examples = build_round2_dataset(
            curated_records=self.mock_curated,
            targeted_records=self.mock_targeted,
            duplicate_targeted=True,
            seed=42,
        )
        self.assertEqual(len(examples), 8)  # 4 curated + 2*2 targeted = 8

        curated = [ex for ex in examples if ex.source == "curated_sft"]
        targeted = [ex for ex in examples if ex.source == "targeted_error_mitigation"]

        self.assertEqual(len(curated), 4)
        self.assertEqual(len(targeted), 4)

        dup_targeted = [ex for ex in targeted if ex.is_duplicate]
        orig_targeted = [ex for ex in targeted if not ex.is_duplicate]

        self.assertEqual(len(orig_targeted), 2)
        self.assertEqual(len(dup_targeted), 2)

        for dup in dup_targeted:
            self.assertTrue(dup.trace_id.endswith("_dup2"))

    def test_canonical_answer_contract_enforcement(self):
        """Verify all completions end in canonical '#### <number>' line and EOS token."""
        examples = build_round2_dataset(
            curated_records=self.mock_curated,
            targeted_records=self.mock_targeted,
            duplicate_targeted=True,
            seed=42,
            eos_token="<|eos|>",
        )
        for ex in examples:
            self.assertTrue(ex.completion.endswith("<|eos|>"))
            clean_comp = ex.completion[:-len("<|eos|>")].strip()
            lines = clean_comp.split("\n")
            self.assertTrue(lines[-1].startswith("#### "))

    def test_test_set_wall_forbidden_ids(self):
        """Verify that any forbidden validation or test problem ID immediately raises TestSetWallError."""
        splits_dir = PROJECT_ROOT / "data" / "splits"
        forbidden_ids = list(load_forbidden_ids(splits_dir=splits_dir))
        self.assertGreater(len(forbidden_ids), 0)

        # Inject a forbidden validation ID into targeted examples
        leaked_id = forbidden_ids[0]
        corrupted_targeted = list(self.mock_targeted)
        corrupted_targeted.append({
            "problem_id": leaked_id,
            "question": "Leaked question from validation split!",
            "solution": "Solution.\n#### 42",
            "gold_answer": "42",
            "target_category": "wrong_setup",
        })

        examples = build_round2_dataset(
            curated_records=self.mock_curated,
            targeted_records=corrupted_targeted,
            duplicate_targeted=False,
            seed=42,
        )

        with self.assertRaises(TestSetWallError):
            verify_round2_dataset_safety(examples=examples, splits_dir=splits_dir)

    def test_freeze_round2_dataset_outputs_and_manifest(self):
        """Verify end-to-end dataset freezing generates valid JSONL, Parquet, and manifest."""
        curated_file = self.test_dir / "curated.jsonl"
        with open(curated_file, "w", encoding="utf-8") as f:
            for r in self.mock_curated:
                f.write(json.dumps(r) + "\n")

        targeted_file = self.test_dir / "targeted.jsonl"
        with open(targeted_file, "w", encoding="utf-8") as f:
            for r in self.mock_targeted:
                f.write(json.dumps(r) + "\n")

        out_dir = self.test_dir / "frozen_out"

        jsonl_path, parquet_path, manifest = freeze_round2_dataset(
            curated_file=curated_file,
            targeted_file=targeted_file,
            output_dir=out_dir,
            duplicate_targeted=True,
            seed=42,
        )

        self.assertTrue(jsonl_path.exists())
        self.assertTrue(parquet_path.exists())
        self.assertTrue((out_dir / "manifest.json").exists())
        self.assertTrue((out_dir / "run_info.json").exists())

        # Check manifest counts
        self.assertIsInstance(manifest, Round2DatasetManifest)
        self.assertEqual(manifest.total_examples, 8)
        self.assertEqual(manifest.curated_count, 4)
        self.assertEqual(manifest.targeted_raw_count, 2)
        self.assertEqual(manifest.targeted_effective_count, 4)
        self.assertTrue(manifest.duplicate_targeted)
        self.assertTrue(manifest.frozen)

        # Check Parquet reading
        df = pd.read_parquet(parquet_path)
        self.assertEqual(len(df), 8)
        self.assertIn("prompt", df.columns)
        self.assertIn("completion", df.columns)
        self.assertIn("text", df.columns)

        # Check SHA-256 consistency
        self.assertEqual(compute_file_sha256(jsonl_path), manifest.sha256_jsonl)
        self.assertEqual(compute_file_sha256(parquet_path), manifest.sha256_parquet)

    def test_cli_runner_sample_mode(self):
        """Verify that CLI runner succeeds with exit code 0 in sample mode."""
        cli_out_dir = self.test_dir / "cli_sample_out"
        exit_code = cli_main([
            "--sample_mode",
            "--output_dir",
            str(cli_out_dir),
            "--duplicate_targeted",
            "--seed",
            "42",
        ])
        self.assertEqual(exit_code, 0)
        self.assertTrue((cli_out_dir / "train_round2.jsonl").exists())
        self.assertTrue((cli_out_dir / "train_round2.parquet").exists())
        self.assertTrue((cli_out_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
