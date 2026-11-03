"""Unit tests for curated reasoning dataset freeze, hashing, and Gate v0.4 verification (Phase 3 Day 25)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.data.curation import (
    ContaminationError,
    CuratedDatasetManifest,
    compute_file_sha256,
    freeze_curated_dataset,
    normalize_reasoning_completion,
)


class TestSHA256Computation(unittest.TestCase):
    """Test suite for file SHA-256 hashing."""

    def test_compute_file_sha256_known_hash(self) -> None:
        """Verify SHA-256 computation against a known string."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test.txt"
            content = b"mathematical reasoning post-training test string\n"
            test_file.write_bytes(content)

            expected = hashlib.sha256(content).hexdigest()
            actual = compute_file_sha256(test_file)
            self.assertEqual(actual, expected)

    def test_compute_file_sha256_missing_file_raises(self) -> None:
        """Attempting to hash a missing file must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            compute_file_sha256("non_existent_file_path_12345.jsonl")


class TestCompletionNormalization(unittest.TestCase):
    """Test suite for format hygiene and normalization."""

    def test_normalize_strips_trailing_whitespace(self) -> None:
        """Trailing whitespace and spaces after answer line should be cleanly stripped."""
        raw = "Step 1: calculate 5 * 6 = 30.\n#### 30   \n\n  "
        normalized = normalize_reasoning_completion(raw)
        self.assertEqual(normalized, "Step 1: calculate 5 * 6 = 30.\n#### 30")

    def test_normalize_ensures_single_newline_before_answer(self) -> None:
        """Multiple newlines before answer delimiter should be collapsed to single newline."""
        raw = "Step 1: add 10 + 20 = 30.\n\n\n#### 30"
        normalized = normalize_reasoning_completion(raw)
        self.assertEqual(normalized, "Step 1: add 10 + 20 = 30.\n#### 30")

    def test_normalize_without_answer_line(self) -> None:
        """Text without answer line should just be cleanly stripped."""
        raw = "   Incomplete text without final answer   "
        normalized = normalize_reasoning_completion(raw)
        self.assertEqual(normalized, "Incomplete text without final answer")


class TestFreezeDatasetLogic(unittest.TestCase):
    """Test suite for freeze_curated_dataset logic."""

    def test_freeze_curated_dataset_success(self) -> None:
        """Freezing dataset normalizes completions, sorts deterministically, and writes manifest."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            curated_file = tmp_path / "traces_input.jsonl"
            records = [
                {
                    "trace_id": "p2_cot_s0",
                    "problem_id": "prob_b",
                    "gold_answer": "50",
                    "completion": "Solution for B: 25 * 2 = 50.  \n\n#### 50   ",
                    "prompt_type": "chain_of_thought",
                    "rank_within_problem": 1,
                    "completion_tokens": 15,
                },
                {
                    "trace_id": "p1_cot_s0",
                    "problem_id": "prob_a",
                    "gold_answer": "42",
                    "completion": "Solution for A: 6 * 7 = 42.\n#### 42",
                    "prompt_type": "chain_of_thought",
                    "rank_within_problem": 1,
                    "completion_tokens": 12,
                },
            ]
            with open(curated_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            splits_dir = tmp_path / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            (splits_dir / "val_ids.json").write_text(json.dumps(["val_1"]), encoding="utf-8")
            (splits_dir / "test_ids.json").write_text(json.dumps(["test_1"]), encoding="utf-8")

            frozen_path, manifest = freeze_curated_dataset(
                curated_file=curated_file,
                splits_dir=splits_dir,
                target_problems=2,
            )

            self.assertTrue(frozen_path.exists())
            self.assertIsInstance(manifest, CuratedDatasetManifest)
            self.assertEqual(manifest.total_traces, 2)
            self.assertEqual(manifest.unique_problems, 2)
            self.assertTrue(manifest.contamination_check_passed)
            self.assertTrue(manifest.sha256)

            # Check deterministic sorting: prob_a must come before prob_b
            lines = [json.loads(line) for line in frozen_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(lines[0]["problem_id"], "prob_a")
            self.assertEqual(lines[1]["problem_id"], "prob_b")

            # Check manifest file written
            manifest_file = tmp_path / "traces_input_manifest.json"
            self.assertTrue(manifest_file.exists())
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest_data["sha256"], manifest.sha256)

    def test_freeze_curated_dataset_rejects_contamination(self) -> None:
        """Freeze must fail immediately if any validation or test ID is present."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            curated_file = tmp_path / "traces_contaminated.jsonl"
            records = [
                {
                    "trace_id": "leak_1",
                    "problem_id": "val_leak_problem",
                    "gold_answer": "10",
                    "completion": "10.\n#### 10",
                }
            ]
            with open(curated_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            splits_dir = tmp_path / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            (splits_dir / "val_ids.json").write_text(json.dumps(["val_leak_problem"]), encoding="utf-8")
            (splits_dir / "test_ids.json").write_text(json.dumps(["test_1"]), encoding="utf-8")

            with self.assertRaises(ContaminationError):
                freeze_curated_dataset(
                    curated_file=curated_file,
                    splits_dir=splits_dir,
                )


class TestColabLauncherAndCLIFreeze(unittest.TestCase):
    """Test suite validating launcher Step 22 and CLI freeze execution."""

    def test_colab_launcher_contains_step22(self) -> None:
        """Verify notebooks/00_colab_launcher.ipynb contains Step 22 for dataset freeze."""
        nb_path = Path("notebooks/00_colab_launcher.ipynb")
        self.assertTrue(nb_path.exists())

        content = json.loads(nb_path.read_text(encoding="utf-8"))
        all_sources = "\n".join("".join(cell.get("source", [])) for cell in content["cells"])
        self.assertIn("Step 22: Phase 3 Day 25 — Dataset Freeze", all_sources)
        self.assertIn("scripts/freeze_curated_data.py", all_sources)

    def test_cli_freeze_script_execution(self) -> None:
        """Run scripts/freeze_curated_data.py via subprocess and verify output artifacts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            curated_file = tmp_path / "traces_test.jsonl"
            records = [
                {
                    "trace_id": "test_t1",
                    "problem_id": "test_p1",
                    "gold_answer": "100",
                    "completion": "100.\n#### 100",
                    "prompt_type": "chain_of_thought",
                    "rank_within_problem": 1,
                    "completion_tokens": 10,
                }
            ]
            with open(curated_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            splits_dir = tmp_path / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            (splits_dir / "val_ids.json").write_text(json.dumps(["val_x"]), encoding="utf-8")
            (splits_dir / "test_ids.json").write_text(json.dumps(["test_x"]), encoding="utf-8")

            manifest_file = tmp_path / "manifest.json"
            reports_dir = tmp_path / "reports"

            cmd = [
                sys.executable,
                "scripts/freeze_curated_data.py",
                "--curated_file",
                str(curated_file),
                "--manifest_file",
                str(manifest_file),
                "--splits_dir",
                str(splits_dir),
                "--reports_dir",
                str(reports_dir),
                "--report_name",
                "test_freeze_report",
                "--target_problems",
                "1",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                result.returncode,
                0,
                f"Script failed with output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )

            self.assertTrue(manifest_file.exists())
            self.assertTrue((reports_dir / "test_freeze_report.md").exists())
            self.assertTrue((reports_dir / "test_freeze_report.json").exists())


if __name__ == "__main__":
    unittest.main()
