"""Unit tests for Phase 5 Day 34: Targeted Example Selection for SFT Round 2."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.selection import (
    TargetedSelectionReport,
    TargetedSelector,
    load_forbidden_ids,
    load_training_pool,
)
from src.evaluation.evaluator import TestSetWallError


class TestForbiddenIDsAndTestSetWall(unittest.TestCase):
    """Test strict isolation of validation and test split IDs."""

    def test_load_forbidden_ids_contains_val_and_test(self):
        forbidden = load_forbidden_ids()
        self.assertGreater(len(forbidden), 1000)

        # Check sample of known val and test IDs from splits
        val_path = Path("data/splits/val_ids.json")
        test_path = Path("data/splits/test_ids.json")
        if val_path.exists():
            val_ids = json.loads(val_path.read_text(encoding="utf-8"))
            for pid in val_ids[:10]:
                self.assertIn(pid, forbidden)
        if test_path.exists():
            test_ids = json.loads(test_path.read_text(encoding="utf-8"))
            for pid in test_ids[:10]:
                self.assertIn(pid, forbidden)

    def test_load_training_pool_filters_forbidden(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_pq = tmp_path / "mock_train.parquet"

            # Create mock train data with 2 valid and 2 forbidden IDs
            forbidden = {"val_p1", "test_p2"}
            df_mock = pd.DataFrame([
                {"problem_id": "valid_01", "question": "What is 2+2?", "cleaned_solution": "2+2=4\n#### 4", "gold_answer": "4"},
                {"problem_id": "val_p1", "question": "What is 3+3?", "cleaned_solution": "3+3=6\n#### 6", "gold_answer": "6"},
                {"problem_id": "valid_02", "question": "What is 5*5?", "cleaned_solution": "5*5=25\n#### 25", "gold_answer": "25"},
                {"problem_id": "test_p2", "question": "What is 7*8?", "cleaned_solution": "7*8=56\n#### 56", "gold_answer": "56"},
            ])
            df_mock.to_parquet(train_pq, index=False)

            clean_df = load_training_pool(train_pq, forbidden_ids=forbidden)
            self.assertEqual(len(clean_df), 2)
            self.assertNotIn("val_p1", clean_df["problem_id"].values)
            self.assertNotIn("test_p2", clean_df["problem_id"].values)
            self.assertIn("valid_01", clean_df["problem_id"].values)
            self.assertIn("valid_02", clean_df["problem_id"].values)

    def test_test_set_path_triggers_wall_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fake_test = tmp_path / "test.parquet"
            fake_test.touch()

            with self.assertRaises(TestSetWallError):
                load_training_pool(fake_test)


class TestTargetedSelectorLogic(unittest.TestCase):
    """Test targeted example selection, attachment, and bounds."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        # 1. Mock Training Pool (6 problems)
        self.mock_train_file = self.tmp_path / "train.parquet"
        self.mock_train_data = [
            {
                "problem_id": "train_01",
                "question": "A car travels 120 miles in 2 hours. What is its speed?",
                "cleaned_solution": "120 / 2 = 60.\n#### 60",
                "gold_answer": "60",
            },
            {
                "problem_id": "train_02",
                "question": "If a train goes 45 miles each hour for 3 hours, how far does it go?",
                "cleaned_solution": "45 * 3 = 135.\n#### 135",
                "gold_answer": "135",
            },
            {
                "problem_id": "train_03",
                "question": "Tom had $50 and spent $20 on lunch. How much does he have left?",
                "cleaned_solution": "50 - 20 = 30.\n#### 30",
                "gold_answer": "30",
            },
            {
                "problem_id": "train_04",
                "question": "A shirt costs $40 with a 10% discount. What is the discount amount?",
                "cleaned_solution": "40 * 0.10 = 4.\n#### 4",
                "gold_answer": "4",
            },
            {
                "problem_id": "train_05",
                "question": "Divide 60 cookies equally among 5 kids.",
                "cleaned_solution": "60 / 5 = 12.\n#### 12",
                "gold_answer": "12",
            },
            {
                "problem_id": "train_06",
                "question": "There are 8 rows of chairs with 6 chairs per row.",
                "cleaned_solution": "8 * 6 = 48.\n#### 48",
                "gold_answer": "48",
            },
        ]
        pd.DataFrame(self.mock_train_data).to_parquet(self.mock_train_file, index=False)

        # 2. Mock Curated Traces (only train_01 has curated trace)
        self.mock_curated_file = self.tmp_path / "traces_curated.jsonl"
        with open(self.mock_curated_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "trace_id": "trace_train_01",
                "problem_id": "train_01",
                "completion": "Let's calculate step by step.\nSpeed = Distance / Time = 120 / 2 = 60 mph.\n#### 60",
                "prompt_type": "chain_of_thought",
                "is_correct": True,
            }) + "\n")

        # 3. Mock Error Records
        self.mock_errors = [
            {
                "problem_id": "err_speed_01",
                "question": "A car travels 180 miles in 3 hours. What is its speed?",
                "gold_answer": "60",
                "category": "wrong_setup",
                "is_error": True,
            },
            {
                "problem_id": "err_money_01",
                "question": "Tom had $100 and spent $35 on a coat. How much money is left?",
                "gold_answer": "65",
                "category": "arithmetic_slip",
                "is_error": True,
            },
        ]

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_curated_trace_attachment_vs_gold_solution(self):
        selector = TargetedSelector(
            train_pool_path=self.mock_train_file,
            curated_traces_path=self.mock_curated_file,
            min_per_category=1,
            max_per_category=5,
            default_per_category=2,
            use_mock=True,
        )

        selected = selector.select_for_errors(self.mock_errors, target_per_category=2)
        self.assertEqual(len(selected), 4)  # 2 categories x 2 examples = 4

        # Verify train_01 attaches curated trace
        train_01_ex = next((ex for ex in selected if ex.problem_id == "train_01"), None)
        self.assertIsNotNone(train_01_ex)
        self.assertEqual(train_01_ex.source, "curated_trace")
        self.assertEqual(train_01_ex.trace_id, "trace_train_01")
        self.assertIn("Speed = Distance / Time", train_01_ex.solution)

        # Verify other problems attach cleaned_gold_solution
        other_ex = next((ex for ex in selected if ex.problem_id != "train_01"), None)
        self.assertIsNotNone(other_ex)
        self.assertEqual(other_ex.source, "cleaned_gold_solution")
        self.assertIsNone(other_ex.trace_id)
        self.assertIn("####", other_ex.solution)

    def test_per_category_clamping_and_bounds(self):
        selector = TargetedSelector(
            train_pool_path=self.mock_train_file,
            curated_traces_path=self.mock_curated_file,
            min_per_category=2,
            max_per_category=4,
            default_per_category=3,
            use_mock=True,
        )
        selected = selector.select_for_errors(self.mock_errors, target_per_category=3)
        self.assertEqual(len(selected), 6)  # 2 categories x 3 examples

    def test_export_targeted_dataset_creates_all_artifacts(self):
        selector = TargetedSelector(
            train_pool_path=self.mock_train_file,
            curated_traces_path=self.mock_curated_file,
            min_per_category=1,
            max_per_category=3,
            default_per_category=2,
            use_mock=True,
        )
        selected = selector.select_for_errors(self.mock_errors, target_per_category=2)

        out_dir = self.tmp_path / "targeted_out"
        report: TargetedSelectionReport = selector.export_targeted_dataset(selected, out_dir)

        self.assertTrue((out_dir / "targeted_examples.jsonl").exists())
        self.assertTrue((out_dir / "targeted_examples.parquet").exists())
        self.assertTrue((out_dir / "manifest.json").exists())
        self.assertTrue((out_dir / "run_info.json").exists())

        self.assertEqual(report.total_selected, 4)
        self.assertEqual(report.examples_per_category["wrong_setup"], 2)
        self.assertEqual(report.examples_per_category["arithmetic_slip"], 2)
        self.assertEqual(report.sources_count["curated_trace"], 1)
        self.assertEqual(report.sources_count["cleaned_gold_solution"], 3)


class TestDay34CLIExecution(unittest.TestCase):
    """Test Day 34 CLI runners and argument wiring."""

    def test_run_day34_selection_cli_sample_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_dir = tmp_path / "targeted"

            cmd = [
                sys.executable,
                "scripts/run_day34_selection.py",
                "--sample_mode",
                "--error_file",
                "data/sample/labeled_30_errors.jsonl",
                "--train_pool",
                "data/sample/train_sample.parquet",
                "--curated_traces",
                "data/synthetic/curated/traces_curated_sft.jsonl",
                "--output_dir",
                str(out_dir),
                "--target_per_category",
                "2",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI error:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

            self.assertTrue((out_dir / "targeted_examples.jsonl").exists())
            self.assertTrue((out_dir / "targeted_examples.parquet").exists())
            self.assertTrue((out_dir / "manifest.json").exists())

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertGreater(manifest["total_selected"], 0)
            self.assertEqual(manifest["contamination_check"], "PASSED (0 validation/test problems)")

    def test_run_day34_selection_cli_with_profile_arg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_dir = tmp_path / "targeted_profile"

            cmd = [
                sys.executable,
                "scripts/run_day34_selection.py",
                "--sample_mode",
                "--profile",
                "local",
                "--error_file",
                "data/sample/labeled_30_errors.jsonl",
                "--train_pool",
                "data/sample/train_sample.parquet",
                "--curated_traces",
                "data/synthetic/curated/traces_curated_sft.jsonl",
                "--output_dir",
                str(out_dir),
                "--target_per_category",
                "2",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI error:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

            self.assertTrue((out_dir / "targeted_examples.jsonl").exists())
            self.assertTrue((out_dir / "manifest.json").exists())

    def test_error_analysis_cli_with_select_targeted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_json = tmp_path / "error_analysis.json"
            out_md = tmp_path / "error_report.md"
            targeted_dir = tmp_path / "targeted_data"

            cmd = [
                sys.executable,
                "scripts/error_analysis.py",
                "--sample_mode",
                "--input_file",
                "data/sample/sample_errors.jsonl",
                "--output_file",
                str(out_json),
                "--report_file",
                str(out_md),
                "--select_targeted",
                "--targeted_output_dir",
                str(targeted_dir),
                "--target_per_category",
                "2",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI error:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

            self.assertTrue((targeted_dir / "targeted_examples.jsonl").exists())
            self.assertTrue((targeted_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
