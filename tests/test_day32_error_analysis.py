"""Unit tests for Phase 5 Day 32: SFT error analysis on val_analysis and 30-sample calibration.

Tests:
1. Design Rule 2 (Test-Set Wall): Strict prevention of evaluation on 'test' split.
2. 30-sample human agreement calibration:
   - Loading 30 labeled error examples.
   - Agreement calculation across all 7 categories.
   - Day 32 Gate verification (agreement >= 80%).
   - Confusion matrix and per-category recall.
3. Agreement Markdown report generation.
4. CLI execution of scripts/calibrate_error_labels.py.
5. CLI execution of scripts/run_day32_error_analysis.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_day32_error_analysis import run_day32_pipeline
from src.evaluation.error_analysis import (
    ALL_ERROR_CATEGORIES,
    ErrorClassifier,
    compute_label_agreement,
    generate_agreement_markdown,
)
from src.evaluation.evaluator import TestSetWallError


class TestDay32Calibration(unittest.TestCase):
    """Test 30-sample human agreement benchmark and calibration reporting."""

    def setUp(self):
        self.calib_file = Path("data/sample/labeled_30_errors.jsonl")
        self.assertTrue(self.calib_file.exists(), f"Calibration benchmark file missing: {self.calib_file}")

        self.records = []
        with open(self.calib_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def test_calibration_dataset_integrity(self):
        self.assertEqual(len(self.records), 30, "Calibration dataset must contain exactly 30 problems")

        categories_represented = set()
        for rec in self.records:
            self.assertIn("problem_id", rec)
            self.assertIn("question", rec)
            self.assertIn("gold_answer", rec)
            self.assertIn("generated_text", rec)
            self.assertIn("manual_label", rec)
            self.assertIn("manual_rationale", rec)
            self.assertIn(rec["manual_label"], ALL_ERROR_CATEGORIES)
            categories_represented.add(rec["manual_label"])

        # All 7 categories must be represented in the 30-sample benchmark
        self.assertEqual(
            categories_represented,
            set(ALL_ERROR_CATEGORIES),
            "All 7 error categories must be represented in the calibration set",
        )

    def test_agreement_computation_and_gate_pass(self):
        classifier = ErrorClassifier(use_mock=True)
        classifications = classifier.classify_records(self.records)

        report = compute_label_agreement(
            classifications=classifications,
            manual_labeled_records=self.records,
            min_agreement_threshold=0.80,
        )

        self.assertEqual(report.total_samples, 30)
        self.assertGreaterEqual(report.agreement_percent, 80.0, "Day 32 Gate: Agreement must be >= 80%")
        self.assertTrue(report.passed_calibration)

        # Check confusion matrix
        self.assertEqual(len(report.confusion_matrix), 7)
        for cat in ALL_ERROR_CATEGORIES:
            self.assertIn(cat, report.confusion_matrix)

        # Check serialization
        d = report.to_dict()
        self.assertIn("agreement_percent", d)
        self.assertIn("per_category", d)
        self.assertIn("confusion_matrix", d)

    def test_agreement_markdown_formatting(self):
        classifier = ErrorClassifier(use_mock=True)
        classifications = classifier.classify_records(self.records[:10])
        report = compute_label_agreement(classifications, self.records[:10])

        md = generate_agreement_markdown(report)
        self.assertIn("# Error Category Agreement & Calibration Report", md)
        self.assertIn("Calibration Verdict", md)
        self.assertIn("`arithmetic_slip`", md)
        self.assertIn("`parse_error`", md)

    def test_test_set_wall_enforced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fake_test_file = tmp_path / "test.parquet"
            fake_test_file.touch()

            with self.assertRaises(TestSetWallError):
                run_day32_pipeline(
                    val_analysis_file=fake_test_file,
                    calibration_file=self.calib_file,
                    adapter_dir=tmp_path / "adapter",
                    output_dir=tmp_path / "out",
                    reports_dir=tmp_path / "rep",
                    use_mock=True,
                )


class TestDay32CLIExecution(unittest.TestCase):
    """Test CLI invocations for Day 32 calibration and analysis runner."""

    def test_calibrate_error_labels_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_json = tmp_path / "agreement.json"
            out_md = tmp_path / "agreement.md"

            cmd = [
                sys.executable,
                "scripts/calibrate_error_labels.py",
                "--calibration_file",
                "data/sample/labeled_30_errors.jsonl",
                "--output_file",
                str(out_json),
                "--report_file",
                str(out_md),
                "--use_mock",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI error:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

            self.assertTrue(out_json.exists())
            self.assertTrue(out_md.exists())

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["agreement"]["passed_calibration"])
            self.assertGreaterEqual(payload["agreement"]["agreement_percent"], 80.0)

    def test_run_day32_pipeline_cli_sample_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_dir = tmp_path / "outputs"
            rep_dir = tmp_path / "reports"

            cmd = [
                sys.executable,
                "scripts/run_day32_error_analysis.py",
                "--sample_mode",
                "--output_dir",
                str(out_dir),
                "--reports_dir",
                str(rep_dir),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI error:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

            err_json = rep_dir / "error_analysis_sft.json"
            err_md = rep_dir / "error_analysis_sft.md"
            cal_json = rep_dir / "error_label_agreement_30.json"
            cal_md = rep_dir / "error_label_agreement_30.md"

            self.assertTrue(err_json.exists())
            self.assertTrue(err_md.exists())
            self.assertTrue(cal_json.exists())
            self.assertTrue(cal_md.exists())


if __name__ == "__main__":
    unittest.main()
