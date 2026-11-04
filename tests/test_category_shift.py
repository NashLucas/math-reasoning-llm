"""Unit tests for Phase 5 Day 38: SFT Round 2 Category Shift Analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_day38_shift_analysis import main as cli_main
from src.evaluation.category_shift import (
    CategoryShiftMetric,
    CategoryShiftSummary,
    append_category_shift_to_markdown_report,
    compute_category_shift,
    generate_category_shift_table,
    run_category_shift_analysis,
)
from src.evaluation.evaluator import TestSetWallError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestCategoryShiftAnalysis(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        self.reports_dir = self.test_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.adapter_dir = self.test_dir / "adapter_model"
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        (self.adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 16}), encoding="utf-8")
        (self.adapter_dir / "adapter_model.bin").write_bytes(b"MOCK_ADAPTER")

        # Mock Round 1 report
        self.r1_report_file = self.reports_dir / "mock_r1_report.json"
        mock_r1_data = {
            "summary": {
                "total_evaluated": 10,
                "correct_count": 6,
                "error_count": 4,
                "accuracy": 60.0,
            },
            "classifications": [
                {"problem_id": "prob_1", "question": "Q1", "gold_answer": "10", "is_correct": False, "category": "incomplete_reasoning"},
                {"problem_id": "prob_2", "question": "Q2", "gold_answer": "20", "is_correct": False, "category": "arithmetic_slip"},
                {"problem_id": "prob_3", "question": "Q3", "gold_answer": "30", "is_correct": False, "category": "misread_question"},
                {"problem_id": "prob_4", "question": "Q4", "gold_answer": "40", "is_correct": False, "category": "wrong_setup"},
                {"problem_id": "prob_5", "question": "Q5", "gold_answer": "50", "is_correct": True, "category": None},
                {"problem_id": "prob_6", "question": "Q6", "gold_answer": "60", "is_correct": True, "category": None},
                {"problem_id": "prob_7", "question": "Q7", "gold_answer": "70", "is_correct": True, "category": None},
                {"problem_id": "prob_8", "question": "Q8", "gold_answer": "80", "is_correct": True, "category": None},
                {"problem_id": "prob_9", "question": "Q9", "gold_answer": "90", "is_correct": True, "category": None},
                {"problem_id": "prob_10", "question": "Q10", "gold_answer": "100", "is_correct": True, "category": None},
            ],
        }
        self.r1_report_file.write_text(json.dumps(mock_r1_data, indent=2), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_category_shift_metrics_and_transitions(self):
        """Verify category shift calculation, frequency reduction, and transition categorization."""
        r1_records = [
            {"problem_id": "prob_1", "question": "Q1", "gold_answer": "10", "is_correct": False, "category": "incomplete_reasoning"},
            {"problem_id": "prob_2", "question": "Q2", "gold_answer": "20", "is_correct": False, "category": "arithmetic_slip"},
            {"problem_id": "prob_3", "question": "Q3", "gold_answer": "30", "is_correct": False, "category": "misread_question"},
            {"problem_id": "prob_4", "question": "Q4", "gold_answer": "40", "is_correct": True, "category": None},
        ]

        # Round 2:
        # prob_1 became correct (fixed)
        # prob_2 remained error with same category (retained_same)
        # prob_3 remained error but shifted category (retained_shifted -> wrong_setup)
        # prob_4 regressed from correct to error (regressed -> arithmetic_slip)
        r2_records = [
            {"problem_id": "prob_1", "question": "Q1", "gold_answer": "10", "is_correct": True, "category": None},
            {"problem_id": "prob_2", "question": "Q2", "gold_answer": "20", "is_correct": False, "category": "arithmetic_slip"},
            {"problem_id": "prob_3", "question": "Q3", "gold_answer": "30", "is_correct": False, "category": "wrong_setup"},
            {"problem_id": "prob_4", "question": "Q4", "gold_answer": "40", "is_correct": False, "category": "arithmetic_slip"},
        ]

        summary = compute_category_shift(r1_records, r2_records, total_evaluated=4)
        self.assertEqual(summary.total_evaluated, 4)
        self.assertEqual(summary.round1_error_count, 3)
        self.assertEqual(summary.round2_error_count, 3)
        self.assertEqual(summary.net_errors_eliminated, 0)

        # Check transition counts
        tc = summary.transition_counts
        self.assertEqual(tc["fixed"], 1)  # prob_1
        self.assertEqual(tc["retained_same"], 1)  # prob_2
        self.assertEqual(tc["retained_shifted"], 1)  # prob_3
        self.assertEqual(tc["regressed"], 1)  # prob_4

        # Check category metrics
        metrics_by_cat = {m.category: m for m in summary.category_metrics}
        inc_m = metrics_by_cat["incomplete_reasoning"]
        self.assertEqual(inc_m.round1_count, 1)
        self.assertEqual(inc_m.round2_count, 0)
        self.assertEqual(inc_m.absolute_change, -1)
        self.assertEqual(inc_m.percent_reduction, 100.0)
        self.assertEqual(inc_m.status, "Reduced / Mitigated")

        setup_m = metrics_by_cat["wrong_setup"]
        self.assertEqual(setup_m.round1_count, 0)
        self.assertEqual(setup_m.round2_count, 1)
        self.assertEqual(setup_m.absolute_change, 1)
        self.assertEqual(setup_m.status, "Regressed")

    def test_generate_category_shift_table(self):
        """Verify Markdown table generation formats correctly."""
        metric = CategoryShiftMetric(
            category="arithmetic_slip",
            round1_count=5,
            round1_percentage=50.0,
            round2_count=2,
            round2_percentage=25.0,
            absolute_change=-3,
            relative_change=-25.0,
            percent_reduction=60.0,
            status="Reduced / Mitigated",
        )
        summary = CategoryShiftSummary(
            total_evaluated=10,
            round1_accuracy=50.0,
            round2_accuracy=80.0,
            accuracy_delta=30.0,
            round1_error_count=5,
            round2_error_count=2,
            error_count_delta=-3,
            net_errors_eliminated=3,
            category_metrics=[metric],
            transition_counts={"fixed": 3},
            transitions=[],
        )
        table = generate_category_shift_table(summary)
        self.assertIn("| `arithmetic_slip` | 5 | 50.0% | 2 | 25.0% | `-3` | `+60.0%` | **Reduced** |", table)

    def test_test_set_wall_enforcement(self):
        """Verify accessing test split raises TestSetWallError."""
        with self.assertRaises(TestSetWallError):
            run_category_shift_analysis(
                val_analysis_file="data/processed/test/test.parquet",
                use_mock=True,
            )

    def test_append_category_shift_to_markdown_report(self):
        """Verify Section 5 is appended or updated in main error analysis report."""
        main_md = self.reports_dir / "error_analysis_sft.md"
        main_md.write_text("# Initial Error Analysis\n\nSome preliminary notes.\n", encoding="utf-8")

        summary = compute_category_shift([], [], total_evaluated=10)
        append_category_shift_to_markdown_report(main_md, summary, "mock/path")

        content = main_md.read_text(encoding="utf-8")
        self.assertIn("## 5. SFT Round 2 Category Shift & Error Mitigation Movement", content)
        self.assertIn("### Before-and-After Category Frequency Comparison", content)

    def test_end_to_end_mock_shift_analysis(self):
        """Verify end-to-end execution of run_category_shift_analysis in mock mode."""
        res = run_category_shift_analysis(
            model_name_or_path="deepseek-ai/deepseek-math-7b-base",
            round2_adapter=self.adapter_dir,
            round1_report=self.r1_report_file,
            limit=5,
            use_mock=True,
            reports_dir=self.reports_dir,
        )

        self.assertIn("summary", res)
        self.assertIn("json_report", res)
        self.assertIn("main_markdown_report", res)
        self.assertIn("standalone_markdown_report", res)

        json_p = Path(res["json_report"])
        self.assertTrue(json_p.exists())
        data = json.loads(json_p.read_text(encoding="utf-8"))
        self.assertEqual(data["stage"], "category_shift_analysis")
        self.assertEqual(data["day"], 38)

    def test_cli_execution_sample_mode(self):
        """Verify scripts/run_day38_shift_analysis.py CLI execution in sample mode."""
        import sys
        from unittest.mock import patch

        cli_args = [
            "run_day38_shift_analysis.py",
            "--round2_adapter", str(self.adapter_dir),
            "--round1_report", str(self.r1_report_file),
            "--reports_dir", str(self.reports_dir),
            "--limit", "5",
            "--sample_mode",
        ]
        with patch.object(sys, "argv", cli_args):
            cli_main()

        self.assertTrue((self.reports_dir / "category_shift_analysis.json").exists())
        self.assertTrue((self.reports_dir / "day38_category_shift_report.md").exists())


if __name__ == "__main__":
    unittest.main()
