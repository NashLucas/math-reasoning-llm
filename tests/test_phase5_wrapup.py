"""Unit tests for Phase 5 Day 39: Phase 5 Gate Verifier & Release v0.6 Wrap-Up."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.verify_phase5_gate import main as cli_main
from src.evaluation.evaluator import TestSetWallError
from src.evaluation.phase5_verifier import (
    generate_phase5_wrapup_report,
    save_phase5_summary_json,
    verify_advancing_adapter,
    verify_all_phase5_gates,
    verify_category_shift,
    verify_dataset_manifest,
    verify_format_compliance,
    verify_round_comparison,
    verify_test_set_wall,
    verify_truncation_rate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase5GateVerifier(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        self.reports_dir = self.test_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.adapter_dir = self.test_dir / "adapter_model"
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        (self.adapter_dir / "adapter_config.json").write_text(
            json.dumps({"r": 16, "peft_type": "LORA"}), encoding="utf-8"
        )
        (self.adapter_dir / "adapter_model.bin").write_bytes(b"MOCK_ADAPTER_WEIGHTS")

        self.dataset_dir = self.test_dir / "sft_round2"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "manifest.json").write_text(
            json.dumps({
                "total_examples": 440,
                "sha256_parquet": "20b2a469800854519b924f25b39a2530209ab754820ba9d4b6b89b08bead08a9",
                "frozen": True,
            }),
            encoding="utf-8",
        )

        self.splits_dir = self.test_dir / "splits"
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        (self.splits_dir / "test_ids.json").write_text(
            json.dumps(["test_id_999", "test_id_1000"]), encoding="utf-8"
        )

        # Mock round comparison report
        self.comp_data = {
            "stage": "round_comparison",
            "phase": 5,
            "day": 37,
            "split": "val_gating",
            "round1": {
                "stage": "sft",
                "accuracy_percent": 80.0,
                "format_compliance_percent": 100.0,
                "truncation_percent": 0.0,
                "average_completion_tokens": 65.0,
            },
            "round2": {
                "stage": "sft_targeted",
                "accuracy_percent": 82.0,
                "format_compliance_percent": 99.0,
                "truncation_percent": 1.0,
                "average_completion_tokens": 64.0,
            },
            "decision": {
                "winner": "round_2",
                "decision": "round_2_wins",
                "advancing_stage": "sft_targeted",
                "rationale": "Round 2 achieved higher accuracy on val_gating.",
            },
        }
        self.comp_file = self.reports_dir / "round_comparison.json"
        self.comp_file.write_text(json.dumps(self.comp_data, indent=2), encoding="utf-8")

        # Mock category shift report
        self.shift_data = {
            "stage": "category_shift_analysis",
            "val_analysis_file": str(self.test_dir / "val_analysis.parquet"),
            "summary": {
                "accuracy_delta": 2.0,
                "net_errors_eliminated": 5,
                "transition_counts": {
                    "fixed": 6,
                    "retained_same": 2,
                    "retained_shifted": 1,
                    "regressed": 1,
                    "consistently_correct": 80,
                },
                "transitions_sample": [
                    {"problem_id": "val_id_1", "transition_type": "fixed"},
                    {"problem_id": "val_id_2", "transition_type": "consistently_correct"},
                ],
            },
        }
        self.shift_file = self.reports_dir / "category_shift_analysis.json"
        self.shift_file.write_text(json.dumps(self.shift_data, indent=2), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_verify_round_comparison_pass(self):
        res = verify_round_comparison(self.comp_data)
        self.assertTrue(res.passed)
        self.assertIn("Round 2", res.observed)

    def test_verify_round_comparison_tie_rule(self):
        # 1% difference treated as tie favoring Round 2
        tie_data = dict(self.comp_data)
        tie_data["round2"] = dict(self.comp_data["round2"], accuracy_percent=79.5)
        tie_data["decision"] = {
            "winner": "tie",
            "decision": "tie_round_2_advances",
            "advancing_stage": "sft_targeted",
            "rationale": "Within 2% tie margin.",
        }
        res = verify_round_comparison(tie_data)
        self.assertTrue(res.passed)

    def test_verify_round_comparison_fail_regression(self):
        # Round 2 collapsed by 5% and Round 1 advances
        fail_data = dict(self.comp_data)
        fail_data["round2"] = dict(self.comp_data["round2"], accuracy_percent=72.0)
        fail_data["decision"] = {
            "winner": "round_1",
            "decision": "round_1_wins",
            "advancing_stage": "sft",
            "rationale": "Round 1 clearly superior.",
        }
        res = verify_round_comparison(fail_data)
        self.assertFalse(res.passed)

    def test_verify_format_compliance(self):
        res = verify_format_compliance(self.comp_data)
        self.assertTrue(res.passed)
        self.assertEqual(res.observed, "99.0%")

        # Non-compliant output < 98%
        bad_format = dict(self.comp_data)
        bad_format["round2"] = dict(self.comp_data["round2"], format_compliance_percent=95.0)
        res_bad = verify_format_compliance(bad_format)
        self.assertFalse(res_bad.passed)

    def test_verify_truncation_rate(self):
        res = verify_truncation_rate(self.comp_data)
        self.assertTrue(res.passed)
        self.assertEqual(res.observed, "1.0%")

        # High truncation > 5%
        bad_trunc = dict(self.comp_data)
        bad_trunc["round2"] = dict(self.comp_data["round2"], truncation_percent=8.5)
        res_bad = verify_truncation_rate(bad_trunc)
        self.assertFalse(res_bad.passed)

    def test_verify_test_set_wall_isolated(self):
        res = verify_test_set_wall(
            splits_dir=self.splits_dir,
            comp_input=self.comp_data,
            shift_input=self.shift_data,
        )
        self.assertTrue(res.passed)
        self.assertIn("0 test leaks detected", res.observed)

    def test_verify_test_set_wall_violation_split(self):
        bad_comp = dict(self.comp_data, split="test")
        with self.assertRaises(TestSetWallError):
            verify_test_set_wall(
                splits_dir=self.splits_dir,
                comp_input=bad_comp,
                shift_input=self.shift_data,
            )

    def test_verify_test_set_wall_violation_problem_id(self):
        leaked_shift = dict(self.shift_data)
        leaked_shift["summary"] = dict(
            self.shift_data["summary"],
            transitions_sample=[{"problem_id": "test_id_999"}],
        )
        with self.assertRaises(TestSetWallError):
            verify_test_set_wall(
                splits_dir=self.splits_dir,
                comp_input=self.comp_data,
                shift_input=leaked_shift,
            )

    def test_verify_category_shift(self):
        res = verify_category_shift(self.shift_data)
        self.assertTrue(res.passed)
        self.assertIn("Net errors eliminated: +5", res.observed)

        # Regressions > Fixed
        bad_shift = dict(self.shift_data)
        bad_shift["summary"] = dict(
            self.shift_data["summary"],
            net_errors_eliminated=-3,
            transition_counts={"fixed": 1, "regressed": 4},
        )
        res_bad = verify_category_shift(bad_shift)
        self.assertFalse(res_bad.passed)

    def test_verify_advancing_adapter(self):
        res = verify_advancing_adapter(self.adapter_dir)
        self.assertTrue(res.passed)

        missing_dir = self.test_dir / "nonexistent_adapter"
        res_missing = verify_advancing_adapter(missing_dir)
        self.assertFalse(res_missing.passed)

    def test_verify_dataset_manifest(self):
        res = verify_dataset_manifest(self.dataset_dir)
        self.assertTrue(res.passed)
        self.assertIn("440 records", res.observed)

        missing_dir = self.test_dir / "nonexistent_data"
        res_missing = verify_dataset_manifest(missing_dir)
        self.assertFalse(res_missing.passed)

    def test_verify_all_phase5_gates_and_reports(self):
        summary = verify_all_phase5_gates(
            comp_input=self.comp_data,
            shift_input=self.shift_data,
            splits_dir=self.splits_dir,
            adapter_path=self.adapter_dir,
            dataset_dir=self.dataset_dir,
        )
        self.assertTrue(summary.all_passed)
        self.assertEqual(summary.verdict, "PASSED")
        self.assertEqual(len(summary.criteria), 7)
        self.assertEqual(summary.summary_metrics["criteria_passed"], 7)

        # Test Markdown report generation
        md_file = self.reports_dir / "test_wrapup_report.md"
        gen_path = generate_phase5_wrapup_report(summary, md_file)
        self.assertTrue(gen_path.exists())
        content = gen_path.read_text(encoding="utf-8")
        self.assertIn("# Phase 5 Error Analysis & Targeted SFT Wrap-Up Report", content)
        self.assertIn("PASSED (MET)", content)
        self.assertIn("Phase 6 Hand-Off Plan", content)

        # Test JSON summary generation
        json_file = self.reports_dir / "test_summary.json"
        json_path = save_phase5_summary_json(summary, json_file)
        self.assertTrue(json_path.exists())
        with open(json_path, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)
        self.assertEqual(loaded_json["verdict"], "PASSED")
        self.assertEqual(loaded_json["release_tag"], "v0.6")

    def test_cli_runner(self):
        cli_args = [
            "scripts/verify_phase5_gate.py",
            "--reports_dir",
            str(self.reports_dir),
            "--adapter_dir",
            str(self.adapter_dir),
            "--dataset_dir",
            str(self.dataset_dir),
            "--splits_dir",
            str(self.splits_dir),
            "--output_report",
            str(self.reports_dir / "cli_wrapup.md"),
            "--output_json",
            str(self.reports_dir / "cli_summary.json"),
            "--quiet",
        ]
        with patch("sys.argv", cli_args):
            exit_code = cli_main()
            self.assertEqual(exit_code, 0)
            self.assertTrue((self.reports_dir / "cli_wrapup.md").exists())
            self.assertTrue((self.reports_dir / "cli_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
