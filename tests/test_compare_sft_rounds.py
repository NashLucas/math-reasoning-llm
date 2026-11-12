"""Unit tests for Phase 5 Day 37: SFT Round Comparison & Gating Evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_sft_rounds import main as cli_main
from src.evaluation.evaluator import TestSetWallError
from src.evaluation.round_comparator import (
    RoundEvalMetrics,
    compare_problem_level_results,
    compute_comparison_decision,
    evaluate_round,
    resolve_round_adapter,
    run_round_comparison,
    stage_advancing_adapter,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRoundComparator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        self.r1_adapter_dir = self.test_dir / "round1_adapter"
        self.r2_adapter_dir = self.test_dir / "round2_adapter"
        self.staging_dir = self.test_dir / "staging_adapter"
        self.reports_dir = self.test_dir / "reports"

        for d in [self.r1_adapter_dir, self.r2_adapter_dir, self.staging_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Mock adapter files
        for ad_dir in [self.r1_adapter_dir, self.r2_adapter_dir]:
            (ad_dir / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}), encoding="utf-8")
            (ad_dir / "adapter_model.bin").write_bytes(b"MOCK_WEIGHTS")
            (ad_dir / "trainer_state.json").write_text(json.dumps({"global_step": 100}), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_decision_rule_round2_wins(self):
        """Verify Round 2 wins definitively when outperforming Round 1 beyond tie margin."""
        r1 = RoundEvalMetrics(
            stage="sft",
            round_name="Round 1 (Curated SFT)",
            adapter_path=str(self.r1_adapter_dir),
            accuracy=0.60,
            accuracy_percent=60.0,
            correct_count=150,
            total_problems=250,
            format_compliance_rate=1.0,
            format_compliance_percent=100.0,
            truncation_rate=0.0,
            truncation_percent=0.0,
            average_completion_tokens=65.0,
        )
        r2 = RoundEvalMetrics(
            stage="sft_targeted",
            round_name="Round 2 (Targeted SFT)",
            adapter_path=str(self.r2_adapter_dir),
            accuracy=0.64,
            accuracy_percent=64.0,
            correct_count=160,
            total_problems=250,
            format_compliance_rate=1.0,
            format_compliance_percent=100.0,
            truncation_rate=0.0,
            truncation_percent=0.0,
            average_completion_tokens=65.0,
        )

        decision = compute_comparison_decision(
            r1=r1,
            r2=r2,
            tie_margin_pct=2.0,
            tie_margin_count=5,
            staging_dir=self.staging_dir,
        )
        self.assertEqual(decision.winner, "round_2")
        self.assertEqual(decision.decision, "round_2_wins")
        self.assertEqual(decision.advancing_stage, "sft_targeted")
        self.assertFalse(decision.is_tie)
        self.assertEqual(decision.difference_accuracy_percent, 4.0)
        self.assertEqual(decision.difference_correct, 10)
        self.assertEqual(decision.advancing_adapter_path, str(self.staging_dir))

    def test_decision_rule_tie_advances_round2(self):
        """Verify that a 1-2 point difference (<=5 problems / <=2.0%) is treated as a tie favoring Round 2."""
        r1 = RoundEvalMetrics(
            stage="sft",
            round_name="Round 1 (Curated SFT)",
            adapter_path=str(self.r1_adapter_dir),
            accuracy=0.60,
            accuracy_percent=60.0,
            correct_count=150,
            total_problems=250,
            format_compliance_rate=1.0,
            format_compliance_percent=100.0,
            truncation_rate=0.0,
            truncation_percent=0.0,
            average_completion_tokens=65.0,
        )
        # Round 2 is 1 point (2 problems) behind or ahead
        r2 = RoundEvalMetrics(
            stage="sft_targeted",
            round_name="Round 2 (Targeted SFT)",
            adapter_path=str(self.r2_adapter_dir),
            accuracy=0.592,
            accuracy_percent=59.2,
            correct_count=148,
            total_problems=250,
            format_compliance_rate=1.0,
            format_compliance_percent=100.0,
            truncation_rate=0.0,
            truncation_percent=0.0,
            average_completion_tokens=65.0,
        )

        decision = compute_comparison_decision(
            r1=r1,
            r2=r2,
            tie_margin_pct=2.0,
            tie_margin_count=5,
            staging_dir=self.staging_dir,
        )
        self.assertEqual(decision.winner, "tie")
        self.assertEqual(decision.decision, "tie_round_2_advances")
        self.assertEqual(decision.advancing_stage, "sft_targeted")
        self.assertTrue(decision.is_tie)
        self.assertEqual(decision.advancing_adapter_path, str(self.staging_dir))

    def test_decision_rule_round1_wins_on_excessive_regression(self):
        """Verify Round 1 wins and advances when Round 2 exhibits excessive regression exceeding margin."""
        r1 = RoundEvalMetrics(
            stage="sft",
            round_name="Round 1 (Curated SFT)",
            adapter_path=str(self.r1_adapter_dir),
            accuracy=0.65,
            accuracy_percent=65.0,
            correct_count=162,
            total_problems=250,
            format_compliance_rate=1.0,
            format_compliance_percent=100.0,
            truncation_rate=0.0,
            truncation_percent=0.0,
            average_completion_tokens=65.0,
        )
        # Round 2 is 4 points (10 problems) behind
        r2 = RoundEvalMetrics(
            stage="sft_targeted",
            round_name="Round 2 (Targeted SFT)",
            adapter_path=str(self.r2_adapter_dir),
            accuracy=0.61,
            accuracy_percent=61.0,
            correct_count=152,
            total_problems=250,
            format_compliance_rate=1.0,
            format_compliance_percent=100.0,
            truncation_rate=0.0,
            truncation_percent=0.0,
            average_completion_tokens=65.0,
        )

        decision = compute_comparison_decision(
            r1=r1,
            r2=r2,
            tie_margin_pct=2.0,
            tie_margin_count=5,
            staging_dir=self.staging_dir,
        )
        self.assertEqual(decision.winner, "round_1")
        self.assertEqual(decision.decision, "round_1_wins")
        self.assertEqual(decision.advancing_stage, "sft")
        self.assertFalse(decision.is_tie)
        self.assertEqual(decision.advancing_adapter_path, str(self.r1_adapter_dir))

    def test_problem_level_breakdown(self):
        """Verify problem-level comparison accurately categorizes recoveries, regressions, and mutual passes."""
        r1_gen_file = self.test_dir / "r1_gen.jsonl"
        r2_gen_file = self.test_dir / "r2_gen.jsonl"

        # Problem 1: Both correct
        # Problem 2: Both incorrect
        # Problem 3: R2 recovery (R1 False, R2 True)
        # Problem 4: R1 regression (R1 True, R2 False)
        probs = [
            ("prob_1", True, True),
            ("prob_2", False, False),
            ("prob_3", False, True),
            ("prob_4", True, False),
        ]

        with open(r1_gen_file, "w", encoding="utf-8") as f1, open(r2_gen_file, "w", encoding="utf-8") as f2:
            for pid, r1_c, r2_c in probs:
                rec1 = {"problem_id": pid, "question": f"Q {pid}", "gold_answer": "42", "extracted_answer": "42" if r1_c else "0", "is_correct": r1_c}
                rec2 = {"problem_id": pid, "question": f"Q {pid}", "gold_answer": "42", "extracted_answer": "42" if r2_c else "0", "is_correct": r2_c}
                f1.write(json.dumps(rec1) + "\n")
                f2.write(json.dumps(rec2) + "\n")

        breakdown = compare_problem_level_results(r1_gen_file, r2_gen_file)
        self.assertEqual(breakdown["both_correct_count"], 1)
        self.assertEqual(breakdown["both_incorrect_count"], 1)
        self.assertEqual(breakdown["r2_recoveries_count"], 1)
        self.assertEqual(breakdown["r1_regressions_count"], 1)
        self.assertEqual(breakdown["net_recovery"], 0)
        self.assertEqual(breakdown["total_compared"], 4)

    def test_test_set_wall_enforcement(self):
        """Verify that attempting to evaluate on the test split raises TestSetWallError."""
        with self.assertRaises(TestSetWallError):
            evaluate_round(
                stage="sft",
                round_name="Round 1",
                model_name_or_path="deepseek-ai/deepseek-math-7b-base",
                adapter_path=self.r1_adapter_dir,
                split="test",
                use_mock=True,
            )

    def test_staging_advancing_adapter(self):
        """Verify advancing adapter files are copied to the destination staging directory."""
        dest = stage_advancing_adapter(self.r1_adapter_dir, self.staging_dir)
        self.assertEqual(dest, self.staging_dir)
        self.assertTrue((self.staging_dir / "adapter_config.json").exists())
        self.assertTrue((self.staging_dir / "adapter_model.bin").exists())

    def test_end_to_end_mock_comparison(self):
        """Verify end-to-end execution of run_round_comparison in mock mode."""
        res = run_round_comparison(
            model_name_or_path="deepseek-ai/deepseek-math-7b-base",
            round1_adapter=self.r1_adapter_dir,
            round2_adapter=self.r2_adapter_dir,
            val_split="val_gating",
            limit=5,
            use_mock=True,
            reports_dir=self.reports_dir,
            staging_dir=self.staging_dir,
        )

        self.assertIn("round1", res)
        self.assertIn("round2", res)
        self.assertIn("decision", res)
        self.assertIn("problem_breakdown", res)

        json_file = Path(res["json_report"])
        md_file = Path(res["md_report"])

        self.assertTrue(json_file.exists())
        self.assertTrue(md_file.exists())

        report_data = json.loads(json_file.read_text(encoding="utf-8"))
        self.assertEqual(report_data["stage"], "round_comparison")
        self.assertEqual(report_data["day"], 37)
        self.assertIn(report_data["decision"]["winner"], ["round_2", "tie", "round_1"])

    def test_cli_execution_sample_mode(self):
        """Verify scripts/compare_sft_rounds.py CLI execution in sample mode."""
        import sys
        from unittest.mock import patch

        cli_args = [
            "compare_sft_rounds.py",
            "--round1_adapter", str(self.r1_adapter_dir),
            "--round2_adapter", str(self.r2_adapter_dir),
            "--reports_dir", str(self.reports_dir),
            "--staging_dir", str(self.staging_dir),
            "--limit", "5",
            "--sample_mode",
        ]
        with patch.object(sys, "argv", cli_args):
            # Should run without error and exit cleanly (or complete main)
            cli_main()

        self.assertTrue((self.reports_dir / "round_comparison.json").exists())
        self.assertTrue((self.reports_dir / "round_comparison.md").exists())

    def test_resolve_round_adapter_explicit(self):
        """Verify resolve_round_adapter discovers an explicit adapter directory."""
        resolved = resolve_round_adapter(
            stage="sft",
            adapter_path=str(self.r1_adapter_dir),
            allow_mock_fallback=False,
        )
        self.assertEqual(resolved, self.r1_adapter_dir.resolve())

    def test_resolve_round_adapter_mock_fallback(self):
        """Verify resolve_round_adapter generates simulated adapter when allowed."""
        non_existent = self.test_dir / "missing_stage" / "adapter_model"
        resolved = resolve_round_adapter(
            stage="missing_stage",
            adapter_path=str(non_existent),
            allow_mock_fallback=True,
        )
        self.assertTrue(resolved.exists())
        self.assertTrue((resolved / "adapter_config.json").exists())

    def test_resolve_round_adapter_checkpoint_fallback(self):
        """Verify resolve_round_adapter discovers completed checkpoint when outputs missing."""
        ckpt_stage_dir = self.test_dir / "checkpoints" / "sft_ckpt_stage"
        ckpt_10 = ckpt_stage_dir / "checkpoint-10"
        ckpt_20 = ckpt_stage_dir / "checkpoint-20"
        ckpt_10.mkdir(parents=True, exist_ok=True)
        ckpt_20.mkdir(parents=True, exist_ok=True)
        (ckpt_20 / "adapter_model.safetensors").write_bytes(b"SAFETENSORS")

        with tempfile.TemporaryDirectory() as tmp_prof:
            # Stage paths where checkpoints points to ckpt_stage_dir
            from unittest.mock import patch
            mock_stage_paths = {
                "outputs": Path(tmp_prof) / "outputs",
                "checkpoints": ckpt_stage_dir,
            }
            with patch("src.config.ConfigManager.get_stage_paths", return_value=mock_stage_paths):
                resolved = resolve_round_adapter(
                    stage="sft_ckpt_stage",
                    adapter_path=None,
                    allow_mock_fallback=False,
                )
                self.assertEqual(resolved, ckpt_20)


if __name__ == "__main__":
    unittest.main()
