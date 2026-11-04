"""Unit tests for Phase 6 Day 46: Post-GRPO Evaluation & Diagnostics.

Validates unbiased pass@k estimation, multi-component reward trajectory scoring,
Answer Contract adherence, Test-Set Wall isolation, comparative gating checks,
visualizations, and CLI execution.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.evaluator import TestSetWallError
from src.evaluation.grpo_evaluator import (
    GRPOComparisonResult,
    ModelEvalSummary,
    compute_maj_at_k_metrics,
    compute_pass_at_k_metrics,
    evaluate_model_for_grpo,
    generate_day46_reports,
    plot_day46_evaluation_curves,
    score_records_with_grpo_rewards,
    unbiased_pass_at_k,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDay46Evaluation(unittest.TestCase):
    """Test suite for Day 46 evaluation metrics, diagnostics, and gating."""

    def test_unbiased_pass_at_k_calculation(self) -> None:
        """Test unbiased pass@k estimation across known mathematical cases."""
        # Case 1: 0 correct out of n
        self.assertAlmostEqual(unbiased_pass_at_k(n=8, c=0, k=1), 0.0)
        self.assertAlmostEqual(unbiased_pass_at_k(n=8, c=0, k=4), 0.0)
        self.assertAlmostEqual(unbiased_pass_at_k(n=8, c=0, k=8), 0.0)

        # Case 2: All correct (c = n)
        self.assertAlmostEqual(unbiased_pass_at_k(n=8, c=8, k=1), 1.0)
        self.assertAlmostEqual(unbiased_pass_at_k(n=8, c=8, k=4), 1.0)
        self.assertAlmostEqual(unbiased_pass_at_k(n=8, c=8, k=8), 1.0)

        # Case 3: n - c < k implies pass@k = 1.0
        # e.g., 8 samples, 6 correct, k=4 -> n-c = 2 < 4
        self.assertAlmostEqual(unbiased_pass_at_k(n=8, c=6, k=4), 1.0)

        # Case 4: 1 correct out of 8, k=1 -> pass@1 = 1/8 = 0.125
        self.assertAlmostEqual(unbiased_pass_at_k(n=8, c=1, k=1), 0.125)

        # Case 5: 4 correct out of 8, k=4
        # 1 - (4*3*2*1)/(8*7*6*5) = 1 - 24/1680 = 1 - 1/70 = 69/70 ~ 0.985714
        expected_p4 = 1.0 - (24.0 / 1680.0)
        self.assertAlmostEqual(unbiased_pass_at_k(n=8, c=4, k=4), expected_p4, places=5)

        # Case 6: Edge bounds n < k or zero values
        self.assertEqual(unbiased_pass_at_k(n=0, c=0, k=1), 0.0)
        self.assertEqual(unbiased_pass_at_k(n=4, c=1, k=8), 1.0)
        self.assertEqual(unbiased_pass_at_k(n=4, c=0, k=8), 0.0)

    def test_compute_pass_at_k_metrics(self) -> None:
        """Test dataset-level pass@k computation across multiple problems."""
        records = [
            {
                "problem_id": "p1",
                "gold_answer": "10",
                "completions": [
                    {"extracted_answer": "10"},
                    {"extracted_answer": "10"},
                    {"extracted_answer": "5"},
                    {"extracted_answer": "5"},
                ],
            },
            {
                "problem_id": "p2",
                "gold_answer": "20",
                "completions": [
                    {"extracted_answer": "3"},
                    {"extracted_answer": "4"},
                    {"extracted_answer": "5"},
                    {"extracted_answer": "6"},
                ],
            },
        ]
        # p1 has n=4, c=2. p2 has n=4, c=0.
        # k=1: p1 -> 2/4 = 0.5, p2 -> 0. Mean = 0.25
        # k=4: p1 -> 1 - 0 = 1.0, p2 -> 0. Mean = 0.5
        metrics = compute_pass_at_k_metrics(records, k_values=(1, 4))
        self.assertAlmostEqual(metrics["pass_at_1"], 0.25)
        self.assertAlmostEqual(metrics["pass_at_4"], 0.50)

    def test_compute_maj_at_k_metrics(self) -> None:
        """Test majority voting accuracy computation."""
        records = [
            {
                "problem_id": "p1",
                "gold_answer": "42",
                "completions": [
                    {"extracted_answer": "42"},
                    {"extracted_answer": "42"},
                    {"extracted_answer": "42"},
                    {"extracted_answer": "10"},
                ],
            },
            {
                "problem_id": "p2",
                "gold_answer": "100",
                "completions": [
                    {"extracted_answer": "50"},
                    {"extracted_answer": "50"},
                    {"extracted_answer": "100"},
                    {"extracted_answer": "100"},
                ],
            },
        ]
        maj_acc = compute_maj_at_k_metrics(records, k=4)
        # p1 vote is 42 (matches gold). p2 vote is tie or majority.
        self.assertGreaterEqual(maj_acc, 0.50)

    def test_test_set_wall_enforcement(self) -> None:
        """Verify strict Test-Set Wall raises TestSetWallError on test split."""
        with self.assertRaises(TestSetWallError):
            evaluate_model_for_grpo(
                stage="grpo",
                model_name="Test Model",
                adapter_path="outputs/grpo/adapter_model",
                split="test",
            )

    def test_reward_scoring_and_stats(self) -> None:
        """Test multi-component reward trajectory scoring."""
        records = [
            {
                "problem_id": "prob_01",
                "gold_answer": "42",
                "completions": [
                    {
                        "text": "Step 1: Compute 40 + 2.\nTherefore, the answer is 42.\n#### 42",
                        "extracted_answer": "42",
                    },
                    {
                        "text": "The answer is 99.\n#### 99",
                        "extracted_answer": "99",
                    },
                ],
            }
        ]

        scored_records, stats = score_records_with_grpo_rewards(records)
        self.assertEqual(len(scored_records), 1)
        self.assertEqual(len(scored_records[0]["scored_completions"]), 2)

        # Completion 1 should have accuracy=1.0, format=1.0, structure=1.0
        c1 = scored_records[0]["scored_completions"][0]
        self.assertEqual(c1["accuracy_reward"], 1.0)
        self.assertEqual(c1["format_reward"], 1.0)

        # Completion 2 should have accuracy=0.0
        c2 = scored_records[0]["scored_completions"][1]
        self.assertEqual(c2["accuracy_reward"], 0.0)

        # Check statistical summary
        self.assertGreaterEqual(stats.total_reward.mean, 0.0)
        self.assertGreaterEqual(stats.format_reward.mean, 0.5)

    def test_mock_model_evaluation(self) -> None:
        """Test single model evaluation in mock mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_adapter = Path(tmpdir) / "mock_adapter"
            tmp_adapter.mkdir()
            (tmp_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

            summary = evaluate_model_for_grpo(
                stage="grpo",
                model_name="GRPO Test Policy",
                adapter_path=tmp_adapter,
                split="val_gating",
                limit=5,
                use_mock=True,
                output_dir=Path(tmpdir) / "outputs",
            )

            self.assertIsInstance(summary, ModelEvalSummary)
            self.assertEqual(summary.total_problems, 5)
            self.assertGreaterEqual(summary.greedy_accuracy, 0.0)
            self.assertGreaterEqual(summary.greedy_format_compliance, 0.0)
            self.assertGreaterEqual(summary.pass_at_8, summary.pass_at_1)

    def test_compare_sft_vs_grpo_pipeline(self) -> None:
        """Test full comparison pipeline between SFT and GRPO."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_adapter = Path(tmpdir) / "sft_adapter"
            sft_adapter.mkdir()
            (sft_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

            grpo_adapter = Path(tmpdir) / "grpo_adapter"
            grpo_adapter.mkdir()
            (grpo_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

            from src.evaluation.grpo_evaluator import compare_sft_vs_grpo

            comparison = compare_sft_vs_grpo(
                sft_adapter_path=sft_adapter,
                grpo_adapter_path=grpo_adapter,
                split="val_gating",
                limit=4,
                use_mock=True,
                output_dir=Path(tmpdir) / "eval_out",
            )

            self.assertIsInstance(comparison, GRPOComparisonResult)
            self.assertIn(comparison.gate_verdict, ["PASSED", "FAIL"])
            self.assertIn("greedy_accuracy_no_regression", comparison.gate_checks)
            self.assertIn("pass_at_8_gain", comparison.gate_checks)
            self.assertIn("answer_contract_preservation", comparison.gate_checks)

    def test_report_generation(self) -> None:
        """Test generating markdown and json evaluation reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_adapter = Path(tmpdir) / "sft_adapter"
            sft_adapter.mkdir()
            (sft_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

            grpo_adapter = Path(tmpdir) / "grpo_adapter"
            grpo_adapter.mkdir()
            (grpo_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

            from src.evaluation.grpo_evaluator import compare_sft_vs_grpo

            comparison = compare_sft_vs_grpo(
                sft_adapter_path=sft_adapter,
                grpo_adapter_path=grpo_adapter,
                split="val_gating",
                limit=3,
                use_mock=True,
                output_dir=Path(tmpdir) / "eval_out",
            )

            md_path, json_path = generate_day46_reports(comparison, reports_dir=tmpdir)
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                self.assertIn("gate_verdict", data)
                self.assertIn("sft_baseline", data)
                self.assertIn("grpo_policy", data)

            with open(md_path, "r", encoding="utf-8") as mf:
                text = mf.read()
                self.assertIn("Phase 6 Day 46: Post-GRPO Evaluation", text)
                self.assertIn("Gate Verification Checklist", text)

    def test_plot_curves_callable(self) -> None:
        """Test that plot_day46_evaluation_curves executes or gracefully skips without crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_adapter = Path(tmpdir) / "sft_adapter"
            sft_adapter.mkdir()
            (sft_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

            grpo_adapter = Path(tmpdir) / "grpo_adapter"
            grpo_adapter.mkdir()
            (grpo_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

            from src.evaluation.grpo_evaluator import compare_sft_vs_grpo

            comparison = compare_sft_vs_grpo(
                sft_adapter_path=sft_adapter,
                grpo_adapter_path=grpo_adapter,
                split="val_gating",
                limit=2,
                use_mock=True,
                output_dir=Path(tmpdir) / "eval_out",
            )

            plot_path = Path(tmpdir) / "test_eval_plot.png"
            res = plot_day46_evaluation_curves(comparison, output_path=plot_path)
            # res is either Path or None (if matplotlib not present)
            self.assertTrue(res is None or isinstance(res, Path))

    def test_cli_execution_sample_mode(self) -> None:
        """Test scripts/evaluate_grpo.py execution via CLI in sample mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_grpo.py"),
                "--sample_mode",
                "--limit",
                "3",
                "--output_dir",
                str(Path(tmpdir) / "eval_out"),
                "--reports_dir",
                str(Path(tmpdir) / "reports"),
            ]
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"CLI execution failed with code {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )
            self.assertTrue((Path(tmpdir) / "reports" / "day46_grpo_evaluation_report.md").exists())
            self.assertTrue((Path(tmpdir) / "reports" / "day46_grpo_evaluation_report.json").exists())


if __name__ == "__main__":
    unittest.main()
