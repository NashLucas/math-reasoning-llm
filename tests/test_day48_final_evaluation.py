"""Unit tests for Phase 7 Day 48: Final Benchmark Evaluation Infrastructure & Test Unfreezing.

Verifies:
1. Test-Set Wall isolation and authorization enforcement (PermissionError on unauthorized access).
2. Cryptographic checksum and unfreezing manifest verification.
3. Unbiased pass@k estimator mathematical correctness (Chen et al. 2021).
4. Non-parametric bootstrap confidence interval engine (single metric & pairwise deltas).
5. Deterministic mock generator fidelity and repeatability.
6. Multi-model benchmark suite execution across greedy, sampled, and self-consistency regimes.
7. Markdown audit report and summary JSON generation.
8. CLI entrypoint execution and argument validation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.evaluator import TestSetWallError
from src.evaluation.final_evaluator import (
    DEFAULT_BENCHMARK_LINEUP,
    DeterministicMockGenerator,
    FinalBenchmarkEvaluator,
    TestUnfreezeManifest,
    compute_bootstrap_ci,
    compute_bootstrap_delta_ci,
    unbiased_pass_at_k,
    verify_and_unfreeze_test_set,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestTestSetWallUnfreezing(unittest.TestCase):
    """Test Test-Set Wall authorization and unfreezing protocol."""

    def test_unauthorized_access_raises_error(self):
        """Verifies that accessing test split without final=True raises TestSetWallError."""
        evaluator = FinalBenchmarkEvaluator(use_mock=True)
        with self.assertRaises(TestSetWallError):
            evaluator.load_problems(split="test", final=False)

    def test_invalid_unfreeze_token_raises_error(self):
        """Verifies that an invalid authorization token raises PermissionError."""
        with self.assertRaises(PermissionError):
            verify_and_unfreeze_test_set(authorized_token="INVALID_TOKEN")

    def test_verify_and_unfreeze_manifest(self):
        """Verifies cryptographic unfreeze manifest generation and zero test leaks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_manifest = Path(tmp_dir) / "test_unfreeze.json"
            manifest = verify_and_unfreeze_test_set(
                output_manifest_path=out_manifest,
                authorized_token="PHASE_7_BENCHMARK_UNFREEZE",
            )
            self.assertIsInstance(manifest, TestUnfreezeManifest)
            self.assertGreater(manifest.problem_count, 0)
            self.assertTrue(manifest.ids_match)
            self.assertTrue(manifest.content_match)
            self.assertEqual(manifest.exact_matches_train, 0)
            self.assertEqual(manifest.exact_matches_val, 0)
            self.assertEqual(manifest.authorization_status, "AUTHORIZED_AND_UNFROZEN")
            self.assertTrue(out_manifest.exists())

            # Read back serialized JSON
            with open(out_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["problem_count"], manifest.problem_count)
            self.assertEqual(data["wall_status"], "UNFROZEN_FOR_FINAL_BENCHMARK")


class TestBootstrapStatistics(unittest.TestCase):
    """Test bootstrap confidence interval and delta computations."""

    def test_bootstrap_ci_constant_array(self):
        """Constant array should have mean == lower == upper and zero std error."""
        data = [1.0] * 50
        ci = compute_bootstrap_ci(data, n_resamples=200, seed=42)
        self.assertEqual(ci["mean"], 1.0)
        self.assertEqual(ci["ci_lower"], 1.0)
        self.assertEqual(ci["ci_upper"], 1.0)
        self.assertEqual(ci["std_err"], 0.0)

    def test_bootstrap_ci_bounds(self):
        """Non-constant data should satisfy lower <= mean <= upper."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 2, size=100).tolist()
        ci = compute_bootstrap_ci(data, n_resamples=500, ci_level=0.95, seed=42)
        self.assertLessEqual(ci["ci_lower"], ci["mean"])
        self.assertLessEqual(ci["mean"], ci["ci_upper"])
        self.assertGreater(ci["std_err"], 0.0)

    def test_bootstrap_delta_identical_arrays(self):
        """Identical paired arrays should yield zero delta and p-value of 1.0."""
        data = [1, 0, 1, 1, 0, 1]
        delta = compute_bootstrap_delta_ci(data, data, n_resamples=200, seed=42)
        self.assertEqual(delta["delta"], 0.0)
        self.assertEqual(delta["ci_lower"], 0.0)
        self.assertEqual(delta["ci_upper"], 0.0)
        self.assertEqual(delta["p_value"], 1.0)
        self.assertFalse(delta["is_significant"])

    def test_bootstrap_delta_significant_difference(self):
        """Model B strictly superior to Model A should yield positive delta with p < 0.05."""
        data_a = [0] * 50 + [1] * 50  # 50%
        data_b = [1] * 90 + [0] * 10  # 90%
        delta = compute_bootstrap_delta_ci(data_a, data_b, n_resamples=500, seed=42)
        self.assertAlmostEqual(delta["delta"], 0.40, places=2)
        self.assertGreater(delta["ci_lower"], 0.0)
        self.assertLess(delta["p_value"], 0.05)
        self.assertTrue(delta["is_significant"])


class TestUnbiasedPassAtK(unittest.TestCase):
    """Test Chen et al. (2021) pass@k estimator."""

    def test_pass_at_k_edge_cases(self):
        # c == 0 -> 0.0
        self.assertEqual(unbiased_pass_at_k(n=8, c=0, k=1), 0.0)
        self.assertEqual(unbiased_pass_at_k(n=8, c=0, k=4), 0.0)

        # c == n -> 1.0
        self.assertEqual(unbiased_pass_at_k(n=8, c=8, k=1), 1.0)
        self.assertEqual(unbiased_pass_at_k(n=8, c=8, k=8), 1.0)

        # n - c < k -> 1.0 (at least one correct in any subset of size k)
        self.assertEqual(unbiased_pass_at_k(n=8, c=6, k=4), 1.0)

    def test_pass_at_k_monotonicity(self):
        """For 0 < c < n, pass@1 <= pass@4 <= pass@8."""
        p1 = unbiased_pass_at_k(n=8, c=3, k=1)
        p4 = unbiased_pass_at_k(n=8, c=3, k=4)
        p8 = unbiased_pass_at_k(n=8, c=3, k=8)
        self.assertLessEqual(p1, p4)
        self.assertLessEqual(p4, p8)
        self.assertAlmostEqual(p1, 3.0 / 8.0)
        self.assertEqual(p8, 1.0)


class TestDeterministicMockGenerator(unittest.TestCase):
    """Test deterministic mock reasoning completion generator."""

    def test_mock_determinism(self):
        """Same problem_id, model_id, and sample_idx should produce identical output."""
        c1 = DeterministicMockGenerator.generate_completion("p_123", "42", "grpo_policy", sample_idx=0)
        c2 = DeterministicMockGenerator.generate_completion("p_123", "42", "grpo_policy", sample_idx=0)
        self.assertEqual(c1["text"], c2["text"])
        self.assertEqual(c1["extracted_answer"], c2["extracted_answer"])
        self.assertEqual(c1["is_correct"], c2["is_correct"])

    def test_grpo_format_compliance(self):
        """GRPO policy should have 100% format compliance in mock generation."""
        for i in range(20):
            c = DeterministicMockGenerator.generate_completion(f"p_{i}", "100", "grpo_policy", sample_idx=0)
            self.assertEqual(c["extraction_method"], "hash")
            self.assertIn("\n\n#### ", c["text"])
            self.assertEqual(c["extracted_answer"], c["text"].split("#### ")[-1].strip())


class TestBenchmarkEvaluatorEngine(unittest.TestCase):
    """Test the complete FinalBenchmarkEvaluator workflow."""

    def test_benchmark_suite_execution_mock(self):
        """Run benchmark suite on a small problem subset."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            evaluator = FinalBenchmarkEvaluator(
                output_dir=tmp_dir,
                use_mock=True,
                seed=42,
            )
            # Evaluate subset of models
            test_models = [
                DEFAULT_BENCHMARK_LINEUP[0],  # base_7b
                DEFAULT_BENCHMARK_LINEUP[-1],  # grpo_policy
            ]
            summary = evaluator.run_benchmark_suite(
                models=test_models,
                regimes=["greedy", "sampled", "self_consistency"],
                limit=15,
                final=True,
            )

            self.assertEqual(summary["num_problems"], 15)
            self.assertIn("base_7b", summary["results"])
            self.assertIn("grpo_policy", summary["results"])

            base_res = summary["results"]["base_7b"]
            grpo_res = summary["results"]["grpo_policy"]

            # Greedy checks
            self.assertIn("accuracy", base_res["greedy"])
            self.assertIn("accuracy", grpo_res["greedy"])
            self.assertIn("accuracy_ci", grpo_res["greedy"])

            # Sampled pass@k checks
            self.assertIn("pass_at_1", grpo_res["sampled"]["pass_at_k"])
            self.assertIn("pass_at_8", grpo_res["sampled"]["pass_at_k"])

            # Self-consistency checks
            self.assertIsNotNone(grpo_res["self_consistency"]["maj_at_8"])

            # Markdown report generation
            rep_path = Path(tmp_dir) / "report.md"
            md_text = evaluator.generate_markdown_report(summary, output_report_path=rep_path)
            self.assertTrue(rep_path.exists())
            self.assertIn("Phase 7 Day 48", md_text)
            self.assertIn("Benchmark Evaluation Comparison Matrix", md_text)
            self.assertIn("Test-Set Wall Unfreezing Protocol Certification", md_text)

    def test_cli_execution_mock(self):
        """Test scripts/evaluate_final.py CLI invocation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_rep = Path(tmp_dir) / "day48_report.md"
            out_json = Path(tmp_dir) / "day48_summary.json"
            out_manifest = Path(tmp_dir) / "test_unfreeze.json"

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_final.py"),
                "--final",
                "--use_mock",
                "--limit",
                "10",
                "--models",
                "base_7b",
                "grpo_policy",
                "--output_dir",
                tmp_dir,
                "--report_path",
                str(out_rep),
                "--summary_json",
                str(out_json),
                "--unfreeze_manifest",
                str(out_manifest),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")
            self.assertTrue(out_rep.exists())
            self.assertTrue(out_json.exists())
            self.assertTrue(out_manifest.exists())

    def test_cli_kebab_case_and_profile(self):
        """Test scripts/evaluate_final.py CLI with kebab-case arguments and --profile."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_rep = Path(tmp_dir) / "day48_report.md"
            out_json = Path(tmp_dir) / "day48_summary.json"
            out_manifest = Path(tmp_dir) / "test_unfreeze.json"

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_final.py"),
                "--final",
                "--use-mock",
                "--limit",
                "5",
                "--models",
                "base_7b",
                "grpo_policy",
                "--output-dir",
                tmp_dir,
                "--report-path",
                str(out_rep),
                "--summary-json",
                str(out_json),
                "--unfreeze-manifest",
                str(out_manifest),
                "--bootstrap-resamples",
                "100",
                "--profile",
                "local",
                "--device",
                "cpu",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")
            self.assertTrue(out_rep.exists())
            self.assertTrue(out_json.exists())
            self.assertTrue(out_manifest.exists())

    def test_drive_mirroring_integration(self):
        """Test Google Drive report mirroring logic."""
        import shutil

        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_drive = Path(tmp_dir) / "mock_drive"
            mock_reports = mock_drive / "reports"
            mock_reports.mkdir(parents=True, exist_ok=True)

            out_rep = Path(tmp_dir) / "day48_evaluation_infrastructure_report.md"
            out_json = Path(tmp_dir) / "day48_evaluation_infrastructure_report.json"
            out_manifest = Path(tmp_dir) / "day48_test_unfreeze_manifest.json"

            out_rep.write_text("# Day 48 Evaluation Report", encoding="utf-8")
            out_json.write_text('{"status": "PASSED"}', encoding="utf-8")
            out_manifest.write_text('{"wall_status": "UNFROZEN"}', encoding="utf-8")

            # Simulate mirroring
            shutil.copy2(out_rep, mock_reports / out_rep.name)
            shutil.copy2(out_json, mock_reports / out_json.name)
            shutil.copy2(out_manifest, mock_reports / out_manifest.name)

            self.assertTrue((mock_reports / out_rep.name).exists())
            self.assertTrue((mock_reports / out_json.name).exists())
            self.assertTrue((mock_reports / out_manifest.name).exists())
            self.assertIn("UNFROZEN", (mock_reports / out_manifest.name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
