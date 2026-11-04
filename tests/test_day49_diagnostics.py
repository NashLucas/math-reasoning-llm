"""Unit tests for Phase 7 Day 49: Full Benchmark Run & Comparative Diagnostics.

Verifies:
1. Reasoning step count extraction and complexity tier partitioning.
2. Stratified test problem loading and Test-Set Wall authorization enforcement.
3. Diagnostic model profiling across complexity tiers, accuracy metrics, and token efficiency.
4. Error taxonomy classification and error reduction calculations.
5. Full diagnostics suite execution with deterministic mock generator.
6. 4-panel publication curves plotting with graceful handling.
7. Markdown report and JSON summary emission.
8. CLI runner invocation and argument validation.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.comparative_diagnostics import (
    ALL_TIERS,
    ComparativeDiagnosticsEngine,
    ComplexityTier,
    ModelDiagnosticProfile,
    extract_reasoning_step_count,
    generate_day49_markdown_report,
    get_problem_complexity_tier,
    plot_day49_final_benchmark_curves,
)
from src.evaluation.evaluator import TestSetWallError
from src.evaluation.final_evaluator import DEFAULT_BENCHMARK_LINEUP

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestComplexityStratification(unittest.TestCase):
    """Test problem reasoning step counting and complexity tiering."""

    def test_extract_reasoning_step_count(self):
        sol_2 = "Step 1: Add 5 + 5 = 10.\nStep 2: Multiply 10 * 2 = 20.\n#### 20"
        self.assertEqual(extract_reasoning_step_count(sol_2), 2)

        sol_5 = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n#### 42"
        self.assertEqual(extract_reasoning_step_count(sol_5), 5)

        sol_empty = ""
        self.assertGreaterEqual(extract_reasoning_step_count(sol_empty), 1)

    def test_get_problem_complexity_tier(self):
        self.assertEqual(
            get_problem_complexity_tier("Line 1\nLine 2\n#### 1"),
            ComplexityTier.TIER_1_SHORT.value,
        )
        self.assertEqual(
            get_problem_complexity_tier("Line 1\nLine 2\nLine 3\nLine 4\n#### 1"),
            ComplexityTier.TIER_2_MEDIUM.value,
        )
        self.assertEqual(
            get_problem_complexity_tier("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\n#### 1"),
            ComplexityTier.TIER_3_COMPLEX.value,
        )


class TestStratifiedProblemLoading(unittest.TestCase):
    """Test dataset loading annotated with complexity tiers."""

    def test_unauthorized_access_raises_error(self):
        engine = ComparativeDiagnosticsEngine(use_mock=True)
        with self.assertRaises(TestSetWallError):
            engine.load_stratified_problems(final=False)

    def test_load_stratified_problems(self):
        engine = ComparativeDiagnosticsEngine(use_mock=True)
        engine.evaluator.unfreeze_test_set()
        problems = engine.load_stratified_problems(limit=15, final=True)
        self.assertEqual(len(problems), 15)
        for p in problems:
            self.assertIn("problem_id", p)
            self.assertIn("question", p)
            self.assertIn("gold_answer", p)
            self.assertIn("step_count", p)
            self.assertIn("complexity_tier", p)
            self.assertIn(p["complexity_tier"], ALL_TIERS)


class TestDiagnosticsEngine(unittest.TestCase):
    """Test model diagnostic profiling and suite execution."""

    def test_diagnose_model_profile(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = ComparativeDiagnosticsEngine(output_dir=tmp_dir, use_mock=True)
            engine.evaluator.unfreeze_test_set()
            problems = engine.load_stratified_problems(limit=10, final=True)

            prof = engine.diagnose_model(DEFAULT_BENCHMARK_LINEUP[0], problems)
            self.assertIsInstance(prof, ModelDiagnosticProfile)
            self.assertEqual(prof.num_problems, 10)
            self.assertGreaterEqual(prof.overall_greedy_acc, 0.0)
            self.assertLessEqual(prof.overall_greedy_acc, 1.0)
            self.assertGreater(prof.token_efficiency, 0.0)
            self.assertIn(ComplexityTier.TIER_1_SHORT.value, prof.tier_performance)
            self.assertIn(ComplexityTier.TIER_2_MEDIUM.value, prof.tier_performance)
            self.assertIn(ComplexityTier.TIER_3_COMPLEX.value, prof.tier_performance)

    def test_run_full_diagnostics_mock(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = ComparativeDiagnosticsEngine(output_dir=tmp_dir, use_mock=True)
            summary = engine.run_full_diagnostics(limit=12, final=True)

            self.assertEqual(summary["num_problems"], 12)
            self.assertIn("profiles", summary)
            self.assertIn("base_7b", summary["profiles"])
            self.assertIn("grpo_policy", summary["profiles"])
            self.assertIn("error_reductions", summary)

            red = summary["error_reductions"]
            self.assertIn("grpo_vs_base", red)
            self.assertIn("grpo_vs_sft_targeted", red)

            # Test report generation
            rep_path = Path(tmp_dir) / "day49_report.md"
            md_text = generate_day49_markdown_report(summary, output_report_path=rep_path)
            self.assertTrue(rep_path.exists())
            self.assertIn("Phase 7 Day 49", md_text)
            self.assertIn("Problem Complexity & Reasoning Step Stratification", md_text)
            self.assertIn("Error Taxonomy Breakdown & Error Reductions", md_text)

            # Test plot function (graceful handling)
            plot_path = Path(tmp_dir) / "day49_curves.png"
            plot_day49_final_benchmark_curves(summary, output_image_path=plot_path)


class TestDay49CLI(unittest.TestCase):
    """Test scripts/run_day49_diagnostics.py CLI execution."""

    def test_cli_execution_mock(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_rep = Path(tmp_dir) / "day49_rep.md"
            out_json = Path(tmp_dir) / "day49_summary.json"
            out_curves = Path(tmp_dir) / "day49_curves.png"

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_day49_diagnostics.py"),
                "--final",
                "--use_mock",
                "--limit",
                "8",
                "--output_dir",
                tmp_dir,
                "--report_path",
                str(out_rep),
                "--summary_json",
                str(out_json),
                "--curves_path",
                str(out_curves),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")
            self.assertTrue(out_rep.exists())
            self.assertTrue(out_json.exists())


if __name__ == "__main__":
    unittest.main()
