"""Unit tests for Reasoning LLM Judge rubric, output parsing, and calibration (Phase 3 Day 21)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.reasoning.judge import (
    ReasoningJudge,
    compute_judge_calibration,
    compute_overall_score,
    compute_pearson_correlation,
    format_judge_prompt,
    parse_judge_output,
)


class TestReasoningJudgeRubric(unittest.TestCase):
    """Test suite for LLM judge prompt formatting, score parsing, and weighting."""

    def test_format_judge_prompt_standard(self) -> None:
        """Standard CoT prompt formatting includes problem, gold answer, and candidate."""
        prompt = format_judge_prompt(
            question="Janet has 5 apples and buys 3 more. How many?",
            gold_answer="8",
            completion="5 + 3 = 8\n#### 8",
            prompt_type="chain_of_thought",
        )
        self.assertIn("Janet has 5 apples", prompt)
        self.assertIn("Gold Answer: 8", prompt)
        self.assertIn("Prompt Strategy Under Evaluation: chain_of_thought", prompt)
        self.assertIn("5 + 3 = 8", prompt)
        self.assertNotIn("Reference Standard Chain-of-Thought Solution", prompt)

    def test_format_judge_prompt_alternative_with_reference(self) -> None:
        """Alternative method prompt formatting includes reference CoT for novelty comparison."""
        prompt = format_judge_prompt(
            question="Janet has 5 apples and buys 3 more.",
            gold_answer="8",
            completion="Let x = 5. Janet buys 3, so total is x + 3 = 8.\n#### 8",
            prompt_type="alternative_method",
            reference_cot="5 + 3 = 8\n#### 8",
        )
        self.assertIn("Reference Standard Chain-of-Thought Solution", prompt)
        self.assertIn("Prompt Strategy Under Evaluation: alternative_method", prompt)

    def test_parse_judge_output_standard_bullets(self) -> None:
        """Parser extracts metrics from standard markdown bullet output."""
        raw = """Evaluation:
- Completeness: 0.90
- Verification Quality: 0.85
- Novelty: N/A
- Overall Score: 0.875
- Rationale: High clarity with rigorous intermediate steps and valid check.
"""
        score = parse_judge_output(raw, prompt_type="solve_then_verify")
        self.assertAlmostEqual(score.completeness, 0.90)
        self.assertAlmostEqual(score.verification_quality or 0.0, 0.85)
        self.assertIsNone(score.novelty)
        self.assertAlmostEqual(score.overall_score, 0.875)
        self.assertEqual(score.parsing_status, "success")
        self.assertIn("High clarity", score.rationale)

    def test_parse_judge_output_json_block(self) -> None:
        """Parser extracts metrics when LLM outputs JSON."""
        raw = """Here is the score:
```json
{
  "completeness": 0.8,
  "verification_quality": "N/A",
  "novelty": 0.7,
  "overall_score": 0.74,
  "rationale": "Distinct algebraic formulation."
}
```
"""
        score = parse_judge_output(raw, prompt_type="alternative_method")
        self.assertAlmostEqual(score.completeness, 0.8)
        self.assertIsNone(score.verification_quality)
        self.assertAlmostEqual(score.novelty or 0.0, 0.7)
        self.assertAlmostEqual(score.overall_score, 0.74)
        self.assertEqual(score.parsing_status, "success")

    def test_parse_judge_output_ratios_and_percentages(self) -> None:
        """Parser normalizes 8/10 and percentage formats (e.g. 90%)."""
        raw = """
Completeness: 9/10
Verification Quality: 85%
Novelty: N/A
Rationale: Very solid derivation.
"""
        score = parse_judge_output(raw, prompt_type="solve_then_verify")
        self.assertAlmostEqual(score.completeness, 0.9)
        self.assertAlmostEqual(score.verification_quality or 0.0, 0.85)
        self.assertIsNone(score.novelty)
        # 0.5 * 0.9 + 0.5 * 0.85 = 0.875
        self.assertAlmostEqual(score.overall_score, 0.875)

    def test_parse_judge_output_empty_fallback(self) -> None:
        """Empty or unparseable judge text defaults gracefully."""
        empty_score = parse_judge_output("", prompt_type="chain_of_thought")
        self.assertEqual(empty_score.parsing_status, "fallback")
        self.assertEqual(empty_score.overall_score, 0.0)

        unparseable = parse_judge_output("Random text without numbers", prompt_type="chain_of_thought")
        self.assertEqual(unparseable.parsing_status, "fallback")
        self.assertAlmostEqual(unparseable.completeness, 0.5)

    def test_compute_overall_score_weightings(self) -> None:
        """Overall score matches prompt-specific weights."""
        # CoT: 100% completeness
        cot_score = compute_overall_score(completeness=0.8, verification=0.9, novelty=0.9, prompt_type="chain_of_thought")
        self.assertAlmostEqual(cot_score, 0.8)

        # Solve-then-verify: 50% completeness + 50% verification
        stv_score = compute_overall_score(completeness=0.8, verification=0.6, prompt_type="solve_then_verify")
        self.assertAlmostEqual(stv_score, 0.7)

        # Alternative method: 40% completeness + 60% novelty
        alt_score = compute_overall_score(completeness=0.5, novelty=1.0, prompt_type="alternative_method")
        self.assertAlmostEqual(alt_score, 0.8)


class TestJudgeCalibrationAndMetrics(unittest.TestCase):
    """Test suite for Pearson correlation and calibration evaluation."""

    def test_pearson_correlation_perfect(self) -> None:
        """Identical series or positive linear relation yields r = 1.0."""
        x = [0.1, 0.3, 0.5, 0.7, 0.9]
        y = [0.2, 0.6, 1.0, 1.4, 1.8]
        r = compute_pearson_correlation(x, y)
        self.assertAlmostEqual(r, 1.0, places=4)

    def test_pearson_correlation_inverse(self) -> None:
        """Perfect inverse linear relation yields r = -1.0."""
        x = [0.1, 0.2, 0.3, 0.4]
        y = [0.9, 0.8, 0.7, 0.6]
        r = compute_pearson_correlation(x, y)
        self.assertAlmostEqual(r, -1.0, places=4)

    def test_pearson_correlation_zero_variance(self) -> None:
        """Constant series produces 0.0 without dividing by zero."""
        x = [0.5, 0.5, 0.5]
        y = [0.2, 0.4, 0.6]
        r = compute_pearson_correlation(x, y)
        self.assertEqual(r, 0.0)

    def test_compute_judge_calibration(self) -> None:
        """Calibration evaluates correlation against threshold r >= 0.60."""
        manual = [0.9, 0.85, 0.5, 0.3, 0.1]
        judge_good = [0.95, 0.80, 0.55, 0.35, 0.15]
        res_good = compute_judge_calibration(judge_good, manual, min_correlation=0.60)
        self.assertTrue(res_good["is_calibrated"])
        self.assertGreater(res_good["pearson_r"], 0.95)
        self.assertLess(res_good["mae"], 0.1)

        judge_bad = [0.1, 0.2, 0.9, 0.8, 0.7]
        res_bad = compute_judge_calibration(judge_bad, manual, min_correlation=0.60)
        self.assertFalse(res_bad["is_calibrated"])


class TestReasoningJudgeMockAndCLI(unittest.TestCase):
    """Test suite for ReasoningJudge mock execution and CLI runner."""

    def test_mock_judge_evaluate_trace(self) -> None:
        """Mock judge provides deterministic, score-differentiated evaluations."""
        judge = ReasoningJudge(use_mock=True)

        # High quality detailed CoT
        detailed_trace = (
            "First, calculate the base rate: 5 * 10 = 50. "
            "Next, add the tax of 10% which is 50 * 0.10 = 5. "
            "Finally, the total cost is 50 + 5 = 55.\n#### 55"
        )
        score_high = judge.evaluate_trace(
            question="What is the total cost?",
            gold_answer="55",
            completion=detailed_trace,
            prompt_type="chain_of_thought",
        )
        self.assertGreaterEqual(score_high.completeness, 0.90)
        self.assertAlmostEqual(score_high.overall_score, score_high.completeness)

        # Short/poor CoT
        score_low = judge.evaluate_trace(
            question="What is the total cost?",
            gold_answer="55",
            completion="55\n#### 55",
            prompt_type="chain_of_thought",
        )
        self.assertLess(score_low.completeness, 0.60)

    def test_mock_judge_evaluate_traces_batch(self) -> None:
        """Batch evaluation returns structured ScoredTrace objects."""
        judge = ReasoningJudge(use_mock=True)
        traces = [
            {
                "trace_id": "t1",
                "problem_id": "p1",
                "prompt_type": "chain_of_thought",
                "question": "Q1",
                "gold_answer": "10",
                "completion": "5 * 2 = 10\n#### 10",
            },
            {
                "trace_id": "t2",
                "problem_id": "p2",
                "prompt_type": "solve_then_verify",
                "question": "Q2",
                "gold_answer": "20",
                "completion": "Solution:\n10 * 2 = 20.\nVerification:\n20 / 2 = 10 condition holds.\n#### 20",
            },
        ]
        scored = judge.evaluate_traces(traces)
        self.assertEqual(len(scored), 2)
        self.assertEqual(scored[0].trace_id, "t1")
        self.assertEqual(scored[1].trace_id, "t2")
        self.assertIsNotNone(scored[1].score.verification_quality)

    def test_benchmark_manual_scores_file_validity(self) -> None:
        """Verify the 20-trace calibration ground-truth file is valid and balanced."""
        benchmark_file = Path("data/calibration/manual_scores_20.json")
        self.assertTrue(benchmark_file.exists(), "Benchmark file data/calibration/manual_scores_20.json missing")

        with open(benchmark_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data), 20, "Benchmark must contain exactly 20 traces")
        types = [t["prompt_type"] for t in data]
        self.assertEqual(types.count("chain_of_thought"), 8)
        self.assertEqual(types.count("solve_then_verify"), 6)
        self.assertEqual(types.count("alternative_method"), 6)

        # Validate score bounds
        for item in data:
            self.assertIn("trace_id", item)
            self.assertIn("manual_completeness", item)
            self.assertIn("manual_overall_score", item)
            c = item["manual_completeness"]
            o = item["manual_overall_score"]
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)
            self.assertGreaterEqual(o, 0.0)
            self.assertLessEqual(o, 1.0)

    def test_cli_calibration_mock_execution(self) -> None:
        """CLI calibration runs end-to-end and satisfies r >= 0.60 gate."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                sys.executable,
                "scripts/score_traces_judge.py",
                "--calibrate",
                "--use_mock",
                "--reports_dir",
                tmp_dir,
                "--report_name",
                "test_calibration",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, f"CLI calibration failed: {result.stderr}")

            md_file = Path(tmp_dir) / "test_calibration.md"
            json_file = Path(tmp_dir) / "test_calibration_report.json"
            self.assertTrue(md_file.exists())
            self.assertTrue(json_file.exists())

            report_data = json.loads(json_file.read_text(encoding="utf-8"))
            calib = report_data["calibration_metrics"]
            self.assertTrue(calib["is_calibrated"])
            self.assertGreaterEqual(calib["pearson_r"], 0.60)


if __name__ == "__main__":
    unittest.main()
