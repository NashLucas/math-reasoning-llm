"""Unit tests for mathematical reasoning metrics, self-consistency voting, and evaluation engine."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.evaluator import MathEvaluator, TestSetWallError
from src.evaluation.math_metrics import (
    compute_accuracy,
    compute_all_metrics,
    compute_format_statistics,
    compute_maj_at_k,
    compute_pass_at_k,
    compute_token_statistics,
)
from src.reasoning.self_consistency import (
    majority_vote,
    majority_vote_from_completions,
    majority_vote_with_details,
)


class TestSelfConsistency(unittest.TestCase):
    """Test suite for majority voting, equivalence clustering, and deterministic tie-breaking."""

    def test_unanimous_vote(self) -> None:
        samples = ["42", "42", "42"]
        winner = majority_vote(samples)
        self.assertEqual(winner, "42")

        details = majority_vote_with_details(samples)
        self.assertEqual(details["winner_count"], 3)
        self.assertEqual(details["total_valid_votes"], 3)
        self.assertEqual(details["agreement_rate"], 1.0)

    def test_clear_majority(self) -> None:
        samples = ["10", "42", "42", "15"]
        winner = majority_vote(samples)
        self.assertEqual(winner, "42")

        details = majority_vote_with_details(samples)
        self.assertEqual(details["winner_count"], 2)
        self.assertEqual(details["total_valid_votes"], 4)
        self.assertAlmostEqual(details["agreement_rate"], 0.5)

    def test_deterministic_tie_breaking_rule(self) -> None:
        """Enforces that ties resolve to the cluster appearing earliest in sampling order."""
        # '10' appears at idx 0, '20' at idx 1 -> '10' wins tie
        self.assertEqual(majority_vote(["10", "20"]), "10")

        # '20' appears at idx 0, '10' at idx 1 -> '20' wins tie
        self.assertEqual(majority_vote(["20", "10"]), "20")

        # 2 votes each: 'apple' at idx 0 vs 'banana' at idx 1
        self.assertEqual(majority_vote(["10", "20", "20", "10"]), "10")
        self.assertEqual(majority_vote(["20", "10", "10", "20"]), "20")

    def test_equivalence_clustering(self) -> None:
        """Syntactically varied but mathematically equivalent answers must cluster together."""
        # 42 in various disguises: float, currency, commas, units
        samples = ["$42.0", "42", "42.00", "42 pages", "15", "15"]
        details = majority_vote_with_details(samples)

        # 42 cluster should have 4 votes; 15 cluster should have 2 votes
        self.assertEqual(details["winner"], "$42.0")  # first representative of the 42 cluster
        self.assertEqual(details["winner_count"], 4)
        self.assertEqual(details["total_valid_votes"], 6)

    def test_unparseable_and_none_handling(self) -> None:
        """Unparseable / None candidates must be ignored without skewing winner selection."""
        samples = [None, "42", None, "10", "42", ""]
        winner = majority_vote(samples)
        self.assertEqual(winner, "42")

        details = majority_vote_with_details(samples)
        self.assertEqual(details["winner_count"], 2)
        self.assertEqual(details["total_valid_votes"], 3)  # only two '42's and one '10'
        self.assertEqual(details["total_samples"], 6)

    def test_all_unparseable_returns_none(self) -> None:
        self.assertIsNone(majority_vote([None, "", "   "]))
        details = majority_vote_with_details([])
        self.assertIsNone(details["winner"])
        self.assertEqual(details["total_valid_votes"], 0)

    def test_majority_vote_from_completions(self) -> None:
        completions = [
            "The answer is 42.\n#### 42",
            "Therefore \\boxed{42}.",
            "Final calculation gives 10.\n#### 10",
        ]
        res = majority_vote_from_completions(completions)
        self.assertEqual(res["winner"], "42")
        self.assertEqual(res["winner_count"], 2)
        self.assertEqual(res["extraction_methods"], ["hash", "boxed", "hash"])


class TestMathMetrics(unittest.TestCase):
    """Test suite for math reasoning metrics on known hand-crafted prediction sets."""

    def setUp(self) -> None:
        # Hand-crafted prediction set with known ground truth outcomes
        self.single_records: List[Dict[str, Any]] = [
            {
                "problem_id": "p1",
                "gold_answer": "42",
                "extracted_answer": "42",
                "extraction_method": "hash",
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            {
                "problem_id": "p2",
                "gold_answer": "100",
                "extracted_answer": "100",
                "extraction_method": "boxed",
                "prompt_tokens": 120,
                "completion_tokens": 60,
            },
            {
                "problem_id": "p3",
                "gold_answer": "5",
                "extracted_answer": "6",
                "extraction_method": "answer_is",
                "prompt_tokens": 80,
                "completion_tokens": 40,
            },
            {
                "problem_id": "p4",
                "gold_answer": "12",
                "extracted_answer": None,
                "extraction_method": "none",
                "prompt_tokens": 100,
                "completion_tokens": 512,  # Truncated
            },
        ]

    def test_accuracy_calculation(self) -> None:
        acc = compute_accuracy(self.single_records)
        # Problems 1 and 2 are correct; 3 and 4 are wrong -> 2 / 4 = 0.50
        self.assertEqual(acc, 0.5)

    def test_format_statistics(self) -> None:
        stats = compute_format_statistics(self.single_records)
        self.assertEqual(stats["total_completions"], 4)
        self.assertEqual(stats["canonical_hash_count"], 1)  # p1
        self.assertEqual(stats["fallback_count"], 2)        # p2 (boxed), p3 (answer_is)
        self.assertEqual(stats["unparseable_count"], 1)     # p4 (none)

        self.assertAlmostEqual(stats["format_compliance_rate"], 0.25)
        self.assertAlmostEqual(stats["fallback_rate"], 0.50)
        self.assertAlmostEqual(stats["unparseable_rate"], 0.25)

    def test_token_statistics_and_truncation(self) -> None:
        tok_stats = compute_token_statistics(self.single_records, max_new_tokens=512)
        # Prompt tokens: (100 + 120 + 80 + 100) / 4 = 100.0
        self.assertEqual(tok_stats["average_prompt_tokens"], 100.0)
        # Completion tokens: (50 + 60 + 40 + 512) / 4 = 165.5
        self.assertEqual(tok_stats["average_completion_tokens"], 165.5)
        self.assertEqual(tok_stats["average_total_tokens"], 265.5)
        # Truncation: only p4 reached 512 -> 1 / 4 = 0.25
        self.assertEqual(tok_stats["truncation_rate"], 0.25)
        self.assertEqual(tok_stats["truncated_count"], 1)

    def test_compute_all_metrics_composite(self) -> None:
        metrics = compute_all_metrics(self.single_records, max_new_tokens=512)
        self.assertEqual(metrics["total_problems"], 4)
        self.assertEqual(metrics["total_completions"], 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["accuracy_percent"], 50.0)
        self.assertEqual(metrics["format_compliance_rate"], 0.25)
        self.assertEqual(metrics["fallback_rate"], 0.50)
        self.assertEqual(metrics["unparseable_rate"], 0.25)
        self.assertEqual(metrics["truncation_rate"], 0.25)

    def test_multi_sample_maj_and_pass_at_k(self) -> None:
        """Verify maj@k and pass@k across diverse sample distributions."""
        multi_records: List[Dict[str, Any]] = [
            # p1: maj=10 (correct), pass=True (10 in samples)
            {
                "problem_id": "p1",
                "gold_answer": "10",
                "extracted_answers": ["10", "10", "20"],
                "extraction_methods": ["hash", "hash", "hash"],
                "prompt_tokens": [100, 100, 100],
                "completion_tokens": [50, 50, 50],
            },
            # p2: maj=10 (incorrect, gold is 20), pass=True (20 in samples)
            {
                "problem_id": "p2",
                "gold_answer": "20",
                "extracted_answers": ["10", "10", "20"],
                "extraction_methods": ["hash", "hash", "hash"],
                "prompt_tokens": [100, 100, 100],
                "completion_tokens": [50, 50, 50],
            },
            # p3: maj=5 (incorrect, gold is 30), pass=False (30 not in samples)
            {
                "problem_id": "p3",
                "gold_answer": "30",
                "extracted_answers": ["5", "6", "7"],
                "extraction_methods": ["hash", "hash", "hash"],
                "prompt_tokens": [100, 100, 100],
                "completion_tokens": [50, 50, 50],
            },
        ]

        maj_3 = compute_maj_at_k(multi_records, k=3)
        pass_3 = compute_pass_at_k(multi_records, k=3)

        # 1 out of 3 problems has correct majority vote -> 1/3
        self.assertAlmostEqual(maj_3, 1 / 3, places=4)

        # 2 out of 3 problems have at least one correct sample -> 2/3
        self.assertAlmostEqual(pass_3, 2 / 3, places=4)

        composite = compute_all_metrics(multi_records, k=3)
        self.assertTrue(composite["is_multi_sample"])
        self.assertEqual(composite["total_problems"], 3)
        self.assertEqual(composite["total_completions"], 9)
        self.assertAlmostEqual(composite["maj_at_k"], 1 / 3, places=4)
        self.assertAlmostEqual(composite["pass_at_k"], 2 / 3, places=4)


class TestEvaluator(unittest.TestCase):
    """Test suite for MathEvaluator, Test-Set Wall enforcement, and output artifacts."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_test_set_wall_enforcement_raises(self) -> None:
        """Evaluator MUST reject evaluation on the 'test' split when final is False."""
        evaluator = MathEvaluator(
            stage="baseline",
            split="test",
            final=False,
            output_dir=self.temp_dir,
            use_mock=True,
        )

        with self.assertRaises(TestSetWallError) as ctx:
            evaluator.evaluate()

        self.assertIn("TEST-SET WALL VIOLATION", str(ctx.exception))

    def test_test_set_wall_authorized_with_final(self) -> None:
        """Evaluator allows evaluation on the 'test' split when final is True."""
        evaluator = MathEvaluator(
            stage="baseline",
            split="test",
            limit=2,
            final=True,
            output_dir=self.temp_dir,
            use_mock=True,
        )

        # Should complete without raising TestSetWallError
        results = evaluator.evaluate()
        self.assertEqual(results["split"], "test")
        self.assertTrue(results["final"])
        self.assertEqual(results["num_problems"], 2)

    def test_evaluator_end_to_end_mock_sample(self) -> None:
        """Evaluates sample problems and verifies generated files and metrics."""
        evaluator = MathEvaluator(
            stage="baseline",
            preset="greedy",
            split="sample",
            limit=5,
            output_dir=self.temp_dir,
            use_mock=True,
        )

        results = evaluator.evaluate()

        # Check return structure
        self.assertEqual(results["stage"], "baseline")
        self.assertEqual(results["preset"], "greedy")
        self.assertEqual(results["split"], "sample")
        self.assertEqual(results["num_problems"], 5)
        self.assertIn("accuracy", results["metrics"])

        # Check files created on disk
        eval_json = Path(self.temp_dir) / "eval_results.json"
        eval_split_json = Path(self.temp_dir) / "eval_results_sample_greedy.json"
        run_info_json = Path(self.temp_dir) / "run_info.json"
        gen_jsonl = Path(self.temp_dir) / "generations_sample_greedy.jsonl"

        self.assertTrue(eval_json.exists(), "eval_results.json missing")
        self.assertTrue(eval_split_json.exists(), "eval_results_sample_greedy.json missing")
        self.assertTrue(run_info_json.exists(), "run_info.json missing")
        self.assertTrue(gen_jsonl.exists(), "generations_sample_greedy.jsonl missing")

        # Verify eval_results content
        with open(eval_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["num_problems"], 5)
        self.assertGreaterEqual(data["metrics"]["format_compliance_rate"], 0.0)

        # Verify run_info content
        with open(run_info_json, "r", encoding="utf-8") as f:
            run_info = json.load(f)
        self.assertEqual(run_info["stage"], "baseline")
        self.assertIn("git", run_info)
        self.assertIn("system", run_info)


if __name__ == "__main__":
    unittest.main()
