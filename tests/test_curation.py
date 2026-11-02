"""Unit tests for reasoning trace curation and correctness filtering (Phase 3 Day 18)."""

from __future__ import annotations

import unittest

from src.data.curation import (
    CorrectnessFilter,
    compute_problem_pass_rates,
    detect_repeated_ngrams,
    evaluate_trace_correctness,
)
from src.data.synthetic import ReasoningTrace


class TestCurationCorrectnessFilter(unittest.TestCase):
    """Test suite for trace correctness verification and curation filtering."""

    def test_evaluate_valid_correct_trace(self) -> None:
        """Verify that a well-formed, correct reasoning trace is kept."""
        trace = {
            "problem_id": "prob_1",
            "gold_answer": "42",
            "completion": "Let us solve this step by step.\n10 + 32 = 42.\n#### 42",
            "completion_tokens": 25,
            "finish_reason": "stop",
        }
        is_kept, reason = evaluate_trace_correctness(trace)
        self.assertTrue(is_kept)
        self.assertIsNone(reason)

    def test_evaluate_incorrect_answer(self) -> None:
        """Verify rejection when the extracted answer does not match the gold answer."""
        trace = {
            "problem_id": "prob_1",
            "gold_answer": "42",
            "completion": "Let us solve this step by step.\n10 + 32 = 45.\n#### 45",
            "completion_tokens": 25,
            "finish_reason": "stop",
        }
        is_kept, reason = evaluate_trace_correctness(trace)
        self.assertFalse(is_kept)
        self.assertEqual(reason, "incorrect_answer")

    def test_evaluate_unparseable_answer(self) -> None:
        """Verify rejection when no valid numeric answer can be extracted."""
        trace = {
            "problem_id": "prob_1",
            "gold_answer": "42",
            "completion": "I don't know how to solve this problem, sorry.",
            "completion_tokens": 15,
            "finish_reason": "stop",
        }
        is_kept, reason = evaluate_trace_correctness(trace)
        self.assertFalse(is_kept)
        self.assertEqual(reason, "unparseable_answer")

    def test_evaluate_empty_completion(self) -> None:
        """Verify rejection on empty completion."""
        trace = {
            "problem_id": "prob_1",
            "gold_answer": "42",
            "completion": "   ",
        }
        is_kept, reason = evaluate_trace_correctness(trace)
        self.assertFalse(is_kept)
        self.assertEqual(reason, "empty_completion")

    def test_evaluate_truncated_response(self) -> None:
        """Verify rejection when response was truncated by max_new_tokens."""
        trace = {
            "problem_id": "prob_1",
            "gold_answer": "42",
            "completion": "Let us compute step 1: 10 + 32 = 42.\n#### 42",
            "completion_tokens": 20,
            "finish_reason": "length",
        }
        is_kept, reason = evaluate_trace_correctness(trace)
        self.assertFalse(is_kept)
        self.assertEqual(reason, "truncated")

    def test_evaluate_too_short(self) -> None:
        """Verify rejection when trace does not show reasoning steps (below min_tokens)."""
        trace = {
            "problem_id": "prob_1",
            "gold_answer": "42",
            "completion": "#### 42",
            "completion_tokens": 2,
            "finish_reason": "stop",
        }
        is_kept, reason = evaluate_trace_correctness(trace, min_tokens=10)
        self.assertFalse(is_kept)
        self.assertEqual(reason, "too_short")

    def test_detect_repeated_ngrams(self) -> None:
        """Verify detection of degenerate repetition loops."""
        normal_text = "First we add 10 to 20 to get 30. Then we subtract 5 to get 25."
        self.assertFalse(detect_repeated_ngrams(normal_text))

        # Looping 4-gram repeated 4 times
        repetitive_text = (
            "We calculate the total sum. "
            "then we add five then we add five then we add five then we add five "
            "to arrive at #### 42"
        )
        self.assertTrue(detect_repeated_ngrams(repetitive_text, ngram_size=4, max_repeats=3))

    def test_correctness_filter_batch(self) -> None:
        """Verify batch filtering with stats breakdown."""
        traces = [
            # 1. Correct
            {"problem_id": "p1", "gold_answer": "10", "completion": "5 + 5 = 10\n#### 10", "completion_tokens": 15, "finish_reason": "stop"},
            # 2. Incorrect
            {"problem_id": "p1", "gold_answer": "10", "completion": "5 + 5 = 12\n#### 12", "completion_tokens": 15, "finish_reason": "stop"},
            # 3. Truncated
            {"problem_id": "p2", "gold_answer": "20", "completion": "10 + 10 = 20\n#### 20", "completion_tokens": 15, "finish_reason": "length"},
            # 4. Correct
            {"problem_id": "p2", "gold_answer": "20", "completion": "10 * 2 = 20\n#### 20", "completion_tokens": 15, "finish_reason": "stop"},
        ]

        filter_engine = CorrectnessFilter(min_tokens=5)
        kept, stats = filter_engine.filter_traces(traces)

        self.assertEqual(len(kept), 2)
        self.assertEqual(stats.total_evaluated, 4)
        self.assertEqual(stats.kept_count, 2)
        self.assertEqual(stats.rejected_count, 2)
        self.assertEqual(stats.retention_rate, 0.5)
        self.assertEqual(stats.rejection_breakdown.get("incorrect_answer"), 1)
        self.assertEqual(stats.rejection_breakdown.get("truncated"), 1)

    def test_compute_problem_pass_rates(self) -> None:
        """Verify per-problem pass rate aggregation."""
        traces = [
            {"problem_id": "prob_a", "is_correct": True},
            {"problem_id": "prob_a", "is_correct": True},
            {"problem_id": "prob_b", "is_correct": True},
            {"problem_id": "prob_b", "is_correct": False},
            {"problem_id": "prob_c", "is_correct": False},
            {"problem_id": "prob_c", "is_correct": False},
        ]

        pass_stats = compute_problem_pass_rates(traces)
        self.assertEqual(pass_stats["prob_a"]["pass_rate"], 1.0)
        self.assertTrue(pass_stats["prob_a"]["is_solved"])

        self.assertEqual(pass_stats["prob_b"]["pass_rate"], 0.5)
        self.assertTrue(pass_stats["prob_b"]["is_solved"])

        self.assertEqual(pass_stats["prob_c"]["pass_rate"], 0.0)
        self.assertFalse(pass_stats["prob_c"]["is_solved"])

    def test_reasoning_trace_object_input(self) -> None:
        """Verify that ReasoningTrace dataclass objects work seamlessly with the filter."""
        trace_obj = ReasoningTrace(
            trace_id="p1_cot_s0",
            problem_id="p1",
            question="What is 7 * 6?",
            gold_answer="42",
            prompt_type="chain_of_thought",
            sample_idx=0,
            teacher_model="test-model",
            prompt="Solve 7 * 6",
            completion="7 * 6 is equal to 42.\n#### 42",
            extracted_answer="42",
            extraction_method="hash",
            is_correct=True,
            format_compliant=True,
            prompt_tokens=10,
            completion_tokens=20,
            finish_reason="stop",
            timestamp_utc="2026-11-02T12:00:00Z",
        )

        filter_engine = CorrectnessFilter()
        kept, stats = filter_engine.filter_traces([trace_obj])
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats.kept_count, 1)


if __name__ == "__main__":
    unittest.main()
