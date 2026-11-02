"""Unit tests for reasoning trace curation, correctness filtering, and yield analysis (Phase 3)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.data.curation import (
    CorrectnessFilter,
    ProblemPassRate,
    YieldAnalysisReport,
    analyze_reasoning_yield,
    classify_difficulty,
    compute_problem_pass_rates,
    detect_repeated_ngrams,
    evaluate_trace_correctness,
    filter_problems_by_pass_rate,
    get_solved_problem_ids,
    load_problem_pass_rates,
    save_problem_pass_rates,
)
from src.data.synthetic import ReasoningTrace


class TestCurationAndYieldAnalysis(unittest.TestCase):
    """Test suite for reasoning trace curation, filter rules, and yield analysis."""

    def test_valid_trace_passes(self) -> None:
        """A well-formed trace with correct answer and canonical format must pass."""
        trace = {
            "trace_id": "p1_cot_s0",
            "problem_id": "p1",
            "gold_answer": "42",
            "completion": "Let's multiply 6 by 7 to get 42.\n#### 42",
            "completion_tokens": 25,
            "finish_reason": "stop",
        }
        is_kept, reason = evaluate_trace_correctness(trace)
        self.assertTrue(is_kept)
        self.assertIsNone(reason)

    def test_wrong_answer_rejected(self) -> None:
        """A trace predicting a wrong answer must be rejected with 'incorrect_answer'."""
        trace = {
            "trace_id": "p2_cot_s0",
            "problem_id": "p2",
            "gold_answer": "42",
            "completion": "Let's multiply 6 by 7 to get 40.\n#### 40",
            "completion_tokens": 25,
            "finish_reason": "stop",
        }
        is_kept, reason = evaluate_trace_correctness(trace)
        self.assertFalse(is_kept)
        self.assertEqual(reason, "incorrect_answer")

    def test_unparseable_answer_rejected(self) -> None:
        """A trace missing any parseable answer must be rejected with 'unparseable_answer'."""
        trace = {
            "trace_id": "p3_cot_s0",
            "problem_id": "p3",
            "gold_answer": "42",
            "completion": "I am working through this math problem carefully...",
            "completion_tokens": 20,
            "finish_reason": "stop",
        }
        is_kept, reason = evaluate_trace_correctness(trace)
        self.assertFalse(is_kept)
        self.assertEqual(reason, "unparseable_answer")

    def test_truncated_trace_rejected(self) -> None:
        """A trace that reached max tokens ('length') must be rejected with 'truncated'."""
        trace = {
            "trace_id": "p4_cot_s0",
            "problem_id": "p4",
            "gold_answer": "42",
            "completion": "Step 1: compute 6*7 = 42\n#### 42",
            "completion_tokens": 512,
            "finish_reason": "length",
        }
        is_kept, reason = evaluate_trace_correctness(trace)
        self.assertFalse(is_kept)
        self.assertEqual(reason, "truncated")

    def test_too_short_trace_rejected(self) -> None:
        """A trace with fewer than min_tokens must be rejected with 'too_short'."""
        trace = {
            "trace_id": "p5_cot_s0",
            "problem_id": "p5",
            "gold_answer": "42",
            "completion": "#### 42",
            "completion_tokens": 4,
            "finish_reason": "stop",
        }
        is_kept, reason = evaluate_trace_correctness(trace, min_tokens=10)
        self.assertFalse(is_kept)
        self.assertEqual(reason, "too_short")

    def test_detect_repeated_ngrams(self) -> None:
        """Repetitive degenerate loops must be identified accurately."""
        # Non-repetitive text
        normal_text = "First we add 5 and 3 to get 8. Then we multiply 8 by 2 to get 16. #### 16"
        self.assertFalse(detect_repeated_ngrams(normal_text))

        # Repetitive 3-gram: "and then we and then we and then we and then we" (repeats 4 times)
        loop_text = "Start " + "and then we " * 5 + "end #### 10"
        self.assertTrue(detect_repeated_ngrams(loop_text, max_repeats=3))

    def test_filter_traces_batch(self) -> None:
        """Batch filtering must return kept traces and comprehensive rejection stats."""
        traces = [
            {"trace_id": "t1", "problem_id": "p1", "gold_answer": "10", "completion": "5 + 5 = 10\n#### 10", "completion_tokens": 15, "finish_reason": "stop"},
            {"trace_id": "t2", "problem_id": "p1", "gold_answer": "10", "completion": "5 + 5 = 12\n#### 12", "completion_tokens": 15, "finish_reason": "stop"},
            {"trace_id": "t3", "problem_id": "p2", "gold_answer": "20", "completion": "10 + 10 = 20\n#### 20", "completion_tokens": 15, "finish_reason": "length"},
            {"trace_id": "t4", "problem_id": "p2", "gold_answer": "20", "completion": "10 + 10 = 20\n#### 20", "completion_tokens": 15, "finish_reason": "stop"},
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
        """Verify per-problem pass rate aggregation and dictionary/attribute access."""
        traces = [
            {"problem_id": "prob_a", "trace_id": "t_a1", "is_correct": True},
            {"problem_id": "prob_a", "trace_id": "t_a2", "is_correct": True},
            {"problem_id": "prob_b", "trace_id": "t_b1", "is_correct": True},
            {"problem_id": "prob_b", "trace_id": "t_b2", "is_correct": False},
            {"problem_id": "prob_c", "trace_id": "t_c1", "is_correct": False},
            {"problem_id": "prob_c", "trace_id": "t_c2", "is_correct": False},
        ]

        pass_stats = compute_problem_pass_rates(traces)
        # Dict access
        self.assertEqual(pass_stats["prob_a"]["pass_rate"], 1.0)
        self.assertTrue(pass_stats["prob_a"]["is_solved"])
        self.assertEqual(pass_stats["prob_a"]["difficulty_bin"], "easy")

        self.assertEqual(pass_stats["prob_b"]["pass_rate"], 0.5)
        self.assertTrue(pass_stats["prob_b"]["is_solved"])
        self.assertEqual(pass_stats["prob_b"]["difficulty_bin"], "medium")

        self.assertEqual(pass_stats["prob_c"]["pass_rate"], 0.0)
        self.assertFalse(pass_stats["prob_c"]["is_solved"])
        self.assertEqual(pass_stats["prob_c"]["difficulty_bin"], "zero")

        # Object attribute access
        self.assertEqual(pass_stats["prob_a"].pass_rate, 1.0)
        self.assertEqual(pass_stats["prob_a"].total_samples, 2)
        self.assertEqual(pass_stats["prob_a"].correct_samples, 2)

    def test_compute_problem_pass_rates_with_problem_coverage(self) -> None:
        """Verify that passing problems sequence ensures complete coverage."""
        all_problems = [
            {"problem_id": "p1", "question": "Q1", "gold_answer": "1"},
            {"problem_id": "p2", "question": "Q2", "gold_answer": "2"},
            {"problem_id": "p3_unattempted", "question": "Q3", "gold_answer": "3"},
        ]
        traces = [
            {"problem_id": "p1", "trace_id": "t1", "is_correct": True},
            {"problem_id": "p2", "trace_id": "t2", "is_correct": False},
        ]
        pass_stats = compute_problem_pass_rates(traces, problems=all_problems)
        self.assertEqual(len(pass_stats), 3)
        self.assertIn("p3_unattempted", pass_stats)
        self.assertEqual(pass_stats["p3_unattempted"].total_samples, 0)
        self.assertEqual(pass_stats["p3_unattempted"].correct_samples, 0)
        self.assertEqual(pass_stats["p3_unattempted"].pass_rate, 0.0)
        self.assertFalse(pass_stats["p3_unattempted"].is_solved)
        self.assertEqual(pass_stats["p3_unattempted"].difficulty_bin, "zero")

    def test_classify_difficulty(self) -> None:
        """Verify difficulty binning logic."""
        self.assertEqual(classify_difficulty(0, 4), "zero")
        self.assertEqual(classify_difficulty(1, 4), "hard")    # 1/4 = 0.25 <= 0.25
        self.assertEqual(classify_difficulty(2, 4), "medium")  # 2/4 = 0.5
        self.assertEqual(classify_difficulty(3, 4), "medium")  # 3/4 = 0.75
        self.assertEqual(classify_difficulty(4, 4), "easy")    # 4/4 = 1.0

    def test_save_and_load_problem_pass_rates(self) -> None:
        """Verify roundtrip saving and loading of problem pass rates to JSON and Parquet."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_pass_rates.json"
            sample_rates = {
                "prob_1": ProblemPassRate(
                    problem_id="prob_1",
                    question="What is 2+2?",
                    gold_answer="4",
                    total_samples=4,
                    correct_samples=3,
                    pass_rate=0.75,
                    is_solved=True,
                    difficulty_bin="medium",
                    correct_trace_ids=["t1", "t2", "t3"],
                    all_trace_ids=["t1", "t2", "t3", "t4"],
                    prompt_types=["chain_of_thought"],
                ),
                "prob_2": ProblemPassRate(
                    problem_id="prob_2",
                    question="What is 5*5?",
                    gold_answer="25",
                    total_samples=4,
                    correct_samples=0,
                    pass_rate=0.0,
                    is_solved=False,
                    difficulty_bin="zero",
                    correct_trace_ids=[],
                    all_trace_ids=["t5", "t6", "t7", "t8"],
                    prompt_types=["chain_of_thought"],
                ),
            }

            json_p, parquet_p = save_problem_pass_rates(sample_rates, out_file)
            self.assertTrue(json_p.exists())
            self.assertIsNotNone(parquet_p)
            self.assertTrue(parquet_p.exists())

            # Load from JSON
            loaded_json = load_problem_pass_rates(json_p)
            self.assertEqual(len(loaded_json), 2)
            self.assertEqual(loaded_json["prob_1"]["pass_rate"], 0.75)
            self.assertEqual(loaded_json["prob_1"]["difficulty_bin"], "medium")
            self.assertEqual(loaded_json["prob_2"]["pass_rate"], 0.0)

            # Load from Parquet
            loaded_parquet = load_problem_pass_rates(parquet_p)
            self.assertEqual(len(loaded_parquet), 2)
            self.assertEqual(loaded_parquet["prob_1"]["pass_rate"], 0.75)
            self.assertEqual(loaded_parquet["prob_2"]["pass_rate"], 0.0)

    def test_analyze_reasoning_yield(self) -> None:
        """Verify full reasoning trace yield analysis and markdown summary."""
        traces = [
            # Problem 1: 4/4 correct (easy)
            {"trace_id": "p1_s0", "problem_id": "p1", "gold_answer": "10", "completion": "5+5=10\n#### 10", "completion_tokens": 20, "finish_reason": "stop"},
            {"trace_id": "p1_s1", "problem_id": "p1", "gold_answer": "10", "completion": "2*5=10\n#### 10", "completion_tokens": 20, "finish_reason": "stop"},
            {"trace_id": "p1_s2", "problem_id": "p1", "gold_answer": "10", "completion": "1+9=10\n#### 10", "completion_tokens": 20, "finish_reason": "stop"},
            {"trace_id": "p1_s3", "problem_id": "p1", "gold_answer": "10", "completion": "12-2=10\n#### 10", "completion_tokens": 20, "finish_reason": "stop"},
            # Problem 2: 2/4 correct (medium)
            {"trace_id": "p2_s0", "problem_id": "p2", "gold_answer": "20", "completion": "10+10=20\n#### 20", "completion_tokens": 20, "finish_reason": "stop"},
            {"trace_id": "p2_s1", "problem_id": "p2", "gold_answer": "20", "completion": "4*5=20\n#### 20", "completion_tokens": 20, "finish_reason": "stop"},
            {"trace_id": "p2_s2", "problem_id": "p2", "gold_answer": "20", "completion": "10+5=15\n#### 15", "completion_tokens": 20, "finish_reason": "stop"},
            {"trace_id": "p2_s3", "problem_id": "p2", "gold_answer": "20", "completion": "10+10=20\n#### 20", "completion_tokens": 20, "finish_reason": "length"}, # truncated
            # Problem 3: 0/4 correct (zero)
            {"trace_id": "p3_s0", "problem_id": "p3", "gold_answer": "30", "completion": "No answer given", "completion_tokens": 20, "finish_reason": "stop"}, # unparseable
            {"trace_id": "p3_s1", "problem_id": "p3", "gold_answer": "30", "completion": "wrong\n#### 1", "completion_tokens": 20, "finish_reason": "stop"},
            {"trace_id": "p3_s2", "problem_id": "p3", "gold_answer": "30", "completion": "wrong\n#### 2", "completion_tokens": 20, "finish_reason": "stop"},
            {"trace_id": "p3_s3", "problem_id": "p3", "gold_answer": "30", "completion": "wrong\n#### 3", "completion_tokens": 20, "finish_reason": "stop"},
        ]

        report, pass_rates, kept = analyze_reasoning_yield(traces)

        self.assertIsInstance(report, YieldAnalysisReport)
        self.assertEqual(report.total_problems, 3)
        self.assertEqual(report.total_traces, 12)
        self.assertEqual(report.correct_traces, 6)
        self.assertEqual(report.trace_yield_rate, 0.5)
        self.assertEqual(report.trace_yield_percent, 50.0)

        self.assertEqual(report.solved_problems, 2)      # p1 and p2
        self.assertAlmostEqual(report.problem_solve_rate, 2 / 3, places=3)
        self.assertEqual(report.unsolved_problems, 1)    # p3
        self.assertAlmostEqual(report.unsolved_rate, 1 / 3, places=3)

        self.assertEqual(report.distribution_by_difficulty.get("easy"), 1)
        self.assertEqual(report.distribution_by_difficulty.get("medium"), 1)
        self.assertEqual(report.distribution_by_difficulty.get("zero"), 1)

        self.assertEqual(report.distribution_by_correct_count.get("4_of_4"), 1)
        self.assertEqual(report.distribution_by_correct_count.get("2_of_4"), 1)
        self.assertEqual(report.distribution_by_correct_count.get("0_of_4"), 1)

        # Markdown report rendering
        md_text = report.to_markdown_summary()
        self.assertIn("# Reasoning Trace Yield Analysis Report", md_text)
        self.assertIn("Total Problems Evaluated:** 3", md_text)
        self.assertIn("50.00% yield", md_text)
        self.assertIn("`4_of_4`", md_text)
        self.assertIn("`2_of_4`", md_text)
        self.assertIn("`0_of_4`", md_text)

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

    def test_analyze_reasoning_yield_cli_mock(self) -> None:
        """Verify executing scripts/analyze_reasoning_yield.py in mock mode via CLI subprocess."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "output"
            cmd = [
                sys.executable,
                "scripts/analyze_reasoning_yield.py",
                "--output_dir",
                str(out_dir),
                "--reports_dir",
                str(out_dir),
                "--mock",
                "--report_name",
                "test_mock_yield",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI command failed:\n{res.stderr}")
            self.assertTrue((out_dir / "problem_pass_rates.json").exists())
            self.assertTrue((out_dir / "problem_pass_rates.parquet").exists())


    def test_get_solved_problem_ids(self) -> None:
        """Verify extraction of solved problem IDs from pass rate records."""
        rates = {
            "p1": {"correct_samples": 3, "is_solved": True},
            "p2": {"correct_samples": 0, "is_solved": False},
            "p3": ProblemPassRate(problem_id="p3", total_samples=4, correct_samples=1, is_solved=True),
            "p4": ProblemPassRate(problem_id="p4", total_samples=4, correct_samples=0, is_solved=False),
        }
        solved_ids = get_solved_problem_ids(rates, min_correct=1)
        self.assertEqual(solved_ids, {"p1", "p3"})

        # Higher threshold: min_correct=2
        solved_ids_2 = get_solved_problem_ids(rates, min_correct=2)
        self.assertEqual(solved_ids_2, {"p1"})

    def test_filter_problems_by_pass_rate(self) -> None:
        """Verify filtering problems sequence using pass rate data."""
        problems = [
            {"problem_id": "p1", "question": "Q1"},
            {"problem_id": "p2", "question": "Q2"},
            {"problem_id": "p3", "question": "Q3"},
        ]
        rates = {
            "p1": {"correct_samples": 2, "is_solved": True},
            "p2": {"correct_samples": 0, "is_solved": False},
            "p3": {"correct_samples": 4, "is_solved": True},
        }
        filtered = filter_problems_by_pass_rate(problems, rates)
        self.assertEqual(len(filtered), 2)
        self.assertEqual([p["problem_id"] for p in filtered], ["p1", "p3"])

    def test_generate_reasoning_data_cli_only_solved_mock(self) -> None:
        """Verify generate_reasoning_data.py CLI with --only_solved and multi-prompt strategies."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "output"
            reports_dir = Path(tmp_dir) / "reports"
            pass_rates_file = Path(tmp_dir) / "pass_rates.json"

            from src.data.synthetic import load_synthetic_problems
            sample_probs = load_synthetic_problems(limit=2, allow_sample_fallback=True)
            p1_id = sample_probs[0]["problem_id"]
            p2_id = sample_probs[1]["problem_id"]

            pass_rates_file.write_text(json.dumps({
                p1_id: {"problem_id": p1_id, "correct_samples": 2, "is_solved": True},
                p2_id: {"problem_id": p2_id, "correct_samples": 0, "is_solved": False},
            }), encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/generate_reasoning_data.py",
                "--output_dir",
                str(out_dir),
                "--reports_dir",
                str(reports_dir),
                "--pass_rates_file",
                str(pass_rates_file),
                "--only_solved",
                "--prompt_types",
                "solve_then_verify,alternative_method",
                "--samples_per_problem",
                "1",
                "--use_mock",
                "--limit",
                "5",
                "--report_name",
                "test_day20_multi",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI command failed:\n{res.stderr}")
            self.assertTrue((reports_dir / "test_day20_multi.md").exists())
            self.assertTrue((reports_dir / "test_day20_multi_report.json").exists())


if __name__ == "__main__":
    unittest.main()
