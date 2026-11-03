"""Unit tests for mathematical error analysis and taxonomy classification (Phase 5 Day 31).

Tests:
1. Rule-based checks:
   - Empty output -> parse_error
   - Whitespace output -> parse_error
   - Unparseable text with no answer -> parse_error
   - Extracted answer is non-numeric -> parse_error
   - Explicit truncation (finish_reason='length') -> truncation
   - Truncated text (unclosed LaTeX boxed or abrupt cutoff) -> truncation
2. Semantic classification:
   - Correct output -> is_error=False, category=None
   - Arithmetic slip -> arithmetic_slip
   - Misread question -> misread_question
   - Wrong setup -> wrong_setup
   - Incomplete reasoning -> incomplete_reasoning
   - Other / degenerate -> other
3. LLM response parser and prompt formatter
4. Aggregate error distribution calculation
5. Markdown report formatting
6. CLI execution of scripts/error_analysis.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.error_analysis import (
    ALL_ERROR_CATEGORIES,
    RULE_BASED_CATEGORIES,
    SEMANTIC_CATEGORIES,
    ErrorCategory,
    ErrorClassification,
    ErrorClassifier,
    check_parse_failure,
    check_truncation,
    classify_semantic_heuristic,
    compute_error_distribution,
    format_error_classification_prompt,
    generate_error_markdown_report,
    parse_llm_classification,
)


class TestErrorAnalysisRuleBased(unittest.TestCase):
    """Test rule-based error detection for parse failures and truncation."""

    def test_category_constants_and_dataclass(self):
        self.assertEqual(len(ALL_ERROR_CATEGORIES), 7)
        self.assertEqual(len(RULE_BASED_CATEGORIES), 2)
        self.assertEqual(len(SEMANTIC_CATEGORIES), 5)

        classification = ErrorClassification(
            problem_id="p1",
            question="What is 1+1?",
            gold_answer="2",
            generated_text="1+1=2\n#### 2",
            extracted_answer="2",
            is_correct=True,
            is_error=False,
            category=None,
            classification_source="ground_truth",
            rationale="Correct",
        )
        d = classification.to_dict()
        self.assertEqual(d["problem_id"], "p1")
        self.assertTrue(d["is_correct"])

    def test_empty_output_is_parse_error(self):
        is_parse, reason = check_parse_failure("")
        self.assertTrue(is_parse)
        self.assertIn("Empty", reason)

        is_parse, reason = check_parse_failure("   \n\t  ")
        self.assertTrue(is_parse)

    def test_unparseable_output_is_parse_error(self):
        text = "I do not know how to solve this problem, sorry."
        is_parse, reason = check_parse_failure(text)
        self.assertTrue(is_parse)
        self.assertIn("No numeric answer", reason)

    def test_non_numeric_extracted_answer_is_parse_error(self):
        text = "The answer is xyz.\n#### xyz"
        is_parse, reason = check_parse_failure(text, extracted_answer="xyz", extraction_method="hash")
        self.assertTrue(is_parse)
        self.assertIn("cannot be parsed into numeric", reason)

    def test_valid_answer_is_not_parse_error(self):
        text = "Calculation leads to:\n#### 42"
        is_parse, reason = check_parse_failure(text, extracted_answer="42", extraction_method="hash")
        self.assertFalse(is_parse)
        self.assertEqual(reason, "Parse successful")

    def test_explicit_truncation_via_metadata(self):
        text = "First step is to calculate 12 + 15 = 27."
        is_trunc, reason = check_truncation(text, metadata={"finish_reason": "length"})
        self.assertTrue(is_trunc)
        self.assertIn("length", reason)

        is_trunc, reason = check_truncation(text, metadata={"is_truncated": True})
        self.assertTrue(is_trunc)

    def test_unclosed_latex_boxed_is_truncation(self):
        text = "We solve 5 * 10 = 50. The answer is \\boxed{50"
        is_trunc, reason = check_truncation(text)
        self.assertTrue(is_trunc)
        self.assertIn("boxed", reason)

    def test_abrupt_cutoff_is_truncation(self):
        text = "The total number of cars is calculated by multiplying 15 by 4 and then"
        is_trunc, reason = check_truncation(text)
        self.assertTrue(is_trunc)
        self.assertIn("cutoff", reason.lower())

    def test_canonical_answer_not_flagged_as_truncated(self):
        text = "Step 1: 10 + 20 = 30.\n#### 30"
        is_trunc, reason = check_truncation(text)
        self.assertFalse(is_trunc)
        self.assertIn("Canonical", reason)


class TestSemanticErrorClassification(unittest.TestCase):
    """Test semantic error classification logic and prompt parsing."""

    def test_parse_llm_classification_standard(self):
        response = "Classification: arithmetic_slip\nRationale: The setup was correct but 7 * 8 was computed as 54."
        cat, rationale = parse_llm_classification(response)
        self.assertEqual(cat, ErrorCategory.ARITHMETIC_SLIP)
        self.assertIn("7 * 8", rationale)

    def test_parse_llm_classification_keyword_fallback(self):
        response = "The model made a misread_question error by confusing 15 with 50."
        cat, rationale = parse_llm_classification(response)
        self.assertEqual(cat, ErrorCategory.MISREAD_QUESTION)

    def test_parse_llm_classification_unknown_defaults_to_other(self):
        response = "This is a random response that does not match any category."
        cat, rationale = parse_llm_classification(response)
        self.assertEqual(cat, ErrorCategory.OTHER)

    def test_prompt_formatting(self):
        prompt = format_error_classification_prompt(
            question="What is 2 + 2?",
            gold_answer="4",
            generated_text="2 + 2 = 5\n#### 5",
            extracted_answer="5",
            gold_solution="2 + 2 = 4\n#### 4",
        )
        self.assertIn("What is 2 + 2?", prompt)
        self.assertIn("Final Answer: 4", prompt)
        self.assertIn("Extracted Model Answer:\n5", prompt)

    def test_heuristic_arithmetic_slip(self):
        question = "Janet has 15 apples and buys 12 more. How many does she have?"
        generated = "Janet starts with 15 apples. She buys 12 more.\n15 + 12 = 28.\n#### 28"
        cat, rationale = classify_semantic_heuristic(
            question=question,
            gold_answer="27",
            generated_text=generated,
            extracted_answer="28",
        )
        self.assertEqual(cat, ErrorCategory.ARITHMETIC_SLIP)

    def test_heuristic_incomplete_reasoning(self):
        question = "A store sells 10 shirts for $20 each and 5 hats for $15 each. What is the total revenue?"
        generated = "The total revenue is 300.\n#### 300"
        cat, rationale = classify_semantic_heuristic(
            question=question,
            gold_answer="275",
            generated_text=generated,
            extracted_answer="300",
        )
        self.assertEqual(cat, ErrorCategory.INCOMPLETE_REASONING)

    def test_heuristic_degenerate_other(self):
        question = "What is 10 + 10?"
        generated = "apple apple apple apple apple apple apple apple apple apple apple apple apple apple apple apple\n#### 10"
        cat, rationale = classify_semantic_heuristic(
            question=question,
            gold_answer="20",
            generated_text=generated,
            extracted_answer="10",
        )
        self.assertEqual(cat, ErrorCategory.OTHER)


class TestErrorClassifierPipeline(unittest.TestCase):
    """Test full ErrorClassifier orchestrator on diverse records."""

    def setUp(self):
        self.classifier = ErrorClassifier(use_mock=True)

    def test_correct_record_is_not_an_error(self):
        res = self.classifier.classify_single(
            question="What is 5 * 5?",
            gold_answer="25",
            generated_text="Step 1: 5 * 5 = 25.\n#### 25",
            extracted_answer="25",
            extraction_method="hash",
            problem_id="prob_correct",
        )
        self.assertTrue(res.is_correct)
        self.assertFalse(res.is_error)
        self.assertIsNone(res.category)
        self.assertEqual(res.classification_source, "ground_truth")

    def test_empty_record_is_parse_error(self):
        res = self.classifier.classify_single(
            question="What is 5 * 5?",
            gold_answer="25",
            generated_text="",
            problem_id="prob_empty",
        )
        self.assertFalse(res.is_correct)
        self.assertTrue(res.is_error)
        self.assertEqual(res.category, ErrorCategory.PARSE_ERROR)
        self.assertEqual(res.classification_source, "rule_based")

    def test_truncated_record(self):
        res = self.classifier.classify_single(
            question="Calculate sum.",
            gold_answer="100",
            generated_text="We add numbers together and",
            problem_id="prob_trunc",
            metadata={"finish_reason": "length"},
        )
        self.assertFalse(res.is_correct)
        self.assertTrue(res.is_error)
        self.assertEqual(res.category, ErrorCategory.TRUNCATION)
        self.assertEqual(res.classification_source, "rule_based")

    def test_custom_backend_classification(self):
        def mock_llm_backend(prompt: str) -> str:
            return "Classification: wrong_setup\nRationale: Formula for area of circle used diameter instead of radius."

        classifier = ErrorClassifier(backend=mock_llm_backend)
        res = classifier.classify_single(
            question="Find the area of circle with diameter 10.",
            gold_answer="78.5",
            generated_text="Area = pi * d^2 = 3.14 * 100 = 314.\n#### 314",
            extracted_answer="314",
            problem_id="prob_circle",
        )
        self.assertFalse(res.is_correct)
        self.assertTrue(res.is_error)
        self.assertEqual(res.category, ErrorCategory.WRONG_SETUP)
        self.assertEqual(res.classification_source, "llm_classifier")
        self.assertIn("radius", res.rationale)

    def test_batch_record_classification(self):
        records = [
            {
                "problem_id": "1",
                "question": "Q1",
                "gold_answer": "10",
                "generated_text": "#### 10",
                "extracted_answer": "10",
            },
            {
                "problem_id": "2",
                "question": "Q2",
                "gold_answer": "20",
                "generated_text": "",
            },
            {
                "problem_id": "3",
                "question": "Q3",
                "gold_answer": "30",
                "generated_text": "First 10 + 20 = and",
                "metadata": {"finish_reason": "length"},
            },
            {
                "problem_id": "4",
                "question": "Q4",
                "gold_answer": "40",
                "generated_text": "10 * 4 = 39.\n#### 39",
                "extracted_answer": "39",
            },
        ]
        classifications = self.classifier.classify_records(records)
        self.assertEqual(len(classifications), 4)

        summary = compute_error_distribution(classifications)
        self.assertEqual(summary["total_evaluated"], 4)
        self.assertEqual(summary["correct_count"], 1)
        self.assertEqual(summary["error_count"], 3)
        self.assertEqual(summary["accuracy"], 25.0)
        self.assertEqual(summary["error_rate"], 75.0)
        self.assertEqual(summary["category_counts"]["parse_error"], 1)
        self.assertEqual(summary["category_counts"]["truncation"], 1)

        # Markdown report generation
        md = generate_error_markdown_report(summary, classifications)
        self.assertIn("# Error Analysis Report", md)
        self.assertIn("`parse_error`", md)
        self.assertIn("`truncation`", md)


class TestErrorAnalysisCLI(unittest.TestCase):
    """Test CLI script execution in scripts/error_analysis.py."""

    def test_cli_execution_with_sample_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_file = tmp_path / "test_generations.jsonl"
            output_file = tmp_path / "error_analysis.json"
            report_file = tmp_path / "error_report.md"

            sample_records = [
                {
                    "problem_id": "cli_1",
                    "question": "What is 10 plus 15?",
                    "gold_answer": "25",
                    "generated_text": "10 + 15 = 25.\n#### 25",
                    "extracted_answer": "25",
                },
                {
                    "problem_id": "cli_2",
                    "question": "What is 20 times 3?",
                    "gold_answer": "60",
                    "generated_text": "",
                },
                {
                    "problem_id": "cli_3",
                    "question": "What is 50 minus 12?",
                    "gold_answer": "38",
                    "generated_text": "50 - 12 = 37.\n#### 37",
                    "extracted_answer": "37",
                },
            ]
            with open(input_file, "w", encoding="utf-8") as f:
                for rec in sample_records:
                    f.write(json.dumps(rec) + "\n")

            cmd = [
                sys.executable,
                "scripts/error_analysis.py",
                "--input_file",
                str(input_file),
                "--output_file",
                str(output_file),
                "--report_file",
                str(report_file),
                "--use_mock",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI exited with error:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

            self.assertTrue(output_file.exists())
            self.assertTrue(report_file.exists())

            payload = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertIn("summary", payload)
            self.assertIn("classifications", payload)
            self.assertEqual(payload["summary"]["total_evaluated"], 3)
            self.assertEqual(payload["summary"]["error_count"], 2)


if __name__ == "__main__":
    unittest.main()
