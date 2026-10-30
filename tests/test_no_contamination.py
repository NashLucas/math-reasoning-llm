"""Unit tests for dataset contamination checks (exact match and n-gram overlap)."""

import unittest

from src.data.formatting import FEW_SHOT_EXEMPLARS
from src.data.loader import load_raw_gsm8k
from src.data.preprocessing import (
    assert_no_contamination,
    check_contamination,
    extract_word_ngrams,
    normalize_text_for_contamination,
    partition_splits,
)


class TestNoContamination(unittest.TestCase):
    def test_text_normalization_for_contamination(self):
        t1 = "What is the TOTAL, after 10% tax??"
        expected = "what is the total after 10 tax"
        self.assertEqual(normalize_text_for_contamination(t1), expected)

    def test_extract_word_ngrams(self):
        text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
        ngrams = extract_word_ngrams(text, n=13)
        self.assertEqual(len(ngrams), 2)
        self.assertIn("one two three four five six seven eight nine ten eleven twelve thirteen", ngrams)

    def test_contamination_detection_triggers(self):
        train_q = [
            "John has five apples and gives two to Mary. How many apples does he have left?",
            "A train travels 60 miles per hour for 3 hours. What is the total distance?",
        ]
        # Test case 1: Exact match (case/punctuation variation)
        test_q_exact = [
            "john has five apples, and gives two to mary! how many apples does he have left?",
        ]
        res = check_contamination(train_q, test_q_exact)
        self.assertTrue(res["is_contaminated"])
        self.assertEqual(res["exact_match_count"], 1)

        with self.assertRaises(AssertionError):
            assert_no_contamination(train_q, test_q_exact)

        # Test case 2: 13-gram overlap with allow_known_template_overlap=False
        test_q_ngram = [
            "A train travels 60 miles per hour for 3 hours. What is the total distance traveled yesterday?",
        ]
        res2 = check_contamination(train_q, test_q_ngram, ngram_size=13)
        self.assertGreaterEqual(res2["ngram_overlap_count"], 1)

        with self.assertRaises(AssertionError):
            assert_no_contamination(train_q, test_q_ngram, ngram_size=13, allow_known_template_overlap=False)

    def test_real_splits_have_no_contamination(self):
        """Verify zero exact matches between train and test splits."""
        raw_train = load_raw_gsm8k("train")
        raw_test = load_raw_gsm8k("test")
        splits = partition_splits(raw_train, raw_test, seed=42)

        train_questions = splits["train"]["question"].tolist()
        test_questions = splits["test"]["question"].tolist()

        # Run full contamination check
        result = check_contamination(train_questions, test_questions, ngram_size=13)
        self.assertFalse(
            result["is_contaminated"],
            f"Contamination discovered! Exact matches: {result['exact_matches']}"
        )
        self.assertEqual(result["exact_match_count"], 0)

    def test_few_shot_exemplars_not_in_test(self):
        """Verify that the 8 fixed few-shot exemplars do not overlap with the test set."""
        raw_test = load_raw_gsm8k("test")
        test_questions = raw_test["question"].tolist()

        exemplar_questions = [ex["question"] for ex in FEW_SHOT_EXEMPLARS]
        result = check_contamination(exemplar_questions, test_questions, ngram_size=13)
        self.assertFalse(
            result["is_contaminated"],
            "A few-shot prompt exemplar leaked into the test set!"
        )
        self.assertEqual(result["exact_match_count"], 0)


if __name__ == "__main__":
    unittest.main()
