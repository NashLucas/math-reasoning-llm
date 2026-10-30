"""Unit tests for tiered answer extraction (hash, boxed, answer_is, last_number)."""

import unittest

from src.math_utils.answer_extraction import (
    extract_and_normalize,
    extract_answer,
    extract_boxed_content,
)


class TestAnswerExtraction(unittest.TestCase):
    def test_extract_hash_line(self):
        text = "Natalia sold 48/2 = 24 clips.\n#### 72"
        ans, method = extract_answer(text)
        self.assertEqual(ans, "72")
        self.assertEqual(method, "hash")

    def test_multiple_hash_takes_last(self):
        # When models write #### more than once, take the last occurrence
        text = "Intermediate step: #### 10\nFinal calculation:\n#### 42"
        ans, method = extract_answer(text)
        self.assertEqual(ans, "42")
        self.assertEqual(method, "hash")

    def test_extract_boxed_simple(self):
        text = "The calculated result is \\boxed{42}."
        ans, method = extract_answer(text)
        self.assertEqual(ans, "42")
        self.assertEqual(method, "boxed")

    def test_extract_boxed_nested(self):
        text = "Hence, we find that x = \\boxed{\\frac{3}{4}}."
        boxed = extract_boxed_content(text)
        self.assertEqual(boxed, "\\frac{3}{4}")

    def test_extract_answer_is(self):
        text1 = "After working through the steps, the final answer is 120."
        ans1, method1 = extract_answer(text1)
        self.assertEqual(ans1, "120")
        self.assertEqual(method1, "answer_is")

        text2 = "Therefore, final answer: $50 dollars."
        ans2, method2 = extract_answer(text2)
        self.assertEqual(ans2, "$50 dollars")
        self.assertEqual(method2, "answer_is")

    def test_extract_last_number_fallback(self):
        text = "We add 10 and 20 to get 30, and then subtract 5 to arrive at 25"
        ans, method = extract_answer(text)
        self.assertEqual(ans, "25")
        self.assertEqual(method, "last_number")

    def test_extract_and_normalize(self):
        text = "Step by step calculation:\n#### $1,500"
        raw, val, method = extract_and_normalize(text)
        self.assertEqual(raw, "$1,500")
        self.assertEqual(val, 1500.0)
        self.assertEqual(method, "hash")

    def test_empty_or_no_number(self):
        ans, method = extract_answer("")
        self.assertIsNone(ans)
        self.assertEqual(method, "none")

        ans, method = extract_answer("No numbers here whatsoever.")
        self.assertIsNone(ans)
        self.assertEqual(method, "none")


if __name__ == "__main__":
    unittest.main()
