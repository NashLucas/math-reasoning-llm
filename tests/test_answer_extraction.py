"""Unit tests for tiered answer extraction, including 50+ hand-written noisy/messy model generations."""

import unittest

from src.math_utils.answer_extraction import (
    extract_and_normalize,
    extract_answer,
    extract_boxed_content,
)
from src.math_utils.normalization import is_equivalent

# 52 Hand-crafted messy, edge-case, and adversarial outputs
MESSY_TEST_CASES = [
    # --- Tier 1: Canonical '####' variations (1-15) ---
    ("The solution is clear.\n#### 42", "42", "hash"),
    ("Steps done.\n#### 1,000", "1,000", "hash"),
    ("Result with currency:\n#### $500", "$500", "hash"),
    ("Result with decimals:\n#### 12.50", "12.50", "hash"),
    ("Result with units:\n#### 15 clips", "15 clips", "hash"),
    ("Result with negative:\n#### -8", "-8", "hash"),
    ("Result with unicode minus:\n#### −25", "−25", "hash"),
    ("Result with percent:\n#### 75%", "75%", "hash"),
    ("Result with trailing punctuation:\n#### 99.", "99.", "hash"),
    ("Result with leading spaces:\n####    123", "123", "hash"),
    ("Multi-hash line (first draft #### 10, self-correction):\n#### 10\nWait, correcting:\n#### 20", "20", "hash"),
    ("Triple hash lines:\n#### 5\n#### 6\n#### 7", "7", "hash"),
    ("Markdown bolding inside hash:\n#### **42**", "**42**", "hash"),
    ("Hash with parenthetical explanation:\n#### 100 (one hundred dollars)", "100 (one hundred dollars)", "hash"),
    ("Hash with fraction:\n#### 3/4", "3/4", "hash"),

    # --- Tier 2: LaTeX \boxed{...} variations (16-25) ---
    ("We compute the integral and obtain \\boxed{35}.", "35", "boxed"),
    ("The final solution is \\boxed{1,200}.", "1,200", "boxed"),
    ("With dollar sign: \\boxed{$45.00}.", "$45.00", "boxed"),
    ("With negative: \\boxed{-14}.", "-14", "boxed"),
    ("Nested LaTeX fraction: \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}", "boxed"),
    ("LaTeX text inside box: \\boxed{60\\text{ miles}}.", "60\\text{ miles}", "boxed"),
    ("Multiple boxes, take last: \\boxed{10} then finally \\boxed{20}.", "20", "boxed"),
    ("Box with percent: \\boxed{50%}.", "50%", "boxed"),
    ("Box with scientific notation: \\boxed{1e4}.", "1e4", "boxed"),
    ("Box with spacing: \\boxed{  88  }.", "88", "boxed"),

    # --- Tier 3: Natural language 'answer is' phrasing (26-37) ---
    ("Therefore, the answer is 64.", "64", "answer_is"),
    ("Hence, the final answer is 125.", "125", "answer_is"),
    ("So, final answer: 300.", "300", "answer_is"),
    ("The answer is $15 dollars.", "$15 dollars", "answer_is"),
    ("The answer equals 18.", "18", "answer_is"),
    ("The answer is -5 degrees.", "-5 degrees", "answer_is"),
    ("Therefore, the answer is 1,500 people.", "1,500 people", "answer_is"),
    ("In summary, final answer: 0.75.", "0.75", "answer_is"),
    ("We conclude that the final answer is 999.", "999", "answer_is"),
    ("The answer is 3/5 of the total.", "3/5 of the total", "answer_is"),
    ("Therefore, the final answer is 48.", "48", "answer_is"),
    ("Thus, answer is 16.", "16", "answer_is"),

    # --- Tier 4: Fallback to last number in free text (38-48) ---
    ("We add 10 and 20 to get 30, then subtract 2 to leave 28", "28", "last_number"),
    ("Total apples: 5 + 7 + 12 = 24", "24", "last_number"),
    ("The count reached 1500", "1500", "last_number"),
    ("She had 3 friends and gave each 4 pencils, arriving at 12", "12", "last_number"),
    ("He walked 5 miles on Monday and 10 on Tuesday, so 15", "15", "last_number"),
    ("The calculated percentage was 35", "35", "last_number"),
    ("She ended up with -10", "-10", "last_number"),
    ("30 divided by 2 gives 15", "15", "last_number"),
    ("After all deductions, remaining is 450", "450", "last_number"),
    ("Cost: 10 + 20 + 30 = 60", "60", "last_number"),
    ("Result: 8.5", "8.5", "last_number"),

    # --- Non-numeric / Empty outputs (49-52) ---
    ("", None, "none"),
    ("   \n\t  ", None, "none"),
    ("I cannot solve this problem because of missing context.", None, "none"),
    ("The problem is incomplete and no number exists.", None, "none"),
]


class TestAnswerExtraction(unittest.TestCase):
    def test_all_52_messy_outputs(self):
        """Verify extraction across 52 diverse, hand-written noisy/messy model responses."""
        for idx, (raw_text, expected_raw, expected_method) in enumerate(MESSY_TEST_CASES, start=1):
            extracted_ans, method = extract_answer(raw_text)
            self.assertEqual(
                method,
                expected_method,
                f"Case #{idx} method mismatch for text: {raw_text!r} (got {method}, expected {expected_method})"
            )
            if expected_raw is None:
                self.assertIsNone(extracted_ans, f"Case #{idx} expected None, got {extracted_ans}")
            else:
                self.assertIsNotNone(extracted_ans, f"Case #{idx} extracted None for: {raw_text!r}")
                # Verify equivalence between extracted and expected
                self.assertTrue(
                    is_equivalent(extracted_ans, expected_raw),
                    f"Case #{idx} value mismatch: got {extracted_ans!r}, expected {expected_raw!r}"
                )

    def test_extract_boxed_nested_curly_braces(self):
        text = "x = \\boxed{\\frac{a + \\sqrt{b}}{2}}"
        boxed = extract_boxed_content(text)
        self.assertEqual(boxed, "\\frac{a + \\sqrt{b}}{2}")

    def test_extract_and_normalize_full(self):
        text = "Calculations complete.\n#### $2,500.00"
        raw, val, method = extract_and_normalize(text)
        self.assertEqual(raw, "$2,500.00")
        self.assertEqual(val, 2500.0)
        self.assertEqual(method, "hash")


if __name__ == "__main__":
    unittest.main()
