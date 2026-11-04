"""Phase 1 parser gate on the real GSM8K gold solutions (see src/math_utils/gold_check.py)."""

import unittest

import pandas as pd

from src.data.loader import load_raw_gsm8k
from src.math_utils.gold_check import NO_HASH_BENCHMARK_FLOOR, check_gold_solutions


class TestGoldParserGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = pd.concat([load_raw_gsm8k("train"), load_raw_gsm8k("test")], ignore_index=True)
        cls.result = check_gold_solutions(raw["answer"], raw["problem_id"])

    def test_all_gold_answers_are_plain_numbers(self):
        self.assertEqual(self.result["total"], 8792)
        self.assertEqual(self.result["gold_format_failures"], 0, self.result["examples"]["gold_format"])

    def test_parser_correct_on_all_gold_solutions(self):
        self.assertEqual(self.result["parser_failures"], 0, self.result["examples"]["parser"])
        self.assertTrue(self.result["gate_passed"])

    def test_fallback_tiers_benchmark_does_not_regress(self):
        bench = self.result["no_hash_benchmark"]
        self.assertGreaterEqual(bench["accuracy"], NO_HASH_BENCHMARK_FLOOR, bench)


class TestGoldCheckCatchesBadData(unittest.TestCase):
    """The gate must fail on bad gold data, unlike the old string-vs-itself comparison."""

    def test_non_numeric_gold_fails(self):
        result = check_gold_solutions(["She has 5 apples.\n#### five"], ["p1"])
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["gold_format_failures"], 1)

    def test_gold_with_units_fails(self):
        result = check_gold_solutions(["5 + 3 = 8\n#### 8 dollars"], ["p1"])
        self.assertFalse(result["gate_passed"])

    def test_two_hash_lines_fail(self):
        result = check_gold_solutions(["#### 3\n5 + 3 = 8\n#### 8"], ["p1"])
        self.assertFalse(result["gate_passed"])

    def test_clean_gold_passes(self):
        result = check_gold_solutions(["She buys 2 + 3 = <<2+3=5>>5 pens.\n#### 1,005"], ["p1"])
        self.assertTrue(result["gate_passed"], result)


if __name__ == "__main__":
    unittest.main()
