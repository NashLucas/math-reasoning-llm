"""Unit tests for answer normalization and numeric tolerance comparison."""

import unittest

from src.math_utils.normalization import (
    is_equivalent,
    normalize_minus,
    parse_numeric_value,
    strip_formatting,
)


class TestNormalization(unittest.TestCase):
    def test_strip_commas(self):
        self.assertEqual(strip_formatting("1,000"), "1000")
        self.assertEqual(strip_formatting("1,234,567"), "1234567")
        self.assertEqual(strip_formatting("48,000.50"), "48000.50")

    def test_strip_currencies(self):
        self.assertEqual(strip_formatting("$100"), "100")
        self.assertEqual(strip_formatting("€50.25"), "50.25")
        self.assertEqual(strip_formatting("£1,200"), "1200")
        self.assertEqual(strip_formatting("₹500"), "500")

    def test_strip_percent(self):
        self.assertEqual(strip_formatting("25%"), "25")
        self.assertEqual(strip_formatting("100 %"), "100")

    def test_strip_units(self):
        self.assertEqual(strip_formatting("42 pages"), "42")
        self.assertEqual(strip_formatting("5 hours"), "5")
        self.assertEqual(strip_formatting("72 clips."), "72")
        self.assertEqual(strip_formatting("$35 dollars"), "35")

    def test_normalize_minus(self):
        self.assertEqual(normalize_minus("−5"), "-5")
        self.assertEqual(normalize_minus("–12"), "-12")
        self.assertEqual(parse_numeric_value("−42"), -42.0)

    def test_parse_fractions(self):
        self.assertEqual(parse_numeric_value("1/2"), 0.5)
        self.assertEqual(parse_numeric_value("3/4"), 0.75)
        self.assertEqual(parse_numeric_value("-1/4"), -0.25)

    def test_is_equivalent(self):
        # Exact integer and float matches
        self.assertTrue(is_equivalent("42", 42))
        self.assertTrue(is_equivalent("1,000", 1000))
        self.assertTrue(is_equivalent("$500", "500"))
        self.assertTrue(is_equivalent("0.5", "1/2"))

        # Floating point tolerance
        self.assertTrue(is_equivalent("10.00001", "10.0", tolerance=1e-4))
        self.assertFalse(is_equivalent("10.01", "10.0", tolerance=1e-4))

        # None handling
        self.assertFalse(is_equivalent(None, "42"))
        self.assertFalse(is_equivalent("42", None))

    def test_equation_keeps_result(self):
        """Phase 1 review: '60 + 25 = 85' used to normalize to its first number, 60."""
        self.assertEqual(strip_formatting("60 + 25 = 85"), "85")
        self.assertEqual(strip_formatting("20 - 5 = 15 dollars"), "15")
        self.assertEqual(strip_formatting("3 x 4 = 12."), "12")
        self.assertEqual(strip_formatting("85 (= 60 + 25)"), "85")
        self.assertEqual(parse_numeric_value("$1000-$200 = $800"), 800.0)
        self.assertEqual(strip_formatting("42"), "42")

    def test_latex_fraction_and_mixed_number(self):
        self.assertEqual(parse_numeric_value(r"\frac{3}{4}"), 0.75)
        self.assertEqual(parse_numeric_value(r"\dfrac{1}{2}"), 0.5)
        self.assertEqual(parse_numeric_value("1 1/2"), 1.5)
        self.assertEqual(parse_numeric_value("2 1/4 cups"), 2.25)
        self.assertEqual(parse_numeric_value("-1 1/2"), -1.5)

    def test_relative_tolerance_is_tight_for_large_numbers(self):
        """A relative tolerance of 1e-4 used to accept 99999.5 for 100000."""
        self.assertFalse(is_equivalent("99999.5", "100000"))
        self.assertFalse(is_equivalent("12.0004", "12"))
        self.assertTrue(is_equivalent("0.3333333", "1/3"))
        self.assertTrue(is_equivalent("192000000", "192,000,000"))


if __name__ == "__main__":
    unittest.main()
