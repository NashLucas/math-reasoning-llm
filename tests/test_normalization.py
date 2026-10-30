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


if __name__ == "__main__":
    unittest.main()
