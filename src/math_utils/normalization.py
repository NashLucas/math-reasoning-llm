"""Answer normalization, numeric parsing, and equivalence comparison for math reasoning."""

from __future__ import annotations

import logging
import math
import re
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Currency symbols to strip
CURRENCY_SYMBOLS = ["$", "€", "£", "¥", "₹", "₩", "¢"]

# Unicode minus variants to normalize
MINUS_SYMBOLS = ["−", "–", "—", "‐", "‑"]

# Trailing common unit words to strip
COMMON_UNITS = [
    "dollars", "dollar", "cents", "cent",
    "pages", "page", "clips", "clip",
    "hours", "hour", "hrs", "hr",
    "minutes", "minute", "mins", "min",
    "seconds", "second", "secs", "sec",
    "days", "day", "weeks", "week", "months", "month", "years", "year",
    "apples", "books", "slices", "trees", "miles", "mph", "meters", "km",
    "pounds", "lbs", "lb", "kg", "grams", "g",
    "percent", "percentage", "%",
]


def normalize_minus(text: str) -> str:
    """Normalize various unicode minus signs to standard ASCII hyphen-minus."""
    for sym in MINUS_SYMBOLS:
        text = text.replace(sym, "-")
    return text


def strip_formatting(text: str) -> str:
    """Strip currency signs, commas in numbers, percentage signs, and trailing unit words."""
    if not text:
        return ""

    text = text.strip()
    text = normalize_minus(text)

    # Strip currency symbols
    for cur in CURRENCY_SYMBOLS:
        text = text.replace(cur, "")

    # Strip commas that act as thousands separators between digits (e.g. 1,000 -> 1000)
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    # Also strip loose commas at end or between digits
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.replace(",", "")

    # Strip percentage signs
    text = text.replace("%", "").strip()

    # Strip common trailing units (e.g. '42 pages' -> '42')
    # Match pattern: number followed by unit word
    tokens = text.split()
    if len(tokens) > 1:
        # Check if first token is a number or fraction
        first_token = tokens[0].rstrip(".:;,")
        if re.match(r"^-?\d+(?:\.\d+)?(?:/\d+)?$", first_token):
            text = first_token

    return text.strip().rstrip(".:;,")


def parse_numeric_value(raw: Union[str, int, float, None]) -> Optional[float]:
    """Parse a normalized string, int, or float into a Python float.

    Handles integers, decimals, scientific notation, and simple fractions (e.g. '1/2').
    """
    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        if math.isnan(raw):
            return None
        return float(raw)

    cleaned = strip_formatting(str(raw))
    if not cleaned:
        return None

    # 1. Simple fraction matching (e.g. '3/4' or '-1/2')
    fraction_match = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", cleaned)
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        if denominator == 0:
            return None
        return numerator / denominator

    # 2. Standard float or integer (including scientific notation e.g. 1e4)
    number_match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", cleaned)
    if number_match:
        try:
            return float(number_match.group(0))
        except ValueError:
            return None

    return None


def is_equivalent(
    pred: Union[str, int, float, None],
    target: Union[str, int, float, None],
    tolerance: float = 1e-4,
) -> bool:
    """Compare a predicted answer to a ground truth target value.

    Applies numeric tolerance comparison when both parse to numbers,
    or normalized string equality as fallback.
    """
    if pred is None or target is None:
        return False

    pred_num = parse_numeric_value(pred)
    target_num = parse_numeric_value(target)

    if pred_num is not None and target_num is not None:
        # Check for integer exact match first
        if pred_num.is_integer() and target_num.is_integer():
            return int(pred_num) == int(target_num)

        # Floating point tolerance comparison: absolute or relative
        diff = abs(pred_num - target_num)
        denom = max(1.0, abs(target_num))
        return (diff <= tolerance) or (diff / denom <= tolerance)

    # Fallback: normalized string comparison
    norm_pred = strip_formatting(str(pred)).lower()
    norm_target = strip_formatting(str(target)).lower()
    return norm_pred == norm_target and len(norm_pred) > 0
