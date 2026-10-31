"""Answer extraction engine implementing tiered pattern extraction for math responses."""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

from src.math_utils.normalization import (
    parse_numeric_value,
    strip_formatting,
)

logger = logging.getLogger(__name__)

# Pattern 1: Canonical final line after '####'
HASH_PATTERN = re.compile(r"####\s*([^\n\r]+)")

# Pattern 3: Common answer phrasing
PHRASE_PATTERNS = [
    re.compile(r"(?:the\s+)?(?:final\s+)?answer\s+(?:is|equals|=)\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"final\s+answer:\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"therefore,?\s*(?:the\s+answer\s+is\s*)?([^\n\r]+)", re.IGNORECASE),
]

# Pattern 4: Fallback to detect numeric tokens (integers, decimals, thousands commas, fractions, negatives)
NUMBER_PATTERN = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract the content of the last LaTeX \\boxed{...} tag with balanced brace matching."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None

    content_start = idx + len(r"\boxed{")
    brace_depth = 1
    i = content_start

    while i < len(text) and brace_depth > 0:
        if text[i] == "{":
            brace_depth += 1
        elif text[i] == "}":
            brace_depth -= 1
        i += 1

    if brace_depth == 0:
        return text[content_start : i - 1].strip()

    return None


def extract_answer(text: str) -> Tuple[Optional[str], str]:
    """Extract answer using the strict tiered priority order:

    1. '####' line (canonical contract; takes the LAST occurrence)
    2. LaTeX \\boxed{...}
    3. Natural language 'answer is' phrasing
    4. Fallback: the last number appearing in the response

    Returns:
        (extracted_answer_string, extraction_method)
    """
    if not text:
        return None, "none"

    # Tier 1: Canonical '####' delimiter
    hash_matches = HASH_PATTERN.findall(text)
    if hash_matches:
        candidate = hash_matches[-1].strip()
        cleaned = strip_formatting(candidate)
        if cleaned:
            return candidate, "hash"

    # Tier 2: LaTeX \boxed{...}
    boxed = extract_boxed_content(text)
    if boxed:
        cleaned = strip_formatting(boxed)
        if cleaned:
            return boxed, "boxed"

    # Tier 3: Natural language phrasing
    for pattern in PHRASE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            candidate = matches[-1].strip().rstrip(".")
            # Verify the candidate contains at least one numeric character
            if re.search(r"\d", candidate):
                cleaned = strip_formatting(candidate)
                if cleaned:
                    return candidate, "answer_is"

    # Tier 4: Fallback last number
    numbers = NUMBER_PATTERN.findall(text)
    if numbers:
        return numbers[-1].strip(), "last_number"

    return None, "none"


def extract_model_answer(text: str) -> Optional[str]:
    """Convenience wrapper returning only the extracted answer string (or None)."""
    ans, _ = extract_answer(text)
    return ans


def extract_and_normalize(text: str) -> Tuple[Optional[str], Optional[float], str]:
    """Extract raw answer, parse into numeric float, and return with extraction method.

    Returns:
        (raw_extracted, numeric_value, method)
    """
    raw, method = extract_answer(text)
    if raw is None:
        return None, None, method

    num_val = parse_numeric_value(raw)
    return raw, num_val, method
