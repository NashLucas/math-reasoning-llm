"""Format reward function for Canonical Answer Contract enforcement."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

DEFAULT_FORMAT_PATTERN = r"(?:\n|^)####\s*(-?[0-9]+(?:\.[0-9]+)?)\s*$"


def score_format(completion: str, pattern: Optional[str] = None) -> float:
    """Verify Canonical Answer Contract adherence (1.0 if strictly ends in #### <number>, else 0.0)."""
    if not completion or not str(completion).strip():
        return 0.0
    pat = pattern or DEFAULT_FORMAT_PATTERN
    regex = re.compile(pat, re.MULTILINE)
    clean_comp = completion.strip()
    return 1.0 if regex.search(clean_comp) else 0.0


def format_reward(
    completions: Sequence[str],
    pattern: Optional[str] = None,
    **kwargs: Any,
) -> List[float]:
    """Batch format reward scoring."""
    return [score_format(c, pattern=pattern) for c in completions]
