"""Quality reward function for reasoning step visibility and length control."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, List, Optional, Sequence

REASONING_INDICATORS = re.compile(
    r"step\s*\d+|because|therefore|since|so|\+|\-|\*|\/|\=|\d+",
    re.IGNORECASE,
)


def score_reasoning_structure(
    completion: str,
    min_tokens: int = 5,
) -> float:
    """Verify presence of step-by-step derivation preceding answer to prevent guess-hacking."""
    if not completion or not str(completion).strip():
        return 0.0

    if "####" in completion:
        derivation = completion.split("####")[0].strip()
    else:
        derivation = completion.strip()

    words = derivation.split()
    if len(words) < min_tokens:
        return 0.0

    has_math_or_step = bool(REASONING_INDICATORS.search(derivation))
    return 1.0 if has_math_or_step else 0.0


def score_length_and_repetition(
    completion: str,
    max_length: int = 512,
    ngram_size: int = 8,
    max_ngram_repetition: int = 3,
    penalty_scale: float = 0.2,
) -> float:
    """Compute penalty for repetitive looping or excessive verbosity."""
    if not completion or not str(completion).strip():
        return 0.0

    penalty = 0.0
    words = completion.split()

    # Length budget check (rough word count ceiling)
    if len(words) > int(max_length * 0.85):
        overrun = len(words) - int(max_length * 0.85)
        penalty += min(0.5, (overrun / 100.0) * penalty_scale)

    # Degenerate repetition check
    if len(words) >= ngram_size:
        ngrams = [tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
        counts = Counter(ngrams)
        for _ng, cnt in counts.items():
            if cnt > max_ngram_repetition:
                excess = cnt - max_ngram_repetition
                penalty += excess * penalty_scale

    return min(1.0, penalty)


def quality_reward(
    completions: Sequence[str],
    min_tokens: int = 5,
    max_length: int = 512,
    **kwargs: Any,
) -> List[float]:
    """Batch quality scoring: reasoning structure minus repetition/length penalty."""
    scores: List[float] = []
    for c in completions:
        struct = score_reasoning_structure(c, min_tokens=min_tokens)
        pen = score_length_and_repetition(c, max_length=max_length)
        scores.append(max(0.0, struct - pen))
    return scores
