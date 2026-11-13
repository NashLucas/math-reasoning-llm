"""Correctness reward function for GRPO mathematical reasoning.

Computes mathematical correctness reward:
- Full credit (1.0) for an exact or mathematically equivalent answer.
- Partial credit (e.g. 0.25 to 0.5) for a close numerical answer (within small margin).
- Penalty for an unparseable completion where no valid answer can be extracted.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

from src.math_utils.answer_extraction import extract_and_normalize
from src.math_utils.normalization import is_equivalent, parse_numeric_value, strip_formatting

logger = logging.getLogger(__name__)


def is_close_numerical(
    pred_val: float,
    gold_val: float,
    rel_tol: float = 0.05,
    abs_tol: float = 1.0,
) -> bool:
    """Check if predicted float is numerically close to gold answer.

    Matches if within absolute tolerance (e.g., off by 1 arithmetic slip)
    or relative tolerance (e.g., within 5%).
    """
    if abs(pred_val - gold_val) <= abs_tol:
        return True
    if abs(gold_val) > 1e-5:
        if abs(pred_val - gold_val) / abs(gold_val) <= rel_tol:
            return True
    return False


def score_correctness(
    completion: str,
    gold_answer: str,
    full_credit: float = 1.0,
    partial_credit: float = 0.25,
    unparseable_penalty: float = 0.0,
    allow_partial: bool = False,
) -> Tuple[float, Optional[str]]:
    """Score correctness of a single completion against gold answer.

    Returns:
        Tuple of (reward_score, extracted_answer_str)
    """
    if not completion or not str(completion).strip():
        return (unparseable_penalty, None)

    norm_str, float_val, _tier = extract_and_normalize(completion)
    if norm_str is None and float_val is None:
        # Unparseable completion
        return (unparseable_penalty, None)

    extracted = norm_str if norm_str is not None else str(float_val)

    # 1. Exact or normalized equivalence -> full credit
    if norm_str is not None and is_equivalent(norm_str, gold_answer):
        return (full_credit, extracted)
    if float_val is not None and is_equivalent(float_val, gold_answer):
        return (full_credit, extracted)

    # 2. Close numerical answer -> partial credit (if enabled)
    if allow_partial and float_val is not None:
        try:
            gold_float = parse_numeric_value(gold_answer)
            if gold_float is not None and is_close_numerical(float_val, gold_float):
                return (partial_credit, extracted)
        except Exception:
            pass

    return (0.0, extracted)


def correctness_reward(
    completions: Sequence[str],
    gold_answers: Sequence[str],
    full_credit: float = 1.0,
    partial_credit: float = 0.25,
    unparseable_penalty: float = 0.0,
    allow_partial: bool = False,
    **kwargs: Any,
) -> List[float]:
    """Batch correctness scoring."""
    scores: List[float] = []
    for comp, gold in zip(completions, gold_answers):
        score, _ = score_correctness(
            completion=comp,
            gold_answer=gold,
            full_credit=full_credit,
            partial_credit=partial_credit,
            unparseable_penalty=unparseable_penalty,
            allow_partial=allow_partial,
        )
        scores.append(score)
    return scores
