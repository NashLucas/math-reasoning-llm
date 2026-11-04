"""Self-consistency majority voting engine for mathematical reasoning.

Aggregates multiple sampled reasoning completions per problem by clustering
extracted answers via numeric/symbolic equivalence, ignoring unparseable outputs,
and resolving ties with a deterministic, fixed tie-breaking rule (earliest sample order).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from src.math_utils.answer_extraction import extract_answer
from src.math_utils.normalization import is_equivalent, strip_formatting

logger = logging.getLogger(__name__)


def is_valid_candidate(ans: Union[str, int, float, None]) -> bool:
    """Check if an extracted answer candidate is non-empty and valid."""
    if ans is None:
        return False
    if isinstance(ans, (int, float)):
        return True
    cleaned = strip_formatting(str(ans))
    return len(cleaned) > 0 and cleaned.lower() != "none"


def majority_vote(answers: Sequence[Union[str, int, float, None]]) -> Optional[str]:
    """Execute majority voting across a sequence of extracted candidate answers.

    Clustering Strategy:
        - Candidate answers are clustered using `is_equivalent(a, b)` which handles
          integer/decimal equivalents ('42' == '42.0'), commas, currency, and tolerances.
        - Unparseable or None candidates are ignored.
        - If all candidates are unparseable, returns None.

    Deterministic Tie-Breaking Rule:
        - If two or more clusters share the highest vote count, the winner is the
          cluster whose representative appeared earliest in the sampling sequence.

    Args:
        answers: Sequence of parsed answers (strings, ints, floats, or None).

    Returns:
        The winning cluster representative string, or None if no valid answers exist.
    """
    details = majority_vote_with_details(answers)
    return details["winner"]


def majority_vote_with_details(
    answers: Sequence[Union[str, int, float, None]],
) -> Dict[str, Any]:
    """Execute majority voting and return comprehensive cluster statistics.

    Returns:
        Dict with keys:
            - 'winner': Winning answer representative (or None)
            - 'winner_count': Number of votes received by winner
            - 'total_valid_votes': Number of non-null valid candidates
            - 'total_samples': Total candidates provided
            - 'agreement_rate': winner_count / total_valid_votes (0.0 if no valid votes)
            - 'confidence': winner_count / total_samples (0.0 if total_samples == 0)
            - 'clusters': List of dicts describing each cluster
    """
    total_samples = len(answers)
    if total_samples == 0:
        return {
            "winner": None,
            "winner_count": 0,
            "total_valid_votes": 0,
            "total_samples": 0,
            "agreement_rate": 0.0,
            "confidence": 0.0,
            "clusters": [],
        }

    clusters: List[Dict[str, Any]] = []

    for idx, ans in enumerate(answers):
        if not is_valid_candidate(ans):
            continue

        cand_str = str(ans).strip()
        matched = False

        for cluster in clusters:
            if is_equivalent(cand_str, cluster["representative"]):
                cluster["count"] += 1
                cluster["members"].append(cand_str)
                matched = True
                break

        if not matched:
            clusters.append(
                {
                    "representative": cand_str,
                    "count": 1,
                    "earliest_index": idx,
                    "members": [cand_str],
                }
            )

    total_valid_votes = sum(c["count"] for c in clusters)

    if not clusters:
        return {
            "winner": None,
            "winner_count": 0,
            "total_valid_votes": 0,
            "total_samples": total_samples,
            "agreement_rate": 0.0,
            "confidence": 0.0,
            "clusters": [],
        }

    # Deterministic sorting:
    # 1. Highest vote count (-count)
    # 2. Earliest appearing in sampling order (earliest_index)
    clusters.sort(key=lambda c: (-c["count"], c["earliest_index"]))

    winner_cluster = clusters[0]
    winner = winner_cluster["representative"]
    winner_count = winner_cluster["count"]

    agreement_rate = winner_count / total_valid_votes if total_valid_votes > 0 else 0.0
    confidence = winner_count / total_samples if total_samples > 0 else 0.0

    return {
        "winner": winner,
        "winner_count": winner_count,
        "total_valid_votes": total_valid_votes,
        "total_samples": total_samples,
        "agreement_rate": round(agreement_rate, 4),
        "confidence": round(confidence, 4),
        "clusters": clusters,
    }


def majority_vote_from_completions(
    completions: Sequence[str],
) -> Dict[str, Any]:
    """Extract answers from raw completion strings and compute majority vote."""
    extracted_answers: List[Optional[str]] = []
    extraction_methods: List[str] = []

    for comp in completions:
        ans, method = extract_answer(comp)
        extracted_answers.append(ans)
        extraction_methods.append(method)

    result = majority_vote_with_details(extracted_answers)
    result["extracted_answers"] = extracted_answers
    result["extraction_methods"] = extraction_methods
    return result
