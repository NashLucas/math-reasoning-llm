"""Mathematical reasoning metrics calculation engine.

Calculates key post-training evaluation metrics:
- Accuracy (greedy / deterministic single-sample accuracy)
- maj@k (majority-vote accuracy across k samples)
- pass@k (fraction of problems with at least one correct sample among k)
- Format compliance rate (fraction of responses following the canonical '#### <num>' contract)
- Fallback extraction rate (fraction falling back to 'boxed', 'answer_is', or 'last_number')
- Unparseable rate (fraction where no answer could be extracted)
- Average prompt, completion, and total token lengths
- Truncation rate (fraction reaching max_new_tokens)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.math_utils.normalization import is_equivalent
from src.reasoning.self_consistency import majority_vote

logger = logging.getLogger(__name__)


def get_gold_answer(record: Dict[str, Any]) -> Optional[str]:
    """Extract gold target answer from diverse record schemas."""
    gold = (
        record.get("gold_answer")
        or record.get("gold")
        or record.get("target")
        or record.get("answer")
    )
    return str(gold).strip() if gold is not None else None


def get_record_completions(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize a record into a standardized list of completion dictionaries.

    Returns:
        List of dicts: [
            {
                "extracted_answer": str | None,
                "extraction_method": str,
                "prompt_tokens": int,
                "completion_tokens": int,
                "is_truncated": bool | None,
            }, ...
        ]
    """
    completions: List[Dict[str, Any]] = []

    # Case 1: Multi-sample record (e.g. self_consistency)
    if "extracted_answers" in record:
        answers = record.get("extracted_answers", [])
        methods = record.get("extraction_methods", [])
        p_tokens = record.get("prompt_tokens", [])
        c_tokens = record.get("completion_tokens", [])

        num_samples = len(answers)
        for i in range(num_samples):
            ans = answers[i] if i < len(answers) else None
            method = methods[i] if i < len(methods) else ("none" if ans is None else "hash")

            pt = p_tokens[i] if (isinstance(p_tokens, list) and i < len(p_tokens)) else (
                p_tokens if isinstance(p_tokens, (int, float)) else 0
            )
            ct = c_tokens[i] if (isinstance(c_tokens, list) and i < len(c_tokens)) else (
                c_tokens if isinstance(c_tokens, (int, float)) else 0
            )

            completions.append({
                "extracted_answer": ans,
                "extraction_method": method,
                "prompt_tokens": int(pt),
                "completion_tokens": int(ct),
                "is_truncated": None,
            })
        return completions

    # Case 2: Single-completion record (e.g. greedy)
    ans = record.get("extracted_answer")
    method = record.get("extraction_method") or ("none" if ans is None else "hash")
    pt = record.get("prompt_tokens", 0)
    ct = record.get("completion_tokens", 0)

    completions.append({
        "extracted_answer": ans,
        "extraction_method": method,
        "prompt_tokens": int(pt) if pt is not None else 0,
        "completion_tokens": int(ct) if ct is not None else 0,
        "is_truncated": record.get("is_truncated"),
    })
    return completions


def compute_accuracy(records: Sequence[Dict[str, Any]]) -> float:
    """Compute top-1 / greedy accuracy across problems."""
    if not records:
        return 0.0

    correct_count = 0
    total = len(records)

    for rec in records:
        gold = get_gold_answer(rec)
        if gold is None:
            continue

        # If already scored
        if "is_correct" in rec and "extracted_answers" not in rec:
            if bool(rec["is_correct"]):
                correct_count += 1
            continue

        completions = get_record_completions(rec)
        if not completions:
            continue

        pred = completions[0]["extracted_answer"]
        if is_equivalent(pred, gold):
            correct_count += 1

    return correct_count / total if total > 0 else 0.0


def compute_maj_at_k(records: Sequence[Dict[str, Any]], k: Optional[int] = None) -> float:
    """Compute majority-vote accuracy (maj@k) across problems.

    For each problem, takes the first k samples (or all if k is None),
    determines the majority vote winner using equivalence clustering,
    and checks if the winner matches the gold target.
    """
    if not records:
        return 0.0

    correct_count = 0
    total = len(records)

    for rec in records:
        gold = get_gold_answer(rec)
        if gold is None:
            continue

        completions = get_record_completions(rec)
        if not completions:
            continue

        slice_comps = completions[:k] if k is not None else completions
        candidate_answers = [c["extracted_answer"] for c in slice_comps]
        winner = majority_vote(candidate_answers)

        if winner is not None and is_equivalent(winner, gold):
            correct_count += 1

    return correct_count / total if total > 0 else 0.0


def compute_pass_at_k(records: Sequence[Dict[str, Any]], k: Optional[int] = None) -> float:
    """Compute empirical pass@k (share of problems with at least 1 correct candidate in k samples)."""
    if not records:
        return 0.0

    pass_count = 0
    total = len(records)

    for rec in records:
        gold = get_gold_answer(rec)
        if gold is None:
            continue

        completions = get_record_completions(rec)
        if not completions:
            continue

        slice_comps = completions[:k] if k is not None else completions
        candidate_answers = [c["extracted_answer"] for c in slice_comps]

        has_correct = any(is_equivalent(ans, gold) for ans in candidate_answers if ans is not None)
        if has_correct:
            pass_count += 1

    return pass_count / total if total > 0 else 0.0


def compute_format_statistics(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Compute format compliance, fallback extraction, and unparseable rates.

    Categories:
        - Format compliance: method == 'hash' (canonical '#### <number>' contract)
        - Fallback: method in ('boxed', 'answer_is', 'last_number')
        - Unparseable: method == 'none' or answer is None
    """
    all_completions: List[Dict[str, Any]] = []
    for rec in records:
        all_completions.extend(get_record_completions(rec))

    total = len(all_completions)
    if total == 0:
        return {
            "format_compliance_rate": 0.0,
            "fallback_rate": 0.0,
            "unparseable_rate": 0.0,
            "canonical_hash_count": 0,
            "fallback_count": 0,
            "unparseable_count": 0,
            "total_completions": 0,
        }

    hash_count = 0
    fallback_count = 0
    unparseable_count = 0

    for comp in all_completions:
        method = comp.get("extraction_method", "none")
        ans = comp.get("extracted_answer")

        if ans is None or method == "none":
            unparseable_count += 1
        elif method == "hash":
            hash_count += 1
        elif method in ("boxed", "answer_is", "last_number"):
            fallback_count += 1
        else:
            # Unknown non-canonical method
            fallback_count += 1

    return {
        "format_compliance_rate": round(hash_count / total, 4),
        "fallback_rate": round(fallback_count / total, 4),
        "unparseable_rate": round(unparseable_count / total, 4),
        "canonical_hash_count": hash_count,
        "fallback_count": fallback_count,
        "unparseable_count": unparseable_count,
        "total_completions": total,
    }


def compute_token_statistics(
    records: Sequence[Dict[str, Any]],
    max_new_tokens: Optional[int] = 512,
) -> Dict[str, float]:
    """Compute average token lengths and truncation rate across all completions."""
    all_completions: List[Dict[str, Any]] = []
    for rec in records:
        all_completions.extend(get_record_completions(rec))

    total = len(all_completions)
    if total == 0:
        return {
            "average_prompt_tokens": 0.0,
            "average_completion_tokens": 0.0,
            "average_total_tokens": 0.0,
            "truncation_rate": 0.0,
            "truncated_count": 0,
        }

    total_prompt = 0
    total_completion = 0
    truncated_count = 0

    for comp in all_completions:
        pt = comp.get("prompt_tokens", 0)
        ct = comp.get("completion_tokens", 0)
        total_prompt += pt
        total_completion += ct

        # Check truncation
        is_trunc = comp.get("is_truncated")
        if is_trunc is True:
            truncated_count += 1
        elif max_new_tokens is not None and ct >= max_new_tokens:
            truncated_count += 1

    avg_pt = total_prompt / total
    avg_ct = total_completion / total
    avg_total = (total_prompt + total_completion) / total

    return {
        "average_prompt_tokens": round(avg_pt, 2),
        "average_completion_tokens": round(avg_ct, 2),
        "average_total_tokens": round(avg_total, 2),
        "truncation_rate": round(truncated_count / total, 4),
        "truncated_count": truncated_count,
    }


def compute_all_metrics(
    records: Sequence[Dict[str, Any]],
    max_new_tokens: Optional[int] = 512,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the full suite of mathematical reasoning metrics.

    Args:
        records: Sequence of generation records (single-completion or multi-sample).
        max_new_tokens: Max new tokens budget used for truncation detection.
        k: Optional sample limit for maj@k and pass@k.

    Returns:
        Comprehensive dictionary of metrics.
    """
    total_problems = len(records)
    if total_problems == 0:
        return {
            "total_problems": 0,
            "total_completions": 0,
            "accuracy": 0.0,
            "accuracy_percent": 0.0,
            "maj_at_k": 0.0,
            "pass_at_k": 0.0,
            "format_compliance_rate": 0.0,
            "fallback_rate": 0.0,
            "unparseable_rate": 0.0,
            "average_prompt_tokens": 0.0,
            "average_completion_tokens": 0.0,
            "average_total_tokens": 0.0,
            "truncation_rate": 0.0,
        }

    # Determine if records are multi-sample
    is_multi_sample = any("extracted_answers" in rec for rec in records)

    acc = compute_accuracy(records)
    maj_k = compute_maj_at_k(records, k=k)
    pass_k = compute_pass_at_k(records, k=k)
    format_stats = compute_format_statistics(records)
    token_stats = compute_token_statistics(records, max_new_tokens=max_new_tokens)

    # For reporting: primary accuracy is maj@k if multi-sample, else greedy accuracy
    primary_accuracy = maj_k if is_multi_sample else acc

    return {
        "total_problems": total_problems,
        "total_completions": format_stats["total_completions"],
        "is_multi_sample": is_multi_sample,
        "accuracy": round(acc, 4),
        "accuracy_percent": round(acc * 100.0, 2),
        "primary_accuracy": round(primary_accuracy, 4),
        "primary_accuracy_percent": round(primary_accuracy * 100.0, 2),
        "maj_at_k": round(maj_k, 4),
        "maj_at_k_percent": round(maj_k * 100.0, 2),
        "pass_at_k": round(pass_k, 4),
        "pass_at_k_percent": round(pass_k * 100.0, 2),
        "format_compliance_rate": format_stats["format_compliance_rate"],
        "format_compliance_percent": round(format_stats["format_compliance_rate"] * 100.0, 2),
        "fallback_rate": format_stats["fallback_rate"],
        "fallback_percent": round(format_stats["fallback_rate"] * 100.0, 2),
        "unparseable_rate": format_stats["unparseable_rate"],
        "unparseable_percent": round(format_stats["unparseable_rate"] * 100.0, 2),
        "canonical_hash_count": format_stats["canonical_hash_count"],
        "fallback_count": format_stats["fallback_count"],
        "unparseable_count": format_stats["unparseable_count"],
        "average_prompt_tokens": token_stats["average_prompt_tokens"],
        "average_completion_tokens": token_stats["average_completion_tokens"],
        "average_total_tokens": token_stats["average_total_tokens"],
        "truncation_rate": token_stats["truncation_rate"],
        "truncation_percent": round(token_stats["truncation_rate"] * 100.0, 2),
        "truncated_count": token_stats["truncated_count"],
    }
