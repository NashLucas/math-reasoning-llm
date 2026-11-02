"""Reasoning trace curation and correctness filtering (Phase 3).

Provides rigorous filtering and quality checks for synthetic reasoning traces:
1. Correctness: verifies extracted answer matches gold answer via is_equivalent.
2. Answer format: ensures final answer is parseable and valid.
3. Quality & length: checks token bounds (min/max tokens) and rejects truncated responses.
4. Degeneracy check: detects repetitive text and looping n-grams.
5. Filter statistics: computes comprehensive retention and rejection breakdowns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.data.synthetic import ReasoningTrace
from src.math_utils.answer_extraction import extract_answer
from src.math_utils.normalization import is_equivalent

logger = logging.getLogger(__name__)

# Default curation thresholds
DEFAULT_MIN_TOKENS = 10
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_NGRAM_REPETITION = 3
DEFAULT_REPETITION_NGRAM_SIZE = 4


@dataclass
class CurationFilterStats:
    """Aggregate statistics for a trace curation filtering pass."""

    total_evaluated: int = 0
    kept_count: int = 0
    rejected_count: int = 0
    retention_rate: float = 0.0
    retention_percent: float = 0.0
    rejection_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluated": self.total_evaluated,
            "kept_count": self.kept_count,
            "rejected_count": self.rejected_count,
            "retention_rate": round(self.retention_rate, 4),
            "retention_percent": round(self.retention_percent, 2),
            "rejection_breakdown": self.rejection_breakdown,
        }


def detect_repeated_ngrams(
    text: str,
    ngram_size: int = DEFAULT_REPETITION_NGRAM_SIZE,
    max_repeats: int = DEFAULT_MAX_NGRAM_REPETITION,
) -> bool:
    """Detect degenerate repetitive loops where an n-gram repeats consecutively.

    Returns True if any phrase of length 2..ngram_size repeats consecutively
    more than max_repeats times.
    """
    if not text:
        return False

    tokens = [t.lower() for t in re.findall(r"\w+", text)]
    num_tokens = len(tokens)
    if num_tokens < 2 * (max_repeats + 1):
        return False

    # Check for repeating consecutive phrases of length L from 2 up to ngram_size
    for L in range(2, ngram_size + 1):
        required_len = L * (max_repeats + 1)
        if num_tokens < required_len:
            continue
        for i in range(num_tokens - required_len + 1):
            chunk = tokens[i : i + L]
            repeats = 1
            idx = i + L
            while idx + L <= num_tokens and tokens[idx : idx + L] == chunk:
                repeats += 1
                if repeats > max_repeats:
                    return True
                idx += L

    return False


def evaluate_trace_correctness(
    trace: Union[Dict[str, Any], ReasoningTrace],
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_ngram_repetition: int = DEFAULT_MAX_NGRAM_REPETITION,
    require_format_compliance: bool = False,
    recheck_answer: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Evaluate a single reasoning trace against correctness and quality criteria.

    Returns:
        (is_kept: bool, rejection_reason: Optional[str])
    """
    if isinstance(trace, ReasoningTrace):
        data = trace.to_dict()
    else:
        data = trace

    completion = data.get("completion", "")
    gold_answer = str(data.get("gold_answer", "")).strip()

    if not completion or not completion.strip():
        return False, "empty_completion"

    # Token length checks
    completion_tokens = data.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = len(completion.split())

    if completion_tokens < min_tokens:
        return False, "too_short"

    if completion_tokens > max_tokens:
        return False, "too_long"

    # Truncation check
    finish_reason = data.get("finish_reason", "stop")
    if finish_reason == "length":
        return False, "truncated"

    # Answer extraction and verification
    if recheck_answer or "extracted_answer" not in data or "extraction_method" not in data:
        extracted_answer, extraction_method = extract_answer(completion)
    else:
        extracted_answer = data.get("extracted_answer")
        extraction_method = data.get("extraction_method", "none")

    if not extracted_answer or extraction_method == "none":
        return False, "unparseable_answer"

    if require_format_compliance and extraction_method != "hash":
        return False, "format_noncompliant"

    if not is_equivalent(extracted_answer, gold_answer):
        return False, "incorrect_answer"

    # Repetition / degeneracy check
    if detect_repeated_ngrams(completion, max_repeats=max_ngram_repetition):
        return False, "repetitive_text"

    return True, None


class CorrectnessFilter:
    """Filters collections of reasoning traces, retaining only verified correct traces."""

    def __init__(
        self,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_ngram_repetition: int = DEFAULT_MAX_NGRAM_REPETITION,
        require_format_compliance: bool = False,
        recheck_answer: bool = True,
    ) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.max_ngram_repetition = max_ngram_repetition
        self.require_format_compliance = require_format_compliance
        self.recheck_answer = recheck_answer

    def filter_trace(
        self, trace: Union[Dict[str, Any], ReasoningTrace]
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate a single trace."""
        return evaluate_trace_correctness(
            trace=trace,
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens,
            max_ngram_repetition=self.max_ngram_repetition,
            require_format_compliance=self.require_format_compliance,
            recheck_answer=self.recheck_answer,
        )

    def filter_traces(
        self,
        traces: Sequence[Union[Dict[str, Any], ReasoningTrace]],
    ) -> Tuple[List[Dict[str, Any]], CurationFilterStats]:
        """Filter a sequence of traces, returning kept traces and curation statistics."""
        kept: List[Dict[str, Any]] = []
        breakdown: Dict[str, int] = {}

        for item in traces:
            raw_dict = item.to_dict() if isinstance(item, ReasoningTrace) else dict(item)
            is_kept, reason = self.filter_trace(raw_dict)

            if is_kept:
                # Ensure updated correctness metadata is saved
                raw_dict["is_correct"] = True
                kept.append(raw_dict)
            else:
                reason_key = reason or "unknown"
                breakdown[reason_key] = breakdown.get(reason_key, 0) + 1

        total = len(traces)
        kept_count = len(kept)
        rejected_count = total - kept_count
        rate = kept_count / total if total > 0 else 0.0

        stats = CurationFilterStats(
            total_evaluated=total,
            kept_count=kept_count,
            rejected_count=rejected_count,
            retention_rate=rate,
            retention_percent=rate * 100.0,
            rejection_breakdown=breakdown,
        )

        return kept, stats


def compute_problem_pass_rates(
    traces: Sequence[Union[Dict[str, Any], ReasoningTrace]],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-problem pass rate statistics across reasoning traces."""
    problem_traces: Dict[str, List[Dict[str, Any]]] = {}

    for t in traces:
        d = t.to_dict() if isinstance(t, ReasoningTrace) else t
        pid = d["problem_id"]
        problem_traces.setdefault(pid, []).append(d)

    pass_rate_stats: Dict[str, Dict[str, Any]] = {}
    for pid, p_list in problem_traces.items():
        total_k = len(p_list)
        correct_count = sum(1 for item in p_list if item.get("is_correct", False))
        rate = correct_count / total_k if total_k > 0 else 0.0
        pass_rate_stats[pid] = {
            "total_samples": total_k,
            "correct_samples": correct_count,
            "pass_rate": round(rate, 4),
            "is_solved": correct_count > 0,
        }

    return pass_rate_stats
