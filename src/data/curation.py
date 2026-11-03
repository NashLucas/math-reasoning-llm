"""Reasoning trace curation, correctness filtering, and yield analysis (Phase 3).

Provides rigorous filtering and quality checks for synthetic reasoning traces:
1. Correctness: verifies extracted answer matches gold answer via is_equivalent.
2. Answer format: ensures final answer is parseable and adheres to canonical answer contract.
3. Quality & length: checks token bounds (min/max tokens) and rejects truncated responses.
4. Degeneracy check: detects repetitive text and looping n-grams.
5. Per-problem pass rates: computes k-sample pass rates and difficulty distributions (Phase 6 input).
6. Yield analysis: computes overall yield, solve rates, unsolved counts, and formatting failures.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass
class ProblemPassRate:
    """Pass rate metrics for a single mathematical problem across k sampled traces."""

    problem_id: str
    question: str = ""
    gold_answer: str = ""
    total_samples: int = 0
    correct_samples: int = 0
    pass_rate: float = 0.0
    is_solved: bool = False
    difficulty_bin: str = "zero"
    correct_trace_ids: List[str] = field(default_factory=list)
    all_trace_ids: List[str] = field(default_factory=list)
    prompt_types: List[str] = field(default_factory=list)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "pass_rate": round(self.pass_rate, 4),
            "is_solved": self.is_solved,
            "difficulty_bin": self.difficulty_bin,
            "correct_trace_ids": self.correct_trace_ids,
            "all_trace_ids": self.all_trace_ids,
            "prompt_types": self.prompt_types,
        }


@dataclass
class YieldAnalysisReport:
    """Comprehensive yield and pass-rate report across generated reasoning traces."""

    total_problems: int = 0
    total_traces: int = 0
    correct_traces: int = 0
    trace_yield_rate: float = 0.0
    trace_yield_percent: float = 0.0
    solved_problems: int = 0
    problem_solve_rate: float = 0.0
    problem_solve_percent: float = 0.0
    unsolved_problems: int = 0
    unsolved_rate: float = 0.0
    unsolved_percent: float = 0.0
    distribution_by_correct_count: Dict[str, int] = field(default_factory=dict)
    distribution_by_difficulty: Dict[str, int] = field(default_factory=dict)
    format_failures: int = 0
    format_failure_rate: float = 0.0
    filter_stats: Optional[CurationFilterStats] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_problems": self.total_problems,
            "total_traces": self.total_traces,
            "correct_traces": self.correct_traces,
            "trace_yield_rate": round(self.trace_yield_rate, 4),
            "trace_yield_percent": round(self.trace_yield_percent, 2),
            "solved_problems": self.solved_problems,
            "problem_solve_rate": round(self.problem_solve_rate, 4),
            "problem_solve_percent": round(self.problem_solve_percent, 2),
            "unsolved_problems": self.unsolved_problems,
            "unsolved_rate": round(self.unsolved_rate, 4),
            "unsolved_percent": round(self.unsolved_percent, 2),
            "distribution_by_correct_count": self.distribution_by_correct_count,
            "distribution_by_difficulty": self.distribution_by_difficulty,
            "format_failures": self.format_failures,
            "format_failure_rate": round(self.format_failure_rate, 4),
            "filter_stats": self.filter_stats.to_dict() if self.filter_stats else {},
        }

    def to_markdown_summary(self) -> str:
        """Render a readable markdown summary of yield analysis findings."""
        lines = [
            "# Reasoning Trace Yield Analysis Report",
            "",
            "## Summary Metrics",
            f"- **Total Problems Evaluated:** {self.total_problems:,}",
            f"- **Total Traces Evaluated:** {self.total_traces:,}",
            f"- **Correct Traces:** {self.correct_traces:,} ({self.trace_yield_percent:.2f}% yield)",
            f"- **Problems Solved (>= 1 correct trace):** {self.solved_problems:,} ({self.problem_solve_percent:.2f}%)",
            f"- **Problems Unsolved (0 correct traces):** {self.unsolved_problems:,} ({self.unsolved_percent:.2f}%)",
            f"- **Format Failures:** {self.format_failures:,} ({self.format_failure_rate * 100:.2f}%)",
            "",
            "## Pass-Rate Distribution (Correct Samples out of k)",
            "| Correct Count | Problem Count | Percentage | Difficulty Category |",
            "|---|---|---|---|",
        ]

        for key, count in sorted(self.distribution_by_correct_count.items()):
            pct = (count / self.total_problems * 100.0) if self.total_problems > 0 else 0.0
            category = "Zero (0/k)" if key.startswith("0") else "Non-zero"
            lines.append(f"| `{key}` | {count:,} | {pct:.2f}% | {category} |")

        lines.extend([
            "",
            "## Difficulty Breakdown (for Phase 6 GRPO / Difficulty Filter)",
            "| Difficulty Zone | Problem Count | Share of Problems | Description |",
            "|---|---|---|---|",
            f"| **Zero (0/k)** | {self.distribution_by_difficulty.get('zero', 0):,} | {(self.distribution_by_difficulty.get('zero', 0) / self.total_problems * 100 if self.total_problems > 0 else 0):.2f}% | Completely failed by teacher model |",
            f"| **Hard (1/k)** | {self.distribution_by_difficulty.get('hard', 0):,} | {(self.distribution_by_difficulty.get('hard', 0) / self.total_problems * 100 if self.total_problems > 0 else 0):.2f}% | High difficulty; only 1 sample succeeded |",
            f"| **Medium (2-3/k)** | {self.distribution_by_difficulty.get('medium', 0):,} | {(self.distribution_by_difficulty.get('medium', 0) / self.total_problems * 100 if self.total_problems > 0 else 0):.2f}% | Learning zone / ideal for GRPO variance |",
            f"| **Easy (4/k)** | {self.distribution_by_difficulty.get('easy', 0):,} | {(self.distribution_by_difficulty.get('easy', 0) / self.total_problems * 100 if self.total_problems > 0 else 0):.2f}% | Mastered by teacher model |",
            "",
        ])

        if self.filter_stats and self.filter_stats.rejection_breakdown:
            lines.extend([
                "## Curation Filter Rejection Breakdown",
                "| Rejection Reason | Count | Share of Rejected Traces |",
                "|---|---|---|",
            ])
            rejected = self.filter_stats.rejected_count
            for reason, cnt in sorted(self.filter_stats.rejection_breakdown.items(), key=lambda x: x[1], reverse=True):
                share = (cnt / rejected * 100.0) if rejected > 0 else 0.0
                lines.append(f"| `{reason}` | {cnt:,} | {share:.2f}% |")
            lines.append("")

        return "\n".join(lines)


def classify_difficulty(correct_samples: int, total_samples: int) -> str:
    """Classify problem difficulty bin based on sample pass rate."""
    if total_samples <= 0 or correct_samples == 0:
        return "zero"
    if correct_samples == total_samples:
        return "easy"
    rate = correct_samples / total_samples
    if rate <= 0.25:
        return "hard"
    return "medium"


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
    problems: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, ProblemPassRate]:
    """Compute per-problem pass rate statistics across reasoning traces.

    Args:
        traces: Sequence of reasoning traces (dicts or ReasoningTrace objects).
        problems: Optional sequence of problems to ensure even problems with zero
            traces or zero correct traces are fully indexed.

    Returns:
        Dictionary mapping problem_id to ProblemPassRate.
    """
    problem_traces: Dict[str, List[Dict[str, Any]]] = {}

    for t in traces:
        d = t.to_dict() if isinstance(t, ReasoningTrace) else dict(t)
        pid = str(d["problem_id"])
        problem_traces.setdefault(pid, []).append(d)

    # Initialize lookup from problem definitions if provided
    pass_rate_stats: Dict[str, ProblemPassRate] = {}
    if problems:
        for p in problems:
            pid = str(p.get("problem_id", ""))
            if not pid:
                continue
            pass_rate_stats[pid] = ProblemPassRate(
                problem_id=pid,
                question=str(p.get("question", "")),
                gold_answer=str(p.get("gold_answer", "")),
                total_samples=0,
                correct_samples=0,
                pass_rate=0.0,
                is_solved=False,
                difficulty_bin="zero",
            )

    # Populate pass rates from traces
    for pid, p_list in problem_traces.items():
        total_k = len(p_list)
        correct_items = [item for item in p_list if item.get("is_correct", False)]
        correct_count = len(correct_items)
        rate = correct_count / total_k if total_k > 0 else 0.0

        first_item = p_list[0]
        q_text = first_item.get("question", "")
        gold_ans = str(first_item.get("gold_answer", ""))

        all_trace_ids = [item.get("trace_id", "") for item in p_list if item.get("trace_id")]
        correct_trace_ids = [
            item.get("trace_id", "") for item in correct_items if item.get("trace_id")
        ]
        prompt_types = list({item.get("prompt_type", "") for item in p_list if item.get("prompt_type")})

        diff_bin = classify_difficulty(correct_count, total_k)

        pass_rate_stats[pid] = ProblemPassRate(
            problem_id=pid,
            question=q_text or (pass_rate_stats[pid].question if pid in pass_rate_stats else ""),
            gold_answer=gold_ans or (pass_rate_stats[pid].gold_answer if pid in pass_rate_stats else ""),
            total_samples=total_k,
            correct_samples=correct_count,
            pass_rate=round(rate, 4),
            is_solved=correct_count > 0,
            difficulty_bin=diff_bin,
            correct_trace_ids=correct_trace_ids,
            all_trace_ids=all_trace_ids,
            prompt_types=prompt_types,
        )

    return pass_rate_stats


def save_problem_pass_rates(
    pass_rates: Dict[str, Union[ProblemPassRate, Dict[str, Any]]],
    output_path: Union[str, Path],
) -> Tuple[Path, Optional[Path]]:
    """Save problem pass rates to JSON and companion Parquet formats.

    Args:
        pass_rates: Mapping of problem_id to ProblemPassRate or dict.
        output_path: Target path (e.g. data/synthetic/generated/problem_pass_rates.json).

    Returns:
        Tuple of (json_path, parquet_path or None).
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_dict: Dict[str, Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []

    for pid, obj in pass_rates.items():
        d = obj.to_dict() if isinstance(obj, ProblemPassRate) else dict(obj)
        json_dict[pid] = d
        records.append(d)

    json_file = out_path if out_path.suffix == ".json" else out_path.with_suffix(".json")
    json_file.write_text(json.dumps(json_dict, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(json_dict)} problem pass rates to: {json_file}")

    parquet_file: Optional[Path] = None
    try:
        import pandas as pd

        parquet_path = json_file.with_suffix(".parquet")
        df = pd.DataFrame(records)
        df.to_parquet(parquet_path, index=False)
        parquet_file = parquet_path
        logger.info(f"Saved companion parquet pass rates to: {parquet_path}")
    except Exception as exc:
        logger.warning(f"Failed to save companion parquet pass rates: {exc}")

    return json_file, parquet_file


def load_problem_pass_rates(
    input_path: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """Load problem pass rates from JSON or Parquet file."""
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Pass rates file not found: {in_path}")

    if in_path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(in_path)
        records = df.to_dict(orient="records")
        return {str(r["problem_id"]): r for r in records}

    content = in_path.read_text(encoding="utf-8")
    data = json.loads(content)
    if isinstance(data, list):
        return {str(r["problem_id"]): r for r in data}
    return data


def analyze_reasoning_yield(
    traces: Sequence[Union[Dict[str, Any], ReasoningTrace]],
    problems: Optional[Sequence[Dict[str, Any]]] = None,
    correctness_filter: Optional[CorrectnessFilter] = None,
) -> Tuple[YieldAnalysisReport, Dict[str, ProblemPassRate], List[Dict[str, Any]]]:
    """Execute complete reasoning trace yield analysis and pass-rate aggregation.

    Args:
        traces: Sequence of traces.
        problems: Optional complete problem list to verify coverage.
        correctness_filter: Optional filter engine (defaults to standard CorrectnessFilter).

    Returns:
        (report, per_problem_pass_rates, kept_traces)
    """
    filter_engine = correctness_filter or CorrectnessFilter()

    # Step 1: Run correctness filtering
    kept_traces, filter_stats = filter_engine.filter_traces(traces)

    # Step 2: Compute format failures on original traces
    format_failures = 0
    for t in traces:
        raw = t.to_dict() if isinstance(t, ReasoningTrace) else t
        comp = raw.get("completion", "")
        method = raw.get("extraction_method")
        if method is None:
            _, method = extract_answer(comp)
        if method != "hash":
            format_failures += 1

    # Step 3: Compute per-problem pass rates
    # Note: we use the filtered flags so is_correct is strictly true
    annotated_traces: List[Dict[str, Any]] = []
    kept_ids = {t["trace_id"] for t in kept_traces if "trace_id" in t}

    for t in traces:
        raw = dict(t.to_dict() if isinstance(t, ReasoningTrace) else t)
        raw_tid = raw.get("trace_id")
        if raw_tid:
            raw["is_correct"] = raw_tid in kept_ids
        else:
            # Fall back to evaluation function if trace_id is missing
            is_ok, _ = filter_engine.filter_trace(raw)
            raw["is_correct"] = is_ok
        annotated_traces.append(raw)

    pass_rates = compute_problem_pass_rates(annotated_traces, problems=problems)

    # Step 4: Aggregate distributions
    total_problems = len(pass_rates)
    total_traces = len(traces)
    correct_traces = len(kept_traces)

    trace_yield_rate = correct_traces / total_traces if total_traces > 0 else 0.0
    format_fail_rate = format_failures / total_traces if total_traces > 0 else 0.0

    solved_count = 0
    unsolved_count = 0
    dist_by_count: Dict[str, int] = {}
    dist_by_diff: Dict[str, int] = {"zero": 0, "hard": 0, "medium": 0, "easy": 0}

    for p in pass_rates.values():
        if p.is_solved:
            solved_count += 1
        else:
            unsolved_count += 1

        count_key = f"{p.correct_samples}_of_{p.total_samples}"
        dist_by_count[count_key] = dist_by_count.get(count_key, 0) + 1

        dbin = p.difficulty_bin
        dist_by_diff[dbin] = dist_by_diff.get(dbin, 0) + 1

    problem_solve_rate = solved_count / total_problems if total_problems > 0 else 0.0
    unsolved_rate = unsolved_count / total_problems if total_problems > 0 else 0.0

    report = YieldAnalysisReport(
        total_problems=total_problems,
        total_traces=total_traces,
        correct_traces=correct_traces,
        trace_yield_rate=trace_yield_rate,
        trace_yield_percent=trace_yield_rate * 100.0,
        solved_problems=solved_count,
        problem_solve_rate=problem_solve_rate,
        problem_solve_percent=problem_solve_rate * 100.0,
        unsolved_problems=unsolved_count,
        unsolved_rate=unsolved_rate,
        unsolved_percent=unsolved_rate * 100.0,
        distribution_by_correct_count=dist_by_count,
        distribution_by_difficulty=dist_by_diff,
        format_failures=format_failures,
        format_failure_rate=format_fail_rate,
        filter_stats=filter_stats,
    )

    return report, pass_rates, kept_traces


def get_solved_problem_ids(
    pass_rates: Union[Dict[str, Any], Path, str],
    min_correct: int = 1,
) -> set[str]:
    """Extract set of problem IDs that have at least min_correct correct traces.

    Args:
        pass_rates: Dict of problem_id to pass rate data, or path to problem_pass_rates.json/.parquet.
        min_correct: Minimum number of correct samples required (default 1).

    Returns:
        Set of solved problem IDs.
    """
    if isinstance(pass_rates, (str, Path)):
        data = load_problem_pass_rates(pass_rates)
    else:
        data = pass_rates

    solved_ids: set[str] = set()
    for pid, info in data.items():
        if isinstance(info, ProblemPassRate):
            if info.correct_samples >= min_correct or (min_correct <= 1 and info.is_solved):
                solved_ids.add(str(pid))
        elif isinstance(info, dict):
            if info.get("correct_samples", 0) >= min_correct or (min_correct <= 1 and info.get("is_solved", False)):
                solved_ids.add(str(pid))
    return solved_ids


def filter_problems_by_pass_rate(
    problems: Sequence[Dict[str, Any]],
    pass_rates: Union[Dict[str, Any], Path, str],
    min_correct: int = 1,
) -> List[Dict[str, Any]]:
    """Filter problem list to only those that achieved at least min_correct correct traces.

    Args:
        problems: Sequence of problem dictionaries (each containing 'problem_id').
        pass_rates: Dict of problem_id to pass rate data, or path to problem_pass_rates.json/.parquet.
        min_correct: Minimum number of correct samples required (default 1).

    Returns:
        Filtered list of problem dictionaries preserving original order.
    """
    solved_ids = get_solved_problem_ids(pass_rates, min_correct=min_correct)
    return [p for p in problems if str(p.get("problem_id", "")) in solved_ids]

# ==============================================================================
# Phase 3 Day 23: Final SFT Dataset Curation, Deduplication, & Contamination Gate
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ContaminationError(ValueError):
    """Raised when validation or test set problem IDs are found in training/curated data."""
    pass


@dataclass
class CuratedTrace:
    """A final curated reasoning trace prepared for SFT training (Phase 3 Day 23)."""

    trace_id: str
    problem_id: str
    prompt_type: str
    question: str
    gold_answer: str
    completion: str
    source: str = "synthetic"  # "synthetic" or "gold_clean"
    is_gold_fallback: bool = False
    rank_within_problem: int = 1
    score: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "problem_id": self.problem_id,
            "prompt_type": self.prompt_type,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "completion": self.completion,
            "source": self.source,
            "is_gold_fallback": self.is_gold_fallback,
            "rank_within_problem": self.rank_within_problem,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class CuratedDatasetStats:
    """Aggregate statistics for the curated SFT reasoning dataset."""

    total_curated_traces: int = 0
    unique_problems_covered: int = 0
    target_problems: int = 3000
    problem_coverage_percent: float = 0.0
    synthetic_trace_count: int = 0
    gold_fallback_count: int = 0
    by_prompt_type: Dict[str, int] = field(default_factory=dict)
    traces_per_problem_dist: Dict[str, int] = field(default_factory=dict)
    near_duplicates_pruned: int = 0
    overlength_pruned: int = 0
    looping_pruned: int = 0
    mean_token_count: float = 0.0
    mean_word_count: float = 0.0
    mean_judge_score: float = 0.0
    contamination_check_passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_curated_traces": self.total_curated_traces,
            "unique_problems_covered": self.unique_problems_covered,
            "target_problems": self.target_problems,
            "problem_coverage_percent": round(self.problem_coverage_percent, 2),
            "synthetic_trace_count": self.synthetic_trace_count,
            "gold_fallback_count": self.gold_fallback_count,
            "by_prompt_type": self.by_prompt_type,
            "traces_per_problem_dist": self.traces_per_problem_dist,
            "near_duplicates_pruned": self.near_duplicates_pruned,
            "overlength_pruned": self.overlength_pruned,
            "looping_pruned": self.looping_pruned,
            "mean_token_count": round(self.mean_token_count, 1),
            "mean_word_count": round(self.mean_word_count, 1),
            "mean_judge_score": round(self.mean_judge_score, 4),
            "contamination_check_passed": self.contamination_check_passed,
        }


def assert_no_contamination(
    problem_ids: Sequence[str],
    splits_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Assert that none of the provided problem IDs belong to validation or test splits.

    Raises:
        ContaminationError: If any problem ID belongs to val_ids.json or test_ids.json.
    """
    s_dir = Path(splits_dir) if splits_dir else (PROJECT_ROOT / "data" / "splits")
    val_file = s_dir / "val_ids.json"
    test_file = s_dir / "test_ids.json"

    val_ids: set[str] = set()
    test_ids: set[str] = set()

    if val_file.exists():
        val_ids = set(json.loads(val_file.read_text(encoding="utf-8")))
    if test_file.exists():
        test_ids = set(json.loads(test_file.read_text(encoding="utf-8")))

    input_ids = set(str(pid) for pid in problem_ids)
    val_overlap = input_ids & val_ids
    test_overlap = input_ids & test_ids

    if val_overlap:
        raise ContaminationError(
            f"TEST-SET WALL CONTAMINATION: {len(val_overlap)} problem IDs belong to validation split: "
            f"{list(val_overlap)[:5]}"
        )
    if test_overlap:
        raise ContaminationError(
            f"TEST-SET WALL CONTAMINATION: {len(test_overlap)} problem IDs belong to test split: "
            f"{list(test_overlap)[:5]}"
        )


def compute_text_jaccard_similarity(
    text_a: str,
    text_b: str,
    ngram_size: int = 4,
) -> float:
    """Compute character n-gram Jaccard similarity between two texts."""
    clean_a = re.sub(r"\s+", " ", text_a.strip().lower())
    clean_b = re.sub(r"\s+", " ", text_b.strip().lower())
    if clean_a == clean_b:
        return 1.0
    if len(clean_a) < ngram_size or len(clean_b) < ngram_size:
        return 1.0 if clean_a == clean_b else 0.0

    grams_a = set(clean_a[i : i + ngram_size] for i in range(len(clean_a) - ngram_size + 1))
    grams_b = set(clean_b[i : i + ngram_size] for i in range(len(clean_b) - ngram_size + 1))
    union = grams_a | grams_b
    if not union:
        return 0.0
    return len(grams_a & grams_b) / len(union)


def is_near_duplicate(
    text_a: str,
    text_b: str,
    threshold: float = 0.85,
    ngram_size: int = 4,
) -> bool:
    """Determine whether two reasoning completions are near-duplicates."""
    if text_a.strip() == text_b.strip():
        return True
    return compute_text_jaccard_similarity(text_a, text_b, ngram_size) >= threshold


def build_gold_fallback_trace(
    problem: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a canonical reasoning trace from a clean human gold solution.

    Used when a problem has no correct synthetic reasoning traces generated.
    """
    pid = str(problem.get("problem_id", ""))
    question = str(problem.get("question", "")).strip()
    gold_answer = str(problem.get("gold_answer", "")).strip()
    cleaned = str(problem.get("cleaned_solution", "")).strip()

    if not cleaned:
        raw_ans = str(problem.get("answer", problem.get("solution", "")))
        from src.data.preprocessing import clean_solution
        cleaned, extracted_ans = clean_solution(raw_ans)
        if not gold_answer and extracted_ans:
            gold_answer = extracted_ans

    # Ensure canonical #### line is present
    canonical_suffix = f"#### {gold_answer}"
    if canonical_suffix not in cleaned:
        cleaned = f"{cleaned}\n#### {gold_answer}"

    words = cleaned.split()
    return {
        "trace_id": f"{pid}_gold_fallback",
        "problem_id": pid,
        "prompt_type": "chain_of_thought",
        "question": question,
        "gold_answer": gold_answer,
        "completion": cleaned,
        "source": "gold_clean",
        "is_gold_fallback": True,
        "rank_within_problem": 1,
        "score": {
            "completeness": 1.0,
            "verification_quality": None,
            "novelty": None,
            "overall_score": 1.0,
            "rationale": "Cleaned human gold reference solution (no synthetic solution available).",
            "parsing_status": "gold",
        },
        "metadata": {
            "is_gold_fallback": True,
            "completion_tokens": len(words),
            "finish_reason": "stop",
        },
    }


def curate_reasoning_dataset(
    traces: Sequence[Union[Dict[str, Any], ReasoningTrace]],
    problems: Sequence[Dict[str, Any]],
    max_traces_per_problem: int = 2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    near_dup_threshold: float = 0.85,
    splits_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[Dict[str, Any]], CuratedDatasetStats]:
    """Execute end-to-end reasoning dataset curation for SFT post-training (Day 23).

    Selection & Quality Rules:
    1. Correctness: Enforces Design Rule 4 (only traces reaching gold answer).
    2. Quality filters: Prunes traces exceeding max_tokens, below min_tokens, or with looping n-grams.
    3. At most 2 traces per problem:
       - Trace 1: Highest judge overall_score among all valid candidates.
       - Trace 2: Highest judge overall_score with a DIFFERENT prompt strategy, skipping near-duplicates.
         If no different strategy exists, takes top non-duplicate of the same strategy.
    4. Clean Gold Fallback: For problems with zero correct synthetic traces, uses the cleaned human gold solution.
    5. Contamination Gate: Asserts that zero validation or test problem IDs appear in the curated dataset.

    Returns:
        (curated_traces, CuratedDatasetStats)
    """
    filter_engine = CorrectnessFilter(
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )

    overlength_pruned = 0
    looping_pruned = 0
    near_duplicates_pruned = 0

    # Step 1: Filter and group candidate traces by problem_id
    valid_candidates_by_pid: Dict[str, List[Dict[str, Any]]] = {}

    for item in traces:
        raw = dict(item.to_dict() if isinstance(item, ReasoningTrace) else item)
        is_correct, reason = filter_engine.filter_trace(raw)
        if not is_correct:
            if reason == "too_long":
                overlength_pruned += 1
            elif reason == "repeated_ngrams":
                looping_pruned += 1
            continue

        pid = str(raw.get("problem_id", ""))
        if not pid:
            continue
        valid_candidates_by_pid.setdefault(pid, []).append(raw)

    # Step 2: Index problems
    problem_lookup: Dict[str, Dict[str, Any]] = {
        str(p.get("problem_id", "")): p for p in problems if str(p.get("problem_id", ""))
    }
    target_count = len(problem_lookup)

    curated_traces: List[Dict[str, Any]] = []
    synthetic_count = 0
    gold_count = 0
    by_ptype: Dict[str, int] = {}
    traces_per_prob_dist: Dict[str, int] = {}
    total_tokens = 0
    total_words = 0
    total_scores = 0.0

    # Step 3: Iterate through all target problems
    for pid, prob in problem_lookup.items():
        candidates = valid_candidates_by_pid.get(pid, [])

        if not candidates:
            # Unsolved problem: Fallback to cleaned human gold solution
            gold_trace = build_gold_fallback_trace(prob)
            curated_traces.append(gold_trace)
            gold_count += 1
            ptype = gold_trace["prompt_type"]
            by_ptype[ptype] = by_ptype.get(ptype, 0) + 1
            traces_per_prob_dist["1_trace"] = traces_per_prob_dist.get("1_trace", 0) + 1

            total_tokens += int(gold_trace.get("metadata", {}).get("completion_tokens", len(gold_trace["completion"].split())))
            total_words += len(gold_trace["completion"].split())
            total_scores += 1.0
            continue

        # Sort candidates: highest judge score first, then longer length
        def _trace_sort_key(t: Dict[str, Any]) -> Tuple[float, int]:
            score_dict = t.get("score")
            score_val = 0.0
            if isinstance(score_dict, dict):
                score_val = float(score_dict.get("overall_score", 0.0))
            elif isinstance(score_dict, (int, float)):
                score_val = float(score_dict)
            return (score_val, len(t.get("completion", "")))

        sorted_candidates = sorted(candidates, key=_trace_sort_key, reverse=True)

        selected_for_problem: List[Dict[str, Any]] = []

        # Trace 1: Best overall candidate
        trace_1 = dict(sorted_candidates[0])
        trace_1["rank_within_problem"] = 1
        trace_1["is_gold_fallback"] = False
        trace_1["source"] = "synthetic"
        selected_for_problem.append(trace_1)

        # Trace 2: Best candidate preferring DIFFERENT prompt strategy, skipping near-duplicates
        if max_traces_per_problem > 1 and len(sorted_candidates) > 1:
            t1_comp = trace_1.get("completion", "")
            t1_type = trace_1.get("prompt_type", "")

            # Tier A: Candidates with different prompt strategy
            diff_type_candidates = [
                c for c in sorted_candidates[1:] if c.get("prompt_type") != t1_type
            ]
            chosen_t2: Optional[Dict[str, Any]] = None

            for cand in diff_type_candidates:
                cand_comp = cand.get("completion", "")
                if is_near_duplicate(cand_comp, t1_comp, threshold=near_dup_threshold):
                    near_duplicates_pruned += 1
                    continue
                chosen_t2 = dict(cand)
                break

            # Tier B: If no different strategy candidate accepted, check same strategy non-duplicates
            if chosen_t2 is None:
                same_type_candidates = [
                    c for c in sorted_candidates[1:] if c.get("prompt_type") == t1_type
                ]
                for cand in same_type_candidates:
                    cand_comp = cand.get("completion", "")
                    if is_near_duplicate(cand_comp, t1_comp, threshold=near_dup_threshold):
                        near_duplicates_pruned += 1
                        continue
                    chosen_t2 = dict(cand)
                    break

            if chosen_t2 is not None:
                chosen_t2["rank_within_problem"] = 2
                chosen_t2["is_gold_fallback"] = False
                chosen_t2["source"] = "synthetic"
                selected_for_problem.append(chosen_t2)

        # Accumulate metrics for selected traces
        num_selected = len(selected_for_problem)
        dist_key = f"{num_selected}_trace" if num_selected == 1 else f"{num_selected}_traces"
        traces_per_prob_dist[dist_key] = traces_per_prob_dist.get(dist_key, 0) + 1

        for st in selected_for_problem:
            curated_traces.append(st)
            synthetic_count += 1
            ptype = st.get("prompt_type", "unknown")
            by_ptype[ptype] = by_ptype.get(ptype, 0) + 1

            comp = st.get("completion", "")
            words = comp.split()
            tok_count = int(st.get("metadata", {}).get("completion_tokens", len(words)))
            total_tokens += tok_count
            total_words += len(words)

            s_dict = st.get("score")
            s_val = float(s_dict.get("overall_score", 0.85)) if isinstance(s_dict, dict) else 0.85
            total_scores += s_val

    # Step 4: Strict Zero-Contamination Assertion
    curated_pids = [t["problem_id"] for t in curated_traces]
    assert_no_contamination(curated_pids, splits_dir=splits_dir)

    # Step 5: Compile Statistics
    total_curated = len(curated_traces)
    unique_problems = len({t["problem_id"] for t in curated_traces})
    cov_pct = (unique_problems / target_count * 100.0) if target_count > 0 else 0.0

    mean_tokens = total_tokens / total_curated if total_curated > 0 else 0.0
    mean_words = total_words / total_curated if total_curated > 0 else 0.0
    mean_score = total_scores / total_curated if total_curated > 0 else 0.0

    stats = CuratedDatasetStats(
        total_curated_traces=total_curated,
        unique_problems_covered=unique_problems,
        target_problems=target_count,
        problem_coverage_percent=cov_pct,
        synthetic_trace_count=synthetic_count,
        gold_fallback_count=gold_count,
        by_prompt_type=by_ptype,
        traces_per_problem_dist=traces_per_prob_dist,
        near_duplicates_pruned=near_duplicates_pruned,
        overlength_pruned=overlength_pruned,
        looping_pruned=looping_pruned,
        mean_token_count=mean_tokens,
        mean_word_count=mean_words,
        mean_judge_score=mean_score,
        contamination_check_passed=True,
    )

    return curated_traces, stats


def save_curated_dataset(
    traces: Sequence[Union[Dict[str, Any], CuratedTrace]],
    output_path: Union[str, Path],
) -> Path:
    """Save curated reasoning dataset to JSONL format."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for t in traces:
            d = t.to_dict() if isinstance(t, CuratedTrace) else dict(t)
            f.write(json.dumps(d) + "\n")
    logger.info(f"Saved {len(traces)} curated traces to: {p}")
    return p


def load_curated_dataset(
    file_path: Union[str, Path],
) -> List[Dict[str, Any]]:
    """Load curated reasoning dataset from JSONL format."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Curated dataset file not found: {p}")
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

