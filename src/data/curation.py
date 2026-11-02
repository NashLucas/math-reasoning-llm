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
