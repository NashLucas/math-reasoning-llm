"""Data processing, formatting, synthetic generation, and curation modules."""

from __future__ import annotations

from src.data.curation import (
    CorrectnessFilter,
    CurationFilterStats,
    compute_problem_pass_rates,
    detect_repeated_ngrams,
    evaluate_trace_correctness,
)
from src.data.formatting import (
    ANSWER_FORMAT_INSTRUCTION,
    FEW_SHOT_EXEMPLARS,
    SYSTEM_PROMPT,
    apply_chat_template,
    format_chat_messages,
    format_few_shot_prompt,
    format_zero_shot_prompt,
)
from src.data.synthetic import (
    ReasoningTrace,
    TraceSampler,
    build_trace_id,
    compute_trace_yield,
    load_existing_traces_by_id,
    load_synthetic_problems,
    load_synthetic_subset_ids,
    load_synthetic_traces,
)

__all__ = [
    "ANSWER_FORMAT_INSTRUCTION",
    "CorrectnessFilter",
    "CurationFilterStats",
    "FEW_SHOT_EXEMPLARS",
    "ReasoningTrace",
    "SYSTEM_PROMPT",
    "TraceSampler",
    "apply_chat_template",
    "build_trace_id",
    "compute_problem_pass_rates",
    "compute_trace_yield",
    "detect_repeated_ngrams",
    "evaluate_trace_correctness",
    "format_chat_messages",
    "format_few_shot_prompt",
    "format_zero_shot_prompt",
    "load_existing_traces_by_id",
    "load_synthetic_problems",
    "load_synthetic_subset_ids",
    "load_synthetic_traces",
]
