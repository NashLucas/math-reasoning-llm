"""Reasoning module: prompt templates, self-consistency, and LLM judge rubric evaluation."""

from __future__ import annotations

from src.reasoning.judge import (
    DEFAULT_WEIGHTS,
    JUDGE_SYSTEM_PROMPT,
    JudgeScore,
    ReasoningJudge,
    ScoredTrace,
    assign_within_problem_ranks,
    compute_judge_calibration,
    compute_overall_score,
    compute_pearson_correlation,
    compute_score_distribution,
    format_judge_prompt,
    parse_judge_output,
)
from src.reasoning.prompting import (
    CANONICAL_ANSWER_INSTRUCTION,
    CANONICAL_LINE_INSTRUCTION,
    PROMPT_CONFIGS,
    ReasoningPromptConfig,
    ReasoningPromptType,
    build_reasoning_prompt_messages,
    format_reasoning_prompt_text,
    get_reasoning_prompt_config,
    get_supported_prompt_types,
    normalize_prompt_type,
    render_reasoning_prompt,
)
from src.reasoning.self_consistency import (
    is_valid_candidate,
    majority_vote,
    majority_vote_from_completions,
    majority_vote_with_details,
)

__all__ = [
    "CANONICAL_ANSWER_INSTRUCTION",
    "CANONICAL_LINE_INSTRUCTION",
    "DEFAULT_WEIGHTS",
    "JUDGE_SYSTEM_PROMPT",
    "JudgeScore",
    "PROMPT_CONFIGS",
    "ReasoningJudge",
    "ReasoningPromptConfig",
    "ReasoningPromptType",
    "ScoredTrace",
    "assign_within_problem_ranks",
    "build_reasoning_prompt_messages",
    "compute_judge_calibration",
    "compute_overall_score",
    "compute_pearson_correlation",
    "compute_score_distribution",
    "format_judge_prompt",
    "format_reasoning_prompt_text",
    "get_reasoning_prompt_config",
    "get_supported_prompt_types",
    "is_valid_candidate",
    "majority_vote",
    "majority_vote_from_completions",
    "majority_vote_with_details",
    "normalize_prompt_type",
    "parse_judge_output",
    "render_reasoning_prompt",
]
