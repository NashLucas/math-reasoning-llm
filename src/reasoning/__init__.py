"""Reasoning module: prompt templates and self-consistency majority voting."""

from __future__ import annotations

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
    "PROMPT_CONFIGS",
    "ReasoningPromptConfig",
    "ReasoningPromptType",
    "build_reasoning_prompt_messages",
    "format_reasoning_prompt_text",
    "get_reasoning_prompt_config",
    "get_supported_prompt_types",
    "is_valid_candidate",
    "majority_vote",
    "majority_vote_from_completions",
    "majority_vote_with_details",
    "normalize_prompt_type",
    "render_reasoning_prompt",
]
