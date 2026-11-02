"""Reasoning prompt templates and builders for synthetic data generation (Phase 3).

Provides three prompt strategies designed to elicit high-quality, diverse reasoning traces:
1. Chain-of-Thought (CoT): standard step-by-step mathematical reasoning.
2. Solve-then-Verify: explicit two-stage generation (initial solution followed by verification/sanity check).
3. Alternative-Method: complementary or alternative mathematical strategy (e.g. algebraic equations,
   working backwards, or proportional reasoning instead of straightforward arithmetic).

Every prompt strategy strictly enforces the canonical GSM8K answer contract:
the completion must end with the final numeric answer formatted as '#### <number>'.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from src.data.formatting import apply_chat_template

logger = logging.getLogger(__name__)


class ReasoningPromptType(str, Enum):
    """Supported reasoning prompt strategies for synthetic data generation."""

    CHAIN_OF_THOUGHT = "chain_of_thought"
    SOLVE_THEN_VERIFY = "solve_then_verify"
    ALTERNATIVE_METHOD = "alternative_method"


# Canonical answer format contract required across all prompts
CANONICAL_ANSWER_INSTRUCTION = (
    "At the end of your response, provide the final numeric answer on a new line "
    "prefixed with '#### ', for example:\n#### 42\n"
    "Write only the plain number after '#### ' (no units, currency symbols, or trailing text)."
)

CANONICAL_LINE_INSTRUCTION = (
    "On the last line, write the final numeric answer as '#### <number>'."
)


@dataclass(frozen=True)
class ReasoningPromptConfig:
    """Configuration and templates for a reasoning prompt strategy."""

    prompt_type: ReasoningPromptType
    name: str
    description: str
    system_prompt: str
    user_template: str
    plain_text_template: str


# Detailed prompt configurations for each strategy
PROMPT_CONFIGS: Dict[ReasoningPromptType, ReasoningPromptConfig] = {
    ReasoningPromptType.CHAIN_OF_THOUGHT: ReasoningPromptConfig(
        prompt_type=ReasoningPromptType.CHAIN_OF_THOUGHT,
        name="Chain-of-Thought",
        description="Direct step-by-step mathematical reasoning and calculation.",
        system_prompt=(
            "You are an expert mathematical problem solver. Solve the following math word problem "
            "step by step, showing all reasoning and arithmetic clearly.\n"
            f"{CANONICAL_ANSWER_INSTRUCTION}"
        ),
        user_template=(
            "Question: {question}\n\n"
            "Please solve this math problem step by step, showing all work clearly.\n"
            f"{CANONICAL_LINE_INSTRUCTION}"
        ),
        plain_text_template=(
            "Question: {question}\n"
            "Solve the problem step by step. "
            f"{CANONICAL_LINE_INSTRUCTION}\n"
            "Answer: Let's think step by step.\n"
        ),
    ),
    ReasoningPromptType.SOLVE_THEN_VERIFY: ReasoningPromptConfig(
        prompt_type=ReasoningPromptType.SOLVE_THEN_VERIFY,
        name="Solve-then-Verify",
        description="Two-stage reasoning: initial solution followed by rigorous verification.",
        system_prompt=(
            "You are an expert mathematical problem solver. Solve the following math word problem "
            "using a rigorous two-stage approach:\n"
            "1. Initial Solution: Work through the problem step by step to find an answer.\n"
            "2. Verification: Carefully verify your reasoning and calculations (for example, by checking "
            "inverse arithmetic, substituting the answer back into the problem conditions, or checking sanity constraints).\n"
            "Write the '#### <number>' line only once, at the very end after Stage 2 verification is complete.\n"
            f"{CANONICAL_ANSWER_INSTRUCTION}"
        ),
        user_template=(
            "Question: {question}\n\n"
            "Please solve this math problem using a two-stage approach:\n"
            "1. Solve: Find the solution step by step.\n"
            "2. Verify: Check your calculations and reasoning to confirm the answer is correct.\n\n"
            "Do not write '####' until after Stage 2 verification is complete. "
            f"{CANONICAL_LINE_INSTRUCTION}"
        ),
        plain_text_template=(
            "Question: {question}\n"
            "Solve the problem in two stages: first find the solution step by step, then verify your work. "
            f"{CANONICAL_LINE_INSTRUCTION}\n"
            "Answer: Stage 1 - Initial Solution:\n"
        ),
    ),
    ReasoningPromptType.ALTERNATIVE_METHOD: ReasoningPromptConfig(
        prompt_type=ReasoningPromptType.ALTERNATIVE_METHOD,
        name="Alternative-Method",
        description="Alternative mathematical strategy (algebraic modeling, proportions, or working backwards).",
        system_prompt=(
            "You are an expert mathematical problem solver. Solve the following math word problem "
            "using an alternative or complementary problem-solving strategy (for example, setting up algebraic "
            "equations, using proportional reasoning, working backwards, or decomposition instead of straightforward "
            "step-by-step arithmetic). Clearly explain your alternative strategy and each step.\n"
            f"{CANONICAL_ANSWER_INSTRUCTION}"
        ),
        user_template=(
            "Question: {question}\n\n"
            "Please solve this math problem using an alternative or complementary method (e.g., algebraic modeling, "
            "proportional reasoning, or working backwards instead of standard arithmetic). Explain each step clearly.\n\n"
            f"{CANONICAL_LINE_INSTRUCTION}"
        ),
        plain_text_template=(
            "Question: {question}\n"
            "Solve the problem using an alternative method (e.g. algebraic equations or working backwards). "
            f"{CANONICAL_LINE_INSTRUCTION}\n"
            "Answer: Alternative Method:\n"
        ),
    ),
}

# Aliases mapping common shorthands to canonical prompt types
PROMPT_ALIASES: Dict[str, ReasoningPromptType] = {
    "cot": ReasoningPromptType.CHAIN_OF_THOUGHT,
    "chain_of_thought": ReasoningPromptType.CHAIN_OF_THOUGHT,
    "chain-of-thought": ReasoningPromptType.CHAIN_OF_THOUGHT,
    "verify": ReasoningPromptType.SOLVE_THEN_VERIFY,
    "solve_then_verify": ReasoningPromptType.SOLVE_THEN_VERIFY,
    "solve-then-verify": ReasoningPromptType.SOLVE_THEN_VERIFY,
    "alt": ReasoningPromptType.ALTERNATIVE_METHOD,
    "alternative": ReasoningPromptType.ALTERNATIVE_METHOD,
    "alternative_method": ReasoningPromptType.ALTERNATIVE_METHOD,
    "alternative-method": ReasoningPromptType.ALTERNATIVE_METHOD,
}


def normalize_prompt_type(prompt_type: Union[str, ReasoningPromptType]) -> ReasoningPromptType:
    """Normalize a prompt type string or enum to its canonical ReasoningPromptType."""
    if isinstance(prompt_type, ReasoningPromptType):
        return prompt_type

    clean = str(prompt_type).strip().lower()
    if clean in PROMPT_ALIASES:
        return PROMPT_ALIASES[clean]

    supported = ", ".join(f"{t.value!r}" for t in ReasoningPromptType)
    raise ValueError(
        f"Unknown reasoning prompt type {prompt_type!r}. Supported types: {supported}"
    )


def get_reasoning_prompt_config(
    prompt_type: Union[str, ReasoningPromptType] = ReasoningPromptType.CHAIN_OF_THOUGHT,
) -> ReasoningPromptConfig:
    """Retrieve configuration and templates for a given reasoning prompt type."""
    canonical_type = normalize_prompt_type(prompt_type)
    return PROMPT_CONFIGS[canonical_type]


def get_supported_prompt_types() -> List[str]:
    """Return a list of supported canonical prompt type strings."""
    return [t.value for t in ReasoningPromptType]


def build_reasoning_prompt_messages(
    question: str,
    prompt_type: Union[str, ReasoningPromptType] = ReasoningPromptType.CHAIN_OF_THOUGHT,
) -> List[Dict[str, str]]:
    """Build OpenAI-compatible chat messages for an instruct teacher model."""
    config = get_reasoning_prompt_config(prompt_type)
    user_content = config.user_template.format(question=question.strip())
    return [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": user_content},
    ]


def format_reasoning_prompt_text(
    question: str,
    prompt_type: Union[str, ReasoningPromptType] = ReasoningPromptType.CHAIN_OF_THOUGHT,
) -> str:
    """Format prompt as plain text for base or non-chat models."""
    config = get_reasoning_prompt_config(prompt_type)
    return config.plain_text_template.format(question=question.strip())


def render_reasoning_prompt(
    question: str,
    prompt_type: Union[str, ReasoningPromptType] = ReasoningPromptType.CHAIN_OF_THOUGHT,
    tokenizer: Optional[Any] = None,
) -> str:
    """Render a reasoning prompt for model input.

    If a tokenizer with a chat template is provided, renders the prompt via the model's
    own chat template (stripping redundant BOS if tokenizer prepends one).
    Otherwise, returns the canonical plain-text prompt format.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = build_reasoning_prompt_messages(question=question, prompt_type=prompt_type)
        return apply_chat_template(messages=messages, tokenizer=tokenizer)

    return format_reasoning_prompt_text(question=question, prompt_type=prompt_type)
