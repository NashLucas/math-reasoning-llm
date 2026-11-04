"""GRPO Reward Functions and Group Advantage Normalization (Phase 6 Day 40).

Implements mathematical reasoning reward functions for Group Relative Policy Optimization:
1. accuracy_reward: Binary verification (1.0 / 0.0) against gold numerical answer.
2. format_reward: Canonical Answer Contract verification (\n#### <number>).
3. reasoning_structure_reward: Derivation presence verification (guards against guess hacking).
4. length_penalty: Repetition detection and token budget overrun penalty.
5. compute_group_relative_advantages: Normalizes rewards within prompt candidate groups.
6. CompositeGRPOReward: Orchestrates composite reward scoring and breakdown tracking.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

from src.math_utils.answer_extraction import extract_and_normalize
from src.math_utils.normalization import is_equivalent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_REWARD_WEIGHTS = {
    "accuracy": 1.0,
    "format": 0.5,
    "reasoning_structure": 0.2,
    "length_penalty": 0.1,
}

DEFAULT_REWARD_PARAMS = {
    "group_size": 4,
    "epsilon": 1e-8,
    "max_completion_length": 512,
    "min_reasoning_tokens": 5,
    "strict_format_pattern": r"(?:\n|^)####\s*(-?[0-9]+(?:\.[0-9]+)?)\s*$",
}

DEFAULT_REPETITION_PARAMS = {
    "enabled": True,
    "ngram_size": 8,
    "max_ngram_repetition": 3,
    "penalty_scale": 0.2,
}


@dataclass
class RewardBreakdown:
    """Detailed score breakdown for a single completion."""

    problem_id: str
    completion: str
    gold_answer: str
    extracted_answer: Optional[str]
    accuracy_reward: float
    format_reward: float
    reasoning_structure_reward: float
    length_penalty: float
    total_reward: float
    group_advantage: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_reward_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load reward configuration from YAML or return default configuration."""
    target_path = Path(config_path) if config_path else (PROJECT_ROOT / "configs" / "reward.yaml")
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            weights = data.get("weights", DEFAULT_REWARD_WEIGHTS)
            params = data.get("parameters", DEFAULT_REWARD_PARAMS)
            rep = data.get("repetition_penalty", DEFAULT_REPETITION_PARAMS)
            return {
                "weights": {**DEFAULT_REWARD_WEIGHTS, **weights},
                "parameters": {**DEFAULT_REWARD_PARAMS, **params},
                "repetition_penalty": {**DEFAULT_REPETITION_PARAMS, **rep},
            }
        except Exception as e:
            logger.warning("Could not read %s, using defaults: %s", target_path, e)

    return {
        "weights": dict(DEFAULT_REWARD_WEIGHTS),
        "parameters": dict(DEFAULT_REWARD_PARAMS),
        "repetition_penalty": dict(DEFAULT_REPETITION_PARAMS),
    }


def accuracy_reward(
    completions: Sequence[str],
    gold_answers: Sequence[str],
    **kwargs: Any,
) -> List[float]:
    """Binary verification (1.0 / 0.0) against ground-truth mathematical answer."""
    scores: List[float] = []
    for comp, gold in zip(completions, gold_answers):
        if not comp or not gold:
            scores.append(0.0)
            continue

        norm_str, float_val, _tier = extract_and_normalize(comp)
        if norm_str is not None and is_equivalent(norm_str, gold):
            scores.append(1.0)
        elif float_val is not None and is_equivalent(float_val, gold):
            scores.append(1.0)
        else:
            scores.append(0.0)

    return scores


def format_reward(
    completions: Sequence[str],
    pattern: Optional[str] = None,
    **kwargs: Any,
) -> List[float]:
    """Verify Canonical Answer Contract adherence (1.0 if strictly ends in #### <number>, else 0.0)."""
    pat = pattern or DEFAULT_REWARD_PARAMS["strict_format_pattern"]
    regex = re.compile(pat, re.MULTILINE)

    scores: List[float] = []
    for comp in completions:
        if not comp:
            scores.append(0.0)
            continue

        clean_comp = comp.strip()
        if regex.search(clean_comp):
            scores.append(1.0)
        else:
            scores.append(0.0)

    return scores


def reasoning_structure_reward(
    completions: Sequence[str],
    min_tokens: int = 5,
    **kwargs: Any,
) -> List[float]:
    """Verify presence of step-by-step derivation preceding answer to prevent guess-hacking."""
    scores: List[float] = []
    for comp in completions:
        if not comp:
            scores.append(0.0)
            continue

        # Extract text preceding '####'
        if "####" in comp:
            derivation = comp.split("####")[0].strip()
        else:
            derivation = comp.strip()

        words = derivation.split()
        if len(words) < min_tokens:
            scores.append(0.0)
            continue

        # Require reasoning indicators or mathematical operations
        has_math_or_step = bool(
            re.search(r"step\s*\d+|because|therefore|since|so|\+|\-|\*|\/|\=|\d+", derivation, re.IGNORECASE)
        )
        scores.append(1.0 if has_math_or_step else 0.0)

    return scores


def length_penalty(
    completions: Sequence[str],
    max_length: int = 512,
    ngram_size: int = 8,
    max_ngram_repetition: int = 3,
    penalty_scale: float = 0.2,
    **kwargs: Any,
) -> List[float]:
    """Compute penalty for repetitive looping or excessive verbosity."""
    penalties: List[float] = []
    for comp in completions:
        if not comp:
            penalties.append(0.0)
            continue

        penalty = 0.0
        words = comp.split()

        # Length budget check (rough word count ceiling)
        if len(words) > (max_length * 0.85):
            overrun = len(words) - int(max_length * 0.85)
            penalty += min(0.5, (overrun / 100.0) * penalty_scale)

        # Degenerate repetition check
        if len(words) >= ngram_size:
            ngrams = [tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
            counts = Counter(ngrams)
            for _ng, cnt in counts.items():
                if cnt > max_ngram_repetition:
                    excess = cnt - max_ngram_repetition
                    penalty += excess * penalty_scale

        penalties.append(min(1.0, penalty))

    return penalties


def compute_group_relative_advantages(
    rewards: Sequence[float],
    group_size: int = 4,
    epsilon: float = 1e-8,
) -> List[float]:
    """Compute group-relative normalized advantages per DeepSeek GRPO formulation.

    A_i = (r_i - mean({r_g})) / (std({r_g}) + epsilon).
    If std({r_g}) <= epsilon (zero variance in group, e.g. all 0s or all 1s), A_i = 0.0.
    """
    if not rewards:
        return []

    if group_size <= 1:
        return [0.0 for _ in rewards]

    total_items = len(rewards)
    advantages: List[float] = [0.0] * total_items

    for start_idx in range(0, total_items, group_size):
        end_idx = min(start_idx + group_size, total_items)
        group = rewards[start_idx:end_idx]
        g_len = len(group)

        if g_len <= 1:
            continue

        mean_val = sum(group) / g_len
        var_val = sum((x - mean_val) ** 2 for x in group) / g_len
        std_val = math.sqrt(var_val)

        if std_val <= epsilon:
            # All rewards in this group are identical: no gradient signal
            for idx in range(start_idx, end_idx):
                advantages[idx] = 0.0
        else:
            for idx, r_val in enumerate(group, start=start_idx):
                advantages[idx] = (r_val - mean_val) / (std_val + epsilon)

    return advantages


class CompositeGRPOReward:
    """Orchestrates composite multi-reward calculation and group advantage estimation."""

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str, Path]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        if isinstance(config, (str, Path)) or config is None:
            self.cfg = load_reward_config(config)
        else:
            self.cfg = config

        # Custom weights override
        if weights is not None:
            self.weights = {**self.cfg["weights"], **weights}
        else:
            self.weights = self.cfg["weights"]

        self.params = self.cfg["parameters"]
        self.rep_params = self.cfg.get("repetition_penalty", DEFAULT_REPETITION_PARAMS)

    def score_single(
        self,
        completion: str,
        gold_answer: str,
        problem_id: str = "",
    ) -> RewardBreakdown:
        """Score a single completion and return a full reward breakdown."""
        acc_list = accuracy_reward([completion], [gold_answer])
        fmt_list = format_reward([completion], pattern=self.params.get("strict_format_pattern"))
        struct_list = reasoning_structure_reward(
            [completion], min_tokens=self.params.get("min_reasoning_tokens", 8)
        )
        pen_list = length_penalty(
            [completion],
            max_length=self.params.get("max_completion_length", 512),
            ngram_size=self.rep_params.get("ngram_size", 8),
            max_ngram_repetition=self.rep_params.get("max_ngram_repetition", 3),
            penalty_scale=self.rep_params.get("penalty_scale", 0.2),
        )

        acc = acc_list[0]
        fmt = fmt_list[0]
        struct = struct_list[0]
        pen = pen_list[0]

        total = (
            self.weights.get("accuracy", 1.0) * acc
            + self.weights.get("format", 0.5) * fmt
            + self.weights.get("reasoning_structure", 0.2) * struct
            - self.weights.get("length_penalty", 0.1) * pen
        )

        norm_str, _val, _tier = extract_and_normalize(completion)

        return RewardBreakdown(
            problem_id=problem_id,
            completion=completion,
            gold_answer=gold_answer,
            extracted_answer=norm_str,
            accuracy_reward=acc,
            format_reward=fmt,
            reasoning_structure_reward=struct,
            length_penalty=pen,
            total_reward=total,
            group_advantage=None,
        )

    def score_batch(
        self,
        completions: Sequence[str],
        gold_answers: Sequence[str],
        problem_ids: Optional[Sequence[str]] = None,
        compute_advantages: bool = True,
        group_size: Optional[int] = None,
    ) -> List[RewardBreakdown]:
        """Score a batch of completions and compute group-relative advantages."""
        pids = problem_ids or [f"p_{i}" for i in range(len(completions))]
        breakdowns: List[RewardBreakdown] = []

        acc_scores = accuracy_reward(completions, gold_answers)
        fmt_scores = format_reward(completions, pattern=self.params.get("strict_format_pattern"))
        struct_scores = reasoning_structure_reward(
            completions, min_tokens=self.params.get("min_reasoning_tokens", 8)
        )
        pen_scores = length_penalty(
            completions,
            max_length=self.params.get("max_completion_length", 512),
            ngram_size=self.rep_params.get("ngram_size", 8),
            max_ngram_repetition=self.rep_params.get("max_ngram_repetition", 3),
            penalty_scale=self.rep_params.get("penalty_scale", 0.2),
        )

        total_rewards: List[float] = []
        for i, (comp, gold, pid) in enumerate(zip(completions, gold_answers, pids)):
            acc = acc_scores[i]
            fmt = fmt_scores[i]
            struct = struct_scores[i]
            pen = pen_scores[i]

            tot = (
                self.weights.get("accuracy", 1.0) * acc
                + self.weights.get("format", 0.5) * fmt
                + self.weights.get("reasoning_structure", 0.2) * struct
                - self.weights.get("length_penalty", 0.1) * pen
            )
            total_rewards.append(tot)

            norm_str, _val, _tier = extract_and_normalize(comp)
            breakdowns.append(
                RewardBreakdown(
                    problem_id=pid,
                    completion=comp,
                    gold_answer=gold,
                    extracted_answer=norm_str,
                    accuracy_reward=acc,
                    format_reward=fmt,
                    reasoning_structure_reward=struct,
                    length_penalty=pen,
                    total_reward=tot,
                    group_advantage=None,
                )
            )

        if compute_advantages:
            g_size = group_size or self.params.get("group_size", 4)
            eps = self.params.get("epsilon", 1e-8)
            advs = compute_group_relative_advantages(total_rewards, group_size=g_size, epsilon=eps)
            for i, adv in enumerate(advs):
                breakdowns[i].group_advantage = adv

        return breakdowns

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        answer: Optional[Sequence[str]] = None,
        gold_answer: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """TRL GRPOTrainer-compatible callable returning scalar rewards."""
        targets = gold_answer if gold_answer is not None else (answer or [""] * len(completions))
        breakdowns = self.score_batch(completions, targets, compute_advantages=False)
        return [b.total_reward for b in breakdowns]


# TRL-compatible standalone function adapters
def trl_accuracy_reward_fn(
    prompts: Sequence[str],
    completions: Sequence[str],
    answer: Optional[Sequence[str]] = None,
    gold_answer: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """TRL GRPOTrainer-compatible reward function for answer accuracy."""
    targets = gold_answer if gold_answer is not None else (answer or [""] * len(completions))
    return accuracy_reward(completions, targets)


def trl_format_reward_fn(
    prompts: Sequence[str],
    completions: Sequence[str],
    **kwargs: Any,
) -> List[float]:
    """TRL GRPOTrainer-compatible reward function for canonical format compliance."""
    return format_reward(completions)


def trl_reasoning_structure_reward_fn(
    prompts: Sequence[str],
    completions: Sequence[str],
    **kwargs: Any,
) -> List[float]:
    """TRL GRPOTrainer-compatible reward function for derivation step presence."""
    return reasoning_structure_reward(completions)
