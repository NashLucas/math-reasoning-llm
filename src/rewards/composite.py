"""Composite GRPO reward scoring and group advantage normalization."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

from src.rewards.correctness import score_correctness
from src.rewards.format import score_format
from src.rewards.quality import score_length_and_repetition, score_reasoning_structure

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_REWARD_WEIGHTS = {
    "accuracy": 1.0,
    "correctness": 1.0,
    "format": 0.5,
    "quality": 0.2,
    "reasoning_structure": 0.2,
    "length_penalty": 0.1,
}

DEFAULT_REWARD_PARAMS = {
    "group_size": 4,
    "epsilon": 1e-8,
    "max_completion_length": 512,
    "min_reasoning_tokens": 5,
    "strict_format_pattern": r"(?:\n|^)####\s*(-?[0-9]+(?:\.[0-9]+)?)\s*$",
    "allow_partial": False,
    "partial_credit": 0.25,
    "unparseable_penalty": 0.0,
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

    @property
    def accuracy(self) -> float:
        return self.accuracy_reward

    @property
    def format(self) -> float:
        return self.format_reward

    @property
    def reasoning_structure(self) -> float:
        return self.reasoning_structure_reward

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_reward_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load reward configuration from YAML or return default configuration."""
    target_path = Path(config_path) if config_path else (PROJECT_ROOT / "configs" / "reward.yaml")
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            weights = data.get("weights", {})
            params = data.get("parameters", {})
            rep = data.get("repetition_penalty", {})
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


def compute_group_relative_advantages(
    rewards: Sequence[float],
    group_size: int = 4,
    epsilon: float = 1e-8,
) -> List[float]:
    """Compute group-relative normalized advantages per DeepSeek GRPO formulation.

    A_i = (r_i - mean({r_g})) / (std({r_g}) + epsilon).
    If std({r_g}) <= epsilon (zero variance), A_i = 0.0.
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
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config or load_reward_config(config_path)
        self.weights = self.config["weights"]
        self.params = self.config["parameters"]
        self.repetition = self.config.get("repetition_penalty", DEFAULT_REPETITION_PARAMS)

        # Allow either 'accuracy' or 'correctness' for weight
        self.w_acc = self.weights.get("accuracy", self.weights.get("correctness", 1.0))
        self.w_fmt = self.weights.get("format", 0.5)
        # Allow either 'reasoning_structure' or 'quality' for structure weight
        self.w_struct = self.weights.get("reasoning_structure", self.weights.get("quality", 0.2))
        self.w_len = self.weights.get("length_penalty", 0.1)

    def score_single(
        self,
        completion: str,
        gold_answer: str,
        problem_id: str = "sample",
    ) -> RewardBreakdown:
        """Score an individual completion and return a full breakdown."""
        acc_score, extracted = score_correctness(
            completion=completion,
            gold_answer=gold_answer,
            allow_partial=self.params.get("allow_partial", False),
            partial_credit=self.params.get("partial_credit", 0.25),
            unparseable_penalty=self.params.get("unparseable_penalty", 0.0),
        )
        fmt_score = score_format(
            completion=completion,
            pattern=self.params.get("strict_format_pattern"),
        )
        struct_score = score_reasoning_structure(
            completion=completion,
            min_tokens=self.params.get("min_reasoning_tokens", 5),
        )

        len_pen = 0.0
        if self.repetition.get("enabled", True):
            len_pen = score_length_and_repetition(
                completion=completion,
                max_length=self.params.get("max_completion_length", 512),
                ngram_size=self.repetition.get("ngram_size", 8),
                max_ngram_repetition=self.repetition.get("max_ngram_repetition", 3),
                penalty_scale=self.repetition.get("penalty_scale", 0.2),
            )

        total = (
            (self.w_acc * acc_score)
            + (self.w_fmt * fmt_score)
            + (self.w_struct * struct_score)
            - (self.w_len * len_pen)
        )

        return RewardBreakdown(
            problem_id=problem_id,
            completion=completion,
            gold_answer=gold_answer,
            extracted_answer=extracted,
            accuracy_reward=round(acc_score, 4),
            format_reward=round(fmt_score, 4),
            reasoning_structure_reward=round(struct_score, 4),
            length_penalty=round(len_pen, 4),
            total_reward=round(total, 4),
        )

    def score(
        self,
        question: str = "",
        completion: str = "",
        gold_answer: str = "",
        problem_id: str = "sample",
    ) -> RewardBreakdown:
        """Alias for score_single supporting question and completion kwargs."""
        return self.score_single(
            completion=completion,
            gold_answer=gold_answer,
            problem_id=problem_id,
        )

    def score_batch(
        self,
        completions: Sequence[str],
        gold_answers: Sequence[str],
        problem_ids: Optional[Sequence[str]] = None,
        normalize_advantages: bool = True,
    ) -> List[RewardBreakdown]:
        """Score a list of completions, optionally computing group-relative advantages."""
        pids = problem_ids or [f"prob_{i}" for i in range(len(completions))]
        breakdowns: List[RewardBreakdown] = []

        for comp, gold, pid in zip(completions, gold_answers, pids):
            breakdowns.append(self.score_single(completion=comp, gold_answer=gold, problem_id=pid))

        if normalize_advantages and breakdowns:
            group_size = self.params.get("group_size", 4)
            eps = self.params.get("epsilon", 1e-8)
            totals = [b.total_reward for b in breakdowns]
            advantages = compute_group_relative_advantages(totals, group_size=group_size, epsilon=eps)
            for b, adv in zip(breakdowns, advantages):
                b.group_advantage = round(adv, 4)

        return breakdowns

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        gold_answer: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """TRL GRPOTrainer-compatible reward function callable.

        Accepts prompts, completions, and gold answers; returns a list of float rewards.
        """
        golds = gold_answer or kwargs.get("answer") or kwargs.get("gold") or [""] * len(completions)
        breakdowns = self.score_batch(completions=completions, gold_answers=golds, normalize_advantages=False)
        return [b.total_reward for b in breakdowns]
