"""Rewards package for Reinforcement Learning and GRPO."""

from src.rewards.composite import (
    CompositeGRPOReward,
    RewardBreakdown,
    compute_group_relative_advantages,
    load_reward_config,
)
from src.rewards.correctness import (
    correctness_reward,
    is_close_numerical,
    score_correctness,
)
from src.rewards.format import format_reward, score_format
from src.rewards.grpo_rewards import (
    accuracy_reward,
    length_penalty,
    reasoning_structure_reward,
    trl_accuracy_reward_fn,
    trl_format_reward_fn,
    trl_reasoning_structure_reward_fn,
)
from src.rewards.quality import (
    quality_reward,
    score_length_and_repetition,
    score_reasoning_structure,
)
from src.rewards.reference_model import (
    KLController,
    KLEstimatorType,
    KLSummary,
    ReferenceModelManager,
    compute_per_token_kl,
    compute_trajectory_kl,
    load_kl_config,
)

__all__ = [
    "accuracy_reward",
    "correctness_reward",
    "score_correctness",
    "is_close_numerical",
    "format_reward",
    "score_format",
    "reasoning_structure_reward",
    "score_reasoning_structure",
    "quality_reward",
    "length_penalty",
    "score_length_and_repetition",
    "compute_group_relative_advantages",
    "RewardBreakdown",
    "CompositeGRPOReward",
    "load_reward_config",
    "trl_accuracy_reward_fn",
    "trl_format_reward_fn",
    "trl_reasoning_structure_reward_fn",
    "KLEstimatorType",
    "KLSummary",
    "compute_per_token_kl",
    "compute_trajectory_kl",
    "KLController",
    "ReferenceModelManager",
    "load_kl_config",
]
