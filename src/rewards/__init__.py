"""Rewards package for Reinforcement Learning and GRPO."""

from src.rewards.grpo_rewards import (
    CompositeGRPOReward,
    RewardBreakdown,
    accuracy_reward,
    compute_group_relative_advantages,
    format_reward,
    length_penalty,
    load_reward_config,
    reasoning_structure_reward,
    trl_accuracy_reward_fn,
    trl_format_reward_fn,
    trl_reasoning_structure_reward_fn,
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
    "format_reward",
    "reasoning_structure_reward",
    "length_penalty",
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
