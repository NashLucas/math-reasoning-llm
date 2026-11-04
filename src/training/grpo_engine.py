"""GRPO Rollout Generation & Policy Update Engine (Phase 6 Day 43).

Implements candidate group rollout sampling (G=4), clipped surrogate policy gradient loss,
group relative advantage weighting, per-token KL divergence penalties (Schulman k3),
and end-to-end GRPO training step execution for mathematical reasoning post-training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.rewards.grpo_rewards import (
    CompositeGRPOReward,
    RewardBreakdown,
    compute_group_relative_advantages,
    load_reward_config,
)
from src.rewards.reference_model import (
    KLEstimatorType,
    ReferenceModelManager,
    compute_per_token_kl,
    load_kl_config,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class GRPORolloutGroup:
    """A group of G candidate rollouts for a single prompt."""

    prompt_id: str
    prompt_text: str
    gold_answer: str
    completions: List[str]
    input_ids: np.ndarray          # [G, seq_len]
    attention_mask: np.ndarray     # [G, seq_len]
    completion_mask: np.ndarray    # [G, seq_len]
    old_logprobs: np.ndarray       # [G, seq_len]
    rewards: List[float]           # Raw composite rewards [G]
    breakdowns: List[RewardBreakdown]
    advantages: List[float]        # Group-relative advantages [G]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "gold_answer": self.gold_answer,
            "completions": self.completions,
            "rewards": self.rewards,
            "advantages": self.advantages,
            "breakdowns": [b.to_dict() for b in self.breakdowns],
            "num_generations": len(self.completions),
        }


@dataclass
class GRPOLossMetrics:
    """Diagnostic metrics from a GRPO policy gradient loss computation."""

    total_loss: float
    policy_loss: float
    kl_penalty: float
    mean_kl: float
    mean_advantage: float
    std_advantage: float
    clip_fraction: float
    mean_ratio: float
    mean_reward: float
    num_sequences: int
    num_valid_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_grpo_loss(
    current_logprobs: np.ndarray,
    old_logprobs: np.ndarray,
    advantages: Union[np.ndarray, List[float]],
    completion_mask: np.ndarray,
    clip_range: float = 0.2,
    ref_logprobs: Optional[np.ndarray] = None,
    beta: float = 0.04,
    kl_estimator: Union[str, KLEstimatorType] = KLEstimatorType.K3,
) -> Tuple[float, GRPOLossMetrics]:
    """Compute Group Relative Policy Optimization (GRPO) clipped surrogate loss.

    Formula (DeepSeekMath / GRPO):
      L_PG(theta) = - (1 / G) * sum_{i=1}^G (1 / |o_i|) * sum_{t in o_i} min(r_{i,t} A_i, clip(r_{i,t}, 1-eps, 1+eps) A_i)
      L_KL(theta) = beta * (1 / G) * sum_{i=1}^G (1 / |o_i|) * sum_{t in o_i} D_KL(o_{i,t})
      L_GRPO(theta) = L_PG(theta) + L_KL(theta)

    Args:
        current_logprobs: Log-probabilities log pi_theta(o_{i,t}) of shape [G, seq_len].
        old_logprobs: Rollout log-probabilities log pi_old(o_{i,t}) of shape [G, seq_len].
        advantages: Group-relative advantages A_i of length G.
        completion_mask: Binary mask [G, seq_len], 1 for completion tokens, 0 elsewhere.
        clip_range: PPO/GRPO clipping epsilon (default: 0.2).
        ref_logprobs: Optional reference log-probabilities log pi_ref(o_{i,t}) of shape [G, seq_len].
        beta: KL penalty coefficient (default: 0.04).
        kl_estimator: Estimator for D_KL ('k1', 'k2', 'k3').

    Returns:
        Tuple of (scalar total_loss, GRPOLossMetrics).
    """
    cur_lp = np.asarray(current_logprobs, dtype=np.float64)
    old_lp = np.asarray(old_logprobs, dtype=np.float64)
    c_mask = np.asarray(completion_mask, dtype=np.float64)
    advs = np.asarray(advantages, dtype=np.float64)

    group_size, seq_len = cur_lp.shape
    if advs.shape[0] != group_size:
        raise ValueError(f"Advantage count ({advs.shape[0]}) does not match group size ({group_size})")

    # Ratio r_{i,t} = exp(cur_lp - old_lp)
    log_ratio = cur_lp - old_lp
    # Clamp log_ratio to [-20, 20] to avoid numerical overflow
    log_ratio = np.clip(log_ratio, -20.0, 20.0)
    ratio = np.exp(log_ratio)

    # Broadcast advantages across sequence length: [G, 1] -> [G, seq_len]
    adv_broadcast = advs[:, np.newaxis]

    # Clipped ratio and surrogates
    clipped_ratio = np.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
    surr1 = ratio * adv_broadcast
    surr2 = clipped_ratio * adv_broadcast

    # Conservative surrogate objective: min(surr1, surr2)
    surrogate_obj = np.minimum(surr1, surr2)

    # Apply completion mask
    masked_surr = surrogate_obj * c_mask

    # Per-sequence completion token counts |o_i|
    seq_lengths = np.sum(c_mask, axis=1)  # [G]
    seq_lengths_safe = np.maximum(seq_lengths, 1.0)

    # Normalize by sequence length per completion, then average over group
    per_seq_pg = np.sum(masked_surr, axis=1) / seq_lengths_safe
    policy_loss = -float(np.mean(per_seq_pg))

    # Diagnostics
    valid_mask = c_mask > 0
    total_valid = int(np.sum(c_mask))
    if total_valid > 0:
        clipped_tokens = np.abs(ratio[valid_mask] - 1.0) > clip_range
        clip_frac = float(np.mean(clipped_tokens))
        mean_ratio = float(np.mean(ratio[valid_mask]))
    else:
        clip_frac = 0.0
        mean_ratio = 1.0

    # KL Penalty computation
    kl_penalty = 0.0
    mean_token_kl = 0.0
    if ref_logprobs is not None:
        ref_lp = np.asarray(ref_logprobs, dtype=np.float64)
        per_token_kl = compute_per_token_kl(cur_lp, ref_lp, estimator=kl_estimator)
        masked_kl = per_token_kl * c_mask

        # Per-sequence KL average
        per_seq_kl = np.sum(masked_kl, axis=1) / seq_lengths_safe
        mean_seq_kl = float(np.mean(per_seq_kl))
        kl_penalty = beta * mean_seq_kl
        if total_valid > 0:
            mean_token_kl = float(np.sum(masked_kl) / total_valid)

    total_loss = policy_loss + kl_penalty

    metrics = GRPOLossMetrics(
        total_loss=total_loss,
        policy_loss=policy_loss,
        kl_penalty=kl_penalty,
        mean_kl=mean_token_kl,
        mean_advantage=float(np.mean(advs)),
        std_advantage=float(np.std(advs)),
        clip_fraction=clip_frac,
        mean_ratio=mean_ratio,
        mean_reward=0.0,
        num_sequences=group_size,
        num_valid_tokens=total_valid,
    )

    return total_loss, metrics


def compute_grpo_loss_torch(
    current_logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    completion_mask: Any,
    clip_range: float = 0.2,
    ref_logprobs: Optional[Any] = None,
    beta: float = 0.04,
) -> Tuple[Any, GRPOLossMetrics]:
    """PyTorch tensor variant of compute_grpo_loss for gradient backpropagation."""
    import torch

    cur_lp = current_logprobs
    old_lp = old_logprobs
    c_mask = completion_mask
    advs = advantages

    group_size = cur_lp.shape[0]

    log_ratio = cur_lp - old_lp
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
    ratio = torch.exp(log_ratio)

    adv_broadcast = advs.view(group_size, 1)

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    surr1 = ratio * adv_broadcast
    surr2 = clipped_ratio * adv_broadcast

    surrogate_obj = torch.min(surr1, surr2)
    masked_surr = surrogate_obj * c_mask

    seq_lengths = torch.sum(c_mask, dim=1)
    seq_lengths_safe = torch.clamp(seq_lengths, min=1.0)

    per_seq_pg = torch.sum(masked_surr, dim=1) / seq_lengths_safe
    policy_loss = -torch.mean(per_seq_pg)

    # KL Penalty (Schulman k3: exp(-delta) + delta - 1)
    kl_penalty = torch.tensor(0.0, device=cur_lp.device, dtype=cur_lp.dtype)
    mean_token_kl = 0.0

    if ref_logprobs is not None:
        delta = cur_lp - ref_logprobs
        delta_clipped = torch.clamp(delta, -20.0, 20.0)
        # Schulman k3: exp(-delta) + delta - 1
        per_token_kl = torch.exp(-delta_clipped) + delta_clipped - 1.0
        per_token_kl = torch.clamp(per_token_kl, min=0.0)
        masked_kl = per_token_kl * c_mask

        per_seq_kl = torch.sum(masked_kl, dim=1) / seq_lengths_safe
        mean_seq_kl = torch.mean(per_seq_kl)
        kl_penalty = beta * mean_seq_kl

        valid_count = torch.sum(c_mask).item()
        if valid_count > 0:
            mean_token_kl = torch.sum(masked_kl).item() / valid_count

    total_loss = policy_loss + kl_penalty

    # Compute diagnostics in non-grad mode
    with torch.no_grad():
        valid_mask = c_mask > 0
        total_valid = int(torch.sum(c_mask).item())
        if total_valid > 0:
            clipped = torch.abs(ratio[valid_mask] - 1.0) > clip_range
            clip_frac = float(torch.mean(clipped.float()).item())
            mean_ratio = float(torch.mean(ratio[valid_mask]).item())
        else:
            clip_frac = 0.0
            mean_ratio = 1.0

        adv_np = advs.detach().cpu().numpy()

    metrics = GRPOLossMetrics(
        total_loss=float(total_loss.item()),
        policy_loss=float(policy_loss.item()),
        kl_penalty=float(kl_penalty.item()),
        mean_kl=mean_token_kl,
        mean_advantage=float(np.mean(adv_np)),
        std_advantage=float(np.std(adv_np)),
        clip_fraction=clip_frac,
        mean_ratio=mean_ratio,
        mean_reward=0.0,
        num_sequences=group_size,
        num_valid_tokens=total_valid,
    )

    return total_loss, metrics


class GRPORolloutSampler:
    """Samples G candidate completions per prompt, scores rewards, and computes advantages."""

    def __init__(
        self,
        reward_calculator: Optional[CompositeGRPOReward] = None,
        group_size: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_completion_length: int = 512,
        mock_mode: bool = True,
    ):
        self.reward_calculator = reward_calculator or CompositeGRPOReward()
        self.group_size = group_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_length = max_completion_length
        self.mock_mode = mock_mode

    def sample_rollout_group(
        self,
        prompt_id: str,
        prompt_text: str,
        gold_answer: str,
        mock_candidates: Optional[List[str]] = None,
    ) -> GRPORolloutGroup:
        """Sample G completions for a prompt, score rewards, and compute group advantages."""
        if mock_candidates is not None and len(mock_candidates) >= self.group_size:
            completions = mock_candidates[: self.group_size]
        else:
            completions = self._generate_mock_completions(gold_answer)

        # 1. Score composite rewards for each completion
        rewards: List[float] = []
        breakdowns: List[RewardBreakdown] = []

        for comp in completions:
            breakdown = self.reward_calculator.score_single(
                completion=comp,
                gold_answer=gold_answer,
                problem_id=prompt_id,
            )
            rewards.append(breakdown.total_reward)
            breakdowns.append(breakdown)

        # 2. Compute group-relative normalized advantages
        eps = 1e-8
        if hasattr(self.reward_calculator, "params"):
            eps = float(self.reward_calculator.params.get("epsilon", 1e-8))

        advantages = compute_group_relative_advantages(
            rewards=rewards,
            group_size=self.group_size,
            epsilon=eps,
        )

        # 3. Build token arrays and completion masks
        seq_len = 16  # mock sequence length
        prompt_len = 6

        input_ids = np.ones((self.group_size, seq_len), dtype=np.int64) * 100
        attention_mask = np.ones((self.group_size, seq_len), dtype=np.int64)

        completion_mask = np.zeros((self.group_size, seq_len), dtype=np.float64)
        completion_mask[:, prompt_len:] = 1.0

        # Deterministic mock logprobs
        rng = np.random.default_rng(seed=42)
        old_logprobs = -0.5 - 1.0 * rng.random((self.group_size, seq_len))
        old_logprobs = old_logprobs * completion_mask

        return GRPORolloutGroup(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            gold_answer=gold_answer,
            completions=completions,
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            old_logprobs=old_logprobs,
            rewards=rewards,
            breakdowns=breakdowns,
            advantages=advantages,
        )

    def _generate_mock_completions(self, gold_answer: str) -> List[str]:
        """Generate diverse mock candidates exhibiting distinct reward profiles."""
        c1 = f"Let's think step by step.\nStep 1: 10 + 5 = 15.\nStep 2: 15 * 2 = 30.\n#### {gold_answer}"
        c2 = f"Step 1: First, compute 50 - 20 = 30.\n#### {gold_answer}"
        c3 = "Step 1: Incorrect calculation yields 999.\n#### 999"
        c4 = f"#### {gold_answer}"  # Guess-hacking shortcut

        candidates = [c1, c2, c3, c4]
        # Adjust to group size if G != 4
        while len(candidates) < self.group_size:
            candidates.append(f"Step 1: Extra reasoning.\n#### {gold_answer}")
        return candidates[: self.group_size]


class GRPOTrainerEngine:
    """Orchestrates GRPO policy rollout collection, advantage normalization, and policy update."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        reward_calculator: Optional[CompositeGRPOReward] = None,
        reference_manager: Optional[ReferenceModelManager] = None,
        mock_mode: bool = True,
    ):
        self.config = config or load_grpo_engine_config()
        self.reward_calculator = reward_calculator or CompositeGRPOReward()
        self.reference_manager = reference_manager or ReferenceModelManager(mock_mode=True)
        self.mock_mode = mock_mode

        # Hyperparameters
        grpo_cfg = self.config.get("grpo", {})
        self.group_size = int(grpo_cfg.get("num_generations", 4))
        self.clip_range = float(grpo_cfg.get("clip_range", 0.2))
        self.beta = float(grpo_cfg.get("beta", 0.04))
        self.learning_rate = float(grpo_cfg.get("learning_rate", 1e-6))
        self.grad_accum_steps = int(grpo_cfg.get("gradient_accumulation_steps", 4))

        self.sampler = GRPORolloutSampler(
            reward_calculator=self.reward_calculator,
            group_size=self.group_size,
            temperature=float(grpo_cfg.get("temperature", 0.7)),
            top_p=float(grpo_cfg.get("top_p", 0.95)),
            max_completion_length=int(grpo_cfg.get("max_completion_length", 512)),
            mock_mode=self.mock_mode,
        )

        self.step_count: int = 0
        self.history: List[Dict[str, Any]] = []

    def train_step(
        self,
        prompt_id: str,
        prompt_text: str,
        gold_answer: str,
        mock_candidates: Optional[List[str]] = None,
    ) -> GRPOLossMetrics:
        """Execute a single full GRPO training step on one prompt candidate group."""
        self.step_count += 1

        # 1. Collect rollout group with rewards and advantages
        group = self.sampler.sample_rollout_group(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            gold_answer=gold_answer,
            mock_candidates=mock_candidates,
        )

        # 2. Extract reference policy log-probabilities
        ref_logprobs = self.reference_manager.get_ref_logprobs(
            input_ids=group.input_ids,
            attention_mask=group.attention_mask,
            completion_mask=group.completion_mask,
        )

        # 3. Current policy log-probabilities (in mock mode, slight shift from old logprobs)
        rng = np.random.default_rng(seed=42 + self.step_count)
        shift = 0.05 * (rng.random(group.old_logprobs.shape) - 0.5)
        current_logprobs = (group.old_logprobs + shift) * group.completion_mask

        # 4. Compute GRPO surrogate loss and metrics
        total_loss, metrics = compute_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=group.old_logprobs,
            advantages=group.advantages,
            completion_mask=group.completion_mask,
            clip_range=self.clip_range,
            ref_logprobs=ref_logprobs,
            beta=self.beta,
            kl_estimator=KLEstimatorType.K3,
        )

        metrics.mean_reward = float(np.mean(group.rewards))

        record = {
            "step": self.step_count,
            "prompt_id": prompt_id,
            "loss": metrics.total_loss,
            "policy_loss": metrics.policy_loss,
            "kl_penalty": metrics.kl_penalty,
            "mean_reward": metrics.mean_reward,
            "mean_advantage": metrics.mean_advantage,
            "clip_fraction": metrics.clip_fraction,
        }
        self.history.append(record)

        return metrics


def load_grpo_engine_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load grpo training arguments and reward configuration."""
    grpo_args_path = PROJECT_ROOT / "configs" / "grpo_args.json"
    reward_yaml_path = PROJECT_ROOT / "configs" / "reward.yaml"
    kl_yaml_path = PROJECT_ROOT / "configs" / "kl_divergence.yaml"

    grpo_dict: Dict[str, Any] = {}
    if grpo_args_path.exists():
        with open(grpo_args_path, "r", encoding="utf-8") as f:
            grpo_dict = json.load(f)

    reward_dict = load_reward_config(reward_yaml_path)
    kl_dict = load_kl_config(kl_yaml_path)

    return {
        "grpo": grpo_dict,
        "rewards": reward_dict,
        "kl": kl_dict,
    }
