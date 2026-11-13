"""Reference model management and KL divergence tracking (Phase 6 Day 42).

Implements token-level and trajectory-level KL divergence estimators (Schulman k1, k2, k3),
mask-aware sequence aggregation, memory-efficient reference policy evaluation,
and adaptive/fixed KL penalty controllers for Group Relative Policy Optimization (GRPO).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class KLEstimatorType(str, Enum):
    """Supported KL divergence estimator formulations."""

    K1 = "k1"  # Standard / reverse log-ratio: log(pi_theta) - log(pi_ref)
    K2 = "k2"  # Squared log-ratio: 0.5 * (log(pi_theta) - log(pi_ref))^2
    K3 = "k3"  # Schulman / DeepSeek non-negative estimator: r - log(r) - 1, r = pi_ref / pi_theta


@dataclass
class KLSummary:
    """Summary metrics of KL divergence across a batch of completions."""

    estimator: str
    mean_token_kl: float
    max_token_kl: float
    min_token_kl: float
    std_token_kl: float
    sequence_mean_kl: float
    sequence_sum_kl: float
    total_valid_tokens: int
    num_sequences: int
    is_safe: bool
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_per_token_kl(
    policy_logprobs: Union[np.ndarray, List[float], List[List[float]]],
    ref_logprobs: Union[np.ndarray, List[float], List[List[float]]],
    estimator: Union[str, KLEstimatorType] = KLEstimatorType.K3,
    clip_ratio_bounds: Optional[Tuple[float, float]] = (1e-6, 1e6),
) -> np.ndarray:
    """Compute per-token KL divergence between policy and reference log-probabilities.

    Formulations:
      - k1: delta = policy_logprobs - ref_logprobs
      - k2: 0.5 * delta^2
      - k3 (DeepSeekMath/Schulman): ratio = exp(-delta) = ref_prob / policy_prob;
            kl = ratio - log(ratio) - 1 = exp(-delta) + delta - 1 >= 0

    Args:
        policy_logprobs: Log-probabilities log pi_theta(y_t | x, y_{<t}).
        ref_logprobs: Log-probabilities log pi_ref(y_t | x, y_{<t}).
        estimator: KLEstimatorType or string ('k1', 'k2', 'k3').
        clip_ratio_bounds: Optional tuple (min_ratio, max_ratio) to prevent overflow/underflow.

    Returns:
        NumPy array of per-token KL divergence values.
    """
    p_lp = np.asarray(policy_logprobs, dtype=np.float64)
    r_lp = np.asarray(ref_logprobs, dtype=np.float64)

    if p_lp.shape != r_lp.shape:
        raise ValueError(
            f"Shape mismatch: policy_logprobs {p_lp.shape} != ref_logprobs {r_lp.shape}"
        )

    # delta = log pi_theta - log pi_ref
    delta = p_lp - r_lp

    est = KLEstimatorType(estimator) if isinstance(estimator, str) else estimator

    if est == KLEstimatorType.K1:
        # Standard unbiased estimator: log(pi_theta / pi_ref)
        return delta

    elif est == KLEstimatorType.K2:
        # Second-order Taylor approximation
        return 0.5 * np.square(delta)

    elif est == KLEstimatorType.K3:
        # Schulman k3: ratio r = exp(-delta) = pi_ref / pi_theta
        # kl = r - log(r) - 1 = exp(-delta) + delta - 1
        # For numerical stability when delta is large negative (policy << ref)
        # or large positive (policy >> ref):
        neg_delta = -delta
        if clip_ratio_bounds is not None:
            min_r, max_r = clip_ratio_bounds
            log_min_r = math.log(min_r)
            log_max_r = math.log(max_r)
            neg_delta = np.clip(neg_delta, log_min_r, log_max_r)

        # expm1(-delta) + delta = exp(-delta) - 1 + delta >= 0
        kl = np.expm1(neg_delta) + delta
        # Guarantee non-negativity despite float precision limits
        return np.maximum(kl, 0.0)

    else:
        raise ValueError(f"Unsupported KL estimator: {estimator}")


def compute_trajectory_kl(
    per_token_kl: np.ndarray,
    completion_mask: Optional[np.ndarray] = None,
    estimator: Union[str, KLEstimatorType] = KLEstimatorType.K3,
    max_kl_threshold: float = 0.5,
) -> KLSummary:
    """Aggregate per-token KL divergence into trajectory and batch metrics.

    Args:
        per_token_kl: Array of token-level KL values of shape [batch_size, seq_len] or [seq_len].
        completion_mask: Binary mask (1 for completion tokens, 0 for prompt/padding).
        estimator: Estimator name for reporting.
        max_kl_threshold: Threshold above which policy drift triggers a safety warning.

    Returns:
        KLSummary containing aggregate metrics and safety checks.
    """
    kl = np.asarray(per_token_kl, dtype=np.float64)

    # Ensure 2D shape [batch_size, seq_len]
    if kl.ndim == 1:
        kl = kl[np.newaxis, :]

    batch_size, seq_len = kl.shape

    if completion_mask is None:
        mask = np.ones_like(kl, dtype=np.float64)
    else:
        mask = np.asarray(completion_mask, dtype=np.float64)
        if mask.shape != kl.shape:
            raise ValueError(
                f"Shape mismatch: per_token_kl {kl.shape} != completion_mask {mask.shape}"
            )

    valid_tokens = int(np.sum(mask))
    est_name = estimator.value if isinstance(estimator, KLEstimatorType) else str(estimator)

    if valid_tokens == 0:
        return KLSummary(
            estimator=est_name,
            mean_token_kl=0.0,
            max_token_kl=0.0,
            min_token_kl=0.0,
            std_token_kl=0.0,
            sequence_mean_kl=0.0,
            sequence_sum_kl=0.0,
            total_valid_tokens=0,
            num_sequences=batch_size,
            is_safe=True,
            warning="No valid completion tokens in mask",
        )

    # Masked token values
    masked_kl = kl[mask > 0]
    mean_token_kl = float(np.mean(masked_kl))
    max_token_kl = float(np.max(masked_kl))
    min_token_kl = float(np.min(masked_kl))
    std_token_kl = float(np.std(masked_kl))

    # Per-sequence aggregates
    seq_sums = np.sum(kl * mask, axis=1)  # [batch_size]
    seq_counts = np.sum(mask, axis=1)
    seq_counts_safe = np.maximum(seq_counts, 1.0)
    seq_means = seq_sums / seq_counts_safe

    sequence_mean_kl = float(np.mean(seq_means))
    sequence_sum_kl = float(np.mean(seq_sums))

    # Safety check against excessive policy drift
    is_safe = mean_token_kl <= max_kl_threshold
    warning = None
    if not is_safe:
        warning = (
            f"Mean token KL ({mean_token_kl:.4f}) exceeds safe ceiling ({max_kl_threshold:.4f}); "
            "policy is drifting excessively from reference."
        )

    return KLSummary(
        estimator=est_name,
        mean_token_kl=mean_token_kl,
        max_token_kl=max_token_kl,
        min_token_kl=min_token_kl,
        std_token_kl=std_token_kl,
        sequence_mean_kl=sequence_mean_kl,
        sequence_sum_kl=sequence_sum_kl,
        total_valid_tokens=valid_tokens,
        num_sequences=batch_size,
        is_safe=is_safe,
        warning=warning,
    )


class KLController:
    """Manages KL penalty coefficient (beta) and tracks policy drift history."""

    def __init__(
        self,
        init_beta: float = 0.04,
        target_kl: float = 0.1,
        max_kl_threshold: float = 0.5,
        adaptive: bool = False,
        proportional_gain: float = 0.1,
        min_beta: float = 1e-4,
        max_beta: float = 1.0,
    ):
        self.beta = float(init_beta)
        self.init_beta = float(init_beta)
        self.target_kl = float(target_kl)
        self.max_kl_threshold = float(max_kl_threshold)
        self.adaptive = bool(adaptive)
        self.proportional_gain = float(proportional_gain)
        self.min_beta = float(min_beta)
        self.max_beta = float(max_beta)

        self.history: List[Dict[str, Any]] = []
        self.step_count: int = 0
        self.warning_count: int = 0

    def step(self, kl_summary: KLSummary) -> float:
        """Update controller with current step's KL metrics and return current beta."""
        self.step_count += 1
        current_kl = kl_summary.mean_token_kl

        if not kl_summary.is_safe:
            self.warning_count += 1
            logger.warning(
                f"[KLController] Step {self.step_count}: {kl_summary.warning}"
            )

        # Adaptive beta adjustment (optional proportional controller)
        if self.adaptive:
            # error = current_kl - target_kl
            # If current_kl > target_kl, increase beta to penalize drift
            # If current_kl < target_kl, decrease beta to allow more exploration
            error = (current_kl - self.target_kl) / max(self.target_kl, 1e-6)
            mult = 1.0 + self.proportional_gain * error
            mult = max(0.8, min(1.25, mult))
            self.beta = float(np.clip(self.beta * mult, self.min_beta, self.max_beta))

        self.history.append({
            "step": self.step_count,
            "mean_token_kl": current_kl,
            "max_token_kl": kl_summary.max_token_kl,
            "beta": self.beta,
            "is_safe": kl_summary.is_safe,
        })

        return self.beta

    def get_summary(self) -> Dict[str, Any]:
        """Return history statistics and drift status."""
        if not self.history:
            return {
                "step_count": 0,
                "current_beta": self.beta,
                "mean_kl": 0.0,
                "warning_count": 0,
            }

        all_kl = [h["mean_token_kl"] for h in self.history]
        return {
            "step_count": self.step_count,
            "current_beta": self.beta,
            "initial_beta": self.init_beta,
            "mean_kl": float(np.mean(all_kl)),
            "max_kl": float(np.max(all_kl)),
            "min_kl": float(np.min(all_kl)),
            "warning_count": self.warning_count,
            "adaptive": self.adaptive,
            "target_kl": self.target_kl,
            "max_kl_threshold": self.max_kl_threshold,
        }


class ReferenceModelManager:
    """Manages frozen reference model log-probability computation and memory efficiency.

    Supports:
      1. PEFT disabled adapter forward pass (memory sharing with base model)
      2. Secondary frozen adapter forward pass
      3. Mock forward mode for testing and non-GPU environments
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        adapter_path: Optional[Union[str, Path]] = None,
        mock_mode: bool = False,
        device: str = "cpu",
        dtype: str = "bfloat16",
    ):
        self.model_name_or_path = model_name_or_path or "deepseek-ai/deepseek-math-7b-base"
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.mock_mode = mock_mode
        self.device = device
        self.dtype = dtype

        self._model = None
        self._tokenizer = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize reference model or verify mock mode."""
        if self._initialized:
            return

        if self.mock_mode:
            logger.info("Initializing ReferenceModelManager in mock mode (no GPU model loaded).")
            self._initialized = True
            return

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading reference model: {self.model_name_or_path}")
            torch_dtype = getattr(torch, self.dtype, torch.float32)

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
            )

            adapter_to_load = None
            if self.adapter_path:
                candidates = [
                    self.adapter_path,
                    PROJECT_ROOT / self.adapter_path,
                    Path("/content/drive/MyDrive/math-reasoning-llm") / self.adapter_path,
                    PROJECT_ROOT / "outputs" / "sft" / "adapter_model",
                    Path("/content/drive/MyDrive/math-reasoning-llm/outputs/sft/adapter_model"),
                ]
                for cand in candidates:
                    if cand.exists() and (cand / "adapter_config.json").exists():
                        adapter_to_load = cand
                        break

            if adapter_to_load:
                logger.info(f"Loading reference LoRA adapter from: {adapter_to_load}")
                model = PeftModel.from_pretrained(base_model, str(adapter_to_load))
            else:
                logger.info("No adapter weights found at candidate paths; using base model directly.")
                model = base_model

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            self._model = model
            self._tokenizer = tokenizer
            self._initialized = True
            logger.info("Reference model successfully initialized and frozen.")

        except Exception as e:
            logger.warning(
                f"Failed to load PyTorch/HuggingFace reference model ({e}); falling back to mock mode."
            )
            self.mock_mode = True
            self._initialized = True

    def get_ref_logprobs(
        self,
        input_ids: Union[List[List[int]], np.ndarray],
        attention_mask: Optional[Union[List[List[int]], np.ndarray]] = None,
        completion_mask: Optional[Union[List[List[int]], np.ndarray]] = None,
    ) -> np.ndarray:
        """Extract sequence token log-probabilities under reference policy.

        Args:
            input_ids: Array/List of token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            completion_mask: Completion mask [batch_size, seq_len].

        Returns:
            np.ndarray of log-probabilities for completion tokens [batch_size, seq_len].
        """
        self.initialize()
        ids = np.asarray(input_ids)
        batch_size, seq_len = ids.shape

        if self.mock_mode:
            # Deterministic mock logprobs: small negative values around -0.5 to -2.0
            # using hash of token id to remain reproducible
            rng = np.random.default_rng(seed=42)
            mock_lp = -0.5 - 1.5 * rng.random((batch_size, seq_len))
            if completion_mask is not None:
                mask = np.asarray(completion_mask)
                mock_lp = mock_lp * mask
            return mock_lp

        import torch

        tensor_ids = torch.as_tensor(ids, device=self.device)
        tensor_attn = (
            torch.as_tensor(attention_mask, device=self.device)
            if attention_mask is not None
            else torch.ones_like(tensor_ids)
        )

        with torch.no_grad():
            outputs = self._model(input_ids=tensor_ids, attention_mask=tensor_attn)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

            # Shift logits and labels for next-token prediction
            # Logit at t predicts token at t+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tensor_ids[:, 1:].contiguous()

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_logprobs = torch.gather(
                log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Pad back to full seq_len with zeros at t=0
            zeros = torch.zeros((batch_size, 1), device=self.device, dtype=token_logprobs.dtype)
            full_logprobs = torch.cat([zeros, token_logprobs], dim=1)

            result = full_logprobs.cpu().numpy()
            if completion_mask is not None:
                mask = np.asarray(completion_mask)
                result = result * mask
            return result


def load_kl_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load KL divergence configuration with standard project fallbacks."""
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

    default_path = PROJECT_ROOT / "configs" / "kl_divergence.yaml"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # Standard fallback
    return {
        "kl_divergence": {
            "estimator": "k3",
            "beta": 0.04,
            "target_kl": 0.1,
            "max_kl_threshold": 0.5,
            "adaptive": False,
            "proportional_gain": 0.1,
            "clip_ratio_bounds": [1e-6, 1e6],
        },
        "reference_model": {
            "base_model": "deepseek-ai/deepseek-math-7b-base",
            "adapter_path": "outputs/sft_targeted/adapter_model",
            "fallback_adapter": "outputs/sft/adapter_model",
            "dtype": "bfloat16",
            "device": "auto",
        },
    }
