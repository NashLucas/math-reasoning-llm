"""Production Serving Engine for Mathematical Reasoning (Phase 7 Day 50).

Provides high-throughput, low-latency inference for the Post-GRPO Policy model:
1. Supports multiple decoding regimes:
   - Greedy decoding (fastest, deterministic single response).
   - Temperature-sampled rollouts (stochastic multi-path exploration).
   - Self-consistency majority voting (maj@k over k sampled paths).
2. Strict Answer Contract enforcement:
   - Validates that outputs terminate with `\\n#### <number>`.
   - Extracts canonical numeric answers and computes format compliance metrics.
3. Production traffic logging & monitoring:
   - Records request timestamp, prompt tokens, completion tokens, latency (ms),
     throughput (tokens/sec), regime, and extracted answers to structured JSONL logs.
4. Flexible runtime support:
   - Runs seamlessly with vLLM / Hugging Face backends on CUDA GPU runtimes.
   - Provides deterministic high-fidelity mock mode for local testing and CPU CI.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.math_utils.answer_extraction import extract_answer
from src.math_utils.normalization import is_equivalent
from src.reasoning.self_consistency import majority_vote_with_details

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ServingConfig:
    """Configuration profile for ReasoningServingEngine."""

    model_name_or_path: str = "deepseek-ai/deepseek-math-7b-base"
    adapter_path: Optional[str] = "outputs/grpo/adapter_model"
    max_new_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.95
    use_mock: bool = True
    log_dir: Optional[str] = "logs/production"
    log_requests: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ServingPrediction:
    """Structured response object for a served mathematical query."""

    problem_id: str
    question: str
    completion: str
    extracted_answer: Optional[str]
    format_compliant: bool
    latency_ms: float
    tokens_per_sec: float
    regime: str
    num_samples: int
    prompt_tokens: int
    completion_tokens: int
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    all_completions: List[str] = field(default_factory=list)
    all_extracted: List[Optional[str]] = field(default_factory=list)
    majority_vote_winner: Optional[str] = None
    majority_vote_confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReasoningServingEngine:
    """Production serving engine delivering post-GRPO mathematical reasoning."""

    def __init__(
        self,
        config: Optional[ServingConfig] = None,
        use_mock: bool = True,
        adapter_path: Optional[str] = None,
        log_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the serving engine."""
        self.config = config or ServingConfig()
        if use_mock is not None:
            self.config.use_mock = use_mock
        if adapter_path is not None:
            self.config.adapter_path = adapter_path
        if log_dir is not None:
            self.config.log_dir = str(log_dir)

        # Setup production logging directory
        if self.config.log_dir:
            self.log_dir_path = Path(self.config.log_dir)
            if not self.log_dir_path.is_absolute():
                self.log_dir_path = PROJECT_ROOT / self.log_dir_path
            self.log_dir_path.mkdir(parents=True, exist_ok=True)
            self.traffic_log_file = self.log_dir_path / "requests.jsonl"
        else:
            self.log_dir_path = None
            self.traffic_log_file = None

        self._traffic_history: List[ServingPrediction] = []
        self._llm = None
        self._tokenizer = None

        logger.info(
            f"Initialized ReasoningServingEngine (mock={self.config.use_mock}, "
            f"adapter={self.config.adapter_path}, log_dir={self.log_dir_path})"
        )

        if not self.config.use_mock:
            self._init_backend()

    def _init_backend(self) -> None:
        """Attempt to initialize hardware-accelerated generation engine."""
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA GPU not available; falling back to mock serving mode."
                )
                self.config.use_mock = True
                return

            from src.generation import GenerationEngine

            self._llm = GenerationEngine(
                model_name=self.config.model_name_or_path,
                adapter_path=self.config.adapter_path,
                max_new_tokens=self.config.max_new_tokens,
            )
            logger.info("Successfully loaded GenerationEngine for serving.")
        except Exception as err:
            logger.warning(
                f"Could not initialize live generation engine ({err}); falling back to mock mode."
            )
            self.config.use_mock = True

    def _synthesize_mock_reasoning(
        self,
        question: str,
        sample_idx: int = 0,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Synthesize high-fidelity post-GRPO mathematical reasoning for local testing/CI."""
        # Extract question numbers to produce contextual arithmetic
        nums = [int(x) for x in re.findall(r"\b\d+\b", question)]
        if len(nums) >= 2:
            n1, n2 = nums[0], nums[1]
            inter = n1 * n2
            final_ans = inter + (nums[2] if len(nums) >= 3 else 10)
        elif len(nums) == 1:
            n1 = nums[0]
            inter = n1 * 2
            final_ans = inter + 5
        else:
            n1, n2 = 12, 5
            inter = 60
            final_ans = 75

        # Stochastic variation under sampling
        if temperature > 0.0 and sample_idx > 0:
            if sample_idx % 4 == 0:
                # Occasional slight deviation in alternative rollout
                final_ans_str = str(final_ans + 1)
            else:
                final_ans_str = str(final_ans)
        else:
            final_ans_str = str(final_ans)

        text = (
            f"To determine the solution, let's break down the problem step-by-step:\n"
            f"Step 1: First, we calculate the primary quantity: {n1} * {n2 if len(nums) >= 2 else 2} = {inter}.\n"
            f"Step 2: Next, we add the remaining component: {inter} + "
            f"{nums[2] if len(nums) >= 3 else (5 if len(nums) == 1 else 15)} = {final_ans_str}.\n"
            f"Step 3: This gives the final required total.\n\n"
            f"#### {final_ans_str}"
        )

        words = text.split()
        comp_tokens = int(len(words) * 1.3)
        prompt_tokens = int(len(question.split()) * 1.3)

        return {
            "text": text,
            "extracted_answer": final_ans_str,
            "format_compliant": True,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
        }

    def predict(
        self,
        question: str,
        regime: str = "greedy",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        num_samples: int = 1,
        problem_id: Optional[str] = None,
    ) -> ServingPrediction:
        """Execute mathematical reasoning inference for a single question."""
        pid = problem_id or f"query_{int(time.time() * 1000)}"
        t0 = time.perf_counter()

        temp = (
            0.0
            if regime == "greedy"
            else (
                temperature
                if temperature is not None
                else self.config.default_temperature
            )
        )
        samples_to_generate = (
            8
            if regime == "self_consistency" and num_samples <= 1
            else max(1, num_samples)
        )
        if regime == "greedy":
            samples_to_generate = 1

        all_completions: List[str] = []
        all_extracted: List[Optional[str]] = []
        total_prompt_tok = 0
        total_comp_tok = 0

        if self.config.use_mock or self._llm is None:
            for s_idx in range(samples_to_generate):
                synth = self._synthesize_mock_reasoning(
                    question=question,
                    sample_idx=s_idx,
                    temperature=temp,
                )
                all_completions.append(synth["text"])
                all_extracted.append(synth["extracted_answer"])
                total_prompt_tok = synth["prompt_tokens"]
                total_comp_tok += synth["completion_tokens"]
        else:
            # Live backend generation
            for _ in range(samples_to_generate):
                out = self._llm.generate(
                    prompt=question,
                    temperature=temp,
                    top_p=top_p or self.config.default_top_p,
                    max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                )
                ans, method = extract_answer(out)
                all_completions.append(out)
                all_extracted.append(ans)
                total_comp_tok += len(out.split())

        t_elapsed = time.perf_counter() - t0
        latency_ms = max(0.1, t_elapsed * 1000.0)
        tokens_per_sec = (total_comp_tok / max(0.001, t_elapsed))

        # Determine primary winner
        if regime == "self_consistency" and len(all_extracted) > 1:
            vote_res = majority_vote_with_details(all_extracted)
            win_ans = vote_res.get("winner")
            vote_conf = float(vote_res.get("confidence", 1.0))
            # Pick completion corresponding to majority vote winner
            primary_completion = all_completions[0]
            for comp, ext in zip(all_completions, all_extracted):
                if is_equivalent(ext, win_ans):
                    primary_completion = comp
                    break
        else:
            primary_completion = all_completions[0]
            win_ans = all_extracted[0]
            vote_conf = 1.0

        # Validate answer contract: canonical terminal line `\n#### <number>`
        fmt_compliant = (
            bool(re.search(r"(?:^|\n)####\s*(-?[\d,]+(?:\.\d+)?)\s*$", primary_completion.strip()))
            and win_ans is not None
        )

        pred = ServingPrediction(
            problem_id=pid,
            question=question,
            completion=primary_completion,
            extracted_answer=win_ans,
            format_compliant=fmt_compliant,
            latency_ms=round(latency_ms, 2),
            tokens_per_sec=round(tokens_per_sec, 2),
            regime=regime,
            num_samples=samples_to_generate,
            prompt_tokens=total_prompt_tok,
            completion_tokens=total_comp_tok,
            all_completions=all_completions,
            all_extracted=all_extracted,
            majority_vote_winner=win_ans,
            majority_vote_confidence=round(vote_conf, 4),
            metadata={
                "model": self.config.model_name_or_path,
                "adapter": self.config.adapter_path,
                "use_mock": self.config.use_mock,
            },
        )

        self._traffic_history.append(pred)
        if self.config.log_requests:
            self.log_traffic(pred)

        return pred

    def predict_batch(
        self,
        questions: List[str],
        regime: str = "greedy",
        temperature: Optional[float] = None,
        num_samples: int = 1,
    ) -> List[ServingPrediction]:
        """Execute mathematical reasoning inference over a batch of questions."""
        results: List[ServingPrediction] = []
        for idx, q in enumerate(questions):
            pred = self.predict(
                question=q,
                regime=regime,
                temperature=temperature,
                num_samples=num_samples,
                problem_id=f"batch_{idx}_{int(time.time())}",
            )
            results.append(pred)
        return results

    def log_traffic(self, prediction: ServingPrediction) -> None:
        """Append a production request record to the structured traffic log."""
        if not self.traffic_log_file:
            return
        try:
            with open(self.traffic_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")
        except Exception as err:
            logger.warning(f"Could not append traffic log ({err})")

    def get_traffic_stats(self) -> Dict[str, Any]:
        """Compute aggregate operational metrics across served traffic."""
        if not self._traffic_history:
            return {
                "total_requests": 0,
                "mean_latency_ms": 0.0,
                "format_compliance_rate": 1.0,
                "total_completion_tokens": 0,
            }

        latencies = [p.latency_ms for p in self._traffic_history]
        formats = [1 if p.format_compliant else 0 for p in self._traffic_history]
        tokens = [p.completion_tokens for p in self._traffic_history]

        return {
            "total_requests": len(self._traffic_history),
            "mean_latency_ms": round(float(sum(latencies) / len(latencies)), 2),
            "format_compliance_rate": round(float(sum(formats) / len(formats)), 4),
            "total_completion_tokens": sum(tokens),
        }
