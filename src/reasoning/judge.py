"""LLM Judge for reasoning traces scoring and rubric evaluation (Phase 3 Day 21).

Scores reasoning traces on three key qualitative dimensions (each 0.0 to 1.0):
1. Completeness: Derivation clarity, presence of all intermediate steps, lack of logical jumps.
2. Verification Quality: Active self-checking, back-substitution, condition checks (for solve_then_verify).
3. Novelty / Diversity: Alternative mathematical formulation compared to standard CoT (for alternative_method).

Design Rule 4:
    "Correctness filters first, the judge second. Curated traces must reach the gold answer.
     Judge scores for completeness, verification quality and novelty only rank traces that
     are already correct, because scores from a 7B judge are noisy."
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Rubric weight presets
DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "chain_of_thought": {"completeness": 1.0},
    "solve_then_verify": {"completeness": 0.5, "verification_quality": 0.5},
    "alternative_method": {"completeness": 0.4, "novelty": 0.6},
}

JUDGE_SYSTEM_PROMPT = """You are an expert mathematical reasoning evaluator and educator.
Your task is to objectively evaluate a mathematical solution according to a strict scoring rubric.
You will evaluate the reasoning on specific criteria, assigning each a continuous score from 0.0 to 1.0.

Scoring Rubric:
1. Completeness (0.0 to 1.0):
   - 1.0: Every step and intermediate calculation is clearly stated, logically followable, and complete.
   - 0.5: Reasoning is understandable but some intermediate steps are compressed or skipped.
   - 0.0: No derivation shown, critical steps missing, or incoherent logic.

2. Verification Quality (0.0 to 1.0, or N/A if not solve-then-verify):
   - 1.0: Active, independent verification (e.g. back-substituting the answer, checking constraints, or alternative sanity check).
   - 0.5: Superficial check (e.g. simply repeating the final line or restating without independent check).
   - 0.0: Verification is missing, invalid, or makes false mathematical claims.

3. Novelty / Method Diversity (0.0 to 1.0, or N/A if not alternative-method):
   - 1.0: Genuinely distinct mathematical strategy compared to standard chain-of-thought (e.g. algebraic equation vs arithmetic breakdown, ratio/unit rate, working backwards).
   - 0.5: Slight reordering or rewording of the same arithmetic sequence.
   - 0.0: Identical method to standard solution or trivial paraphrase.

Always structure your final evaluation in the following exact format:
Evaluation:
- Completeness: <score between 0.0 and 1.0>
- Verification Quality: <score between 0.0 and 1.0, or N/A>
- Novelty: <score between 0.0 and 1.0, or N/A>
- Overall Score: <score between 0.0 and 1.0>
- Rationale: <1-3 sentences explaining the assigned scores>
"""


@dataclass
class JudgeScore:
    """Individual evaluation score and breakdown from the LLM judge."""

    completeness: float = 0.0
    verification_quality: Optional[float] = None
    novelty: Optional[float] = None
    overall_score: float = 0.0
    rationale: str = ""
    raw_output: str = ""
    parsing_status: str = "success"  # "success", "partial", "fallback"

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completeness": round(self.completeness, 3),
            "verification_quality": round(self.verification_quality, 3) if self.verification_quality is not None else None,
            "novelty": round(self.novelty, 3) if self.novelty is not None else None,
            "overall_score": round(self.overall_score, 3),
            "rationale": self.rationale,
            "parsing_status": self.parsing_status,
        }


@dataclass
class ScoredTrace:
    """A reasoning trace paired with its judge score."""

    trace_id: str
    problem_id: str
    prompt_type: str
    question: str
    gold_answer: str
    completion: str
    score: JudgeScore
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "problem_id": self.problem_id,
            "prompt_type": self.prompt_type,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "completion": self.completion,
            "score": self.score.to_dict(),
            "metadata": self.metadata,
        }


def format_judge_prompt(
    question: str,
    gold_answer: str,
    completion: str,
    prompt_type: str = "chain_of_thought",
    reference_cot: Optional[str] = None,
) -> str:
    """Format the evaluation prompt for the LLM judge."""
    lines = [
        f"Problem: {question}",
        f"Gold Answer: {gold_answer}",
        f"Prompt Strategy Under Evaluation: {prompt_type}",
        "",
        "Candidate Solution:",
        completion.strip(),
        "",
    ]

    if prompt_type == "alternative_method" and reference_cot:
        lines.extend([
            "Reference Standard Chain-of-Thought Solution (for comparing novelty):",
            reference_cot.strip(),
            "",
        ])

    lines.extend([
        "Evaluate the candidate solution using the scoring rubric and output your evaluation:",
    ])

    return "\n".join(lines)


def _clamp_score(val: float) -> float:
    """Clamp float to [0.0, 1.0]."""
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return max(0.0, min(1.0, float(val)))


def parse_judge_output(
    raw_output: str,
    prompt_type: str = "chain_of_thought",
    weights: Optional[Dict[str, float]] = None,
) -> JudgeScore:
    """Extract completeness, verification quality, novelty, and overall score from judge text.

    Robust to various LLM response shapes:
    - Standard markdown bullet format (`- Completeness: 0.9`)
    - JSON embedded code block (`{"completeness": 0.9, ...}`)
    - Key-value lines (`Completeness: 0.9 / 1.0` or `Completeness = 0.9`)
    - 0-10 or 0-5 scale normalization
    """
    if not raw_output or not raw_output.strip():
        return JudgeScore(raw_output=raw_output, parsing_status="fallback")

    text = raw_output.strip()
    status = "success"

    # 1. Attempt JSON parsing if JSON block is present
    json_match = re.search(r"\{[\s\S]*?\}", text)
    if json_match:
        try:
            d = json.loads(json_match.group(0))
            if "completeness" in d or "overall_score" in d:
                c_val = float(d.get("completeness", 0.0))
                v_val = float(d["verification_quality"]) if d.get("verification_quality") not in (None, "N/A", "n/a") else None
                n_val = float(d["novelty"]) if d.get("novelty") not in (None, "N/A", "n/a") else None
                o_val = float(d.get("overall_score", 0.0)) if "overall_score" in d else None
                rat = str(d.get("rationale", "")).strip()

                # Scale normalization if scores > 1.0 (e.g. 0-10 scale)
                scale = 10.0 if (c_val > 1.0 or (v_val and v_val > 1.0) or (n_val and n_val > 1.0)) else 1.0
                c = _clamp_score(c_val / scale)
                v = _clamp_score(v_val / scale) if v_val is not None else None
                n = _clamp_score(n_val / scale) if n_val is not None else None

                overall = _clamp_score(o_val / scale) if o_val is not None else compute_overall_score(c, v, n, prompt_type, weights)
                return JudgeScore(
                    completeness=c,
                    verification_quality=v,
                    novelty=n,
                    overall_score=overall,
                    rationale=rat,
                    raw_output=raw_output,
                    parsing_status="success",
                )
        except Exception:
            pass

    # 2. Regex pattern extraction for bullet or line-based format
    clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    clean_text = re.sub(r"__(.*?)__", r"\1", clean_text)

    def _extract_metric(metric_name: str) -> Optional[float]:
        # Matches: "- Completeness: 0.8", "Completeness = 8/10", "Completeness: 85%", "**Completeness:** 1.0"
        patterns = [
            rf"(?:^|\n)[ \t*\-\d.]*{metric_name}[ \t]*[:=][ \t]*([0-9]+(?:\.[0-9]+)?)[ \t]*/[ \t]*([0-9]+(?:\.[0-9]+)?)",
            rf"(?:^|\n)[ \t*\-\d.]*{metric_name}[ \t]*[:=][ \t]*([0-9]+(?:\.[0-9]+)?)[ \t]*%",
            rf"(?:^|\n)[ \t*\-\d.]*{metric_name}[ \t]*[:=][ \t]*([0-9]+(?:\.[0-9]+)?)",
        ]

        # Check for explicit N/A
        na_match = re.search(rf"(?:^|\n)[ \t*\-\d.]*{metric_name}[ \t]*[:=][ \t]*(?:N/A|n/a|None|none)", clean_text)
        if na_match:
            return None

        # Ratio match: e.g. 8/10 or 4/5
        m_ratio = re.search(patterns[0], clean_text, re.IGNORECASE)
        if m_ratio:
            num = float(m_ratio.group(1))
            den = float(m_ratio.group(2))
            return _clamp_score(num / den) if den > 0 else 0.0

        # Percentage match: e.g. 85%
        m_pct = re.search(patterns[1], clean_text, re.IGNORECASE)
        if m_pct:
            return _clamp_score(float(m_pct.group(1)) / 100.0)

        # Direct number match: e.g. 0.85 or 8.5
        m_num = re.search(patterns[2], clean_text, re.IGNORECASE)
        if m_num:
            val = float(m_num.group(1))
            if val > 1.0 and val <= 10.0:
                val = val / 10.0
            elif val > 10.0 and val <= 100.0:
                val = val / 100.0
            return _clamp_score(val)

        return None

    completeness = _extract_metric("Completeness")
    verification = _extract_metric("Verification(?: Quality)?")
    novelty = _extract_metric("Novelty(?: / Method Diversity)?")
    extracted_overall = _extract_metric("Overall Score")

    # Extract rationale
    rationale_match = re.search(r"(?:^|\n)[ \t*\-\d.]*Rationale[ \t]*[:=][ \t]*(.+)", clean_text, re.DOTALL | re.IGNORECASE)
    rationale = rationale_match.group(1).strip() if rationale_match else ""
    # Truncate rationale if multi-paragraph
    if "\n\n" in rationale:
        rationale = rationale.split("\n\n")[0].strip()

    # Fallback assignment if parsing failed
    if completeness is None and extracted_overall is not None:
        completeness = extracted_overall
        status = "partial"
    elif completeness is None:
        completeness = 0.5  # Neutral fallback
        status = "fallback"

    overall = extracted_overall if extracted_overall is not None else compute_overall_score(
        completeness=completeness,
        verification=verification,
        novelty=novelty,
        prompt_type=prompt_type,
        weights=weights,
    )

    return JudgeScore(
        completeness=completeness,
        verification_quality=verification,
        novelty=novelty,
        overall_score=overall,
        rationale=rationale,
        raw_output=raw_output,
        parsing_status=status,
    )


def compute_overall_score(
    completeness: float,
    verification: Optional[float] = None,
    novelty: Optional[float] = None,
    prompt_type: str = "chain_of_thought",
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute overall score as weighted average according to prompt strategy."""
    w = weights or DEFAULT_WEIGHTS.get(prompt_type, {"completeness": 1.0})

    total_weight = 0.0
    weighted_sum = 0.0

    if "completeness" in w:
        c_w = w["completeness"]
        weighted_sum += completeness * c_w
        total_weight += c_w

    if "verification_quality" in w and verification is not None:
        v_w = w["verification_quality"]
        weighted_sum += verification * v_w
        total_weight += v_w

    if "novelty" in w and novelty is not None:
        n_w = w["novelty"]
        weighted_sum += novelty * n_w
        total_weight += n_w

    if total_weight <= 0.0:
        return _clamp_score(completeness)

    return _clamp_score(weighted_sum / total_weight)


class ReasoningJudge:
    """Manages LLM judge model loading, batch prompt formatting, and score parsing."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        backend: str = "auto",
        device: Optional[str] = None,
        dtype: str = "auto",
        use_mock: bool = False,
        seed: int = 42,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.use_mock = use_mock
        self.seed = seed
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit

        self.model: Any = None
        self.tokenizer: Any = None
        self._is_initialized = False

    def _init_model(self) -> None:
        if self._is_initialized or self.use_mock:
            return

        from src.model import load_model_and_tokenizer

        resolved_dtype = self.dtype
        if resolved_dtype == "auto":
            try:
                import torch

                resolved_dtype = "bfloat16" if torch.cuda.is_available() else "float32"
            except ImportError:
                resolved_dtype = "float32"

        # Low-VRAM auto-detection for 7B/8B models on <=16GB GPUs
        auto_4bit = self.load_in_4bit
        if not auto_4bit and not self.load_in_8bit:
            try:
                import torch
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    if vram_gb <= 16.0 and any(m in self.model_name_or_path.lower() for m in ["7b", "8b"]):
                        logger.info(
                            f"Auto-enabling 4-bit quantization for judge model {self.model_name_or_path} on {vram_gb:.1f}GB GPU"
                        )
                        auto_4bit = True
            except Exception:
                pass

        logger.info(f"Initializing Judge model: {self.model_name_or_path} ({resolved_dtype})")
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name_or_path=self.model_name_or_path,
            torch_dtype=resolved_dtype,
            device_map=self.device or "auto",
            load_in_4bit=auto_4bit,
            load_in_8bit=self.load_in_8bit,
        )
        self._is_initialized = True

    def generate_mock_score(
        self,
        completion: str,
        prompt_type: str = "chain_of_thought",
    ) -> JudgeScore:
        """Generate deterministic score for testing and mock runs."""
        text = completion.lower()
        words = text.split()
        num_words = len(words)

        # Completeness based on step detail and intermediate calculations
        has_equals = "=" in text
        if num_words >= 35 and has_equals:
            c_score = 0.95
        elif num_words >= 20:
            c_score = 0.85
        elif num_words >= 12:
            c_score = 0.65
        else:
            c_score = 0.45

        # Verification quality
        v_score: Optional[float] = None
        if prompt_type == "solve_then_verify":
            has_verif = "verif" in text or "check" in text
            if has_verif and any(k in text for k in ["substitut", "condition", "average", "match", "difference", "back"]):
                v_score = 0.95
            elif has_verif and any(k in text for k in [">", "<", "math holds"]):
                v_score = 0.50
            elif has_verif and any(k in text for k in ["none", "the answer is"]):
                v_score = 0.10
            elif has_verif:
                v_score = 0.70
            else:
                v_score = 0.0

        # Novelty / diversity
        n_score: Optional[float] = None
        if prompt_type == "alternative_method":
            distinct_markers = ["assumption", "factor", "pairing", "multiplicative", "excess", "ratio"]
            if any(m in text for m in distinct_markers):
                n_score = 0.90
            elif "another way" in text or "step" in text:
                n_score = 0.20
            else:
                n_score = 0.60

        overall = compute_overall_score(c_score, v_score, n_score, prompt_type)
        return JudgeScore(
            completeness=c_score,
            verification_quality=v_score,
            novelty=n_score,
            overall_score=overall,
            rationale="Mock judge: intermediate derivation steps and prompt-specific criteria evaluated.",
            raw_output=(
                f"Evaluation:\n"
                f"- Completeness: {c_score}\n"
                f"- Verification Quality: {v_score if v_score is not None else 'N/A'}\n"
                f"- Novelty: {n_score if n_score is not None else 'N/A'}\n"
                f"- Overall Score: {overall}\n"
                f"- Rationale: Mock judge: intermediate derivation steps and prompt-specific criteria evaluated.\n"
            ),
            parsing_status="success",
        )

    def evaluate_trace(
        self,
        question: str,
        gold_answer: str,
        completion: str,
        prompt_type: str = "chain_of_thought",
        reference_cot: Optional[str] = None,
    ) -> JudgeScore:
        """Evaluate a single reasoning trace."""
        if self.use_mock:
            return self.generate_mock_score(completion, prompt_type)

        self._init_model()
        user_prompt = format_judge_prompt(
            question=question,
            gold_answer=gold_answer,
            completion=completion,
            prompt_type=prompt_type,
            reference_cot=reference_cot,
        )

        from src.data.formatting import apply_chat_template, format_chat_messages
        from src.generation import generate_batch_hf

        messages = format_chat_messages(
            question=user_prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
        )
        rendered_prompt = apply_chat_template(messages, self.tokenizer)

        results = generate_batch_hf(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=[rendered_prompt],
            max_new_tokens=256,
            temperature=0.1,  # Low temperature for judge consistency
            do_sample=False,
        )

        raw_output = results[0][0]["completion"]
        return parse_judge_output(raw_output, prompt_type=prompt_type)

    def evaluate_traces(
        self,
        traces: Sequence[Union[Dict[str, Any], Any]],
        reference_traces_by_pid: Optional[Dict[str, str]] = None,
        batch_size: int = 10,
    ) -> List[ScoredTrace]:
        """Evaluate a batch of traces, returning ScoredTrace objects."""
        ref_lookup = reference_traces_by_pid or {}
        scored_traces: List[ScoredTrace] = []

        if self.use_mock:
            for item in traces:
                d = item if isinstance(item, dict) else item.to_dict()
                ptype = d.get("prompt_type", "chain_of_thought")
                score = self.generate_mock_score(d.get("completion", ""), ptype)
                scored_traces.append(
                    ScoredTrace(
                        trace_id=d.get("trace_id", ""),
                        problem_id=d.get("problem_id", ""),
                        prompt_type=ptype,
                        question=d.get("question", ""),
                        gold_answer=str(d.get("gold_answer", "")),
                        completion=d.get("completion", ""),
                        score=score,
                    )
                )
            return scored_traces

        self._init_model()
        from src.data.formatting import apply_chat_template, format_chat_messages
        from src.generation import generate_batch_hf

        prompts: List[str] = []
        raw_items: List[Dict[str, Any]] = []

        for item in traces:
            d = item if isinstance(item, dict) else item.to_dict()
            raw_items.append(d)
            pid = str(d.get("problem_id", ""))
            ptype = d.get("prompt_type", "chain_of_thought")
            ref_cot = ref_lookup.get(pid)

            user_p = format_judge_prompt(
                question=d.get("question", ""),
                gold_answer=str(d.get("gold_answer", "")),
                completion=d.get("completion", ""),
                prompt_type=ptype,
                reference_cot=ref_cot,
            )
            messages = format_chat_messages(
                question=user_p,
                system_prompt=JUDGE_SYSTEM_PROMPT,
            )
            prompts.append(apply_chat_template(messages, self.tokenizer))

        # Batch generation
        for i in range(0, len(prompts), batch_size):
            chunk_prompts = prompts[i : i + batch_size]
            chunk_items = raw_items[i : i + batch_size]
            results = generate_batch_hf(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=chunk_prompts,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
            )

            for item_dict, res_group in zip(chunk_items, results):
                raw_out = res_group[0]["completion"]
                ptype = item_dict.get("prompt_type", "chain_of_thought")
                score = parse_judge_output(raw_out, prompt_type=ptype)
                scored_traces.append(
                    ScoredTrace(
                        trace_id=item_dict.get("trace_id", ""),
                        problem_id=item_dict.get("problem_id", ""),
                        prompt_type=ptype,
                        question=item_dict.get("question", ""),
                        gold_answer=str(item_dict.get("gold_answer", "")),
                        completion=item_dict.get("completion", ""),
                        score=score,
                    )
                )

        return scored_traces


def compute_pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Pearson correlation coefficient r between two series."""
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    diff_x = [val - mean_x for val in x]
    diff_y = [val - mean_y for val in y]

    cov = sum(dx * dy for dx, dy in zip(diff_x, diff_y))
    var_x = sum(dx * dx for dx in diff_x)
    var_y = sum(dy * dy for dy in diff_y)

    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0

    return cov / denom


def compute_judge_calibration(
    judge_scores: Sequence[Union[JudgeScore, Dict[str, Any], float]],
    manual_scores: Sequence[Union[JudgeScore, Dict[str, Any], float]],
    min_correlation: float = 0.60,
) -> Dict[str, Any]:
    """Compute calibration metrics between LLM judge scores and manual ground truth scores.

    Day 22 Gate requirement:
        "Check calibration against your 20 manual scores: if the correlation is below 0.6,
         simplify the rubric or give more examples in the prompt."

    Returns:
        Dictionary with pearson_r, mae, mse, is_calibrated (True if r >= 0.60), and n_samples.
    """
    def _extract_val(s: Union[JudgeScore, Dict[str, Any], float]) -> float:
        if isinstance(s, (int, float)):
            return float(s)
        if isinstance(s, JudgeScore):
            return s.overall_score
        return float(s.get("overall_score", s.get("score", 0.0)))

    vals_judge = [_extract_val(s) for s in judge_scores]
    vals_manual = [_extract_val(s) for s in manual_scores]

    n = min(len(vals_judge), len(vals_manual))
    if n < 2:
        return {
            "n_samples": n,
            "pearson_r": 0.0,
            "mae": 0.0,
            "mse": 0.0,
            "is_calibrated": False,
        }

    v_j = vals_judge[:n]
    v_m = vals_manual[:n]

    r = compute_pearson_correlation(v_j, v_m)
    mae = sum(abs(j - m) for j, m in zip(v_j, v_m)) / n
    mse = sum((j - m) ** 2 for j, m in zip(v_j, v_m)) / n

    return {
        "n_samples": n,
        "pearson_r": round(r, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "is_calibrated": r >= min_correlation,
    }

def compute_score_distribution(
    scores: Sequence[Union[JudgeScore, Dict[str, Any], float]],
) -> Dict[str, Any]:
    """Compute summary statistics and histogram buckets for a collection of judge scores.

    Useful for Day 22 score spread analysis and detecting judge score clustering (>0.90).
    """
    def _extract_val(s: Union[JudgeScore, Dict[str, Any], float]) -> float:
        if isinstance(s, (int, float)):
            return float(s)
        if isinstance(s, JudgeScore):
            return s.overall_score
        return float(s.get("overall_score", s.get("score", 0.0)))

    vals = sorted([_extract_val(s) for s in scores])
    n = len(vals)
    if n == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "iqr": 0.0,
            "pct_above_0_90": 0.0,
            "buckets": {"<0.50": 0, "0.50-0.69": 0, "0.70-0.79": 0, "0.80-0.89": 0, ">=0.90": 0},
            "high_clustering": False,
        }

    mean_val = sum(vals) / n
    variance = sum((v - mean_val) ** 2 for v in vals) / n
    std_val = math.sqrt(variance)

    def _percentile(p: float) -> float:
        idx = int(p * (n - 1))
        return vals[idx]

    med_val = _percentile(0.50)
    q25 = _percentile(0.25)
    q75 = _percentile(0.75)
    iqr = q75 - q25

    # Buckets
    b_lt_50 = sum(1 for v in vals if v < 0.50)
    b_50_69 = sum(1 for v in vals if 0.50 <= v < 0.70)
    b_70_79 = sum(1 for v in vals if 0.70 <= v < 0.80)
    b_80_89 = sum(1 for v in vals if 0.80 <= v < 0.90)
    b_ge_90 = sum(1 for v in vals if v >= 0.90)
    pct_above_90 = round((b_ge_90 / n) * 100.0, 2)

    return {
        "count": n,
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "median": round(med_val, 4),
        "min": round(vals[0], 4),
        "max": round(vals[-1], 4),
        "q25": round(q25, 4),
        "q75": round(q75, 4),
        "iqr": round(iqr, 4),
        "pct_above_0_90": pct_above_90,
        "buckets": {
            "<0.50": b_lt_50,
            "0.50-0.69": b_50_69,
            "0.70-0.79": b_70_79,
            "0.80-0.89": b_80_89,
            ">=0.90": b_ge_90,
        },
        "high_clustering": pct_above_90 >= 60.0,
    }


def assign_within_problem_ranks(
    scored_traces: Sequence[Union[ScoredTrace, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Group scored traces by problem_id and assign 1-based relative ranks.

    Rank 1 is the highest overall_score for that problem.
    Resolves ties deterministically by prompt type diversity and trace length.
    """
    by_problem: Dict[str, List[Dict[str, Any]]] = {}
    for item in scored_traces:
        d = item.to_dict() if hasattr(item, "to_dict") else dict(item)
        pid = str(d.get("problem_id", ""))
        by_problem.setdefault(pid, []).append(d)

    ranked_results: List[Dict[str, Any]] = []
    for pid, group in by_problem.items():
        # Sort descending by overall_score, then secondary by length
        sorted_group = sorted(
            group,
            key=lambda t: (
                float(t.get("score", {}).get("overall_score", 0.0)),
                len(t.get("completion", "")),
            ),
            reverse=True,
        )
        for rank, trace_dict in enumerate(sorted_group, start=1):
            if "metadata" not in trace_dict or not isinstance(trace_dict["metadata"], dict):
                trace_dict["metadata"] = {}
            trace_dict["metadata"]["rank_within_problem"] = rank
            trace_dict["rank_within_problem"] = rank
            ranked_results.append(trace_dict)

    return ranked_results

