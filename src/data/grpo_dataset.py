"""GRPO dataset preparation and difficulty filtering (Phase 6 Day 41).

Implements difficulty profiling, empirical pass-rate filtering (0.20 <= pass_rate <= 0.80),
reasoning step-complexity stratification, strict Test-Set Wall validation, and canonical
GRPO prompt formatting for post-training reinforcement learning.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import yaml

from src.data.formatting import format_zero_shot_prompt
from src.data.preprocessing import (
    load_data_config,
    verify_no_contamination,
)
from src.run_info import save_run_info

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestSetWallError(Exception):
    """Raised when validation or test problems leak into training data."""

    __test__ = False


class DifficultyTier(str, Enum):
    """Problem difficulty categories for GRPO."""

    EASY = "easy"          # High pass rate (> 0.80) or <= 2 reasoning steps (zero-variance risk)
    MEDIUM = "medium"      # Intermediate pass rate (0.20 - 0.80) or 3-6 reasoning steps (optimal)
    HARD = "hard"          # Low pass rate (< 0.20) or > 6 reasoning steps (zero-variance / timeout)
    UNOBSERVED = "unobserved"  # No empirical pass rate recorded


@dataclass
class GRPOProblemProfile:
    """Difficulty and candidate profile for a math problem."""

    problem_id: str
    question: str
    gold_answer: str
    cleaned_solution: str
    step_count: int
    char_length: int
    pass_rate: Optional[float]
    difficulty_tier: str
    is_grpo_candidate: bool
    selection_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def count_reasoning_steps(solution: str) -> int:
    """Count non-empty derivation steps in a cleaned solution string.

    Ignores blank lines and the terminal '#### <number>' answer marker.
    """
    if not solution or not solution.strip():
        return 0

    lines = solution.splitlines()
    step_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("####")
    ]
    return len(step_lines)


def profile_problem_difficulty(
    problem_id: str,
    question: str,
    gold_answer: str,
    cleaned_solution: str,
    pass_rate: Optional[float] = None,
    min_pass_rate: float = 0.20,
    max_pass_rate: float = 0.80,
    min_steps: int = 3,
    max_steps: int = 6,
    max_prompt_chars: int = 1000,
    filter_mode: str = "hybrid",
) -> GRPOProblemProfile:
    """Profile problem difficulty and determine eligibility for GRPO candidate groups.

    Args:
        problem_id: Unique problem identifier.
        question: Question prompt text.
        gold_answer: Ground truth target numeric answer.
        cleaned_solution: Step-by-step derivation text.
        pass_rate: Optional empirical pass rate (0.0 to 1.0) from multi-sample rollouts.
        min_pass_rate: Minimum empirical pass rate (inclusive).
        max_pass_rate: Maximum empirical pass rate (inclusive).
        min_steps: Minimum solution reasoning steps (inclusive).
        max_steps: Maximum solution reasoning steps (inclusive).
        max_prompt_chars: Maximum allowed characters in question prompt.
        filter_mode: 'hybrid' (pass-rate if present, else steps),
                     'pass_rate_only', or 'steps_only'.

    Returns:
        GRPOProblemProfile containing tier, eligibility flag, and reason.
    """
    step_count = count_reasoning_steps(cleaned_solution)
    char_len = len(question.strip())

    # Check question length ceiling
    if char_len > max_prompt_chars:
        return GRPOProblemProfile(
            problem_id=problem_id,
            question=question,
            gold_answer=gold_answer,
            cleaned_solution=cleaned_solution,
            step_count=step_count,
            char_length=char_len,
            pass_rate=pass_rate,
            difficulty_tier=DifficultyTier.HARD.value,
            is_grpo_candidate=False,
            selection_reason=f"question_exceeds_max_chars_{char_len}_{max_prompt_chars}",
        )

    # 1. Mode: Pass rate only
    if filter_mode == "pass_rate_only":
        if pass_rate is None:
            return GRPOProblemProfile(
                problem_id=problem_id,
                question=question,
                gold_answer=gold_answer,
                cleaned_solution=cleaned_solution,
                step_count=step_count,
                char_length=char_len,
                pass_rate=None,
                difficulty_tier=DifficultyTier.UNOBSERVED.value,
                is_grpo_candidate=False,
                selection_reason="missing_empirical_pass_rate",
            )
        return _evaluate_pass_rate(
            problem_id, question, gold_answer, cleaned_solution,
            step_count, char_len, pass_rate, min_pass_rate, max_pass_rate,
        )

    # 2. Mode: Steps only
    if filter_mode == "steps_only":
        return _evaluate_steps(
            problem_id, question, gold_answer, cleaned_solution,
            step_count, char_len, pass_rate, min_steps, max_steps,
        )

    # 3. Mode: Hybrid (default)
    # If pass_rate is recorded, empirical performance dictates eligibility
    if pass_rate is not None:
        return _evaluate_pass_rate(
            problem_id, question, gold_answer, cleaned_solution,
            step_count, char_len, pass_rate, min_pass_rate, max_pass_rate,
        )

    # Otherwise fallback to solution step complexity
    return _evaluate_steps(
        problem_id, question, gold_answer, cleaned_solution,
        step_count, char_len, pass_rate, min_steps, max_steps,
    )


def _evaluate_pass_rate(
    problem_id: str,
    question: str,
    gold_answer: str,
    cleaned_solution: str,
    step_count: int,
    char_len: int,
    pass_rate: float,
    min_pass_rate: float,
    max_pass_rate: float,
) -> GRPOProblemProfile:
    """Evaluate candidate eligibility from empirical pass rate."""
    if pass_rate > max_pass_rate:
        return GRPOProblemProfile(
            problem_id=problem_id,
            question=question,
            gold_answer=gold_answer,
            cleaned_solution=cleaned_solution,
            step_count=step_count,
            char_length=char_len,
            pass_rate=pass_rate,
            difficulty_tier=DifficultyTier.EASY.value,
            is_grpo_candidate=False,
            selection_reason=f"pass_rate_too_high_{pass_rate:.2f}_zero_variance",
        )
    if pass_rate < min_pass_rate:
        return GRPOProblemProfile(
            problem_id=problem_id,
            question=question,
            gold_answer=gold_answer,
            cleaned_solution=cleaned_solution,
            step_count=step_count,
            char_length=char_len,
            pass_rate=pass_rate,
            difficulty_tier=DifficultyTier.HARD.value,
            is_grpo_candidate=False,
            selection_reason=f"pass_rate_too_low_{pass_rate:.2f}_zero_variance",
        )

    return GRPOProblemProfile(
        problem_id=problem_id,
        question=question,
        gold_answer=gold_answer,
        cleaned_solution=cleaned_solution,
        step_count=step_count,
        char_length=char_len,
        pass_rate=pass_rate,
        difficulty_tier=DifficultyTier.MEDIUM.value,
        is_grpo_candidate=True,
        selection_reason=f"optimal_pass_rate_{pass_rate:.2f}_high_variance",
    )


def _evaluate_steps(
    problem_id: str,
    question: str,
    gold_answer: str,
    cleaned_solution: str,
    step_count: int,
    char_len: int,
    pass_rate: Optional[float],
    min_steps: int,
    max_steps: int,
) -> GRPOProblemProfile:
    """Evaluate candidate eligibility from solution reasoning step count."""
    if step_count < min_steps:
        return GRPOProblemProfile(
            problem_id=problem_id,
            question=question,
            gold_answer=gold_answer,
            cleaned_solution=cleaned_solution,
            step_count=step_count,
            char_length=char_len,
            pass_rate=pass_rate,
            difficulty_tier=DifficultyTier.EASY.value,
            is_grpo_candidate=False,
            selection_reason=f"too_few_steps_{step_count}_trivial",
        )
    if step_count > max_steps:
        return GRPOProblemProfile(
            problem_id=problem_id,
            question=question,
            gold_answer=gold_answer,
            cleaned_solution=cleaned_solution,
            step_count=step_count,
            char_length=char_len,
            pass_rate=pass_rate,
            difficulty_tier=DifficultyTier.HARD.value,
            is_grpo_candidate=False,
            selection_reason=f"excessive_steps_{step_count}_complex",
        )

    return GRPOProblemProfile(
        problem_id=problem_id,
        question=question,
        gold_answer=gold_answer,
        cleaned_solution=cleaned_solution,
        step_count=step_count,
        char_length=char_len,
        pass_rate=pass_rate,
        difficulty_tier=DifficultyTier.MEDIUM.value,
        is_grpo_candidate=True,
        selection_reason=f"optimal_reasoning_steps_{step_count}",
    )


def format_grpo_prompt(question: str) -> str:
    """Format a question into the canonical zero-shot prompt with Answer Contract instruction."""
    return format_zero_shot_prompt(question)


def compute_sha256(path: Union[str, Path]) -> str:
    """Calculate SHA256 hexadecimal digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_grpo_dataset_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load grpo dataset yaml configuration with standard fallbacks."""
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

    default_path = PROJECT_ROOT / "configs" / "grpo_dataset.yaml"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    return {
        "filtering": {
            "mode": "hybrid",
            "min_pass_rate": 0.20,
            "max_pass_rate": 0.80,
            "min_steps": 3,
            "max_steps": 6,
            "max_prompt_chars": 1000,
        },
        "sampling": {
            "target_train_size": 1000,
            "val_size": 100,
            "seed": 42,
        },
        "paths": {
            "train_input": "data/processed/train/train.parquet",
            "val_input": "data/processed/val/val_gating.parquet",
            "test_input": "data/processed/test/test.parquet",
            "pass_rates_input": "data/synthetic/generated/problem_pass_rates.parquet",
            "output_dir": "data/processed/grpo",
        },
    }


def prepare_grpo_dataset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pass_rates_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute full difficulty filtering, Test-Set Wall verification, and dataset freezing.

    Args:
        train_df: Training pool DataFrame.
        val_df: Validation pool DataFrame for held-out GRPO validation prompts.
        test_df: Forbidden test split DataFrame for contamination assertion.
        pass_rates_df: Optional DataFrame with 'problem_id' and 'pass_rate'.
        output_dir: Directory where grpo_train and grpo_val datasets will be saved.
        config: GRPO dataset configuration dictionary.
        seed: Random seed for deterministic candidate sampling.

    Returns:
        Dictionary containing output paths, counts, tier distributions, and manifest.
    """
    cfg = config or load_grpo_dataset_config()
    filt_cfg = cfg.get("filtering", {})
    samp_cfg = cfg.get("sampling", {})

    mode = filt_cfg.get("mode", "hybrid")
    min_pass = float(filt_cfg.get("min_pass_rate", 0.20))
    max_pass = float(filt_cfg.get("max_pass_rate", 0.80))
    min_steps = int(filt_cfg.get("min_steps", 3))
    max_steps = int(filt_cfg.get("max_steps", 6))
    max_chars = int(filt_cfg.get("max_prompt_chars", 1000))
    target_train_size = int(samp_cfg.get("target_train_size", 1000))
    val_size = int(samp_cfg.get("val_size", 100))

    out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "data" / "processed" / "grpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. STRICT TEST-SET WALL ENFORCEMENT
    test_ids: Set[str] = set(test_df["problem_id"].tolist())
    val_ids: Set[str] = set(val_df["problem_id"].tolist())

    train_test_overlap = set(train_df["problem_id"].tolist()).intersection(test_ids)
    if train_test_overlap:
        raise TestSetWallError(
            f"Test-Set Wall violated! {len(train_test_overlap)} test problems detected in training pool: {list(train_test_overlap)[:5]}"
        )

    # Check n-gram / exact question contamination
    try:
        data_cfg = load_data_config()
        reviewed_test_ids = set(data_cfg.get("contamination", {}).get("reviewed_template_overlaps", []))
    except Exception:
        reviewed_test_ids = set()

    verify_no_contamination(train_df, test_df, reviewed_test_ids=reviewed_test_ids)

    # 2. Build pass rate lookup if provided
    pass_rate_lookup: Dict[str, float] = {}
    if pass_rates_df is not None and not pass_rates_df.empty:
        for _, row in pass_rates_df.iterrows():
            pid = str(row["problem_id"])
            pr = float(row["pass_rate"])
            pass_rate_lookup[pid] = pr

    # 3. Profile all training problems
    profiles: List[GRPOProblemProfile] = []
    candidates: List[GRPOProblemProfile] = []

    for _, row in train_df.iterrows():
        pid = str(row["problem_id"])
        # Ensure never in test or val
        if pid in test_ids or pid in val_ids:
            continue

        q = str(row["question"])
        gold = str(row.get("gold_answer", ""))
        sol = str(row.get("cleaned_solution", row.get("answer", "")))
        pr = pass_rate_lookup.get(pid, None)

        profile = profile_problem_difficulty(
            problem_id=pid,
            question=q,
            gold_answer=gold,
            cleaned_solution=sol,
            pass_rate=pr,
            min_pass_rate=min_pass,
            max_pass_rate=max_pass,
            min_steps=min_steps,
            max_steps=max_steps,
            max_prompt_chars=max_chars,
            filter_mode=mode,
        )
        profiles.append(profile)
        if profile.is_grpo_candidate:
            candidates.append(profile)

    tier_counts = {
        DifficultyTier.EASY.value: sum(1 for p in profiles if p.difficulty_tier == DifficultyTier.EASY.value),
        DifficultyTier.MEDIUM.value: sum(1 for p in profiles if p.difficulty_tier == DifficultyTier.MEDIUM.value),
        DifficultyTier.HARD.value: sum(1 for p in profiles if p.difficulty_tier == DifficultyTier.HARD.value),
        DifficultyTier.UNOBSERVED.value: sum(1 for p in profiles if p.difficulty_tier == DifficultyTier.UNOBSERVED.value),
    }

    # 4. Sample target train candidates deterministically
    candidate_df = pd.DataFrame([p.to_dict() for p in candidates])
    if len(candidate_df) > target_train_size:
        # Sample deterministically by problem_id sorting first
        candidate_df = candidate_df.sort_values("problem_id").sample(
            n=target_train_size, random_state=seed
        ).reset_index(drop=True)
    else:
        candidate_df = candidate_df.sort_values("problem_id").reset_index(drop=True)

    # Attach canonical formatted prompts
    candidate_df["prompt"] = candidate_df["question"].apply(format_grpo_prompt)

    # 5. Prepare held-out validation prompts
    val_candidates = []
    for _, row in val_df.iterrows():
        pid = str(row["problem_id"])
        if pid in test_ids:
            continue
        q = str(row["question"])
        gold = str(row.get("gold_answer", ""))
        sol = str(row.get("cleaned_solution", row.get("answer", "")))
        val_candidates.append({
            "problem_id": pid,
            "question": q,
            "gold_answer": gold,
            "cleaned_solution": sol,
            "step_count": count_reasoning_steps(sol),
            "char_length": len(q),
            "pass_rate": pass_rate_lookup.get(pid, None),
            "difficulty_tier": "validation",
            "is_grpo_candidate": True,
            "selection_reason": "held_out_validation_prompt",
            "prompt": format_grpo_prompt(q),
        })

    val_candidates_df = pd.DataFrame(val_candidates)
    if len(val_candidates_df) > val_size:
        val_candidates_df = val_candidates_df.sort_values("problem_id").sample(
            n=val_size, random_state=seed
        ).reset_index(drop=True)
    else:
        val_candidates_df = val_candidates_df.sort_values("problem_id").reset_index(drop=True)

    # Re-verify test set wall on final outputs
    final_train_ids = set(candidate_df["problem_id"].tolist())
    final_val_ids = set(val_candidates_df["problem_id"].tolist())

    if final_train_ids.intersection(test_ids):
        raise TestSetWallError("Final GRPO train set leaks test problem IDs!")
    if final_val_ids.intersection(test_ids):
        raise TestSetWallError("Final GRPO val set leaks test problem IDs!")
    if final_train_ids.intersection(final_val_ids):
        raise TestSetWallError("Train and validation sets overlap in GRPO dataset!")

    # 6. Save Parquet and JSONL datasets
    train_parquet_path = out_dir / "grpo_train.parquet"
    train_jsonl_path = out_dir / "grpo_train.jsonl"
    val_parquet_path = out_dir / "grpo_val.parquet"
    val_jsonl_path = out_dir / "grpo_val.jsonl"

    candidate_df.to_parquet(train_parquet_path, index=False)
    candidate_df.to_json(train_jsonl_path, orient="records", lines=True, force_ascii=False)

    val_candidates_df.to_parquet(val_parquet_path, index=False)
    val_candidates_df.to_json(val_jsonl_path, orient="records", lines=True, force_ascii=False)

    # Compute hashes
    manifest = {
        "dataset_name": "grpo_gsm8k_filtered",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "files": {
            "grpo_train_parquet": {
                "path": str(train_parquet_path),
                "sha256": compute_sha256(train_parquet_path),
                "count": len(candidate_df),
            },
            "grpo_train_jsonl": {
                "path": str(train_jsonl_path),
                "sha256": compute_sha256(train_jsonl_path),
                "count": len(candidate_df),
            },
            "grpo_val_parquet": {
                "path": str(val_parquet_path),
                "sha256": compute_sha256(val_parquet_path),
                "count": len(val_candidates_df),
            },
            "grpo_val_jsonl": {
                "path": str(val_jsonl_path),
                "sha256": compute_sha256(val_jsonl_path),
                "count": len(val_candidates_df),
            },
        },
        "filtering_criteria": {
            "mode": mode,
            "min_pass_rate": min_pass,
            "max_pass_rate": max_pass,
            "min_steps": min_steps,
            "max_steps": max_steps,
            "max_prompt_chars": max_chars,
        },
        "statistics": {
            "total_profiled": len(profiles),
            "eligible_candidates": len(candidates),
            "sampled_train_count": len(candidate_df),
            "sampled_val_count": len(val_candidates_df),
            "tier_counts": tier_counts,
            "step_distribution": candidate_df["step_count"].value_counts().to_dict(),
            "mean_steps": float(candidate_df["step_count"].mean()),
            "mean_char_length": float(candidate_df["char_length"].mean()),
        },
        "test_set_wall": {
            "test_problems_checked": len(test_df),
            "exact_matches_test": 0,
            "forbidden_id_leaks": 0,
            "passed": True,
        },
        "frozen": True,
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "train_parquet_path": train_parquet_path,
        "train_jsonl_path": train_jsonl_path,
        "val_parquet_path": val_parquet_path,
        "val_jsonl_path": val_jsonl_path,
        "train_count": len(candidate_df),
        "val_count": len(val_candidates_df),
        "tier_counts": tier_counts,
    }
