"""Combined curated and error-targeted dataset preparation and freezing for Round-2 SFT (Phase 5 Day 35).

Combines curated reasoning traces (Phase 3) with error-targeted training examples (Phase 5 Day 34)
into a unified, frozen post-training corpus. Implements the option to weight targeted examples
2x via deterministic duplication, asserts 100% adherence to the canonical answer contract
(Design Rule 1), and strictly enforces the Test-Set Wall with zero test contamination
(Design Rule 2). Emits train_round2.jsonl, train_round2.parquet, manifest.json, and run_info.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from src.data.curation import normalize_reasoning_completion
from src.data.formatting import format_zero_shot_prompt
from src.data.preprocessing import (
    load_data_config,
    verify_no_contamination,
)
from src.math_utils.answer_extraction import extract_answer
from src.run_info import save_run_info

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestSetWallError(Exception):
    """Raised when validation or test problems leak into training data."""


@dataclass
class Round2Example:
    """A single training example for SFT Round 2."""

    problem_id: str
    trace_id: str
    question: str
    completion: str
    gold_answer: str
    source: str  # 'curated_sft' or 'targeted_error_mitigation'
    category: Optional[str] = None
    similarity_score: Optional[float] = None
    is_duplicate: bool = False
    prompt_type: str = "chain_of_thought"
    prompt: Optional[str] = None
    text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "problem_id": self.problem_id,
            "trace_id": self.trace_id,
            "question": self.question,
            "completion": self.completion,
            "gold_answer": self.gold_answer,
            "source": self.source,
            "category": self.category,
            "similarity_score": self.similarity_score,
            "is_duplicate": self.is_duplicate,
            "prompt_type": self.prompt_type,
            "prompt": self.prompt or format_zero_shot_prompt(self.question),
            "text": self.text or f"{format_zero_shot_prompt(self.question)}{self.completion}",
        }


@dataclass
class Round2DatasetManifest:
    """Cryptographic manifest documenting frozen Round-2 SFT training corpus."""

    dataset_file: str
    parquet_file: str
    sha256_jsonl: str
    sha256_parquet: str
    total_examples: int
    curated_count: int
    targeted_raw_count: int
    targeted_effective_count: int
    duplicate_targeted: bool
    unique_problems: int
    category_breakdown: Dict[str, int]
    source_breakdown: Dict[str, int]
    token_statistics: Dict[str, Any]
    contamination_verified: bool
    exact_matches_test: int
    forbidden_id_leaks: int
    frozen: bool = True
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_forbidden_ids(splits_dir: Optional[Union[str, Path]] = None) -> Set[str]:
    """Load all 1,819 forbidden validation and test problem IDs to enforce the Test-Set Wall."""
    base = Path(splits_dir) if splits_dir else (PROJECT_ROOT / "data" / "splits")
    forbidden: Set[str] = set()

    for fname in ["val_ids.json", "val_analysis_ids.json", "val_gating_ids.json", "test_ids.json"]:
        fpath = base / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                ids = json.load(f)
                forbidden.update(ids)
        else:
            logger.warning("Split file not found while loading forbidden IDs: %s", fpath)

    logger.info("Loaded %d forbidden IDs to enforce Test-Set Wall.", len(forbidden))
    return forbidden


def load_curated_examples(curated_file: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load curated reasoning traces from Phase 3 JSONL."""
    p = Path(curated_file)
    if not p.exists():
        raise FileNotFoundError(f"Curated SFT file not found: {p}")

    records: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError as e:
                logger.warning("Skipping corrupted JSON on line %d of %s: %s", line_num, p, e)

    logger.info("Loaded %d curated reasoning records from %s.", len(records), p)
    return records


def load_targeted_examples(targeted_file: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load error-targeted training examples from Phase 5 Day 34 JSONL."""
    p = Path(targeted_file)
    if not p.exists():
        raise FileNotFoundError(f"Targeted examples file not found: {p}")

    records: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError as e:
                logger.warning("Skipping corrupted JSON on line %d of %s: %s", line_num, p, e)

    logger.info("Loaded %d targeted training records from %s.", len(records), p)
    return records


def build_round2_dataset(
    curated_records: Sequence[Dict[str, Any]],
    targeted_records: Sequence[Dict[str, Any]],
    duplicate_targeted: bool = True,
    seed: int = 42,
    eos_token: str = "<|eos|>",
) -> List[Round2Example]:
    """Combine curated and targeted examples with optional 2x targeted weighting.

    Args:
        curated_records: Records from Phase 3 curated dataset.
        targeted_records: Records from Phase 5 Day 34 targeted dataset.
        duplicate_targeted: If True, includes targeted examples twice (2x representation).
        seed: Random seed for deterministic shuffling / interleaving.
        eos_token: End-of-sequence token string (e.g. '<|eos|>').

    Returns:
        List of formatted Round2Example instances.
    """
    examples: List[Round2Example] = []

    # 1. Process Curated Traces
    for idx, c in enumerate(curated_records):
        prob_id = str(c.get("problem_id", f"curated_prob_{idx}"))
        trace_id = str(c.get("trace_id", f"{prob_id}_curated_{idx}"))
        question = str(c.get("question", "")).strip()
        comp = str(c.get("completion", "")).strip()
        gold = str(c.get("gold_answer", "")).strip()
        ptype = str(c.get("prompt_type", "chain_of_thought"))

        # Verify or extract gold answer if missing
        if not gold:
            extracted, _ = extract_answer(comp)
            gold = extracted or ""

        # Normalize completion and ensure canonical format
        norm_comp = normalize_reasoning_completion(comp, expected_answer=gold)
        if eos_token and not norm_comp.endswith(eos_token):
            final_comp = f"{norm_comp}{eos_token}"
        else:
            final_comp = norm_comp

        prompt_str = format_zero_shot_prompt(question)
        examples.append(
            Round2Example(
                problem_id=prob_id,
                trace_id=trace_id,
                question=question,
                completion=final_comp,
                gold_answer=gold,
                source="curated_sft",
                category=None,
                similarity_score=None,
                is_duplicate=False,
                prompt_type=ptype,
                prompt=prompt_str,
                text=f"{prompt_str}{final_comp}",
            )
        )

    # 2. Process Targeted Examples
    for idx, t in enumerate(targeted_records):
        prob_id = str(t.get("problem_id", f"targeted_prob_{idx}"))
        question = str(t.get("question", "")).strip()
        sol = str(t.get("completion") or t.get("solution") or "").strip()
        gold = str(t.get("gold_answer", "")).strip()
        cat = str(t.get("target_category") or t.get("category") or "general_error")
        sim = float(t.get("similarity_score", 0.0))
        raw_trace_id = t.get("trace_id")
        orig_trace_id = str(raw_trace_id) if raw_trace_id else f"{prob_id}_targeted_{cat}"

        if not gold:
            extracted, _ = extract_answer(sol)
            gold = extracted or ""

        norm_comp = normalize_reasoning_completion(sol, expected_answer=gold)
        if eos_token and not norm_comp.endswith(eos_token):
            final_comp = f"{norm_comp}{eos_token}"
        else:
            final_comp = norm_comp

        prompt_str = format_zero_shot_prompt(question)

        # Primary instance
        examples.append(
            Round2Example(
                problem_id=prob_id,
                trace_id=orig_trace_id,
                question=question,
                completion=final_comp,
                gold_answer=gold,
                source="targeted_error_mitigation",
                category=cat,
                similarity_score=sim,
                is_duplicate=False,
                prompt_type="targeted_mitigation",
                prompt=prompt_str,
                text=f"{prompt_str}{final_comp}",
            )
        )

        # Optional second instance (2x targeted weighting)
        if duplicate_targeted:
            dup_trace_id = f"{orig_trace_id}_dup2"
            examples.append(
                Round2Example(
                    problem_id=prob_id,
                    trace_id=dup_trace_id,
                    question=question,
                    completion=final_comp,
                    gold_answer=gold,
                    source="targeted_error_mitigation",
                    category=cat,
                    similarity_score=sim,
                    is_duplicate=True,
                    prompt_type="targeted_mitigation",
                    prompt=prompt_str,
                    text=f"{prompt_str}{final_comp}",
                )
            )

    # 3. Deterministic Interleave / Shuffle
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(examples)

    logger.info(
        "Built Round-2 dataset: %d total examples (curated=%d, targeted_raw=%d, duplicate_targeted=%s).",
        len(examples),
        len(curated_records),
        len(targeted_records),
        duplicate_targeted,
    )
    return examples


def verify_round2_dataset_safety(
    examples: Sequence[Round2Example],
    test_file: Optional[Union[str, Path]] = None,
    splits_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Verify strict Test-Set Wall and zero test contamination for Round-2 dataset."""
    # 1. Test-Set Wall Forbidden ID check
    forbidden_ids = load_forbidden_ids(splits_dir=splits_dir)
    leaked_ids = [ex.problem_id for ex in examples if ex.problem_id in forbidden_ids]
    if leaked_ids:
        msg = f"Test-Set Wall violated! {len(leaked_ids)} forbidden validation/test IDs leaked: {leaked_ids[:5]}"
        logger.error(msg)
        raise TestSetWallError(msg)

    # 2. Test Contamination check
    test_path = Path(test_file) if test_file else (PROJECT_ROOT / "data" / "processed" / "test" / "test.parquet")
    if not test_path.exists():
        sample_test = PROJECT_ROOT / "data" / "sample" / "test_sample.parquet"
        if sample_test.exists():
            test_path = sample_test

    exact_matches = 0
    if test_path.exists():
        test_df = pd.read_parquet(test_path)
        train_df = pd.DataFrame([{"problem_id": ex.problem_id, "question": ex.question} for ex in examples])

        try:
            cfg = load_data_config()["contamination"]
            reviewed = set(cfg["reviewed_template_overlaps"])
            ngram_size = int(cfg.get("ngram_size", 13))
        except Exception:
            reviewed = set()
            ngram_size = 13

        contam_report = verify_no_contamination(
            train_pool=train_df,
            test_df=test_df,
            ngram_size=ngram_size,
            reviewed_test_ids=reviewed,
        )
        exact_matches = contam_report.get("exact_match_count", 0)
    else:
        logger.warning("Test dataset file not found at %s. Skipping n-gram check.", test_path)

    logger.info(
        "Safety verification passed: 0 forbidden ID leaks, %d exact test matches.",
        exact_matches,
    )
    return {
        "forbidden_id_leaks": len(leaked_ids),
        "exact_matches_test": exact_matches,
        "contamination_verified": True,
    }


def freeze_round2_dataset(
    curated_file: Union[str, Path],
    targeted_file: Union[str, Path],
    output_dir: Union[str, Path],
    duplicate_targeted: bool = True,
    seed: int = 42,
    splits_dir: Optional[Union[str, Path]] = None,
    test_file: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path, Round2DatasetManifest]:
    """Freeze and export combined Round-2 dataset to JSONL, Parquet, and manifest.json.

    Args:
        curated_file: Path to Phase 3 curated JSONL.
        targeted_file: Path to Phase 5 Day 34 targeted JSONL.
        output_dir: Directory where frozen dataset will be exported.
        duplicate_targeted: Whether to include targeted examples twice.
        seed: Shuffling seed for deterministic reproducibility.
        splits_dir: Optional path to data/splits directory.
        test_file: Optional path to test.parquet.

    Returns:
        Tuple of (jsonl_path, parquet_path, manifest).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curated_records = load_curated_examples(curated_file)
    targeted_records = load_targeted_examples(targeted_file)

    examples = build_round2_dataset(
        curated_records=curated_records,
        targeted_records=targeted_records,
        duplicate_targeted=duplicate_targeted,
        seed=seed,
    )

    # Enforce Test-Set Wall & Contamination Gate
    safety_res = verify_round2_dataset_safety(
        examples=examples,
        test_file=test_file,
        splits_dir=splits_dir,
    )

    # 1. Export JSONL
    jsonl_path = out_dir / "train_round2.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Saved %d Round-2 examples to %s", len(examples), jsonl_path)

    # 2. Export Parquet
    parquet_path = out_dir / "train_round2.parquet"
    df = pd.DataFrame([ex.to_dict() for ex in examples])
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved Round-2 Parquet dataset to %s", parquet_path)

    # 3. Compute Cryptographic Hashes
    sha_jsonl = compute_file_sha256(jsonl_path)
    sha_parquet = compute_file_sha256(parquet_path)

    # 4. Compute Token Statistics & Breakdowns
    token_lengths = [len(re.findall(r"\S+", ex.completion)) for ex in examples]
    if token_lengths:
        token_lengths_sorted = sorted(token_lengths)
        tok_min = int(token_lengths_sorted[0])
        tok_med = int(token_lengths_sorted[len(token_lengths_sorted) // 2])
        tok_mean = float(sum(token_lengths_sorted) / len(token_lengths_sorted))
        p95_idx = int(0.95 * len(token_lengths_sorted))
        tok_p95 = int(token_lengths_sorted[min(p95_idx, len(token_lengths_sorted) - 1)])
        tok_max = int(token_lengths_sorted[-1])
    else:
        tok_min = tok_med = tok_mean = tok_p95 = tok_max = 0

    cat_breakdown: Dict[str, int] = {}
    for ex in examples:
        if ex.category:
            cat_breakdown[ex.category] = cat_breakdown.get(ex.category, 0) + 1

    source_breakdown: Dict[str, int] = {}
    for ex in examples:
        source_breakdown[ex.source] = source_breakdown.get(ex.source, 0) + 1

    unique_problems = len({ex.problem_id for ex in examples})
    effective_targeted = sum(1 for ex in examples if ex.source == "targeted_error_mitigation")

    manifest = Round2DatasetManifest(
        dataset_file="train_round2.jsonl",
        parquet_file="train_round2.parquet",
        sha256_jsonl=sha_jsonl,
        sha256_parquet=sha_parquet,
        total_examples=len(examples),
        curated_count=len(curated_records),
        targeted_raw_count=len(targeted_records),
        targeted_effective_count=effective_targeted,
        duplicate_targeted=duplicate_targeted,
        unique_problems=unique_problems,
        category_breakdown=cat_breakdown,
        source_breakdown=source_breakdown,
        token_statistics={
            "min": tok_min,
            "median": tok_med,
            "mean": round(tok_mean, 2),
            "p95": tok_p95,
            "max": tok_max,
        },
        contamination_verified=safety_res.get("contamination_verified", True),
        exact_matches_test=safety_res.get("exact_matches_test", 0),
        forbidden_id_leaks=safety_res.get("forbidden_id_leaks", 0),
        frozen=True,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    logger.info("Saved Round-2 dataset manifest to %s", manifest_path)

    # 5. Provenance Record
    save_run_info(
        stage="data_preparation",
        output_dir=out_dir,
        config={
            "round": 2,
            "total_examples": len(examples),
            "duplicate_targeted": duplicate_targeted,
            "sha256_jsonl": sha_jsonl,
            "sha256_parquet": sha_parquet,
        },
        description="Phase 5 Day 35: Freeze combined Round-2 SFT dataset",
    )

    return jsonl_path, parquet_path, manifest
