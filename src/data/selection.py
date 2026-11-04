"""Targeted example selection for error mitigation and Round-2 SFT (Phase 5 Day 34).

Selects training problems most similar to observed error questions in each category/topic,
strictly enforces the Test-Set Wall (zero validation/test overlap), and attaches each
problem's curated reasoning trace (or its cleaned gold solution if none exists).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.data.preprocessing import frozen_splits_dir
from src.evaluation.error_analysis import (
    ALL_ERROR_CATEGORIES,
    ErrorClassification,
    QuestionEmbedder,
)
from src.evaluation.evaluator import TestSetWallError
from src.run_info import save_run_info

logger = logging.getLogger(__name__)


@dataclass
class TargetedExample:
    """A training problem selected to mitigate a specific category of error."""

    problem_id: str
    question: str
    solution: str
    gold_answer: str
    target_category: str
    similarity_score: float
    source: str  # 'curated_trace' or 'cleaned_gold_solution'
    trace_id: Optional[str] = None
    prompt_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "question": self.question,
            "solution": self.solution,
            "gold_answer": self.gold_answer,
            "target_category": self.target_category,
            "similarity_score": round(self.similarity_score, 4),
            "source": self.source,
            "trace_id": self.trace_id,
            "prompt_type": self.prompt_type,
        }


@dataclass
class TargetedSelectionReport:
    """Metadata report and manifest for targeted example selection."""

    total_selected: int
    examples_per_category: Dict[str, int]
    sources_count: Dict[str, int]
    embedding_backend: str
    output_dir: str
    target_categories: List[str]
    timestamp_utc: str
    contamination_check: str = "PASSED (0 validation/test problems)"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_forbidden_ids(splits_dir: Optional[Path] = None) -> Set[str]:
    """Load all problem IDs belonging to validation and test splits.

    Strictly enforces Design Rule 2 (The Test-Set Wall).
    """
    if splits_dir is None:
        try:
            splits_dir = frozen_splits_dir()
        except Exception:
            splits_dir = Path("data/splits")

    forbidden: Set[str] = set()
    split_files = [
        "val_ids.json",
        "val_analysis_ids.json",
        "val_gating_ids.json",
        "test_ids.json",
    ]
    for fname in split_files:
        p = splits_dir / fname
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    forbidden.update(data)
                elif isinstance(data, dict):
                    forbidden.update(data.get("ids", []))
            except Exception as e:
                logger.warning("Could not read split file %s: %s", p, e)

    return forbidden


def load_curated_traces_index(curated_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Index curated traces by problem_id for fast lookup."""
    if curated_path is None:
        curated_path = Path("data/synthetic/curated/traces_curated_sft.jsonl")

    index: Dict[str, Dict[str, Any]] = {}
    if not curated_path.exists():
        logger.warning("Curated traces file %s not found; will use gold solutions.", curated_path)
        return index

    with open(curated_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                pid = item.get("problem_id")
                # Store first valid trace per problem
                if pid and pid not in index:
                    index[pid] = item

    logger.info("Indexed %d curated traces from %s", len(index), curated_path)
    return index


def load_training_pool(
    train_pool_path: Optional[Path] = None,
    forbidden_ids: Optional[Set[str]] = None,
    use_mock: bool = False,
) -> pd.DataFrame:
    """Load the candidate training pool while strictly filtering out validation & test problems."""
    if train_pool_path is None:
        train_pool_path = Path("data/processed/train/train.parquet")

    # Enforce Test-Set Wall on provided path
    if "test" in str(train_pool_path).lower() and "train" not in str(train_pool_path).lower():
        raise TestSetWallError(
            f"TEST-SET WALL VIOLATION: Cannot load test data into training pool: {train_pool_path}"
        )

    if not train_pool_path.exists():
        if use_mock:
            sample_train = Path("data/sample/train_sample.parquet")
            if sample_train.exists():
                logger.warning(
                    "Training pool %s not found; mock run falling back to %s.",
                    train_pool_path,
                    sample_train,
                )
                train_pool_path = sample_train
            else:
                raise FileNotFoundError(f"Neither {train_pool_path} nor {sample_train} exists.")
        else:
            raise FileNotFoundError(f"Training pool file not found: {train_pool_path}")

    df = pd.read_parquet(train_pool_path)
    initial_count = len(df)

    if forbidden_ids:
        # Strictly exclude validation and test IDs
        df = df[~df["problem_id"].isin(forbidden_ids)].copy()
        removed = initial_count - len(df)
        if removed > 0:
            logger.info("Filtered %d forbidden validation/test problems from training pool.", removed)

    # Validate essential columns
    for col in ["problem_id", "question"]:
        if col not in df.columns:
            raise ValueError(f"Training pool missing required column: '{col}'")

    if "cleaned_solution" not in df.columns and "answer" in df.columns:
        # Clean answer if cleaned_solution not present
        df["cleaned_solution"] = df["answer"].apply(lambda a: re.sub(r"<<.*?>>", "", str(a)))

    return df


class TargetedSelector:
    """Selects targeted training examples similar to error questions."""

    def __init__(
        self,
        embedder: Optional[QuestionEmbedder] = None,
        train_pool_path: Optional[Path] = None,
        curated_traces_path: Optional[Path] = None,
        splits_dir: Optional[Path] = None,
        min_per_category: int = 20,
        max_per_category: int = 50,
        default_per_category: int = 30,
        use_mock: bool = False,
    ):
        self.embedder = embedder or QuestionEmbedder(use_fallback=use_mock)
        self.train_pool_path = train_pool_path
        self.curated_traces_path = curated_traces_path
        self.splits_dir = splits_dir
        self.min_per_category = min_per_category
        self.max_per_category = max_per_category
        self.default_per_category = default_per_category
        self.use_mock = use_mock

        # Load forbidden IDs
        self.forbidden_ids = load_forbidden_ids(self.splits_dir)
        logger.info("Loaded %d forbidden IDs to enforce Test-Set Wall.", len(self.forbidden_ids))

        # Index curated traces
        self.curated_index = load_curated_traces_index(self.curated_traces_path)

        # Load clean training pool
        self.train_df = load_training_pool(
            self.train_pool_path,
            forbidden_ids=self.forbidden_ids,
            use_mock=self.use_mock,
        )
        logger.info("Active training pool size: %d candidate problems.", len(self.train_df))

        # Precompute training pool embeddings
        self._train_embs: Optional[np.ndarray] = None
        self._backend_name: str = self.embedder.backend

    def get_train_embeddings(self) -> Tuple[np.ndarray, str]:
        """Compute or retrieve cached training pool embeddings."""
        if self._train_embs is None:
            questions = self.train_df["question"].tolist()
            logger.info("Computing embeddings for %d training questions...", len(questions))
            self._train_embs, self._backend_name = self.embedder.embed(questions)
        return self._train_embs, self._backend_name

    def select_for_errors(
        self,
        error_records: Sequence[Union[ErrorClassification, Dict[str, Any]]],
        target_per_category: Optional[int] = None,
        categories: Optional[Sequence[str]] = None,
        deduplicate_across_categories: bool = True,
    ) -> List[TargetedExample]:
        """Select top similar training examples for each error category.

        Args:
            error_records: Sequence of error records or ErrorClassification objects.
            target_per_category: Number of training examples per category (clamped to 20-50).
            categories: Optional subset of categories to target. If None, targets all active categories.
            deduplicate_across_categories: Whether each selected problem ID must be globally unique.

        Returns:
            List of TargetedExample instances.
        """
        # 1. Group error questions by category
        errors_by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for r in error_records:
            if isinstance(r, ErrorClassification):
                if not r.is_error:
                    continue
                cat = r.category.value if r.category else "other"
                item = r.to_dict()
            elif isinstance(r, dict):
                if not r.get("is_error", True):
                    continue
                cat_val = r.get("category") or r.get("gold_category") or r.get("manual_label") or "other"
                cat = cat_val.value if hasattr(cat_val, "value") else str(cat_val)
                item = r
            else:
                continue

            errors_by_cat.setdefault(cat, []).append(item)

        if not errors_by_cat:
            logger.warning("No error questions provided for targeted selection.")
            return []

        active_categories = list(categories) if categories else list(errors_by_cat.keys())
        # Filter to categories with actual errors
        active_categories = [c for c in active_categories if c in errors_by_cat and errors_by_cat[c]]

        logger.info(
            "Targeting %d active error categories: %s",
            len(active_categories),
            active_categories,
        )

        train_embs, _ = self.get_train_embeddings()
        num_train = len(self.train_df)

        # Determine target count per category
        k = target_per_category or self.default_per_category
        # Clamp between min and max (or pool size if small)
        if num_train < self.min_per_category:
            k = num_train
        else:
            k = max(self.min_per_category, min(self.max_per_category, k))
            k = min(k, num_train)

        selected_examples: List[TargetedExample] = []
        selected_problem_ids: Set[str] = set()

        for cat in active_categories:
            cat_errors = errors_by_cat[cat]
            error_questions = [e.get("question", "") for e in cat_errors]

            logger.info(
                "Embedding %d error questions for category '%s'...",
                len(error_questions),
                cat,
            )
            cat_embs, _ = self.embedder.embed(error_questions)

            # Ensure dimensional alignment between train and error embeddings
            if cat_embs.shape[1] != train_embs.shape[1]:
                aligned = np.zeros((cat_embs.shape[0], train_embs.shape[1]), dtype=np.float32)
                common_d = min(cat_embs.shape[1], train_embs.shape[1])
                aligned[:, :common_d] = cat_embs[:, :common_d]
                cat_embs = aligned

            # Compute category centroid and normalize
            centroid = np.mean(cat_embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            # Compute cosine similarity between all training questions and category centroid
            sim_scores = np.dot(train_embs, centroid)

            # Sort indices descending by similarity
            sorted_indices = np.argsort(sim_scores)[::-1]

            cat_selected = 0
            for idx in sorted_indices:
                if cat_selected >= k:
                    break

                row = self.train_df.iloc[int(idx)]
                pid = str(row["problem_id"])

                # Safety: verify Test-Set Wall
                if pid in self.forbidden_ids:
                    continue

                if deduplicate_across_categories and pid in selected_problem_ids:
                    continue

                sim = float(sim_scores[idx])

                # Check curated traces first, fall back to cleaned gold solution
                if pid in self.curated_index:
                    trace = self.curated_index[pid]
                    sol_text = trace.get("completion") or trace.get("solution") or row.get("cleaned_solution", "")
                    src = "curated_trace"
                    trace_id = trace.get("trace_id")
                    prompt_type = trace.get("prompt_type")
                else:
                    sol_text = row.get("cleaned_solution", "")
                    src = "cleaned_gold_solution"
                    trace_id = None
                    prompt_type = None

                # Guarantee canonical format ending (Design Rule 1)
                gold_ans = str(row.get("gold_answer", ""))
                if "####" not in sol_text and gold_ans:
                    sol_text = sol_text.rstrip() + f"\n#### {gold_ans}"

                example = TargetedExample(
                    problem_id=pid,
                    question=str(row["question"]),
                    solution=sol_text,
                    gold_answer=gold_ans,
                    target_category=cat,
                    similarity_score=sim,
                    source=src,
                    trace_id=trace_id,
                    prompt_type=prompt_type,
                )
                selected_examples.append(example)
                selected_problem_ids.add(pid)
                cat_selected += 1

            logger.info("Selected %d training examples for category '%s'.", cat_selected, cat)

        return selected_examples

    def export_targeted_dataset(
        self,
        examples: Sequence[TargetedExample],
        output_dir: Path,
    ) -> TargetedSelectionReport:
        """Export targeted examples to JSONL, Parquet, and manifest.json."""
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "targeted_examples.jsonl"
        parquet_path = output_dir / "targeted_examples.parquet"
        manifest_path = output_dir / "manifest.json"

        # 1. Write JSONL
        records = [ex.to_dict() for ex in examples]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Saved %d targeted examples to: %s", len(records), jsonl_path)

        # 2. Write Parquet
        df_targeted = pd.DataFrame(records)
        df_targeted.to_parquet(parquet_path, index=False)
        logger.info("Saved targeted Parquet dataset to: %s", parquet_path)

        # 3. Compute Manifest Counts
        cat_counts: Dict[str, int] = {}
        src_counts: Dict[str, int] = {}
        for ex in examples:
            cat_counts[ex.target_category] = cat_counts.get(ex.target_category, 0) + 1
            src_counts[ex.source] = src_counts.get(ex.source, 0) + 1

        report = TargetedSelectionReport(
            total_selected=len(examples),
            examples_per_category=cat_counts,
            sources_count=src_counts,
            embedding_backend=self._backend_name,
            output_dir=str(output_dir),
            target_categories=sorted(cat_counts.keys()),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Saved targeted dataset manifest to: %s", manifest_path)

        # Save run provenance
        save_run_info(
            stage="sft_targeted",
            output_dir=output_dir,
            config={
                "day": 34,
                "task": "targeted_example_selection",
                "total_selected": len(examples),
                "examples_per_category": cat_counts,
                "sources_count": src_counts,
                "output_dir": str(output_dir),
            },
        )

        return report
