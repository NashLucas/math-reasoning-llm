"""Safety prompt set assembly for Llama Guard screening (Day 15, reused in Phase 7).

Builds a fixed, seeded set of harmful red-teaming prompts and benign math questions:
- Harmful: stratified sample from JailbreakBench JBB-Behaviors (an equal number per category).
- Benign: GSM8K questions drawn from the training split only (never validation or test).

Each record: {safety_id, prompt, label ('harmful' | 'benign'), category, source}.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

HARMFUL_SOURCE = "JailbreakBench/JBB-Behaviors"
HARMFUL_CONFIG = "behaviors"
HARMFUL_SPLIT = "harmful"
BENIGN_SOURCE = "openai/gsm8k (train split)"


def make_safety_id(label: str, prompt: str) -> str:
    """Stable ID from the label and a hash of the prompt text."""
    digest = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:16]
    return f"safety_{label}_{digest}"


def sample_harmful_prompts(
    rows: Sequence[Dict[str, Any]],
    n_total: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Stratified sample of harmful behaviors, an equal share from each category.

    Args:
        rows: JBB-Behaviors rows with at least 'Goal' and 'Category'.
        n_total: Target number of prompts; must divide evenly across categories.
        seed: Sampling seed.
    """
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["Category"]].append(row)

    categories = sorted(by_category)
    if not categories:
        return []
    if n_total % len(categories) != 0:
        raise ValueError(f"n_total={n_total} does not divide evenly across {len(categories)} categories")
    per_category = n_total // len(categories)

    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    for cat in categories:
        pool = sorted(by_category[cat], key=lambda r: r["Goal"])
        if len(pool) < per_category:
            raise ValueError(f"Category '{cat}' has {len(pool)} rows, fewer than {per_category} requested")
        for row in rng.sample(pool, per_category):
            prompt = row["Goal"].strip()
            records.append({
                "safety_id": make_safety_id("harmful", prompt),
                "prompt": prompt,
                "label": "harmful",
                "category": cat,
                "source": HARMFUL_SOURCE,
            })
    return records


def sample_benign_prompts(
    train_rows: Sequence[Dict[str, Any]],
    n_total: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample benign GSM8K questions from the training split.

    Rows from any split other than 'train' are rejected so the validation and test walls hold.
    """
    for row in train_rows:
        split = row.get("split", "train")
        if split != "train":
            raise ValueError(f"Benign prompts must come from the train split, got split='{split}'")

    pool = sorted(train_rows, key=lambda r: r["problem_id"])
    if len(pool) < n_total:
        raise ValueError(f"Train pool has {len(pool)} rows, fewer than {n_total} requested")

    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    for row in rng.sample(pool, n_total):
        prompt = row["question"].strip()
        records.append({
            "safety_id": make_safety_id("benign", prompt),
            "prompt": prompt,
            "label": "benign",
            "category": "math_word_problem",
            "source": BENIGN_SOURCE,
            "problem_id": row["problem_id"],
            "gold_answer": str(row.get("gold_answer", "")),
        })
    return records


def build_safety_prompt_set(
    harmful_rows: Sequence[Dict[str, Any]],
    train_rows: Sequence[Dict[str, Any]],
    n_harmful: int = 50,
    n_benign: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Combine harmful and benign samples into one ordered prompt set."""
    return sample_harmful_prompts(harmful_rows, n_harmful, seed) + sample_benign_prompts(
        train_rows, n_benign, seed
    )


def save_safety_prompt_set(
    records: Sequence[Dict[str, Any]],
    output_dir: Union[str, Path],
    seed: int = 42,
) -> Dict[str, Any]:
    """Write safety_prompts.jsonl and a manifest with counts and file hash."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "safety_prompts.jsonl"

    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    content = "\n".join(lines) + "\n"
    jsonl_path.write_text(content, encoding="utf-8")

    by_category: Dict[str, int] = defaultdict(int)
    for r in records:
        if r["label"] == "harmful":
            by_category[r["category"]] += 1

    manifest = {
        "file": jsonl_path.name,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "seed": seed,
        "total": len(records),
        "harmful": sum(1 for r in records if r["label"] == "harmful"),
        "benign": sum(1 for r in records if r["label"] == "benign"),
        "harmful_source": f"{HARMFUL_SOURCE} ({HARMFUL_CONFIG}/{HARMFUL_SPLIT})",
        "benign_source": BENIGN_SOURCE,
        "harmful_by_category": dict(sorted(by_category.items())),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_safety_prompt_set(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load safety prompt records from a JSONL file."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
