#!/usr/bin/env python3
"""Execution runner for Phase 5 Day 34: Targeted Example Selection for SFT Round 2.

Selects training problems most similar to error questions for each active category,
enforcing:
1. Design Rule 2 (The Test-Set Wall): zero validation and zero test set problems are returned.
2. Attaches curated reasoning trace (or cleaned gold solution if none exists).
3. Selects 20-50 problems per error category.
4. Exports targeted dataset to 'data/targeted/' as JSONL and Parquet with manifest.json.
5. Saves run provenance record.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.selection import TargetedSelector  # noqa: E402
from src.evaluation.evaluator import TestSetWallError  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_day34_selection")


def load_input_errors(error_file: Path, sample_mode: bool = False) -> List[Dict[str, Any]]:
    """Load error records from error analysis JSON or labeled JSONL."""
    if not error_file.exists():
        if sample_mode:
            sample_candidate = PROJECT_ROOT / "data" / "sample" / "labeled_30_errors.jsonl"
            if sample_candidate.exists():
                logger.warning("Error file %s not found; falling back to %s.", error_file, sample_candidate)
                error_file = sample_candidate
            else:
                raise FileNotFoundError(f"Neither {error_file} nor {sample_candidate} exists.")
        else:
            raise FileNotFoundError(f"Error file not found: {error_file}")

    suffix = error_file.suffix.lower()
    records: List[Dict[str, Any]] = []

    if suffix == ".jsonl":
        with open(error_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if item.get("is_error", True):
                        if "is_error" not in item:
                            item["is_error"] = True
                        if "category" not in item and "gold_category" in item:
                            item["category"] = item["gold_category"]
                        elif "category" not in item and "manual_label" in item:
                            item["category"] = item["manual_label"]
                        records.append(item)
    elif suffix == ".json":
        data = json.loads(error_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if item.get("is_error", True):
                    records.append(item)
        elif isinstance(data, dict):
            classifications = data.get("classifications", [])
            for item in classifications:
                if item.get("is_error", False):
                    records.append(item)
            if not records and "records" in data:
                for item in data["records"]:
                    if not item.get("is_correct", False):
                        records.append(item)

    logger.info("Loaded %d error records from %s", len(records), error_file)
    return records


def run_day34_pipeline(
    error_file: Path,
    train_pool_path: Optional[Path] = None,
    curated_traces_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    target_per_category: int = 30,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_mock: bool = False,
    sample_mode: bool = False,
) -> Dict[str, Any]:
    """Execute Day 34 targeted example selection pipeline."""
    logger.info("==================================================")
    logger.info("Phase 5 Day 34: Targeted Example Selection for SFT Round 2")
    logger.info("Error File:           %s", error_file)
    logger.info("Train Pool Path:      %s", train_pool_path)
    logger.info("Curated Traces Path:  %s", curated_traces_path)
    logger.info("Output Directory:     %s", output_dir)
    logger.info("Target Per Category:  %d", target_per_category)
    logger.info("Embedding Model:      %s (Mock: %s)", embedding_model, use_mock or sample_mode)
    logger.info("==================================================")

    # 1. Enforce Test-Set Wall
    if "test" in str(error_file).lower() and "val" not in str(error_file).lower():
        raise TestSetWallError(
            "CRITICAL: Targeted selection cannot be derived from test set errors! "
            "Validation analysis errors only."
        )

    out_path = output_dir or (PROJECT_ROOT / "data" / "targeted")

    # 2. Load Error Records
    errors = load_input_errors(error_file, sample_mode=(sample_mode or use_mock))
    if not errors and (sample_mode or use_mock):
        sample_candidate = PROJECT_ROOT / "data" / "sample" / "labeled_30_errors.jsonl"
        errors = load_input_errors(sample_candidate, sample_mode=True)

    if not errors:
        raise ValueError(f"No error records found to target from {error_file}.")

    # 3. Instantiate TargetedSelector
    selector = TargetedSelector(
        train_pool_path=train_pool_path,
        curated_traces_path=curated_traces_path,
        default_per_category=target_per_category,
        use_mock=(use_mock or sample_mode),
    )

    # 4. Select Similar Training Examples
    targeted_examples = selector.select_for_errors(
        error_records=errors,
        target_per_category=target_per_category,
    )

    if not targeted_examples:
        raise RuntimeError("Targeted selection produced 0 examples.")

    # 5. Export Targeted Dataset
    report = selector.export_targeted_dataset(
        examples=targeted_examples,
        output_dir=out_path,
    )

    return {
        "passed": True,
        "total_selected": report.total_selected,
        "examples_per_category": report.examples_per_category,
        "sources_count": report.sources_count,
        "manifest": report.to_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select targeted training examples for Round-2 SFT mitigation (Phase 5 Day 34)."
    )
    parser.add_argument(
        "--error_file",
        type=str,
        default="reports/error_analysis_sft.json",
        help="Path to classified errors JSON or JSONL file.",
    )
    parser.add_argument(
        "--train_pool",
        type=str,
        default="data/processed/train/train.parquet",
        help="Path to candidate training pool Parquet.",
    )
    parser.add_argument(
        "--curated_traces",
        type=str,
        default="data/synthetic/curated/traces_curated_sft.jsonl",
        help="Path to Phase 3 curated reasoning traces JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/targeted",
        help="Directory where targeted dataset will be written.",
    )
    parser.add_argument(
        "--target_per_category",
        type=int,
        default=30,
        help="Number of training examples per category (20-50, default: 30).",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model checkpoint for semantic embeddings.",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use deterministic mock/TF-IDF embedder instead of loading transformer model.",
    )
    parser.add_argument(
        "--sample_mode",
        action="store_true",
        help="Run quickly on sample data in mock mode for CI / testing.",
    )
    return parser


def main(args_list: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)

    error_path = Path(args.error_file)
    train_pool = Path(args.train_pool)
    curated_traces = Path(args.curated_traces)
    output_dir = Path(args.output_dir)

    try:
        res = run_day34_pipeline(
            error_file=error_path,
            train_pool_path=train_pool,
            curated_traces_path=curated_traces,
            output_dir=output_dir,
            target_per_category=args.target_per_category,
            embedding_model=args.embedding_model,
            use_mock=args.use_mock or args.sample_mode,
            sample_mode=args.sample_mode,
        )
        logger.info(
            "Targeted example selection completed successfully: %d examples selected.",
            res["total_selected"],
        )
        return 0 if res["passed"] else 1
    except Exception as e:
        logger.error("Day 34 targeted selection failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
