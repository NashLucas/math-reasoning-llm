#!/usr/bin/env python3
"""CLI runner for mathematical error analysis and taxonomy classification (Phase 5 Day 31).

Evaluates model reasoning errors against a 7-category taxonomy:
1. Rule-based checks (evaluated first):
   - parse_error: Empty, unparseable, or missing numeric answer.
   - truncation: Output hit maximum generation tokens or cut off abruptly.
2. Semantic reasoning errors (evaluated via 7B instruct or heuristic classifier):
   - arithmetic_slip: Correct mathematical setup, calculation mistake.
   - misread_question: Misread numbers, entities, or solved for wrong quantity.
   - wrong_setup: Incorrect formula, equation modeling, or conceptual error.
   - incomplete_reasoning: Stopped midway, omitted steps, or guessed answer.
   - other: Degenerate repetition, language mixing, or incoherent failure.
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

import pandas as pd  # noqa: E402

from src.config import ConfigManager  # noqa: E402
from src.data.selection import TargetedSelector  # noqa: E402
from src.evaluation.error_analysis import (  # noqa: E402
    ALL_ERROR_CATEGORIES,
    ErrorClassifier,
    ErrorClusterer,
    QuestionEmbedder,
    compute_error_distribution,
    generate_error_markdown_report,
)
from src.run_info import save_run_info  # noqa: E402

try:
    from src.model import patch_peft_torchao_compatibility
    patch_peft_torchao_compatibility()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_input_path(input_path: Path, profile: Optional[str] = None) -> Path:
    """Resolve input file looking across local workspace, Google Drive profile, and checkpoint folders."""
    if input_path.exists():
        return input_path

    # Try relative to PROJECT_ROOT
    if not input_path.is_absolute() and (PROJECT_ROOT / input_path).exists():
        return PROJECT_ROOT / input_path

    # Try profile outputs dir (e.g. Google Drive)
    try:
        mgr = ConfigManager()
        active_profile = profile or "local"
        stage_paths = mgr.get_stage_paths("sft", profile_name=active_profile)
        drive_adapter_eval = stage_paths["outputs"] / "adapter_model" / "eval_val_greedy" / input_path.name
        if drive_adapter_eval.exists():
            return drive_adapter_eval

        # Check under stage checkpoints dir
        ckpt_dir = stage_paths.get("checkpoints")
        if ckpt_dir and ckpt_dir.exists():
            for sub in sorted(ckpt_dir.glob("checkpoint-*/eval_val_greedy/eval_results.json"), reverse=True):
                if sub.exists():
                    logger.info("Resolved input file from checkpoint: %s", sub)
                    return sub

        # Try local checkpoints dir as well
        local_ckpt_dir = PROJECT_ROOT / "checkpoints" / "sft"
        if local_ckpt_dir.exists():
            for sub in sorted(local_ckpt_dir.glob("checkpoint-*/eval_val_greedy/eval_results.json"), reverse=True):
                if sub.exists():
                    logger.info("Resolved input file from local checkpoint: %s", sub)
                    return sub
    except Exception as exc:
        logger.warning("Error during input path resolution: %s", exc)

    # Try data/sample/sample_errors.jsonl as fallback
    sample_errors = PROJECT_ROOT / "data" / "sample" / "sample_errors.jsonl"
    if sample_errors.exists():
        logger.warning("Input file %s not found; falling back to sample errors: %s", input_path, sample_errors)
        return sample_errors

    return input_path


def load_input_records(input_path: Path, profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load generation or evaluation records from JSON, JSONL, or Parquet."""
    resolved_path = resolve_input_path(input_path, profile=profile)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input records file does not exist: {input_path}")
    input_path = resolved_path

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    elif suffix == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # If an eval_results summary pointing to generations_file is passed
            if "generations_file" in data:
                gen_file = Path(data["generations_file"])
                if gen_file.exists():
                    return load_input_records(gen_file, profile=profile)
                sibling = input_path.parent / gen_file.name
                if sibling.exists():
                    return load_input_records(sibling, profile=profile)
            # Check common result containers
            if "records" in data and isinstance(data["records"], list):
                return data["records"]
            elif "generations" in data and isinstance(data["generations"], list):
                return data["generations"]
            elif "per_problem" in data and isinstance(data["per_problem"], list):
                return data["per_problem"]
            return [data]
        return []
    elif suffix == ".parquet":
        df = pd.read_parquet(input_path)
        return df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported input format: {suffix} (expected .jsonl, .json, or .parquet)")


def run_error_analysis(
    input_file: Path,
    output_file: Path,
    report_file: Path,
    model_name_or_path: str = "deepseek-ai/deepseek-math-7b-instruct",
    use_mock: bool = False,
    limit: Optional[int] = None,
    stage_name: str = "SFT Round 1",
    profile: Optional[str] = None,
    cluster: bool = False,
    num_clusters: int = 5,
    select_targeted: bool = False,
    target_per_category: int = 30,
    targeted_output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full error classification pipeline over input generations."""
    logger.info("==================================================")
    logger.info("Phase 5 Day 31: Error Analysis & Taxonomy Classification")
    logger.info("Input File:        %s", input_file)
    logger.info("Output JSON:       %s", output_file)
    logger.info("Markdown Report:   %s", report_file)
    logger.info("Classifier Model:  %s (Mock: %s)", model_name_or_path, use_mock)
    logger.info("==================================================")

    records = load_input_records(input_file, profile=profile)
    logger.info("Loaded %d records from %s", len(records), input_file)

    if limit and limit > 0:
        records = records[:limit]
        logger.info("Subsetting to %d records per --limit", limit)

    classifier = ErrorClassifier(
        model_name_or_path=model_name_or_path,
        use_mock=use_mock,
    )

    classifications = classifier.classify_records(records)
    summary = compute_error_distribution(classifications)

    logger.info("--------------------------------------------------")
    logger.info("Classification Results Summary:")
    logger.info("  Total Evaluated:    %d", summary["total_evaluated"])
    logger.info("  Correct:            %d (%.1f%%)", summary["correct_count"], summary["accuracy"])
    logger.info("  Errors Analyzed:    %d (%.1f%%)", summary["error_count"], summary["error_rate"])
    logger.info("  Rule-Based Errors:  %d (%.1f%%)", summary["rule_based_count"], summary["rule_based_percentage"])
    logger.info("  Semantic Errors:    %d (%.1f%%)", summary["semantic_count"], summary["semantic_percentage"])
    logger.info("--------------------------------------------------")
    logger.info("Category Breakdown:")
    for cat in ALL_ERROR_CATEGORIES:
        cnt = summary["category_counts"].get(cat, 0)
        pct = summary["category_percentages"].get(cat, 0.0)
        logger.info("  %-22s: %4d (%5.1f%%)", cat, cnt, pct)
    logger.info("==================================================")

    # Persist JSON report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    full_payload = {
        "summary": summary,
        "classifications": [c.to_dict() for c in classifications],
        "config": {
            "input_file": str(input_file),
            "model_name_or_path": model_name_or_path,
            "use_mock": use_mock,
            "stage_name": stage_name,
        },
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    logger.info("Saved error analysis JSON report to: %s", output_file)

    # Optional topic clustering (Phase 5 Day 33)
    cluster_report = None
    if cluster:
        embedder = QuestionEmbedder(use_fallback=use_mock)
        clusterer = ErrorClusterer(embedder=embedder, num_clusters=num_clusters, random_state=42)
        cluster_report = clusterer.cluster(classifications, num_clusters=num_clusters)
        full_payload["clusters"] = cluster_report.to_dict()

    # Persist Markdown report
    report_file.parent.mkdir(parents=True, exist_ok=True)
    md_content = generate_error_markdown_report(
        summary, classifications, stage_name=stage_name, cluster_report=cluster_report
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved error analysis Markdown report to: %s", report_file)

    # Optional targeted example selection (Phase 5 Day 34)
    if select_targeted:
        targeted_dir = targeted_output_dir or (PROJECT_ROOT / "data" / "targeted")
        logger.info("Executing targeted example selection (target_per_category=%d)...", target_per_category)
        train_pool = PROJECT_ROOT / "data" / "processed" / "train" / "train.parquet"
        if not train_pool.exists():
            sample_train = PROJECT_ROOT / "data" / "sample" / "train_sample.parquet"
            if sample_train.exists():
                train_pool = sample_train

        selector = TargetedSelector(
            train_pool_path=train_pool,
            default_per_category=target_per_category,
            use_mock=use_mock,
        )
        targeted_examples = selector.select_for_errors(
            error_records=classifications,
            target_per_category=target_per_category,
        )
        if targeted_examples:
            selector.export_targeted_dataset(targeted_examples, output_dir=targeted_dir)
            full_payload["targeted_selection"] = {
                "total_selected": len(targeted_examples),
                "output_dir": str(targeted_dir),
            }

    # Save run provenance
    run_info_dir = output_file.parent
    save_run_info(
        stage="error_analysis",
        output_dir=run_info_dir,
        config={
            "input_file": str(input_file),
            "output_file": str(output_file),
            "report_file": str(report_file),
            "model_name_or_path": model_name_or_path,
            "use_mock": use_mock,
            "stage_name": stage_name,
            "summary": summary,
        },
    )

    return full_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify mathematical reasoning errors into the 7-category taxonomy (Phase 5 Day 31)."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="outputs/sft/adapter_model/eval_val_greedy/eval_results.json",
        help="Path to evaluation generations or results (JSON, JSONL, or Parquet).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="reports/error_analysis_sft.json",
        help="Path where error analysis JSON results will be written.",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="reports/error_analysis_sft.md",
        help="Path where error analysis Markdown report will be written.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Teacher model for semantic error classification (defaults to config: models.teacher).",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use deterministic mock/heuristic classifier instead of loading 7B LLM.",
    )
    parser.add_argument(
        "--sample_mode",
        action="store_true",
        help="Run quickly on sample data in mock mode for CI / testing.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Perform semantic topic clustering on error questions (Phase 5 Day 33).",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=5,
        help="Number of semantic topic clusters K (default: 5).",
    )
    parser.add_argument(
        "--select_targeted",
        action="store_true",
        help="Select similar training examples for error mitigation (Phase 5 Day 34).",
    )
    parser.add_argument(
        "--target_per_category",
        type=int,
        default=30,
        help="Number of targeted training examples per category (20-50, default: 30).",
    )
    parser.add_argument(
        "--targeted_output_dir",
        type=str,
        default="data/targeted",
        help="Output directory where targeted dataset will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to classify.",
    )
    parser.add_argument(
        "--stage_name",
        type=str,
        default="SFT Round 1",
        help="Stage name displayed in report titles.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Configuration profile (e.g. 'local', 'colab').",
    )
    return parser


def main(args_list: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)

    cfg = ConfigManager()
    model_name = args.model_name_or_path or cfg.base_config.get("models", {}).get(
        "teacher", "deepseek-ai/deepseek-math-7b-instruct"
    )

    use_mock = args.use_mock or args.sample_mode

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    report_path = Path(args.report_file)

    # In sample_mode without explicit input file override, use sample_errors.jsonl if available
    if args.sample_mode and (not input_path.exists() or args.input_file == "outputs/sft/adapter_model/eval_val_greedy/eval_results.json"):
        sample_errors = Path("data/sample/sample_errors.jsonl")
        sample_fallback = Path("data/sample/val_sample.parquet")
        if sample_errors.exists():
            input_path = sample_errors
            logger.info("Sample mode: using %s as input", input_path)
        elif sample_fallback.exists():
            input_path = sample_fallback
            logger.info("Sample mode fallback: using %s as input", input_path)

    try:
        run_error_analysis(
            input_file=input_path,
            output_file=output_path,
            report_file=report_path,
            model_name_or_path=model_name,
            use_mock=use_mock,
            limit=args.limit,
            stage_name=args.stage_name,
            profile=args.profile,
            cluster=args.cluster,
            num_clusters=args.num_clusters,
            select_targeted=args.select_targeted,
            target_per_category=args.target_per_category,
            targeted_output_dir=Path(args.targeted_output_dir) if args.targeted_output_dir else None,
        )
        return 0
    except Exception as e:
        logger.error("Error analysis failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
