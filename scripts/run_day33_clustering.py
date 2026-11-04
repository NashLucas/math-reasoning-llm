#!/usr/bin/env python3
"""End-to-end execution runner for Phase 5 Day 33: Error Question Semantic Clustering & Topic Reporting.

Enforces:
1. Design Rule 2 (The Test-Set Wall): Error analysis and clustering operate strictly on
   validation analysis errors or golden sample errors. The test split is strictly forbidden.
2. Embeds mathematical error questions using 'sentence-transformers/all-MiniLM-L6-v2' (or TF-IDF fallback).
3. Clusters error questions into distinct mathematical semantic topics (percentages, rates, money, time, counting).
4. Selects 2 representative examples per cluster closest to the cluster centroid.
5. Emits 'reports/error_clusters.json' and updates 'reports/error_analysis_sft.md' with complete cluster profiles.
6. Records run provenance with save_run_info.
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

from src.evaluation.error_analysis import (  # noqa: E402
    ClusterAnalysisReport,
    ErrorClusterer,
    QuestionEmbedder,
    generate_cluster_markdown,
)
from src.evaluation.evaluator import TestSetWallError  # noqa: E402
from src.run_info import save_run_info  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_day33_clustering")


def load_error_records(input_path: Path) -> List[Dict[str, Any]]:
    """Load error records from a JSON file, JSONL file, or analysis report."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input errors file not found: {input_path}")

    suffix = input_path.suffix.lower()
    records: List[Dict[str, Any]] = []

    if suffix == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Treat labeled error examples as errors
                    if item.get("is_error", True):
                        if "is_error" not in item:
                            item["is_error"] = True
                        if "category" not in item and "gold_category" in item:
                            item["category"] = item["gold_category"]
                        elif "category" not in item and "manual_label" in item:
                            item["category"] = item["manual_label"]
                        records.append(item)
    elif suffix == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if item.get("is_error", True):
                    records.append(item)
        elif isinstance(data, dict):
            # Could be error_analysis_sft.json
            classifications = data.get("classifications", [])
            for item in classifications:
                if item.get("is_error", False):
                    records.append(item)
            # If no classifications flagged as errors, check for records
            if not records and "records" in data:
                for item in data["records"]:
                    if not item.get("is_correct", False):
                        records.append(item)

    logger.info("Loaded %d error records from %s", len(records), input_path)
    return records


def run_day33_clustering(
    input_file: Path,
    output_clusters_file: Path,
    report_file: Path,
    num_clusters: int = 5,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_mock: bool = False,
    sample_mode: bool = False,
    existing_analysis_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute Day 33 question semantic clustering and report generation."""
    logger.info("==================================================")
    logger.info("Phase 5 Day 33: Error Question Semantic Clustering & Topic Reporting")
    logger.info("Input Error File:       %s", input_file)
    logger.info("Output Clusters File:   %s", output_clusters_file)
    logger.info("Report File:            %s", report_file)
    logger.info("Target Clusters (K):    %d", num_clusters)
    logger.info("Embedding Model:        %s (Mock/Fallback: %s)", embedding_model, use_mock or sample_mode)
    logger.info("==================================================")

    # 1. Enforce Test-Set Wall (Design Rule 2)
    if "test" in str(input_file).lower() and "val" not in str(input_file).lower():
        raise TestSetWallError(
            "CRITICAL: Day 33 error clustering is strictly forbidden on the test set! "
            "Use validation analysis half or calibration benchmarks only."
        )

    # If input file doesn't exist and sample_mode is on, fall back to labeled_30_errors.jsonl
    if not input_file.exists():
        if sample_mode or use_mock:
            sample_candidate = PROJECT_ROOT / "data" / "sample" / "labeled_30_errors.jsonl"
            if sample_candidate.exists():
                logger.warning(
                    "Input file %s not found; falling back to sample errors %s.",
                    input_file,
                    sample_candidate,
                )
                input_file = sample_candidate
            else:
                raise FileNotFoundError(f"Input file not found: {input_file}")
        else:
            raise FileNotFoundError(f"Input file not found: {input_file}")

    # 2. Load Error Records
    records = load_error_records(input_file)
    if not records and (sample_mode or use_mock):
        sample_candidate = PROJECT_ROOT / "data" / "sample" / "labeled_30_errors.jsonl"
        logger.warning("No errors found in %s; falling back to %s.", input_file, sample_candidate)
        records = load_error_records(sample_candidate)

    if not records:
        raise ValueError(f"No error records found in {input_file} to cluster.")

    # 3. Perform Semantic Topic Clustering
    embedder = QuestionEmbedder(
        model_name_or_path=embedding_model,
        use_fallback=(use_mock or sample_mode),
    )
    clusterer = ErrorClusterer(
        embedder=embedder,
        num_clusters=num_clusters,
        random_state=42,
    )
    cluster_report: ClusterAnalysisReport = clusterer.cluster(records, num_clusters=num_clusters)

    # 4. Save Cluster JSON Artifacts
    output_clusters_file.parent.mkdir(parents=True, exist_ok=True)
    cluster_payload = {
        "cluster_report": cluster_report.to_dict(),
        "input_file": str(input_file),
        "total_errors": len(records),
        "num_clusters": cluster_report.num_clusters,
    }
    with open(output_clusters_file, "w", encoding="utf-8") as f:
        json.dump(cluster_payload, f, indent=2)
    logger.info("Saved cluster analysis JSON to: %s", output_clusters_file)

    # Also update error_analysis_sft.json if present
    analysis_json_file = existing_analysis_file or (PROJECT_ROOT / "reports" / "error_analysis_sft.json")
    if analysis_json_file.exists():
        try:
            analysis_data = json.loads(analysis_json_file.read_text(encoding="utf-8"))
            analysis_data["clusters"] = cluster_report.to_dict()
            with open(analysis_json_file, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, indent=2)
            logger.info("Updated %s with topic cluster analysis.", analysis_json_file)
        except Exception as e:
            logger.warning("Could not update analysis JSON %s: %s", analysis_json_file, e)

    # 5. Generate and Update Markdown Report
    report_file.parent.mkdir(parents=True, exist_ok=True)
    cluster_md = generate_cluster_markdown(cluster_report)

    # If reports/error_analysis_sft.md already exists, append or update section 4
    if report_file.exists():
        existing_md = report_file.read_text(encoding="utf-8")
        if "## 4. Semantic Topic Clusters of Reasoning Errors" in existing_md:
            # Replace existing section 4
            parts = existing_md.split("## 4. Semantic Topic Clusters of Reasoning Errors")
            updated_md = parts[0].rstrip() + "\n\n" + cluster_md + "\n"
        else:
            updated_md = existing_md.rstrip() + "\n\n" + cluster_md + "\n"
        report_file.write_text(updated_md, encoding="utf-8")
        logger.info("Updated existing Markdown report at: %s", report_file)
    else:
        report_file.write_text(cluster_md, encoding="utf-8")
        logger.info("Created new Markdown report at: %s", report_file)

    # 6. Save Run Provenance
    save_run_info(
        stage="error_analysis",
        output_dir=output_clusters_file.parent,
        config={
            "day": 33,
            "task": "error_question_topic_clustering",
            "input_file": str(input_file),
            "output_clusters_file": str(output_clusters_file),
            "report_file": str(report_file),
            "num_clusters": cluster_report.num_clusters,
            "total_errors": len(records),
            "embedding_model": cluster_report.embedding_model,
            "clusters": [
                {
                    "id": c.cluster_id,
                    "name": c.name,
                    "size": c.size,
                    "percentage": c.percentage,
                    "top_keywords": c.top_keywords,
                }
                for c in cluster_report.clusters
            ],
        },
    )

    return {
        "passed": True,
        "total_errors": len(records),
        "num_clusters": cluster_report.num_clusters,
        "clusters": cluster_report.to_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster mathematical reasoning error questions by semantic topic (Phase 5 Day 33)."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="reports/error_analysis_sft.json",
        help="Path to input errors JSON or JSONL file.",
    )
    parser.add_argument(
        "--output_clusters_file",
        type=str,
        default="reports/error_clusters.json",
        help="Path where cluster JSON results will be written.",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="reports/error_analysis_sft.md",
        help="Path where Markdown report with cluster profiles will be updated/written.",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=5,
        help="Target number of semantic topic clusters K (default: 5).",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model checkpoint for question representations.",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use deterministic TF-IDF text embedding fallback instead of sentence-transformers.",
    )
    parser.add_argument(
        "--sample_mode",
        action="store_true",
        help="Run quickly on sample error benchmarks in mock mode.",
    )
    return parser


def main(args_list: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)

    input_path = Path(args.input_file)
    # If default input_file doesn't exist or has no errors, fallback gracefully in sample mode
    if args.sample_mode and (not input_path.exists()):
        input_path = PROJECT_ROOT / "data" / "sample" / "labeled_30_errors.jsonl"

    try:
        res = run_day33_clustering(
            input_file=input_path,
            output_clusters_file=Path(args.output_clusters_file),
            report_file=Path(args.report_file),
            num_clusters=args.num_clusters,
            embedding_model=args.embedding_model,
            use_mock=args.use_mock or args.sample_mode,
            sample_mode=args.sample_mode,
        )
        return 0 if res["passed"] else 1
    except Exception as e:
        logger.error("Day 33 clustering execution failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
