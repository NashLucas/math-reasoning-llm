#!/usr/bin/env python3
"""End-to-end execution runner for Phase 5 Day 32: Classify SFT Model Errors on Validation Analysis Half.

Enforces:
1. Design Rule 2 (The Test-Set Wall): Error analysis operates strictly on the 250 problems
   of the analysis half of validation ('val_analysis.parquet'). Test set is strictly forbidden.
2. Evaluates SFT adapter model with greedy decoding on val_analysis.
3. Classifies all observed errors into the 7-category taxonomy.
4. Executes the 30-sample manual agreement calibration to verify category definitions.
5. Saves all audit artifacts and provenance records.
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

from src.config import ConfigManager  # noqa: E402
from src.evaluation.error_analysis import (  # noqa: E402
    ErrorClassifier,
    compute_error_distribution,
    compute_label_agreement,
    generate_agreement_markdown,
    generate_error_markdown_report,
)
from src.evaluation.evaluator import MathEvaluator, TestSetWallError  # noqa: E402
from src.run_info import save_run_info  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_day32")


def run_day32_pipeline(
    val_analysis_file: Path,
    calibration_file: Path,
    adapter_dir: Path,
    output_dir: Path,
    reports_dir: Path,
    model_name_or_path: str = "deepseek-ai/deepseek-math-7b-base",
    teacher_model: str = "deepseek-ai/deepseek-math-7b-instruct",
    use_mock: bool = False,
    limit: Optional[int] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Day 32 evaluation, error classification, and 30-sample calibration."""
    logger.info("==================================================")
    logger.info("Phase 5 Day 32: SFT Error Analysis on Validation Analysis Half")
    logger.info("Analysis Split:     %s", val_analysis_file)
    logger.info("SFT Adapter Dir:    %s", adapter_dir)
    logger.info("Calibration File:   %s", calibration_file)
    logger.info("Teacher Model:      %s (Mock: %s)", teacher_model, use_mock)
    logger.info("==================================================")

    # 1. Enforce Test-Set Wall (Design Rule 2)
    if "test" in str(val_analysis_file).lower() and "val" not in str(val_analysis_file).lower():
        raise TestSetWallError(
            "CRITICAL: Day 32 error analysis is strictly restricted to the validation analysis half! "
            "Access to test split is forbidden."
        )

    if not val_analysis_file.exists():
        raise FileNotFoundError(f"Validation analysis file not found: {val_analysis_file}")

    # 2. Run / Load Greedy Evaluation on val_analysis
    eval_stage_dir = output_dir / "eval_val_analysis_greedy"
    eval_results_file = eval_stage_dir / "eval_results.json"

    if eval_results_file.exists() and not use_mock:
        logger.info("Found existing evaluation results at %s; skipping re-generation.", eval_results_file)
    else:
        logger.info("Running greedy evaluation on %s...", val_analysis_file)
        evaluator = MathEvaluator(
            stage="sft",
            checkpoint=model_name_or_path,
            adapter_path=str(adapter_dir) if adapter_dir.exists() else None,
            preset="greedy",
            split="val_analysis",
            output_dir=str(eval_stage_dir),
            use_mock=use_mock,
            limit=limit,
        )
        evaluator.evaluate()

    # Load evaluated problem records
    generations_file = eval_stage_dir / "generations_val_analysis_greedy.jsonl"
    if not generations_file.exists():
        # Fallback to eval_results.json records
        data = json.loads(eval_results_file.read_text(encoding="utf-8"))
        generations_file = Path(data.get("generations_file", generations_file))

    records: List[Dict[str, Any]] = []
    if generations_file.exists():
        with open(generations_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info("Loaded %d generation records from %s", len(records), generations_file)
    else:
        logger.warning("Generations file not found; using calibration benchmark directly.")

    # 3. Classify Errors on val_analysis
    classifier = ErrorClassifier(
        model_name_or_path=teacher_model,
        use_mock=use_mock,
    )
    classifications = classifier.classify_records(records)
    error_summary = compute_error_distribution(classifications)

    reports_dir.mkdir(parents=True, exist_ok=True)
    error_json_file = reports_dir / "error_analysis_sft.json"
    error_md_file = reports_dir / "error_analysis_sft.md"

    with open(error_json_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": error_summary,
                "classifications": [c.to_dict() for c in classifications],
                "stage": "val_analysis_greedy",
            },
            f,
            indent=2,
        )
    logger.info("Saved error analysis JSON to: %s", error_json_file)

    md_report = generate_error_markdown_report(
        summary=error_summary,
        classifications=classifications,
        stage_name="SFT Round 1 — Validation Analysis Half (250 Problems)",
    )
    with open(error_md_file, "w", encoding="utf-8") as f:
        f.write(md_report)
    logger.info("Saved error analysis Markdown to: %s", error_md_file)

    # 4. Execute 30-Sample Agreement Calibration Benchmark
    logger.info("Running 30-sample agreement calibration from %s...", calibration_file)
    calib_records: List[Dict[str, Any]] = []
    with open(calibration_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                calib_records.append(json.loads(line))

    calib_classifications = classifier.classify_records(calib_records)
    agreement_report = compute_label_agreement(
        classifications=calib_classifications,
        manual_labeled_records=calib_records,
        min_agreement_threshold=0.80,
    )

    calib_json_file = reports_dir / "error_label_agreement_30.json"
    calib_md_file = reports_dir / "error_label_agreement_30.md"

    with open(calib_json_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "agreement": agreement_report.to_dict(),
                "calibration_file": str(calibration_file),
            },
            f,
            indent=2,
        )
    logger.info("Saved 30-sample calibration JSON to: %s", calib_json_file)

    agreement_md = generate_agreement_markdown(agreement_report)
    with open(calib_md_file, "w", encoding="utf-8") as f:
        f.write(agreement_md)
    logger.info("Saved 30-sample calibration Markdown to: %s", calib_md_file)

    # Append to main Markdown report
    appendix = (
        f"\n\n## 4. Human Agreement Calibration (30 Samples)\n\n"
        f"- **Calibration Status:** **{'PASSED' if agreement_report.passed_calibration else 'FAILED'}**\n"
        f"- **Agreement Rate:** {agreement_report.agreed_count}/{agreement_report.total_samples} "
        f"({agreement_report.agreement_percent}%)\n"
        f"- **Target Requirement:** $\\ge 80\\%$\n\n"
        f"Detailed per-category confusion analysis is documented in `reports/error_label_agreement_30.md`.\n"
    )
    with open(error_md_file, "a", encoding="utf-8") as f:
        f.write(appendix)

    # 5. Provenance Record
    save_run_info(
        stage="error_analysis",
        output_dir=reports_dir,
        config={
            "val_analysis_file": str(val_analysis_file),
            "adapter_dir": str(adapter_dir),
            "calibration_file": str(calibration_file),
            "error_summary": error_summary,
            "agreement_percent": agreement_report.agreement_percent,
            "passed_calibration": agreement_report.passed_calibration,
        },
    )

    logger.info("==================================================")
    logger.info("Day 32 Results Summary:")
    logger.info("  Analysis Problems Evaluated: %d", error_summary["total_evaluated"])
    logger.info("  Analysis Errors Categorized: %d", error_summary["error_count"])
    logger.info("  30-Sample Agreement Rate:    %.1f%% (Gate: >= 80.0%%)", agreement_report.agreement_percent)
    logger.info("  Day 32 Gate Status:          %s", "PASSED" if agreement_report.passed_calibration else "FAILED")
    logger.info("==================================================")

    return {
        "error_summary": error_summary,
        "agreement": agreement_report.to_dict(),
        "passed": agreement_report.passed_calibration,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Day 32 SFT error analysis on val_analysis and 30-sample calibration benchmark."
    )
    parser.add_argument(
        "--val_analysis_file",
        type=str,
        default="data/processed/val/val_analysis.parquet",
        help="Path to val_analysis.parquet split file.",
    )
    parser.add_argument(
        "--calibration_file",
        type=str,
        default="data/sample/labeled_30_errors.jsonl",
        help="Path to 30-example reference labeled errors JSONL.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="outputs/sft/adapter_model",
        help="Path to trained SFT adapter model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/sft",
        help="Output directory for intermediate evaluation generations.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Directory where error analysis and calibration reports will be saved.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="deepseek-ai/deepseek-math-7b-base",
        help="Base model checkpoint.",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default=None,
        help="Teacher model for semantic classification (defaults to config: models.teacher).",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Run deterministic mock generator and heuristic classifier.",
    )
    parser.add_argument(
        "--sample_mode",
        action="store_true",
        help="Run quickly on sample problems in mock mode for CI / testing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of evaluation problems.",
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
    teacher = args.teacher_model or cfg.base_config.get("models", {}).get(
        "teacher", "deepseek-ai/deepseek-math-7b-instruct"
    )

    use_mock = args.use_mock or args.sample_mode
    limit = 10 if (args.sample_mode and args.limit is None) else args.limit

    try:
        res = run_day32_pipeline(
            val_analysis_file=Path(args.val_analysis_file),
            calibration_file=Path(args.calibration_file),
            adapter_dir=Path(args.adapter_dir),
            output_dir=Path(args.output_dir),
            reports_dir=Path(args.reports_dir),
            model_name_or_path=args.model_name_or_path,
            teacher_model=teacher,
            use_mock=use_mock,
            limit=limit,
            profile=args.profile,
        )
        return 0 if res["passed"] else 1
    except Exception as e:
        logger.error("Day 32 pipeline execution failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
