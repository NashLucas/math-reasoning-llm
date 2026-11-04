#!/usr/bin/env python3
"""CLI runner for 30-sample error label agreement calibration (Phase 5 Day 32).

Evaluates automatic error classification against human expert ground-truth labels on 30 representative
mathematical error problems across all 7 categories:
  - arithmetic_slip
  - misread_question
  - wrong_setup
  - incomplete_reasoning
  - parse_error
  - truncation
  - other

Enforces Day 32 gate:
  - Automatic categories must agree with most (>= 80%) of the 30 manual labels.
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
    ALL_ERROR_CATEGORIES,
    ErrorClassifier,
    compute_label_agreement,
    generate_agreement_markdown,
)
from src.run_info import save_run_info  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("calibrate_error_labels")


def load_labeled_records(path: Path) -> List[Dict[str, Any]]:
    """Load reference labeled records from JSONL or JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    records = []
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("records", [data])
    else:
        raise ValueError(f"Unsupported calibration file format: {path.suffix}")

    return records


def run_calibration(
    calibration_file: Path,
    output_file: Path,
    report_file: Path,
    model_name_or_path: str = "deepseek-ai/deepseek-math-7b-instruct",
    use_mock: bool = False,
    min_agreement: float = 0.80,
    update_main_report: bool = True,
    main_report_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute 30-sample error classification calibration benchmark."""
    logger.info("==================================================")
    logger.info("Phase 5 Day 32: 30-Sample Error Label Agreement Calibration")
    logger.info("Benchmark File:     %s", calibration_file)
    logger.info("Output JSON:        %s", output_file)
    logger.info("Markdown Report:    %s", report_file)
    logger.info("Classifier Model:   %s (Mock: %s)", model_name_or_path, use_mock)
    logger.info("Required Agreement: %.1f%%", min_agreement * 100.0)
    logger.info("==================================================")

    records = load_labeled_records(calibration_file)
    logger.info("Loaded %d labeled error records from %s", len(records), calibration_file)

    classifier = ErrorClassifier(
        model_name_or_path=model_name_or_path,
        use_mock=use_mock,
    )

    classifications = classifier.classify_records(records)
    agreement_report = compute_label_agreement(
        classifications=classifications,
        manual_labeled_records=records,
        min_agreement_threshold=min_agreement,
    )

    logger.info("--------------------------------------------------")
    logger.info("Agreement Results:")
    logger.info("  Total Problems:     %d", agreement_report.total_samples)
    logger.info("  Agreed Count:       %d", agreement_report.agreed_count)
    logger.info("  Disagreed Count:    %d", agreement_report.disagreed_count)
    logger.info("  Agreement Rate:     %.2f%%", agreement_report.agreement_percent)
    logger.info(
        "  Calibration Status: %s",
        "PASSED" if agreement_report.passed_calibration else "FAILED",
    )
    logger.info("--------------------------------------------------")
    logger.info("Per-Category Ground Truth Recall:")
    for cat in ALL_ERROR_CATEGORIES:
        stats = agreement_report.per_category.get(cat, {})
        g_cnt = stats.get("gold_count", 0)
        tp = stats.get("true_positives", 0)
        rec = stats.get("agreement_rate", 0.0)
        logger.info("  %-22s: %2d / %2d (Recall: %5.1f%%)", cat, tp, g_cnt, rec)
    logger.info("==================================================")

    # Persist JSON report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agreement": agreement_report.to_dict(),
        "config": {
            "calibration_file": str(calibration_file),
            "model_name_or_path": model_name_or_path,
            "use_mock": use_mock,
            "min_agreement_threshold": min_agreement,
        },
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved calibration JSON report to: %s", output_file)

    # Persist dedicated Markdown report
    report_file.parent.mkdir(parents=True, exist_ok=True)
    md_content = generate_agreement_markdown(agreement_report)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved calibration Markdown report to: %s", report_file)

    # Optionally append agreement section to main error analysis report
    target_main = main_report_file or Path("reports/error_analysis_sft.md")
    if update_main_report and target_main.exists():
        existing_text = target_main.read_text(encoding="utf-8")
        if "## 4. Human Agreement Calibration (30 Samples)" not in existing_text:
            appendix = (
                f"\n\n## 4. Human Agreement Calibration (30 Samples)\n\n"
                f"- **Calibration Status:** **{'PASSED' if agreement_report.passed_calibration else 'FAILED'}**\n"
                f"- **Overall Agreement:** {agreement_report.agreed_count}/{agreement_report.total_samples} "
                f"({agreement_report.agreement_percent}%)\n"
                f"- **Target Gate Threshold:** $\\ge {int(min_agreement * 100)}\\%$\n\n"
                f"For full per-category confusion details, see `reports/error_label_agreement_30.md`.\n"
            )
            with open(target_main, "a", encoding="utf-8") as f:
                f.write(appendix)
            logger.info("Appended calibration section to main report: %s", target_main)

    # Save run provenance
    save_run_info(
        stage="error_analysis_calibration",
        output_dir=output_file.parent,
        config={
            "calibration_file": str(calibration_file),
            "output_file": str(output_file),
            "report_file": str(report_file),
            "agreement_percent": agreement_report.agreement_percent,
            "passed_calibration": agreement_report.passed_calibration,
        },
    )

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate error classification against human expert ground-truth labels (Phase 5 Day 32)."
    )
    parser.add_argument(
        "--calibration_file",
        type=str,
        default="data/sample/labeled_30_errors.jsonl",
        help="Path to JSONL file containing 30 manually labeled error examples.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="reports/error_label_agreement_30.json",
        help="Path where calibration agreement JSON will be written.",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="reports/error_label_agreement_30.md",
        help="Path where calibration Markdown report will be written.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Teacher model for classification (defaults to config: models.teacher).",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use deterministic heuristic classifier instead of GPU model.",
    )
    parser.add_argument(
        "--min_agreement",
        type=float,
        default=0.80,
        help="Minimum agreement threshold (0.0 to 1.0) required to pass calibration.",
    )
    return parser


def main(args_list: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)

    cfg = ConfigManager()
    model_name = args.model_name_or_path or cfg.base_config.get("models", {}).get(
        "teacher", "deepseek-ai/deepseek-math-7b-instruct"
    )

    try:
        res = run_calibration(
            calibration_file=Path(args.calibration_file),
            output_file=Path(args.output_file),
            report_file=Path(args.report_file),
            model_name_or_path=model_name,
            use_mock=args.use_mock,
            min_agreement=args.min_agreement,
        )
        passed = res["agreement"]["passed_calibration"]
        return 0 if passed else 1
    except Exception as e:
        logger.error("Calibration failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
