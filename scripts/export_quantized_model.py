#!/usr/bin/env python3
"""Phase 8 Day 52: Model Quantization & Compression CLI Runner.

Evaluates and exports quantization profiles for the Post-GRPO Policy model:
- BitsAndBytes 4-bit / 8-bit
- AWQ 4-bit (vLLM)
- GGUF Q4_K_M (llama.cpp)

Emits:
- reports/day52_quantization_report.md
- reports/day52_quantization.json

Usage:
    python scripts/export_quantized_model.py
    python scripts/export_quantized_model.py --adapter_path outputs/grpo/adapter_model
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_info import save_run_info  # noqa: E402
from src.serving.quantization import run_quantization_audit  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_quantized_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and export model quantization profiles for lightweight deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adapter_path",
        "--adapter-path",
        dest="adapter_path",
        type=str,
        default="outputs/grpo/adapter_model",
        help="Path to post-GRPO policy adapter",
    )
    parser.add_argument(
        "--output_report",
        "--output-report",
        dest="output_report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "day52_quantization_report.md",
        help="Path where markdown report will be written",
    )
    parser.add_argument(
        "--output_json",
        "--output-json",
        dest="output_json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "day52_quantization.json",
        help="Path where JSON summary will be written",
    )
    parser.add_argument(
        "--use_mock",
        "--use-mock",
        dest="use_mock",
        action="store_true",
        default=True,
        help="Run deterministic mock engine for verification",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="auto",
        choices=["local", "colab", "auto"],
        help="Execution environment profile",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_report = Path(args.output_report)
    if not out_report.is_absolute():
        out_report = PROJECT_ROOT / out_report

    out_json = Path(args.output_json)
    if not out_json.is_absolute():
        out_json = PROJECT_ROOT / out_json

    out_dir = out_report.parent

    logger.info("========================================================================")
    logger.info("PHASE 8 DAY 52: MODEL QUANTIZATION & COMPRESSION AUDIT")
    logger.info("========================================================================")
    logger.info("Adapter Path:        %s", args.adapter_path)
    logger.info("Target Markdown:     %s", out_report)
    logger.info("Target JSON Summary: %s", out_json)
    logger.info("Backend Mode:        %s", "Deterministic Mock" if args.use_mock else "CUDA GPU")
    logger.info("Environment Profile: %s", args.profile)
    logger.info("========================================================================")

    try:
        report = run_quantization_audit(
            adapter_path=args.adapter_path,
            use_mock=args.use_mock,
            output_report_path=out_report,
            output_json_path=out_json,
        )
    except Exception as err:
        logger.error("Quantization audit failed: %s", err, exc_info=True)
        return 1

    # Record run provenance
    try:
        save_run_info(
            stage="quantization_audit",
            output_dir=out_dir,
            config=report.to_dict(),
            description="Phase 8 Day 52 Model Quantization & Compression Audit",
        )
    except Exception as exc:
        logger.warning("Could not save run_info: %s", exc)

    # Mirror deliverables to Google Drive if in Colab environment
    drive_base = Path("/content/drive/MyDrive/math-reasoning-llm")
    if drive_base.exists() or args.profile == "colab":
        try:
            drive_reports = drive_base / "reports"
            drive_reports.mkdir(parents=True, exist_ok=True)
            if out_report.exists():
                shutil.copy2(out_report, drive_reports / out_report.name)
            if out_json.exists():
                shutil.copy2(out_json, drive_reports / out_json.name)
            run_info_file = out_dir / "run_info.json"
            if run_info_file.exists():
                shutil.copy2(run_info_file, drive_reports / run_info_file.name)
            logger.info("Mirrored Day 52 quantization reports to Google Drive at: %s", drive_reports)
        except Exception as exc:
            logger.warning("Could not mirror reports to Drive: %s", exc)

    if not args.quiet:
        print("\n" + "=" * 80)
        print("PHASE 8 DAY 52: MODEL QUANTIZATION AUDIT SUMMARY")
        print("=" * 80)
        print(f"Timestamp:              {report.timestamp}")
        print(f"Base Model:             {report.base_model}")
        print(f"Advancing Adapter:      {report.adapter_path}")
        print(f"Recommended GPU Regime: {report.recommended_gpu_profile}")
        print(f"Recommended CPU Regime: {report.recommended_cpu_profile}")
        print("-" * 80)
        print("Quantization Profiles:")
        for p in report.profiles_evaluated:
            print(
                f"  - {p.display_name:<30} | {p.bits:>2}-bit | Size: {p.estimated_size_gb:>4.1f} GB | "
                f"Comp: {p.compression_ratio:>4.2f}x | Contract: {p.format_compliance*100:>5.1f}% | [{p.status}]"
            )
        print("=" * 80)
        print("Reports Generated:")
        print(f"  Markdown: {out_report}")
        print(f"  JSON:     {out_json}")
        print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
