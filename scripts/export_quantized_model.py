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
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.quantization import run_quantization_audit  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_quantized_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and export model quantization profiles for lightweight deployment."
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="outputs/grpo/adapter_model",
        help="Path to post-GRPO policy adapter",
    )
    parser.add_argument(
        "--output_report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "day52_quantization_report.md",
        help="Path where markdown report will be written",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "day52_quantization.json",
        help="Path where JSON summary will be written",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        default=True,
        help="Run deterministic mock engine for verification",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("========================================================================")
    logger.info("PHASE 8 DAY 52: MODEL QUANTIZATION & COMPRESSION AUDIT")
    logger.info("========================================================================")
    logger.info("Adapter Path:        %s", args.adapter_path)
    logger.info("Target Markdown:     %s", args.output_report)
    logger.info("Target JSON Summary: %s", args.output_json)
    logger.info("Backend Mode:        %s", "Deterministic Mock" if args.use_mock else "CUDA GPU")
    logger.info("========================================================================")

    try:
        report = run_quantization_audit(
            adapter_path=args.adapter_path,
            use_mock=args.use_mock,
            output_report_path=args.output_report,
            output_json_path=args.output_json,
        )
    except Exception as err:
        logger.error("Quantization audit failed: %s", err, exc_info=True)
        return 1

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
        print(f"  Markdown: {args.output_report}")
        print(f"  JSON:     {args.output_json}")
        print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
