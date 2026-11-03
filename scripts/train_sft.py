#!/usr/bin/env python3
"""SFT training executable workflow (Phase 4 Day 26).

Post-trains deepseek-math-7b-base (or development model) on curated reasoning traces
using LoRA adapters and completion-only loss masking.

Usage:
    python scripts/train_sft.py \
        --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
        --curated_file data/synthetic/curated/traces_curated_sft.jsonl \
        --output_dir checkpoints/sft \
        --adapter_dir outputs/sft/adapter_model \
        --sample_mode \
        --device cpu \
        --max_steps 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.sft_trainer import run_sft_training  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_sft")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SFT training with LoRA on curated math reasoning traces (Phase 4 Day 26)."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model Hub ID or local path (default: Qwen/Qwen2.5-1.5B-Instruct).",
    )
    parser.add_argument(
        "--curated_file",
        type=str,
        default="data/synthetic/curated/traces_curated_sft.jsonl",
        help="Path to curated reasoning dataset JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/sft",
        help="Directory to save training step checkpoints.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="outputs/sft/adapter_model",
        help="Directory to export the final trained LoRA adapter.",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="configs/lora_config.yaml",
        help="Path to LoRA configuration YAML.",
    )
    parser.add_argument(
        "--training_args",
        type=str,
        default="configs/training_args.json",
        help="Path to training arguments JSON.",
    )
    parser.add_argument(
        "--sample_mode",
        action="store_true",
        help="Run in sample mode on a small subset (e.g. 20 examples) for quick verification.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=20,
        help="Number of examples to use in sample mode (default: 20).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional max training steps override.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from existing checkpoints in output_dir.",
    )
    args = parser.parse_args()

    c_path = Path(args.curated_file)
    if not c_path.is_absolute():
        c_path = PROJECT_ROOT / c_path

    # Fallback to sample data if curated dataset is missing
    if not c_path.exists():
        sample_path = PROJECT_ROOT / "data" / "sample" / "train_sample.jsonl"
        if sample_path.exists():
            logger.warning(f"Curated dataset not found at {c_path}. Falling back to sample set: {sample_path}")
            c_path = sample_path

    logger.info("==================================================")
    logger.info("Starting SFT Training Workflow (Phase 4 Day 26)")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Curated Data: {c_path}")
    logger.info(f"Output Checkpoints: {args.output_dir}")
    logger.info(f"Final Adapter Dir: {args.adapter_dir}")
    logger.info(f"Sample Mode: {args.sample_mode} (limit: {args.sample_limit if args.sample_mode else 'all'})")
    logger.info(f"Device: {args.device}")
    logger.info(f"Resume: {not args.no_resume}")
    logger.info("==================================================")

    trainer, run_info = run_sft_training(
        model_name_or_path=args.model_name_or_path,
        curated_data_path=c_path,
        output_dir=args.output_dir,
        final_adapter_dir=args.adapter_dir,
        lora_config_path=args.lora_config,
        training_args_path=args.training_args,
        resume=not args.no_resume,
        sample_limit=args.sample_limit if args.sample_mode else None,
        device=args.device,
        max_steps=args.max_steps,
    )

    logger.info("SFT Training Run completed successfully!")


if __name__ == "__main__":
    main()
