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

from src.config import ConfigManager, get_default_profile  # noqa: E402
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
        "--load_in_4bit",
        action="store_true",
        default=False,
        help="Load model in 4-bit NF4 quantization for memory-constrained GPUs (e.g. T4/L4/V100).",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        default=False,
        help="Load model in 8-bit quantization.",
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
        "--learning_rate",
        type=float,
        default=None,
        help="Optional learning rate override (e.g. 5e-5, 1e-4, 2e-4).",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from existing checkpoints in output_dir.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Step interval for running the greedy validation evaluation callback (default: None/off).",
    )
    parser.add_argument(
        "--eval_limit",
        type=int,
        default=100,
        help="Number of validation problems to evaluate per callback step (default: 100).",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/splits/val.parquet",
        help="Path to validation dataset for periodic evaluation callback.",
    )
    parser.add_argument(
        "--generations_dir",
        type=str,
        default="logs/generations",
        help="Directory to save callback generation records and summaries.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["local", "colab"],
        help="Runtime profile (local or colab for Google Drive persistence).",
    )
    args = parser.parse_args()

    mgr = ConfigManager()
    active_profile = args.profile or get_default_profile()
    stage_paths = mgr.get_stage_paths("sft", profile_name=active_profile)
    profile_data_dir = mgr.get_profile_dir("data_dir", profile_name=active_profile)

    # Resolve output_dir, adapter_dir, generations_dir if using colab profile or defaults
    output_dir = args.output_dir
    adapter_dir = args.adapter_dir
    generations_dir = args.generations_dir
    if args.profile == "colab" or active_profile == "colab":
        if output_dir == "checkpoints/sft":
            output_dir = str(stage_paths["checkpoints"])
        if adapter_dir == "outputs/sft/adapter_model":
            adapter_dir = str(stage_paths["outputs"] / "adapter_model")
        if generations_dir == "logs/generations":
            generations_dir = str(stage_paths["logs"] / "generations")

    c_path = Path(args.curated_file)
    if not c_path.is_absolute():
        drive_curated = profile_data_dir / "synthetic" / "curated" / c_path.name
        local_curated = PROJECT_ROOT / c_path
        if drive_curated.exists() and local_curated.exists():
            if drive_curated.stat().st_size >= local_curated.stat().st_size:
                c_path = drive_curated
            else:
                c_path = local_curated
        elif drive_curated.exists():
            c_path = drive_curated
        elif local_curated.exists():
            c_path = local_curated
        elif (profile_data_dir / c_path).exists():
            c_path = profile_data_dir / c_path
        else:
            c_path = PROJECT_ROOT / c_path

    # Fallback to sample data if curated dataset is missing
    if not c_path.exists():
        sample_path = PROJECT_ROOT / "data" / "sample" / "train_sample.jsonl"
        if sample_path.exists():
            logger.warning(f"Curated dataset not found at {c_path}. Falling back to sample set: {sample_path}")
            c_path = sample_path

    # Auto-detect VRAM constraint for 7B models when on CUDA
    load_4bit = args.load_in_4bit
    if not load_4bit and not args.load_in_8bit and args.device != "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_mem <= 16.5 and "7b" in args.model_name_or_path.lower():
                    logger.info(
                        f"Detected GPU with {total_mem:.1f} GiB VRAM (<= 16 GiB). "
                        "Auto-enabling --load_in_4bit (QLoRA) to prevent CUDA OOM."
                    )
                    load_4bit = True
        except Exception:
            pass

    logger.info("==================================================")
    logger.info("Starting SFT Training Workflow (Phase 4 Day 26)")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Curated Data: {c_path}")
    logger.info(f"Output Checkpoints: {output_dir}")
    logger.info(f"Final Adapter Dir: {adapter_dir}")
    logger.info(f"Sample Mode: {args.sample_mode} (limit: {args.sample_limit if args.sample_mode else 'all'})")
    logger.info(f"Device: {args.device}")
    logger.info(f"4-bit QLoRA: {load_4bit}, 8-bit: {args.load_in_8bit}")
    logger.info(f"Resume: {not args.no_resume}")
    logger.info(f"Eval Steps: {args.eval_steps} (limit: {args.eval_limit})")
    logger.info(f"Generations Dir: {generations_dir}")
    logger.info("==================================================")

    trainer, run_info = run_sft_training(
        model_name_or_path=args.model_name_or_path,
        curated_data_path=c_path,
        output_dir=output_dir,
        final_adapter_dir=adapter_dir,
        lora_config_path=args.lora_config,
        training_args_path=args.training_args,
        resume=not args.no_resume,
        sample_limit=args.sample_limit if args.sample_mode else None,
        device=args.device,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
        eval_limit=args.eval_limit,
        val_path=args.val_file,
        generations_dir=generations_dir,
        load_in_4bit=load_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    logger.info("SFT Training Run completed successfully!")


if __name__ == "__main__":
    main()
