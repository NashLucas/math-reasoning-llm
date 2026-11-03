"""SFT Trainer orchestration with LoRA adapters and completion-only loss masking (Phase 4 Day 26).

Wraps TRL's SFTTrainer with PEFT LoRA adapters, loads configurations from configs/lora_config.yaml
and configs/training_args.json, handles automatic checkpoint resumption, and executes training
runs on GPU or CPU dry-run environments.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from src.data.sft_dataset import (
    load_sft_dataset,
    tokenize_sft_example,
)
from src.run_info import collect_run_info, save_run_info
from src.training.checkpointing import (
    find_latest_checkpoint,
    prune_checkpoints,
)

logger = logging.getLogger(__name__)

# Check optional deep learning dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from transformers import Trainer, TrainingArguments
    from transformers.utils import is_accelerate_available
    HAS_TRANSFORMERS = True
except ImportError:
    Trainer = None
    TrainingArguments = None

    def is_accelerate_available() -> bool:
        return False

    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    LoraConfig = None
    get_peft_model = None
    HAS_PEFT = False

try:
    from trl import SFTConfig, SFTTrainer
    HAS_TRL = True
except ImportError:
    SFTConfig = None
    SFTTrainer = None
    HAS_TRL = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_lora_config(
    config_path: Optional[Union[str, Path]] = None,
    **overrides: Any,
) -> Any:
    """Load LoRA configuration from YAML and return a peft.LoraConfig or dictionary.

    Args:
        config_path: Path to lora_config.yaml. Defaults to configs/lora_config.yaml.
        **overrides: Optional overrides for LoRA parameters (r, lora_alpha, etc.).

    Returns:
        peft.LoraConfig instance if PEFT is available, otherwise dictionary.
    """
    p = Path(config_path) if config_path else (PROJECT_ROOT / "configs" / "lora_config.yaml")
    if not p.exists():
        raise FileNotFoundError(f"LoRA config file not found at: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.update(overrides)

    if HAS_PEFT and LoraConfig is not None:
        target_mods = cfg.get("target_modules")
        return LoraConfig(
            r=int(cfg.get("r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias=str(cfg.get("bias", "none")),
            task_type=str(cfg.get("task_type", "CAUSAL_LM")),
            target_modules=target_mods,
        )

    return cfg


def load_training_arguments(
    args_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    **overrides: Any,
) -> Any:
    """Load SFT training arguments from JSON and return TrainingArguments/SFTConfig or dictionary.

    Args:
        args_path: Path to training_args.json. Defaults to configs/training_args.json.
        output_dir: Optional override for checkpoint output directory.
        device: Device string ('cuda', 'cpu'). If 'cpu', forces bf16/fp16 to False.
        **overrides: Optional overrides.

    Returns:
        TrainingArguments or SFTConfig instance if transformers is available, otherwise dictionary.
    """
    p = Path(args_path) if args_path else (PROJECT_ROOT / "configs" / "training_args.json")
    if not p.exists():
        raise FileNotFoundError(f"Training arguments file not found at: {p}")

    with open(p, "r", encoding="utf-8") as f:
        args_dict = json.load(f)

    if output_dir is not None:
        args_dict["output_dir"] = str(output_dir)

    args_dict.update(overrides)

    # Device hardware adjustments
    is_cpu = device == "cpu" or (HAS_TORCH and not torch.cuda.is_available())
    if is_cpu:
        args_dict["bf16"] = False
        args_dict["fp16"] = False
        args_dict["gradient_checkpointing"] = False

    if HAS_TRANSFORMERS and TrainingArguments is not None:
        import inspect
        target_cls = SFTConfig if (HAS_TRL and SFTConfig is not None) else TrainingArguments
        valid_params = set(inspect.signature(target_cls.__init__).parameters.keys())

        clean_args = dict(args_dict)
        if "max_seq_length" not in valid_params:
            clean_args.pop("max_seq_length", None)
        if "warmup_ratio" not in valid_params and "warmup_steps" in valid_params:
            clean_args["warmup_steps"] = clean_args.pop("warmup_ratio", 10)
            if isinstance(clean_args["warmup_steps"], float) and clean_args["warmup_steps"] < 1.0:
                clean_args["warmup_steps"] = 10

        filtered = {k: v for k, v in clean_args.items() if k in valid_params}
        return target_cls(**filtered)

    return args_dict


class CompletionOnlyDataCollator:
    """Collates pre-tokenized SFT examples into padded PyTorch tensors.

    Pads sequences dynamically to the longest sequence in the batch using tokenizer.pad_token_id.
    Padded positions in 'labels' are set to -100 so they do not contribute to the loss.
    Crucially: preserves genuine EOS tokens in completions and does NOT mask them.
    """

    def __init__(self, tokenizer: Any, pad_to_multiple_of: Optional[int] = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(tokenizer, "eos_token_id", 0)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            input_ids = list(f["input_ids"])
            attention_mask = list(f.get("attention_mask", [1] * len(input_ids)))
            labels = list(f.get("labels", [-100] * len(input_ids)))

            padding_len = max_len - len(input_ids)
            # Right-padding for batch collation during causal training
            input_ids.extend([self.pad_token_id] * padding_len)
            attention_mask.extend([0] * padding_len)
            labels.extend([-100] * padding_len)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        if HAS_TORCH:
            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
            }

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }


class MockModel:
    """Lightweight mock model for testing when PyTorch is not available."""

    def __init__(self) -> None:
        self.config = type("Config", (), {"vocab_size": 1000})()

    def eval(self) -> MockModel:
        return self

    def parameters(self) -> List[Any]:
        return []


class MockTokenizer:
    """Lightweight mock tokenizer for testing when Transformers is not available."""

    def __init__(self) -> None:
        self.pad_token = "<|pad|>"
        self.pad_token_id = 0
        self.eos_token = "<|eos|>"
        self.eos_token_id = 1
        self.bos_token = "<|bos|>"
        self.bos_token_id = 2
        self.name_or_path = "mock-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        words = text.split()
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        for w in words:
            tokens.append(abs(hash(w)) % 1000 + 10)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return f"mock_decoded_{len(token_ids)}_tokens"

    def save_pretrained(self, out_dir: Union[str, Path]) -> None:
        pass


class MockSFTTrainer:
    """Lightweight deterministic training simulator for CPU and non-torch test environments."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        max_steps: int = 5,
        callbacks: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.output_dir = Path(output_dir)
        self.max_steps = max_steps
        self.callbacks = callbacks or []
        self.global_step = 0
        self.kwargs = kwargs

    def train(self, resume_from_checkpoint: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Simulate training loop, create mock checkpoints, and return train metrics."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        start_step = 0
        if resume_from_checkpoint:
            logger.info(f"Resuming MockSFTTrainer from checkpoint: {resume_from_checkpoint}")
            start_step = 2

        logger.info(f"MockSFTTrainer: training on {len(self.train_dataset)} examples for {self.max_steps} steps...")
        loss_history = []
        for step in range(start_step + 1, self.max_steps + 1):
            simulated_loss = round(max(0.005, 2.5 * (0.6 ** step)), 4)
            loss_history.append({"step": step, "loss": simulated_loss})
            self.global_step = step

            # Trigger registered callbacks
            if self.callbacks:
                mock_state = type("TrainerState", (), {"global_step": step, "log_history": []})()
                for cb in self.callbacks:
                    if hasattr(cb, "on_step_end"):
                        cb.on_step_end(None, mock_state, None, model=self.model, tokenizer=self.tokenizer)

            # Save periodic mock checkpoint
            ckpt_dir = self.output_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "trainer_state.json").write_text(
                json.dumps({"global_step": step, "loss": simulated_loss, "best_metric": None}),
                encoding="utf-8",
            )
            (ckpt_dir / "adapter_config.json").write_text(
                json.dumps({"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]}),
                encoding="utf-8",
            )

        return {
            "global_step": self.global_step,
            "training_loss": loss_history[-1]["loss"] if loss_history else 0.5,
            "loss_history": loss_history,
        }

    def save_model(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Save mock adapter weights and tokenizer."""
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text(
            json.dumps({"r": 16, "lora_alpha": 32, "peft_type": "LORA", "task_type": "CAUSAL_LM"}, indent=2),
            encoding="utf-8",
        )
        (out / "adapter_model.bin").write_text("mock_adapter_weights", encoding="utf-8")
        logger.info(f"Saved mock adapter model to: {out}")


def build_sft_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: List[Dict[str, Any]],
    eval_dataset: Optional[List[Dict[str, Any]]] = None,
    lora_config: Optional[Any] = None,
    training_args: Optional[Any] = None,
    output_dir: Optional[Union[str, Path]] = None,
    max_steps: Optional[int] = None,
    callbacks: Optional[List[Any]] = None,
) -> Any:
    """Build and return an SFTTrainer instance, or MockSFTTrainer in non-torch / mock environments."""
    collator = CompletionOnlyDataCollator(tokenizer=tokenizer)

    is_mock = (
        isinstance(model, MockModel)
        or getattr(model, "__class__", None).__name__ == "MockModel"
        or not HAS_TORCH
        or not HAS_TRANSFORMERS
        or Trainer is None
        or (HAS_TORCH and torch is not None and not isinstance(model, torch.nn.Module))
        or not is_accelerate_available()
    )

    if is_mock:
        logger.info("Building MockSFTTrainer for testing/dry-run environment.")
        return MockSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            output_dir=output_dir or "checkpoints/sft",
            max_steps=max_steps or 5,
            callbacks=callbacks,
        )

    # If PEFT and LoraConfig are available, apply LoRA to model
    if HAS_PEFT and lora_config is not None and get_peft_model is not None:
        if isinstance(lora_config, LoraConfig):
            logger.info("Applying LoRA adapter to base model via PEFT...")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    # Build TRL SFTTrainer if available
    if HAS_TRL and SFTTrainer is not None and isinstance(training_args, SFTConfig):
        logger.info("Instantiating TRL SFTTrainer...")
        return SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=callbacks,
        )

    # Fallback to standard Hugging Face Trainer
    logger.info("Instantiating Hugging Face Trainer with CompletionOnlyDataCollator...")
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )


def run_sft_training(
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    curated_data_path: Union[str, Path] = "data/synthetic/curated/traces_curated_sft.jsonl",
    output_dir: Union[str, Path] = "checkpoints/sft",
    final_adapter_dir: Union[str, Path] = "outputs/sft/adapter_model",
    lora_config_path: Optional[Union[str, Path]] = None,
    training_args_path: Optional[Union[str, Path]] = None,
    resume: bool = True,
    sample_limit: Optional[int] = None,
    device: Optional[str] = None,
    max_steps: Optional[int] = None,
    callbacks: Optional[List[Any]] = None,
    eval_steps: Optional[int] = None,
    eval_limit: int = 100,
    val_path: Optional[Union[str, Path]] = None,
    generations_dir: Union[str, Path] = "logs/generations",
) -> Tuple[Any, Dict[str, Any]]:
    """Execute the complete SFT training pipeline with LoRA and automatic checkpoint resumption.

    Args:
        model_name_or_path: Base model Hub ID or local path.
        curated_data_path: Path to curated SFT JSONL dataset.
        output_dir: Directory to save step checkpoints.
        final_adapter_dir: Directory to copy final trained LoRA adapter.
        lora_config_path: Path to lora_config.yaml.
        training_args_path: Path to training_args.json.
        resume: Whether to resume from latest checkpoint.
        sample_limit: Optional limit for testing/dry runs.
        device: Device string ('cuda', 'cpu').
        max_steps: Optional step limit for testing.
        callbacks: Optional list of Trainer callbacks.
        eval_steps: Optional step interval for periodic validation evaluation.
        eval_limit: Number of validation problems to evaluate per callback step.
        val_path: Optional path to validation dataset file.
        generations_dir: Output directory for callback generations and summaries.

    Returns:
        Tuple of (trainer, run_metadata_dict).
    """
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    adapter_out_path = Path(final_adapter_dir)
    adapter_out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Model and Tokenizer
    logger.info(f"Initializing model and tokenizer: {model_name_or_path}")
    if model_name_or_path.startswith("mock"):
        tokenizer = MockTokenizer()
        model = MockModel()
    else:
        try:
            from src.model import load_tokenizer
            tokenizer = load_tokenizer(model_name_or_path, padding_side="right")
        except Exception as exc:
            logger.warning(f"Could not load real tokenizer ({exc}). Using mock tokenizer.")
            tokenizer = MockTokenizer()

        try:
            from src.model import load_model
            model = load_model(
                model_name_or_path,
                torch_dtype="bfloat16" if device != "cpu" else "float32",
                device_map="auto" if device != "cpu" else "cpu",
            )
        except Exception as exc:
            logger.warning(f"Could not load real model ({exc}). Using mock model.")
            model = MockModel()

    eos_str = getattr(tokenizer, "eos_token", "<|eos|>") or "<|eos|>"

    # 2. Load and Tokenize Curated Dataset
    c_path = Path(curated_data_path)
    if not c_path.is_absolute():
        c_path = PROJECT_ROOT / c_path

    logger.info(f"Loading SFT prompt-completion examples from: {c_path}")
    sft_examples = load_sft_dataset(c_path, eos_token=eos_str, limit=sample_limit)
    logger.info(f"Loaded {len(sft_examples)} SFT examples. Pre-tokenizing with completion-only loss masking...")

    tokenized_dataset = []
    for ex in sft_examples:
        tok_ex = tokenize_sft_example(ex, tokenizer, max_seq_length=512)
        tokenized_dataset.append(tok_ex)

    # 3. Load Configurations
    lora_cfg = load_lora_config(config_path=lora_config_path)
    train_args = load_training_arguments(
        args_path=training_args_path,
        output_dir=out_dir_path,
        device=device,
        max_steps=max_steps if max_steps else -1,
    )

    # 4. Check for Checkpoint Resumption
    latest_checkpoint: Optional[Path] = None
    if resume and out_dir_path.exists():
        latest_checkpoint = find_latest_checkpoint(out_dir_path, complete_only=True)
        if latest_checkpoint:
            logger.info(f"Found completed checkpoint to resume from: {latest_checkpoint}")

    # 5. Build Callbacks and Trainer
    active_callbacks = list(callbacks or [])
    if eval_steps is not None:
        from src.training.callbacks import ValidationEvalCallback
        val_callback = ValidationEvalCallback(
            val_path=val_path,
            tokenizer=tokenizer,
            eval_steps=eval_steps,
            output_dir=generations_dir,
            eval_limit=eval_limit,
        )
        active_callbacks.append(val_callback)

    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        lora_config=lora_cfg,
        training_args=train_args,
        output_dir=out_dir_path,
        max_steps=max_steps,
        callbacks=active_callbacks if active_callbacks else None,
    )

    logger.info("Starting SFT training run...")
    train_result = trainer.train(resume_from_checkpoint=str(latest_checkpoint) if latest_checkpoint else None)

    # 6. Save Adapter & Prune
    logger.info(f"Saving final trained adapter to: {adapter_out_path}")
    if hasattr(trainer, "save_model"):
        trainer.save_model(str(adapter_out_path))

    # Save tokenizer with adapter
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(adapter_out_path))

    # Prune older checkpoints to keep only the 5 most recent
    pruned = prune_checkpoints(out_dir_path, keep=5)
    if pruned:
        logger.info(f"Pruned {len(pruned)} older checkpoints.")

    # 7. Record Provenance (Design Rule 5)
    config_snapshot = {
        "model_name_or_path": str(model_name_or_path),
        "curated_data_path": str(c_path),
        "total_examples": len(sft_examples),
        "lora_config": lora_cfg.to_dict() if hasattr(lora_cfg, "to_dict") else lora_cfg,
        "train_result": train_result,
    }
    run_record = collect_run_info(stage="sft", config=config_snapshot, description="sft_training_run")
    save_run_info(stage="sft", output_dir=adapter_out_path, config=config_snapshot, description="sft_training_run")
    logger.info(f"Saved run provenance to: {adapter_out_path / 'run_info.json'}")

    return trainer, run_record
