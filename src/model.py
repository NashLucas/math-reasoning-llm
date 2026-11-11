"""Model loading, LoRA adapter attachment, and tokenizer preparation.

Supports loading causal language models in bfloat16, attaching LoRA adapters
via PEFT, merging adapters, and configuring left-padding for batched decoder-only generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Optional torch import to allow clean importing in non-torch environments
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    PeftModel = None
    HAS_PEFT = False


def resolve_torch_dtype(dtype: Optional[Union[str, Any]] = "bfloat16") -> Any:
    """Resolve a string or torch.dtype representation to a PyTorch dtype.

    Falls back to float32 if torch is available but CUDA/bfloat16 is unsupported,
    or returns the string representation if torch is not installed.
    """
    if not HAS_TORCH:
        return dtype

    if dtype is None or dtype == "auto":
        return "auto"

    if isinstance(dtype, str):
        dtype_lower = dtype.lower().strip()
        if dtype_lower in ["bfloat16", "bf16"]:
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                logger.warning("CUDA is available but bfloat16 is not supported on this GPU. Falling back to float16.")
                return torch.float16
            return torch.bfloat16
        if dtype_lower in ["float16", "fp16"]:
            return torch.float16
        if dtype_lower in ["float32", "fp32"]:
            return torch.float32

    return dtype


def resolve_device_map(device_preference: Optional[Union[str, Dict[str, Any]]] = "auto") -> Union[str, Dict[str, Any]]:
    """Determine device placement for model loading ('auto', 'cuda', 'cpu', or custom map)."""
    if not HAS_TORCH:
        return device_preference or "cpu"

    if device_preference is None or device_preference == "auto":
        return "auto" if torch.cuda.is_available() else "cpu"

    return device_preference


def load_tokenizer(
    model_name_or_path: Union[str, Path],
    padding_side: str = "left",
    hf_cache_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Any:
    """Load and configure a tokenizer with left-padding for decoder-only batched generation.

    Args:
        model_name_or_path: Hugging Face hub ID or local directory path.
        padding_side: Tokenizer padding side ('left' required for causal LM batch generation).
        hf_cache_dir: Optional cache directory for model downloads.
        **kwargs: Extra keyword arguments forwarded to AutoTokenizer.from_pretrained.

    Returns:
        Configured AutoTokenizer instance.
    """
    if not HAS_TRANSFORMERS:
        raise RuntimeError("Transformers is not installed. Please install transformers to load tokenizers.")

    model_path_str = str(model_name_or_path)
    cache_dir_str = str(hf_cache_dir) if hf_cache_dir else None

    logger.info(f"Loading tokenizer from: {model_path_str} (padding_side={padding_side})")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_str,
        cache_dir=cache_dir_str,
        trust_remote_code=kwargs.pop("trust_remote_code", True),
        **kwargs,
    )

    # Enforce left padding for decoder-only generation
    tokenizer.padding_side = padding_side

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Set tokenizer.pad_token to eos_token ('{tokenizer.eos_token}', id={tokenizer.eos_token_id})")
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            logger.info("Added special token '<|pad|>' as tokenizer.pad_token")

    return tokenizer


def load_model(
    model_name_or_path: Union[str, Path],
    adapter_path: Optional[Union[str, Path]] = None,
    torch_dtype: Optional[Union[str, Any]] = "bfloat16",
    device_map: Optional[Union[str, Dict[str, Any]]] = "auto",
    hf_cache_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Any:
    """Load a causal language model in bfloat16, with or without a LoRA adapter.

    Args:
        model_name_or_path: Base model Hugging Face ID or local path.
        adapter_path: Optional local path or Hub ID for a LoRA adapter.
        torch_dtype: Desired torch dtype ('bfloat16', 'float16', 'float32', or 'auto').
        device_map: Device allocation strategy ('auto', 'cuda', 'cpu', or custom dict).
        hf_cache_dir: Optional local directory for caching model weights.
        **kwargs: Additional parameters forwarded to AutoModelForCausalLM.from_pretrained.

    Returns:
        PreTrainedModel or PeftModel in evaluation mode.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed. Please install torch to load models.")
    if not HAS_TRANSFORMERS:
        raise RuntimeError("Transformers is not installed. Please install transformers to load models.")

    model_path_str = str(model_name_or_path)
    resolved_dtype = resolve_torch_dtype(torch_dtype)
    resolved_device = resolve_device_map(device_map)
    cache_dir_str = str(hf_cache_dir) if hf_cache_dir else None

    load_in_4bit = kwargs.pop("load_in_4bit", False)
    load_in_8bit = kwargs.pop("load_in_8bit", False)

    from_pretrained_kwargs = {
        "device_map": resolved_device,
        "cache_dir": cache_dir_str,
        "trust_remote_code": kwargs.pop("trust_remote_code", True),
    }

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            bf16_ok = (
                HAS_TORCH
                and torch.cuda.is_available()
                and hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            )
            if isinstance(resolved_dtype, torch.dtype):
                if resolved_dtype == torch.bfloat16 and not bf16_ok:
                    compute_dtype = torch.float16
                else:
                    compute_dtype = resolved_dtype
            else:
                compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

            from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info(f"Loading base model '{model_path_str}' in 4-bit NF4 (compute_dtype={compute_dtype}, device={resolved_device})")
        except Exception as e:
            logger.warning(f"BitsAndBytesConfig failed ({e}); passing load_in_4bit=True directly")
            from_pretrained_kwargs["load_in_4bit"] = True
    elif load_in_8bit:
        logger.info(f"Loading base model '{model_path_str}' in 8-bit (device={resolved_device})")
        from_pretrained_kwargs["load_in_8bit"] = True
    else:
        bf16_ok = (
            HAS_TORCH
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        if HAS_TORCH and resolved_dtype == torch.bfloat16 and not bf16_ok:
            logger.info("bfloat16 requested but CUDA device does not support it (compute < 8.0). Falling back to float16.")
            resolved_dtype = torch.float16

        logger.info(f"Loading base model '{model_path_str}' (dtype={resolved_dtype}, device={resolved_device})")
        from_pretrained_kwargs["dtype"] = resolved_dtype

    from_pretrained_kwargs.update(kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        model_path_str,
        **from_pretrained_kwargs,
    )

    # Attach LoRA adapter if provided
    if adapter_path is not None:
        if not HAS_PEFT:
            raise RuntimeError("PEFT is not installed. Please install peft to attach LoRA adapters.")

        adapter_path_str = str(adapter_path)
        logger.info(f"Attaching LoRA adapter from '{adapter_path_str}'...")
        model = PeftModel.from_pretrained(model, adapter_path_str)
        logger.info("LoRA adapter successfully attached.")

    model.eval()
    return model


def merge_adapter_to_base(model: Any) -> Any:
    """Merge LoRA adapter weights into the base model and unload the adapter (for Phase 8 export)."""
    if HAS_PEFT and isinstance(model, PeftModel):
        logger.info("Merging LoRA adapter weights into base model...")
        return model.merge_and_unload()
    logger.info("Model is not a PeftModel. Returning model as is.")
    return model


def load_model_and_tokenizer(
    model_name_or_path: Union[str, Path],
    adapter_path: Optional[Union[str, Path]] = None,
    torch_dtype: Optional[Union[str, Any]] = "bfloat16",
    device_map: Optional[Union[str, Dict[str, Any]]] = "auto",
    padding_side: str = "left",
    hf_cache_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Load both model and tokenizer with consistent settings and left-padding.

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = load_tokenizer(
        model_name_or_path,
        padding_side=padding_side,
        hf_cache_dir=hf_cache_dir,
    )
    model = load_model(
        model_name_or_path,
        adapter_path=adapter_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        hf_cache_dir=hf_cache_dir,
        **kwargs,
    )
    return model, tokenizer
