"""Generation engine for mathematical reasoning (Hugging Face and vLLM backends).

Features:
- Batched inference with left-padding for decoder-only causal language models.
- Stop rules: EOS token, stop strings ('Question:' for few-shot prompts, triple newline) applied
  during generation on both backends, and truncation at the line after '#### <answer>'.
- Per-completion token counts and finish reason ('stop' or 'length'), so truncation is measured
  from what the model actually generated, not from batch padding.
- LoRA adapters on both backends (vLLM via LoRARequest).
- Resumable chunked JSONL output that skips finished problems on rerun. With a run_config, the
  output file is tied to one run (model, adapter, prompt, decoding, seed): resuming with a
  different run raises instead of silently reusing another model's generations.
- Seeded sampling for reproducible sampled and self-consistency runs.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.math_utils.answer_extraction import extract_answer
from src.math_utils.normalization import is_equivalent

logger = logging.getLogger(__name__)

DEFAULT_STOP_SEQUENCES = ["Question:", "\n\n\n"]

# vLLM only accepts these values for max_lora_rank
VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# Compatibility shim for PyTorch 2.14+ where torch.ops.aten.cholesky was removed
# but vLLM's internal env_override._patch_inductor_fallback_allow_list expects it
if HAS_TORCH:
    try:
        _ = torch.ops.aten.cholesky
    except AttributeError:
        try:
            torch.ops.aten.__dict__["cholesky"] = getattr(
                torch.ops.aten, "linalg_cholesky", lambda *a, **kw: None
            )
        except Exception:
            pass

# Optional vllm import
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAS_VLLM = True
except Exception:
    LLM = None
    SamplingParams = None
    LoRARequest = None
    HAS_VLLM = False


class GenerationRunMismatchError(RuntimeError):
    """Raised when an existing generations file belongs to a different run."""


def is_vllm_available() -> bool:
    """Check if vLLM package is installed and a CUDA GPU is present."""
    return HAS_VLLM and HAS_TORCH and torch.cuda.is_available()


def resolve_generation_backend(backend: str = "auto") -> str:
    """Resolve generation backend: 'auto', 'vllm', or 'hf'.

    - 'auto': chooses 'vllm' if available with CUDA GPU, otherwise falls back to 'hf'.
    - 'vllm': requires vllm; falls back with warning to 'hf' if not installed or no CUDA.
    - 'hf': always uses Hugging Face Transformers backend.
    """
    req = (backend or "auto").lower().strip()
    if req == "auto":
        if is_vllm_available():
            logger.info("Auto-selected generation backend: 'vllm' (vLLM installed with CUDA GPU)")
            return "vllm"
        return "hf"

    if req == "vllm":
        if not HAS_VLLM:
            logger.warning("vLLM requested but not installed. Falling back to Hugging Face backend ('hf').")
            return "hf"
        if not (HAS_TORCH and torch.cuda.is_available()):
            logger.warning("vLLM requested but CUDA GPU is not available. Falling back to Hugging Face backend ('hf').")
            return "hf"
        return "vllm"

    return "hf"


def read_adapter_rank(adapter_path: Union[str, Path]) -> int:
    """LoRA rank 'r' from a PEFT adapter directory (adapter_config.json)."""
    config_file = Path(adapter_path) / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_path}; is this a PEFT LoRA adapter?")
    return int(json.loads(config_file.read_text(encoding="utf-8"))["r"])


def vllm_max_lora_rank(rank: int) -> int:
    """Smallest rank vLLM accepts that fits the adapter."""
    for allowed in VLLM_LORA_RANKS:
        if rank <= allowed:
            return allowed
    raise ValueError(f"LoRA rank {rank} exceeds vLLM's maximum ({VLLM_LORA_RANKS[-1]})")


def load_vllm_model(
    model_name_or_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 2048,
    trust_remote_code: bool = True,
    adapter_path: Optional[Union[str, Path]] = None,
    seed: int = 0,
    **kwargs: Any,
) -> Any:
    """Load an LLM instance from vLLM. With an adapter path, LoRA support is enabled for it."""
    if not HAS_VLLM:
        raise ImportError(
            "vllm is not installed. Run 'pip install -r requirements-vllm.txt' "
            "or use backend='hf'."
        )
    if adapter_path is not None:
        kwargs.setdefault("enable_lora", True)
        kwargs.setdefault("max_lora_rank", vllm_max_lora_rank(read_adapter_rank(adapter_path)))
        kwargs.setdefault("max_loras", 1)
    return LLM(
        model=model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        seed=seed,
        **kwargs,
    )


def make_lora_request(adapter_path: Optional[Union[str, Path]]) -> Any:
    """vLLM LoRARequest for an adapter directory, or None without one."""
    if adapter_path is None:
        return None
    if LoRARequest is None:
        raise RuntimeError("vllm is required to evaluate a LoRA adapter on the vLLM backend.")
    return LoRARequest("adapter", 1, str(adapter_path))


def shutdown_model(model: Any, backend: str) -> None:
    """Shut down a vLLM engine so its GPU memory can be reclaimed.

    A vLLM engine holds gpu_memory_utilization of the card until it is shut down; creating a
    second engine in the same process without this runs out of memory. The caller must also
    drop its own reference and then call free_gpu_memory().
    """
    if model is None or backend != "vllm":
        return
    engine = getattr(model, "llm_engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception as e:  # best effort: the API differs between vLLM versions
            logger.debug(f"vLLM engine shutdown failed: {e}")
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        logger.debug(f"vLLM parallel state cleanup skipped: {e}")


def free_gpu_memory() -> None:
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        logger.info(f"GPU memory after release: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")


def generate_batch_vllm(
    llm: Any,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    do_sample: bool = False,
    stop_sequences: Optional[Sequence[str]] = None,
    num_return_sequences: int = 1,
    seed: Optional[int] = None,
    lora_request: Any = None,
) -> List[List[Dict[str, Any]]]:
    """Execute batched generation using vLLM for high-throughput inference.

    Returns:
        List of lists (one list of completion dicts per input prompt).
    """
    if not HAS_VLLM:
        raise RuntimeError("vllm is required for vLLM generation backend.")

    stops = list(stop_sequences) if stop_sequences else list(DEFAULT_STOP_SEQUENCES)

    params: Dict[str, Any] = {
        "n": num_return_sequences,
        "temperature": temperature if do_sample else 0.0,
        "top_p": top_p if do_sample else 1.0,
        "top_k": top_k if (do_sample and top_k > 0) else -1,
        "max_tokens": max_new_tokens,
        "stop": stops,
    }
    if do_sample and seed is not None:
        # Per-request seed: each prompt's samples are reproducible regardless of batch composition
        params["seed"] = seed
    sampling_params = SamplingParams(**params)

    if lora_request is not None:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    results: List[List[Dict[str, Any]]] = []
    for req_output in outputs:
        prompt_len = len(req_output.prompt_token_ids)
        completions_list: List[Dict[str, Any]] = []

        for comp in req_output.outputs:
            raw_text = comp.text
            clean_text = truncate_at_stop_sequences(raw_text, stop_sequences=stops)
            extracted, method = extract_answer(clean_text)
            finish_reason = getattr(comp, "finish_reason", None)

            completions_list.append({
                "completion_raw": raw_text,
                "completion": clean_text,
                "extracted_answer": extracted,
                "extraction_method": method,
                "prompt_tokens": prompt_len,
                "completion_tokens": len(comp.token_ids),
                "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
            })
        results.append(completions_list)

    return results


def truncate_at_stop_sequences(
    text: str,
    stop_sequences: Optional[Sequence[str]] = None,
) -> str:
    """Truncate generated text at stop sequences and after the '#### <answer>' line.

    Stop Rules:
    1. Truncate at explicit stop strings (e.g. 'Question:', '\n\n\n').
    2. Truncate at the newline immediately following '#### <number>'.
    """
    if not text:
        return ""

    truncated = text

    # Rule 1: Truncate at explicit stop sequences
    stops = list(stop_sequences) if stop_sequences else list(DEFAULT_STOP_SEQUENCES)
    for stop in stops:
        idx = truncated.find(stop)
        if idx != -1:
            truncated = truncated[:idx]

    # Rule 2: Truncate at newline after '#### <number>' line
    hash_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", truncated)
    if hash_match:
        end_idx = hash_match.end()
        newline_idx = truncated.find("\n", end_idx)
        if newline_idx != -1:
            truncated = truncated[:newline_idx]

    return truncated.strip()


def load_existing_generations(output_file: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Read an existing JSONL generation file and index by problem_id."""
    path = Path(output_file)
    if not path.exists():
        return {}

    existing: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                pid = record.get("problem_id")
                if pid:
                    existing[pid] = record
            except json.JSONDecodeError as e:
                logger.warning(f"Malformed JSON on line {line_num} in {path}: {e}")

    return existing


def append_generation_records(
    output_file: Union[str, Path],
    records: List[Dict[str, Any]],
) -> None:
    """Append a list of generation records to a JSONL file."""
    if not records:
        return

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_meta_path(output_file: Union[str, Path]) -> Path:
    path = Path(output_file)
    return path.with_name(path.stem + ".meta.json")


def run_config_key(run_config: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(run_config, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def check_or_write_run_meta(output_file: Union[str, Path], run_config: Dict[str, Any], has_records: bool) -> None:
    """Tie a generations file to one run. Raise if existing records came from a different run."""
    meta_path = run_meta_path(output_file)
    key = run_config_key(run_config)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("run_key") != key:
            old = meta.get("run_config", {})
            diff = {k: (old.get(k), run_config.get(k)) for k in sorted(set(old) | set(run_config)) if old.get(k) != run_config.get(k)}
            raise GenerationRunMismatchError(
                f"{output_file} holds generations from a different run; differing settings (old, new): {diff}. "
                "Use a different output directory, or move the old file aside."
            )
        return
    if has_records:
        raise GenerationRunMismatchError(
            f"{output_file} has generations but no {meta_path.name}, so their model and settings are unknown. "
            "Move the file aside (or use a different output directory) and rerun."
        )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"run_key": key, "run_config": run_config}, indent=2, default=str), encoding="utf-8")


def _eos_token_ids(model: Any, tokenizer: Any) -> set:
    ids = set()
    for source in (getattr(getattr(model, "generation_config", None), "eos_token_id", None), tokenizer.eos_token_id):
        if isinstance(source, int):
            ids.add(source)
        elif isinstance(source, (list, tuple)):
            ids.update(int(x) for x in source)
    return ids


def generate_batch_hf(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    do_sample: bool = False,
    stop_sequences: Optional[Sequence[str]] = None,
    num_return_sequences: int = 1,
) -> List[List[Dict[str, Any]]]:
    """Execute batched generation using Hugging Face Transformers.

    Stop strings end generation (not just the decoded text), and each completion's token count
    stops at its own end: batch padding after a finished sequence is not counted.

    Returns:
        List of lists (one list of completion dicts per input prompt).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for Hugging Face model generation.")

    if tokenizer.padding_side != "left":
        logger.warning(f"Tokenizer padding_side is '{tokenizer.padding_side}'. Forcing to 'left'.")
        tokenizer.padding_side = "left"

    stops = list(stop_sequences) if stop_sequences else list(DEFAULT_STOP_SEQUENCES)

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")

    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_lengths = [int(mask.sum().item()) for mask in attention_mask]

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "num_return_sequences": num_return_sequences,
        "stop_strings": stops,
        "tokenizer": tokenizer,
    }

    if do_sample and temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        # 0 disables top-k explicitly; leaving it unset would apply the model's default (often 50)
        gen_kwargs["top_k"] = top_k if top_k > 0 else 0
    else:
        gen_kwargs["do_sample"] = False
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None
        gen_kwargs["top_k"] = None

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    eos_ids = _eos_token_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id
    prompt_len = input_ids.shape[1]  # left padding: every prompt ends at the same position
    results: List[List[Dict[str, Any]]] = [[] for _ in range(len(prompts))]

    for idx in range(len(outputs)):
        prompt_idx = idx // num_return_sequences
        new_ids = outputs[idx][prompt_len:].tolist()

        # The sequence ends at its first EOS (inclusive); padding added after it (or after a
        # stop string) while longer sequences in the batch kept going is not generation.
        n_gen, hit_eos = len(new_ids), False
        for j, token_id in enumerate(new_ids):
            if token_id in eos_ids:
                n_gen, hit_eos = j + 1, True
                break
            if token_id == pad_id:
                n_gen = j
                break

        completion_text = tokenizer.decode(new_ids[:n_gen], skip_special_tokens=True)
        hit_stop_string = any(s in completion_text for s in stops)
        finish_reason = "stop" if (hit_eos or hit_stop_string or n_gen < max_new_tokens) else "length"

        clean_text = truncate_at_stop_sequences(completion_text, stop_sequences=stops)
        extracted, method = extract_answer(clean_text)

        results[prompt_idx].append({
            "completion_raw": completion_text,
            "completion": clean_text,
            "extracted_answer": extracted,
            "extraction_method": method,
            "prompt_tokens": prompt_lengths[prompt_idx],
            "completion_tokens": n_gen,
            "finish_reason": finish_reason,
        })

    return results


def _decoding_params(preset: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if preset == "greedy":
        return {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "do_sample": False, "num_return_sequences": 1}
    if preset == "sampled":
        return {
            "temperature": cfg.get("temperature", 0.7), "top_p": cfg.get("top_p", 0.95),
            "top_k": cfg.get("top_k", -1), "do_sample": True, "num_return_sequences": 1,
        }
    if preset == "self_consistency":
        return {
            "temperature": cfg.get("temperature", 0.7), "top_p": cfg.get("top_p", 0.95),
            "top_k": cfg.get("top_k", -1), "do_sample": True, "num_return_sequences": cfg.get("num_samples", 8),
        }
    return {
        "temperature": cfg.get("temperature", 0.0), "top_p": cfg.get("top_p", 1.0), "top_k": cfg.get("top_k", -1),
        "do_sample": cfg.get("do_sample", False), "num_return_sequences": cfg.get("num_return_sequences", 1),
    }


def run_generation_pipeline(
    problems: List[Dict[str, Any]],
    output_file: Union[str, Path],
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    model_name_or_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    backend: str = "auto",
    batch_size: int = 4,
    preset: str = "greedy",
    generation_config: Optional[Dict[str, Any]] = None,
    prompt_formatter: Optional[Any] = None,
    stop_sequences: Optional[Sequence[str]] = None,
    seed: int = 42,
    vllm_chunk_size: int = 128,
    run_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run generation on a list of problems with chunked resumable JSONL output.

    Skips problems that are already completed in `output_file` on rerun.

    Args:
        problems: List of problem dicts containing 'problem_id', 'question', and 'gold_answer'.
        output_file: Path to destination JSONL file.
        model: Pre-loaded model (optional; loads via model_name_or_path if None, and is then
            released at the end so the next run can load another model).
        tokenizer: Pre-loaded tokenizer (optional; loads via model_name_or_path if None).
        model_name_or_path: Model ID or path if model/tokenizer not passed.
        adapter_path: Optional LoRA adapter path (both backends).
        batch_size: Batch size for the Hugging Face backend.
        preset: Generation preset name ('greedy', 'sampled', 'self_consistency').
        generation_config: Optional config dict overriding defaults.
        prompt_formatter: Function mapping problem to prompt string.
        stop_sequences: Stop sequences for truncation.
        seed: Sampling seed.
        vllm_chunk_size: Prompts per vLLM call (each chunk is saved before the next starts).
        run_config: Identity of this run (model, adapter, prompt format, decoding, seed). When
            given, the output file is tied to it and resuming with a different run raises.

    Returns:
        Full list of result records (combining previously existing and newly generated).
    """
    output_path = Path(output_file)
    cfg = generation_config or {}
    max_new_tokens = cfg.get("max_new_tokens", 512)
    stops = list(stop_sequences or cfg.get("stop_sequences", DEFAULT_STOP_SEQUENCES))
    decoding = _decoding_params(preset, cfg)
    active_backend = resolve_generation_backend(backend)

    # A model passed in decides the backend under 'auto': a Hugging Face model with its tokenizer
    # must not be handed to vLLM just because vLLM is installed on this runtime
    if model is not None and (backend or "auto").lower() == "auto":
        is_vllm_engine = isinstance(LLM, type) and isinstance(model, LLM)
        if is_vllm_engine:
            active_backend = "vllm"
        elif tokenizer is not None:
            active_backend = "hf"

    # Default prompt formatter: 8-shot few-shot prompt
    if prompt_formatter is None:
        from src.data.formatting import format_few_shot_prompt

        def default_formatter(prob: Dict[str, Any]) -> str:
            return format_few_shot_prompt(prob["question"])

        prompt_formatter = default_formatter

    # 1. Check existing generations for resumption
    existing_records = load_existing_generations(output_path)
    if run_config is not None:
        full_config = dict(
            run_config, backend=active_backend, preset=preset, max_new_tokens=max_new_tokens,
            stop_sequences=stops, seed=seed, **decoding,
        )
        check_or_write_run_meta(output_path, full_config, has_records=bool(existing_records))

    problems_by_id = {p.get("problem_id"): p for p in problems}
    for pid, rec in existing_records.items():
        if "prompt" in rec and pid in problems_by_id and rec["prompt"] != prompt_formatter(problems_by_id[pid]):
            raise GenerationRunMismatchError(
                f"{output_path}: the stored prompt for {pid} differs from the current prompt; "
                "the file was generated with another prompt format. Use a different output directory."
            )

    already_done_ids = set(existing_records.keys())
    pending_problems = [p for p in problems if p.get("problem_id") not in already_done_ids]
    logger.info(
        f"Total problems: {len(problems)}. Already completed: {len(already_done_ids)}. "
        f"Remaining: {len(pending_problems)}."
    )

    if not pending_problems:
        logger.info(f"All {len(problems)} problems already completed in {output_path}. Skipping generation.")
        return [existing_records[p["problem_id"]] for p in problems if p.get("problem_id") in existing_records]

    # 2. Lazy load model (and release it at the end if loaded here)
    owns_model = model is None
    lora_request = None
    if active_backend == "vllm":
        if model is None:
            if not model_name_or_path:
                raise ValueError("Must provide either model or model_name_or_path for vLLM.")
            logger.info(f"Loading vLLM model: {model_name_or_path}" + (f" + adapter {adapter_path}" if adapter_path else ""))
            model = load_vllm_model(model_name_or_path, adapter_path=adapter_path, seed=seed)
        lora_request = make_lora_request(adapter_path)
    elif model is None or tokenizer is None:
        if not model_name_or_path:
            raise ValueError("Must provide either (model, tokenizer) or model_name_or_path.")
        from src.model import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(model_name_or_path=model_name_or_path, adapter_path=adapter_path)
        owns_model = True

    # 3. Generate chunk by chunk, saving each chunk before starting the next
    chunk_size = vllm_chunk_size if active_backend == "vllm" else batch_size
    try:
        for i in range(0, len(pending_problems), chunk_size):
            batch_probs = pending_problems[i : i + chunk_size]
            batch_prompts = [prompt_formatter(p) for p in batch_probs]

            if active_backend == "vllm":
                batch_outputs = generate_batch_vllm(
                    llm=model,
                    prompts=batch_prompts,
                    max_new_tokens=max_new_tokens,
                    stop_sequences=stops,
                    seed=seed,
                    lora_request=lora_request,
                    **decoding,
                )
            else:
                if decoding["do_sample"] and HAS_TORCH:
                    from transformers import set_seed

                    set_seed(seed + i)  # per chunk, so a resumed run samples the same as an uninterrupted one
                batch_outputs = generate_batch_hf(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    max_new_tokens=max_new_tokens,
                    stop_sequences=stops,
                    **decoding,
                )

            batch_records = [
                _build_record(prob, prompt, completions, preset, decoding["num_return_sequences"])
                for prob, prompt, completions in zip(batch_probs, batch_prompts, batch_outputs)
            ]
            for rec in batch_records:
                existing_records[rec["problem_id"]] = rec

            # Immediately append chunk to disk for crash resilience
            append_generation_records(output_path, batch_records)
    finally:
        if owns_model:
            shutdown_model(model, active_backend)
            model = None
            tokenizer = None
            free_gpu_memory()

    # Return full ordered records
    return [existing_records[p["problem_id"]] for p in problems if p.get("problem_id") in existing_records]


def _build_record(
    prob: Dict[str, Any], prompt: str, completions: List[Dict[str, Any]], preset: str, num_return_sequences: int
) -> Dict[str, Any]:
    gold = prob.get("gold_answer") or prob.get("answer")
    base = {
        "problem_id": prob.get("problem_id"),
        "question": prob.get("question"),
        "gold_answer": str(gold) if gold is not None else None,
        "prompt": prompt,
        "preset": preset,
    }
    if num_return_sequences == 1:
        comp = completions[0]
        extracted = comp["extracted_answer"]
        return dict(
            base,
            generated_text=comp["completion"],
            extracted_answer=extracted,
            extraction_method=comp.get("extraction_method", "none"),
            is_correct=is_equivalent(extracted, gold) if gold is not None else False,
            prompt_tokens=comp["prompt_tokens"],
            completion_tokens=comp["completion_tokens"],
            finish_reason=comp.get("finish_reason"),
        )
    return dict(
        base,
        generated_texts=[c["completion"] for c in completions],
        extracted_answers=[c["extracted_answer"] for c in completions],
        extraction_methods=[c.get("extraction_method", "none") for c in completions],
        prompt_tokens=[c.get("prompt_tokens", 0) for c in completions],
        completion_tokens=[c.get("completion_tokens", 0) for c in completions],
        finish_reasons=[c.get("finish_reason") for c in completions],
        num_samples=len(completions),
    )
