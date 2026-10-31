"""Generation engine for mathematical reasoning with Hugging Face backend.

Features:
- Batched inference with left-padding for decoder-only causal language models.
- Stop rules: EOS token, new 'Question:' prompt hallucination stop, and line after '####'.
- Resumable chunked JSONL output: automatically skips finished problems on rerun.
- Configurable decoding presets: greedy, sampled, and self-consistency.
- Answer extraction and correctness verification against gold targets.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.math_utils.answer_extraction import extract_answer
from src.math_utils.normalization import is_equivalent

logger = logging.getLogger(__name__)

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# Optional vllm import
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    LLM = None
    SamplingParams = None
    HAS_VLLM = False


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


def load_vllm_model(
    model_name_or_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 2048,
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> Any:
    """Load an LLM instance from vLLM for high-throughput inference."""
    if not HAS_VLLM:
        raise ImportError(
            "vllm is not installed. Run 'pip install -r requirements-vllm.txt' "
            "or use backend='hf'."
        )
    return LLM(
        model=model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


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
) -> List[List[Dict[str, Any]]]:
    """Execute batched generation using vLLM for high-throughput inference.

    Args:
        llm: Loaded vLLM LLM instance.
        prompts: List of prompt strings.
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        top_k: Top-k filtering limit.
        do_sample: Whether to sample or decode greedily.
        stop_sequences: List of strings to truncate upon.
        num_return_sequences: Number of completions per prompt.

    Returns:
        List of lists (one list of completion dicts per input prompt).
    """
    if not HAS_VLLM:
        raise RuntimeError("vllm is required for vLLM generation backend.")

    stops = list(stop_sequences) if stop_sequences else ["Question:", "\n\n\n"]

    sampling_params = SamplingParams(
        n=num_return_sequences,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if (do_sample and top_k > 0) else -1,
        max_tokens=max_new_tokens,
        stop=stops,
    )

    outputs = llm.generate(prompts, sampling_params)

    results: List[List[Dict[str, Any]]] = []
    for req_output in outputs:
        prompt_len = len(req_output.prompt_token_ids)
        completions_list: List[Dict[str, Any]] = []

        for comp in req_output.outputs:
            raw_text = comp.text
            clean_text = truncate_at_stop_sequences(raw_text, stop_sequences=stop_sequences)
            extracted, method = extract_answer(clean_text)

            completions_list.append({
                "completion_raw": raw_text,
                "completion": clean_text,
                "extracted_answer": extracted,
                "extraction_method": method,
                "prompt_tokens": prompt_len,
                "completion_tokens": len(comp.token_ids),
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
    stops = list(stop_sequences) if stop_sequences else ["Question:", "\n\n\n"]
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
    """Read an existing JSONL generation file and index by problem_id.

    Returns:
        Dict mapping problem_id to the existing result record.
    """
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

    Args:
        model: Loaded AutoModelForCausalLM instance.
        tokenizer: Configured AutoTokenizer with left-padding.
        prompts: List of prompt strings.
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).
        top_p: Nucleus sampling probability.
        top_k: Top-k filtering limit.
        do_sample: Whether to sample or decode greedily.
        stop_sequences: List of strings to truncate upon.
        num_return_sequences: Number of completions per prompt.

    Returns:
        List of lists (one list of completion dicts per input prompt).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for Hugging Face model generation.")

    if tokenizer.padding_side != "left":
        logger.warning(f"Tokenizer padding_side is '{tokenizer.padding_side}'. Forcing to 'left'.")
        tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_lengths = [int(mask.sum().item()) for mask in attention_mask]

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": num_return_sequences,
    }

    if do_sample and temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
    else:
        gen_kwargs["do_sample"] = False
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    # outputs shape: (batch_size * num_return_sequences, seq_len)
    batch_size = len(prompts)
    results: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]

    for idx in range(len(outputs)):
        prompt_idx = idx // num_return_sequences
        raw_output_ids = outputs[idx]

        # Extract only the newly generated tokens (slice off prompt tokens)
        # Left-padding means prompt tokens are at the end of input_ids
        prompt_len = input_ids.shape[1]
        new_token_ids = raw_output_ids[prompt_len:]
        completion_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

        clean_text = truncate_at_stop_sequences(completion_text, stop_sequences=stop_sequences)
        extracted, method = extract_answer(clean_text)

        results[prompt_idx].append({
            "completion_raw": completion_text,
            "completion": clean_text,
            "extracted_answer": extracted,
            "extraction_method": method,
            "prompt_tokens": prompt_lengths[prompt_idx],
            "completion_tokens": len(new_token_ids),
        })

    return results


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
) -> List[Dict[str, Any]]:
    """Run generation on a list of problems with chunked resumable JSONL output.

    Skips problems that are already completed in `output_file` on rerun.

    Args:
        problems: List of problem dicts containing 'problem_id', 'question', and 'gold_answer'.
        output_file: Path to destination JSONL file.
        model: Pre-loaded model (optional; loads via model_name_or_path if None).
        tokenizer: Pre-loaded tokenizer (optional; loads via model_name_or_path if None).
        model_name_or_path: Model ID or path if model/tokenizer not passed.
        adapter_path: Optional LoRA adapter path.
        batch_size: Batch size for model generation.
        preset: Generation preset name ('greedy', 'sampled', 'self_consistency').
        generation_config: Optional config dict overriding defaults.
        prompt_formatter: Function mapping problem to prompt string.
        stop_sequences: Stop sequences for truncation.

    Returns:
        Full list of result records (combining previously existing and newly generated).
    """
    output_path = Path(output_file)

    # 1. Check existing generations for resumption
    existing_records = load_existing_generations(output_path)
    already_done_ids = set(existing_records.keys())

    pending_problems = [p for p in problems if p.get("problem_id") not in already_done_ids]
    logger.info(
        f"Total problems: {len(problems)}. Already completed: {len(already_done_ids)}. "
        f"Remaining: {len(pending_problems)}."
    )

    if not pending_problems:
        logger.info(f"All {len(problems)} problems already completed in {output_path}. Skipping generation.")
        return [existing_records[p["problem_id"]] for p in problems if p.get("problem_id") in existing_records]

    # 2. Resolve backend and lazy load model
    active_backend = resolve_generation_backend(backend)

    if active_backend == "vllm":
        if model is None:
            if not model_name_or_path:
                raise ValueError("Must provide either model or model_name_or_path for vLLM.")
            logger.info(f"Loading vLLM model: {model_name_or_path}")
            model = load_vllm_model(model_name_or_path)
    else:
        # Hugging Face backend
        if model is None or tokenizer is None:
            if not model_name_or_path:
                raise ValueError("Must provide either (model, tokenizer) or model_name_or_path.")
            from src.model import load_model_and_tokenizer
            model, tokenizer = load_model_and_tokenizer(
                model_name_or_path=model_name_or_path,
                adapter_path=adapter_path,
            )

    # 3. Resolve generation parameters
    cfg = generation_config or {}
    max_new_tokens = cfg.get("max_new_tokens", 512)
    stops = stop_sequences or cfg.get("stop_sequences", ["Question:", "\n\n\n"])

    if preset == "greedy":
        temperature = 0.0
        top_p = 1.0
        top_k = -1
        do_sample = False
        num_return_sequences = 1
    elif preset == "sampled":
        temperature = cfg.get("temperature", 0.7)
        top_p = cfg.get("top_p", 0.95)
        top_k = cfg.get("top_k", 50)
        do_sample = True
        num_return_sequences = 1
    elif preset == "self_consistency":
        temperature = cfg.get("temperature", 0.7)
        top_p = cfg.get("top_p", 0.95)
        top_k = cfg.get("top_k", 50)
        do_sample = True
        num_return_sequences = cfg.get("num_samples", 8)
    else:
        temperature = cfg.get("temperature", 0.0)
        top_p = cfg.get("top_p", 1.0)
        top_k = cfg.get("top_k", -1)
        do_sample = cfg.get("do_sample", False)
        num_return_sequences = cfg.get("num_return_sequences", 1)

    # Default prompt formatter: 8-shot few-shot prompt
    if prompt_formatter is None:
        from src.data.formatting import format_few_shot_prompt

        def default_formatter(prob: Dict[str, Any]) -> str:
            return format_few_shot_prompt(prob["question"])

        prompt_formatter = default_formatter

    # 4. Generate in batches and append chunk by chunk
    newly_completed_records: List[Dict[str, Any]] = []

    for i in range(0, len(pending_problems), batch_size):
        batch_probs = pending_problems[i : i + batch_size]
        batch_prompts = [prompt_formatter(p) for p in batch_probs]

        if active_backend == "vllm":
            batch_outputs = generate_batch_vllm(
                llm=model,
                prompts=batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                stop_sequences=stops,
                num_return_sequences=num_return_sequences,
            )
        else:
            batch_outputs = generate_batch_hf(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                stop_sequences=stops,
                num_return_sequences=num_return_sequences,
            )

        batch_records: List[Dict[str, Any]] = []
        for prob, completions in zip(batch_probs, batch_outputs):
            gold = prob.get("gold_answer") or prob.get("answer")

            if num_return_sequences == 1:
                comp = completions[0]
                extracted = comp["extracted_answer"]
                correct = is_equivalent(extracted, gold) if gold is not None else False

                rec = {
                    "problem_id": prob.get("problem_id"),
                    "question": prob.get("question"),
                    "gold_answer": str(gold) if gold is not None else None,
                    "prompt": prompt_formatter(prob),
                    "generated_text": comp["completion"],
                    "extracted_answer": extracted,
                    "extraction_method": comp.get("extraction_method", "none"),
                    "is_correct": correct,
                    "prompt_tokens": comp["prompt_tokens"],
                    "completion_tokens": comp["completion_tokens"],
                    "preset": preset,
                }
            else:
                # Multiple completions (self-consistency)
                generated_texts = [c["completion"] for c in completions]
                extracted_answers = [c["extracted_answer"] for c in completions]
                extraction_methods = [c.get("extraction_method", "none") for c in completions]
                prompt_tokens_list = [c.get("prompt_tokens", 0) for c in completions]
                completion_tokens_list = [c.get("completion_tokens", 0) for c in completions]
                rec = {
                    "problem_id": prob.get("problem_id"),
                    "question": prob.get("question"),
                    "gold_answer": str(gold) if gold is not None else None,
                    "prompt": prompt_formatter(prob),
                    "generated_texts": generated_texts,
                    "extracted_answers": extracted_answers,
                    "extraction_methods": extraction_methods,
                    "prompt_tokens": prompt_tokens_list,
                    "completion_tokens": completion_tokens_list,
                    "num_samples": len(completions),
                    "preset": preset,
                }

            batch_records.append(rec)
            existing_records[prob["problem_id"]] = rec

        # Immediately append chunk to disk for crash resilience
        append_generation_records(output_path, batch_records)
        newly_completed_records.extend(batch_records)

    # Return full ordered records
    return [existing_records[p["problem_id"]] for p in problems if p.get("problem_id") in existing_records]
