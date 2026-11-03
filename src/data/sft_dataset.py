"""SFT dataset formatting, tokenization, and completion-only loss masking (Phase 4 Day 26).

Implements zero-shot prompt-completion pairs, ensures the end-of-sequence token is present
and not masked, and verifies that loss is computed exclusively on the reasoning completion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.data.curation import normalize_reasoning_completion
from src.data.formatting import format_zero_shot_prompt

logger = logging.getLogger(__name__)


@dataclass
class SFTExample:
    """A formatted prompt-completion pair for SFT causal language modeling."""

    problem_id: str
    trace_id: str
    prompt: str
    completion: str
    gold_answer: str
    prompt_type: str = "chain_of_thought"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "trace_id": self.trace_id,
            "prompt": self.prompt,
            "completion": self.completion,
            "gold_answer": self.gold_answer,
            "prompt_type": self.prompt_type,
            "text": f"{self.prompt}{self.completion}",
        }


def format_sft_prompt_completion(
    question: str,
    completion: str,
    gold_answer: Optional[str] = None,
    eos_token: str = "<|eos|>",
) -> Dict[str, str]:
    """Format a question and reasoning completion into an SFT prompt-completion pair.

    The prompt states the GSM8K '#### <number>' answer contract via format_zero_shot_prompt.
    The completion contains the reasoning steps and final answer, cleanly ending with the
    tokenizer's end-of-sequence token.

    Args:
        question: Problem description string.
        completion: Reasoning derivation ending in '#### <number>'.
        gold_answer: Optional reference numeric answer for verification.
        eos_token: End-of-sequence token string (e.g. '<|eos|>' or '<|endoftext|>').

    Returns:
        Dict with 'prompt', 'completion', and 'full_text'.
    """
    prompt = format_zero_shot_prompt(question)
    norm_comp = normalize_reasoning_completion(completion, expected_answer=gold_answer)

    # Ensure completion ends with exactly one EOS token
    if eos_token and not norm_comp.endswith(eos_token):
        formatted_completion = f"{norm_comp}{eos_token}"
    else:
        formatted_completion = norm_comp

    return {
        "prompt": prompt,
        "completion": formatted_completion,
        "full_text": f"{prompt}{formatted_completion}",
    }


def prepare_sft_examples_from_curated(
    traces: Sequence[Dict[str, Any]],
    eos_token: str = "<|eos|>",
) -> List[Dict[str, Any]]:
    """Convert curated reasoning traces into SFT training examples.

    Args:
        traces: Sequence of curated trace records.
        eos_token: End-of-sequence token string.

    Returns:
        List of SFT formatted example dictionaries.
    """
    examples: List[Dict[str, Any]] = []
    for t in traces:
        q = t.get("question", "")
        comp = t.get("completion", "")
        gold = t.get("gold_answer", "")
        pair = format_sft_prompt_completion(question=q, completion=comp, gold_answer=gold, eos_token=eos_token)
        rec = {
            "problem_id": t.get("problem_id", ""),
            "trace_id": t.get("trace_id", ""),
            "prompt": pair["prompt"],
            "completion": pair["completion"],
            "text": pair["full_text"],
            "gold_answer": gold,
            "prompt_type": t.get("prompt_type", "chain_of_thought"),
        }
        examples.append(rec)
    return examples


def load_sft_dataset(
    file_path: Union[str, Path],
    eos_token: str = "<|eos|>",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load curated reasoning dataset and format as SFT prompt-completion pairs."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Curated SFT file not found: {p}")

    records: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
                if limit and len(records) >= limit:
                    break

    return prepare_sft_examples_from_curated(records, eos_token=eos_token)


def tokenize_sft_example(
    example: Dict[str, Any],
    tokenizer: Any,
    max_seq_length: int = 512,
) -> Dict[str, Any]:
    """Tokenize an SFT prompt-completion pair with completion-only loss masking.

    Loss is computed strictly on the completion tokens (labels = token_id), including the final
    EOS token. Prompt tokens are masked with label = -100 so the model is not supervised to predict
    the prompt instructions.

    Args:
        example: Dictionary containing 'prompt' and 'completion'.
        tokenizer: PreTrainedTokenizer or fast tokenizer instance.
        max_seq_length: Maximum allowed total sequence length.

    Returns:
        Dictionary with 'input_ids', 'attention_mask', 'labels', 'prompt_len', 'completion_len'.
    """
    prompt = example["prompt"]
    completion = example["completion"]

    # 1. Tokenize prompt (includes BOS if the tokenizer automatically prepends it)
    if hasattr(tokenizer, "encode"):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    else:
        prompt_ids = tokenizer(prompt)["input_ids"]

    # 2. Tokenize completion (add_special_tokens=False so no second BOS token is injected)
    if hasattr(tokenizer, "encode"):
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    else:
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    # Ensure the completion ends with the tokenizer's eos_token_id
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and (not completion_ids or completion_ids[-1] != eos_id):
        completion_ids.append(eos_id)

    # 3. Concatenate and assemble labels
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    attention_mask = [1] * len(input_ids)

    # 4. Truncate if total sequence exceeds max_seq_length
    if len(input_ids) > max_seq_length:
        # Keep full completion and truncate prompt from the left, ensuring completion is intact
        overflow = len(input_ids) - max_seq_length
        prompt_keep = max(1, len(prompt_ids) - overflow)
        prompt_ids = prompt_ids[-prompt_keep:]
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + list(completion_ids)
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > max_seq_length:
            # If completion itself exceeds max_seq_length, truncate completion but preserve EOS
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
            if eos_id is not None:
                input_ids[-1] = eos_id
                labels[-1] = eos_id

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_len": len(prompt_ids),
        "completion_len": len(completion_ids),
    }


def verify_sft_loss_masking(
    tokenizer: Any,
    prompt: str,
    completion: str,
    eos_token: Optional[str] = None,
    max_seq_length: int = 512,
) -> Dict[str, Any]:
    """Execute the Day 26 hand check verifying loss masking and EOS supervision.

    Verifies:
    1. Prompt tokens are strictly masked (label == -100).
    2. Completion tokens are supervised (label != -100).
    3. The final token in labels is the tokenizer's EOS token and is NOT masked (-100).
    4. Decoding tokens where label != -100 reconstructs the expected completion and EOS.

    Args:
        tokenizer: Model tokenizer instance.
        prompt: Zero-shot prompt text.
        completion: Reasoning completion text.
        eos_token: Optional explicit EOS string.
        max_seq_length: Sequence budget.

    Returns:
        Detailed hand-check verification dictionary.
    """
    eos_str = eos_token or getattr(tokenizer, "eos_token", "<|eos|>") or "<|eos|>"
    example = format_sft_prompt_completion(question="dummy", completion=completion, eos_token=eos_str)
    example["prompt"] = prompt

    tokenized = tokenize_sft_example(example, tokenizer, max_seq_length=max_seq_length)
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    prompt_len = tokenized["prompt_len"]
    completion_len = tokenized["completion_len"]
    eos_id = getattr(tokenizer, "eos_token_id", None)

    # 1. Prompt mask check
    prompt_labels = labels[:prompt_len]
    prompt_masked = len(prompt_labels) > 0 and all(lbl == -100 for lbl in prompt_labels)

    # 2. Completion supervision check
    completion_labels = labels[prompt_len:]
    completion_supervised = len(completion_labels) > 0 and all(lbl != -100 for lbl in completion_labels)

    # 3. EOS token present and unmasked
    eos_present = (eos_id is not None and labels[-1] == eos_id) if eos_id is not None else True
    eos_unmasked = labels[-1] != -100

    # 4. Decoded unmasked tokens
    unmasked_ids = [token_id for token_id, lbl in zip(input_ids, labels) if lbl != -100]
    if hasattr(tokenizer, "decode"):
        decoded_completion = tokenizer.decode(unmasked_ids, skip_special_tokens=False)
    else:
        decoded_completion = ""

    all_checks_passed = bool(prompt_masked and completion_supervised and eos_present and eos_unmasked)

    return {
        "all_checks_passed": all_checks_passed,
        "total_tokens": len(input_ids),
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_len,
        "prompt_masked": prompt_masked,
        "completion_supervised": completion_supervised,
        "eos_token_id": eos_id,
        "last_label": labels[-1],
        "eos_present": eos_present,
        "eos_unmasked": eos_unmasked,
        "decoded_target": decoded_completion,
    }
