"""Reasoning trace sampling and synthetic data generation engine (Phase 3).

Generates multiple diverse reasoning traces per mathematical problem across different
prompt strategies (Chain-of-Thought, Solve-then-Verify, Alternative-Method), parses
answers, evaluates correctness against gold answers, and persists complete records
in chunked, resumable JSONL files.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.config import ConfigManager
from src.generation import (
    DEFAULT_STOP_SEQUENCES,
    LLM,
    check_or_write_run_meta,
    free_gpu_memory,
    generate_batch_hf,
    generate_batch_vllm,
    is_vllm_available,
    load_vllm_model,
    resolve_generation_backend,
    shutdown_model,
)
from src.math_utils.answer_extraction import extract_answer
from src.math_utils.normalization import is_equivalent
from src.reasoning.prompting import (
    ReasoningPromptType,
    normalize_prompt_type,
    render_reasoning_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class ReasoningTrace:
    """Complete record of a generated mathematical reasoning trace."""

    trace_id: str
    problem_id: str
    question: str
    gold_answer: str
    prompt_type: str
    sample_idx: int
    teacher_model: str
    prompt: str
    completion: str
    extracted_answer: Optional[str]
    extraction_method: str
    is_correct: bool
    format_compliant: bool
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningTrace:
        return cls(**data)


def load_synthetic_traces(output_file: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read all reasoning traces from a synthetic JSONL file."""
    path = Path(output_file)
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError as e:
                logger.warning(f"Malformed JSON on line {line_num} in {path}: {e}")

    return records


def load_existing_traces_by_id(output_file: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Read existing synthetic JSONL records and index by unique trace_id."""
    records = load_synthetic_traces(output_file)
    existing: Dict[str, Dict[str, Any]] = {}
    for r in records:
        tid = r.get("trace_id")
        if tid:
            existing[tid] = r
    return existing


def build_trace_id(problem_id: str, prompt_type: str, sample_idx: int) -> str:
    """Build a deterministic unique identifier for a reasoning trace."""
    clean_type = normalize_prompt_type(prompt_type).value
    return f"{problem_id}_{clean_type}_s{sample_idx}"


def select_synthetic_subset_ids(
    train_ids: Sequence[str],
    subset_size: int = 3000,
    seed: int = 42,
) -> List[str]:
    """Deterministically select the Phase 3 synthetic subset from training problem IDs.

    Sorts train_ids before sampling with random.Random(seed) so the subset is
    reproducible across platforms and matches data/splits/synthetic_subset_ids.json.
    """
    unique_sorted = sorted(set(train_ids))
    if subset_size > len(unique_sorted):
        raise ValueError(
            f"Requested subset_size={subset_size} exceeds training pool size ({len(unique_sorted)})."
        )
    rng = random.Random(seed)
    return rng.sample(unique_sorted, subset_size)


def load_synthetic_subset_ids(subset_file: Optional[Union[str, Path]] = None) -> List[str]:
    """Load the 3,000 frozen problem IDs for synthetic data generation."""
    if subset_file is None:
        mgr = ConfigManager()
        data_cfg = mgr.load_config("data.yaml")
        path_str = data_cfg.get("synthetic", {}).get("subset_file", "data/splits/synthetic_subset_ids.json")
        path = mgr.project_root / path_str
    else:
        path = Path(subset_file)

    if not path.exists():
        raise FileNotFoundError(f"Synthetic subset file not found: {path}")

    ids = json.loads(path.read_text(encoding="utf-8"))
    return list(ids)


def load_synthetic_problems(
    split_path: Optional[Union[str, Path]] = None,
    subset_file: Optional[Union[str, Path]] = None,
    limit: Optional[int] = None,
    allow_sample_fallback: bool = False,
) -> List[Dict[str, Any]]:
    """Load the 3,000 synthetic training problems, verifying zero val/test overlap.

    Args:
        split_path: Explicit path to a training Parquet file. Defaults to the profile's
            processed train split (`data/processed/train/train.parquet`).
        subset_file: Path to the frozen subset ID list (`data/splits/synthetic_subset_ids.json`).
        limit: Optional maximum number of problems to return.
        allow_sample_fallback: If True (mock/test runs only), fall back to
            `data/sample/train_sample.parquet` when the full processed split is absent.
            Real runs never silently fall back to the 20-problem sample set.
    """
    import pandas as pd

    mgr = ConfigManager()
    project_root = mgr.project_root

    # Resolve train split
    used_sample_fallback = False
    if split_path is None:
        train_file = mgr.data_path("processed", "train", "train.parquet")
        if not train_file.exists():
            sample_file = project_root / "data" / "sample" / "train_sample.parquet"
            if allow_sample_fallback and sample_file.exists():
                train_file = sample_file
                used_sample_fallback = True
                logger.warning(
                    f"Processed train split not found; mock run falling back to sample set: {train_file}"
                )
            else:
                raise FileNotFoundError(
                    f"Processed training split not found: {train_file}. Run scripts/prepare_data.py first."
                )
    else:
        train_file = Path(split_path)
        if not train_file.exists():
            raise FileNotFoundError(f"Training split file not found: {train_file}")

    df_train = pd.read_parquet(train_file)

    # Load subset IDs
    subset_ids = load_synthetic_subset_ids(subset_file)
    subset_set = set(subset_ids)

    # Contamination check: assert zero overlap with validation or test IDs
    test_ids_file = project_root / "data" / "splits" / "test_ids.json"
    val_ids_file = project_root / "data" / "splits" / "val_ids.json"

    if test_ids_file.exists():
        test_ids = set(json.loads(test_ids_file.read_text(encoding="utf-8")))
        test_overlap = subset_set.intersection(test_ids)
        if test_overlap:
            raise ValueError(
                f"Contamination error: {len(test_overlap)} synthetic problems overlap with test set!"
            )

    if val_ids_file.exists():
        val_ids = set(json.loads(val_ids_file.read_text(encoding="utf-8")))
        val_overlap = subset_set.intersection(val_ids)
        if val_overlap:
            raise ValueError(
                f"Contamination error: {len(val_overlap)} synthetic problems overlap with val set!"
            )

    # Filter to subset problems in frozen subset_ids order
    filtered = df_train[df_train["problem_id"].isin(subset_set)]
    if len(filtered) == 0:
        if used_sample_fallback:
            records = df_train.to_dict(orient="records")
        else:
            raise ValueError(
                f"No problems in {train_file} matched the synthetic subset IDs ({len(subset_ids)} IDs)."
            )
    else:
        id_to_record = {r["problem_id"]: r for r in filtered.to_dict(orient="records")}
        records = [id_to_record[pid] for pid in subset_ids if pid in id_to_record]

    if limit is not None and limit > 0:
        records = records[:limit]

    return records


def generate_mock_trace_completion(
    question: str,
    gold_answer: str,
    prompt_type: ReasoningPromptType,
    sample_idx: int,
) -> str:
    """Generate synthetic completion adhering to each prompt strategy (mock/plumbing mode only)."""
    gold_str = str(gold_answer).strip()

    if prompt_type == ReasoningPromptType.CHAIN_OF_THOUGHT:
        return (
            "Let's think step by step to solve this problem.\n"
            f"Question analysis: {question[:90]}...\n"
            "Step 1: Set up the known quantities and variables.\n"
            f"Step 2: Calculate the intermediate values carefully (sample {sample_idx + 1}).\n"
            f"Step 3: Combine all operations to obtain {gold_str}.\n"
            f"#### {gold_str}"
        )
    elif prompt_type == ReasoningPromptType.SOLVE_THEN_VERIFY:
        return (
            "Stage 1 - Initial Solution:\n"
            f"Working through the question step by step yields an initial solution of {gold_str}.\n\n"
            "Stage 2 - Verification:\n"
            f"Double checking with inverse operations and constraint validation (sample {sample_idx + 1}).\n"
            f"The verification confirms the value {gold_str} satisfies all constraints.\n"
            f"#### {gold_str}"
        )
    elif prompt_type == ReasoningPromptType.ALTERNATIVE_METHOD:
        return (
            "Alternative Strategy (Algebraic Formulation / Ratio Analysis):\n"
            "Rather than sequential arithmetic, let x denote the unknown quantity.\n"
            "Formulating the equation according to problem constraints:\n"
            f"Solving for x directly gives x = {gold_str} (sample {sample_idx + 1}).\n"
            f"#### {gold_str}"
        )
    else:
        return f"Step-by-step reasoning concludes with {gold_str}.\n#### {gold_str}"


class TraceSampler:
    """Manages reasoning trace sampling across problems, prompt types, and decodings."""

    def __init__(
        self,
        model_name_or_path: str = "deepseek-ai/deepseek-math-7b-instruct",
        backend: str = "auto",
        output_dir: Optional[Union[str, Path]] = None,
        chunk_size: int = 10,
        batch_size: Optional[int] = None,
        seed: int = 42,
        use_mock: bool = False,
        torch_dtype: str = "bfloat16",
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.backend = backend
        self.chunk_size = max(1, int(chunk_size))
        self.seed = seed
        self.use_mock = use_mock
        self.torch_dtype = torch_dtype

        mgr = ConfigManager()
        self.project_root = mgr.project_root
        self.model_slug = Path(str(model_name_or_path).rstrip("/\\")).name or "model"

        if output_dir is None:
            self.output_dir = mgr.data_path("synthetic", "generated")
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Batch sizes from configs/generation.yaml
        gen_cfg = mgr.load_config("generation.yaml")
        configured_batch_sizes = gen_cfg.get("batch_sizes", {})
        if batch_size is not None:
            self.batch_size = max(1, int(batch_size))
        elif is_vllm_available() and self.backend != "hf" and model is None:
            self.batch_size = int(configured_batch_sizes.get("vllm", 128))
        else:
            self.batch_size = int(configured_batch_sizes.get("transformers_local", 4))

        self.model: Any = model
        self.tokenizer: Any = tokenizer
        self._resolved_backend: Optional[str] = None
        self._owns_model: bool = False

    def _init_model(self) -> None:
        """Initialize generation backend and model if not already loaded."""
        if self.use_mock:
            return

        if self.model is not None:
            if self._resolved_backend is None:
                is_vllm_engine = isinstance(LLM, type) and isinstance(self.model, LLM)
                if is_vllm_engine:
                    self._resolved_backend = "vllm"
                elif self.tokenizer is not None or (self.backend or "auto").lower() == "hf":
                    self._resolved_backend = "hf"
                else:
                    self._resolved_backend = resolve_generation_backend(self.backend)
            return

        resolved = resolve_generation_backend(self.backend)
        self._resolved_backend = resolved
        logger.info(f"Initializing {resolved} backend for model {self.model_name_or_path}")

        if resolved == "vllm":
            self.model = load_vllm_model(
                model_name_or_path=self.model_name_or_path,
                seed=self.seed,
            )
            self.tokenizer = self.model.get_tokenizer()
        else:
            from src.model import load_model_and_tokenizer

            self.model, self.tokenizer = load_model_and_tokenizer(
                model_name_or_path=self.model_name_or_path,
                torch_dtype=self.torch_dtype,
                device_map="auto",
            )
        self._owns_model = True

    def sample_traces(
        self,
        problems: List[Dict[str, Any]],
        prompt_types: Sequence[Union[str, ReasoningPromptType]] = (ReasoningPromptType.CHAIN_OF_THOUGHT,),
        samples_per_problem: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        stop_sequences: Optional[Sequence[str]] = None,
        filename_suffix: Optional[str] = None,
    ) -> Path:
        """Sample reasoning traces for all problems across specified prompt strategies.

        Resumable: reads existing traces from disk and skips finished (problem_id, prompt_type, sample_idx).
        Batches prompts for high throughput and flushes newly finished chunks to disk every
        `self.chunk_size` problems.
        """
        canonical_types = [normalize_prompt_type(pt) for pt in prompt_types]
        type_str = "_".join(t.value for t in canonical_types) if len(canonical_types) <= 2 else "multi_prompt"
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        out_filename = f"traces_{self.model_slug}_{type_str}{suffix}.jsonl"
        out_path = self.output_dir / out_filename

        stops = list(stop_sequences) if stop_sequences is not None else list(DEFAULT_STOP_SEQUENCES)

        # Establish and verify run metadata for resumption safety
        run_config = {
            "model_name_or_path": self.model_name_or_path,
            "prompt_types": [t.value for t in canonical_types],
            "samples_per_problem": samples_per_problem,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "seed": self.seed,
            "backend": self.backend,
            "use_mock": self.use_mock,
        }
        has_existing = out_path.exists() and out_path.stat().st_size > 0
        check_or_write_run_meta(out_path, run_config, has_records=has_existing)

        # Load existing traces indexed by trace_id for resumption
        existing_traces: Dict[str, Dict[str, Any]] = {}
        if out_path.exists():
            existing_traces = load_existing_traces_by_id(out_path)
            logger.info(f"Loaded {len(existing_traces)} existing traces from {out_path}")

        # Build problem-level items that still need at least one sample
        pending_items: List[Tuple[Dict[str, Any], ReasoningPromptType, Tuple[int, ...]]] = []
        total_missing_traces = 0
        for prob in problems:
            pid = prob["problem_id"]
            for ptype in canonical_types:
                missing_idxs = tuple(
                    s_idx
                    for s_idx in range(samples_per_problem)
                    if build_trace_id(pid, ptype.value, s_idx) not in existing_traces
                )
                if missing_idxs:
                    pending_items.append((prob, ptype, missing_idxs))
                    total_missing_traces += len(missing_idxs)

        total_needed = len(problems) * len(canonical_types) * samples_per_problem
        logger.info(
            f"Total traces required: {total_needed} ({len(existing_traces)} exist, "
            f"{total_missing_traces} to generate across {len(pending_items)} problem-prompt items)"
        )

        if not pending_items:
            logger.info("All requested traces are already generated.")
            return out_path

        # Initialize model if needed
        self._init_model()

        start_time = time.time()
        generated_traces_count = 0
        batch_call_idx = 0

        # Process in chunks of self.chunk_size problem-prompt items and flush after each chunk
        for chunk_start in range(0, len(pending_items), self.chunk_size):
            chunk_items = pending_items[chunk_start:chunk_start + self.chunk_size]
            chunk_records: List[Dict[str, Any]] = []

            # Group items in this chunk by their missing sample_idx tuple so each batch call
            # requests a uniform num_return_sequences
            groups: Dict[Tuple[int, ...], List[Tuple[Dict[str, Any], ReasoningPromptType]]] = {}
            for prob, ptype, missing_idxs in chunk_items:
                groups.setdefault(missing_idxs, []).append((prob, ptype))

            for missing_idxs, group_pairs in groups.items():
                num_return = len(missing_idxs)
                for sub_start in range(0, len(group_pairs), self.batch_size):
                    sub_pairs = group_pairs[sub_start:sub_start + self.batch_size]
                    rendered_prompts = [
                        render_reasoning_prompt(
                            question=prob["question"],
                            prompt_type=ptype,
                            tokenizer=self.tokenizer,
                        )
                        for prob, ptype in sub_pairs
                    ]

                    if self.use_mock or self.model is None:
                        batch_completions: List[List[Dict[str, Any]]] = []
                        for (prob, ptype), r_prompt in zip(sub_pairs, rendered_prompts):
                            q = prob["question"]
                            gold = str(prob.get("gold_answer", "")).strip()
                            comp_list: List[Dict[str, Any]] = []
                            for s_idx in missing_idxs:
                                text = generate_mock_trace_completion(q, gold, ptype, s_idx)
                                parsed_ans, method = extract_answer(text)
                                comp_list.append({
                                    "completion": text,
                                    "extracted_answer": parsed_ans,
                                    "extraction_method": method,
                                    "prompt_tokens": len(r_prompt.split()),
                                    "completion_tokens": len(text.split()),
                                    "finish_reason": "stop",
                                })
                            batch_completions.append(comp_list)
                    else:
                        batch_seed = self.seed + batch_call_idx
                        batch_call_idx += 1
                        batch_completions = self._generate_batch(
                            prompts=rendered_prompts,
                            num_return_sequences=num_return,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=max_new_tokens,
                            stop_sequences=stops,
                            seed=batch_seed,
                        )

                    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    for (prob, ptype), r_prompt, comp_list in zip(
                        sub_pairs, rendered_prompts, batch_completions
                    ):
                        q = prob["question"]
                        gold = str(prob.get("gold_answer", "")).strip()
                        pid = prob["problem_id"]

                        for s_idx, item in zip(missing_idxs, comp_list):
                            completion_text = item["completion"]
                            parsed_ans = item.get("extracted_answer")
                            method = item.get("extraction_method")
                            if parsed_ans is None and method is None:
                                parsed_ans, method = extract_answer(completion_text)
                            method = method or "none"

                            is_compliant = method == "hash"
                            is_correct = is_equivalent(parsed_ans, gold) if parsed_ans else False

                            record = ReasoningTrace(
                                trace_id=build_trace_id(pid, ptype.value, s_idx),
                                problem_id=pid,
                                question=q,
                                gold_answer=gold,
                                prompt_type=ptype.value,
                                sample_idx=s_idx,
                                teacher_model=self.model_name_or_path,
                                prompt=r_prompt,
                                completion=completion_text,
                                extracted_answer=parsed_ans,
                                extraction_method=method,
                                is_correct=is_correct,
                                format_compliant=is_compliant,
                                prompt_tokens=int(item.get("prompt_tokens", 0)),
                                completion_tokens=int(item.get("completion_tokens", 0)),
                                finish_reason=str(item.get("finish_reason") or "stop"),
                                timestamp_utc=timestamp,
                            )
                            chunk_records.append(record.to_dict())

            self._append_chunk(out_path, chunk_records)
            generated_traces_count += len(chunk_records)
            elapsed = time.time() - start_time
            rate = generated_traces_count / elapsed if elapsed > 0 else 0.0
            logger.info(
                f"Progress: [{generated_traces_count}/{total_missing_traces}] traces generated "
                f"({rate:.1f} traces/sec) -> flushed to {out_path.name}"
            )

        return out_path

    def _generate_batch(
        self,
        prompts: List[str],
        num_return_sequences: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        stop_sequences: Sequence[str],
        seed: int,
    ) -> List[List[Dict[str, Any]]]:
        """Execute batched generation on the active backend (vLLM or Hugging Face)."""
        do_sample = temperature > 0.0

        if self._resolved_backend == "vllm":
            return generate_batch_vllm(
                llm=self.model,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=-1,
                do_sample=do_sample,
                stop_sequences=stop_sequences,
                num_return_sequences=num_return_sequences,
                seed=seed,
            )

        if do_sample:
            from transformers import set_seed

            set_seed(seed)

        return generate_batch_hf(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=-1,
            do_sample=do_sample,
            stop_sequences=stop_sequences,
            num_return_sequences=num_return_sequences,
        )

    def _append_chunk(self, path: Path, records: List[Dict[str, Any]]) -> None:
        """Append records to JSONL file with immediate disk sync."""
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    def close(self) -> None:
        """Release model and free GPU memory."""
        if self.model is not None and self._owns_model:
            shutdown_model(self.model, self._resolved_backend or "hf")
            self.model = None
            self.tokenizer = None
            self._owns_model = False
            free_gpu_memory()


def compute_trace_yield(traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute yield metrics, per-problem pass rates, and format compliance statistics."""
    total_traces = len(traces)
    if total_traces == 0:
        return {
            "total_traces": 0,
            "total_problems": 0,
            "format_compliance_rate": 0.0,
            "format_compliance_percent": 0.0,
            "correct_traces": 0,
            "trace_accuracy": 0.0,
            "trace_accuracy_percent": 0.0,
            "problems_with_at_least_one_correct": 0,
            "problem_coverage_rate": 0.0,
            "pass_rate_distribution": {},
        }

    correct_traces = sum(1 for t in traces if t.get("is_correct", False))
    compliant_traces = sum(1 for t in traces if t.get("format_compliant", False))

    # Group by problem_id
    problem_traces: Dict[str, List[Dict[str, Any]]] = {}
    for t in traces:
        pid = t["problem_id"]
        problem_traces.setdefault(pid, []).append(t)

    total_problems = len(problem_traces)
    pass_rates: Dict[str, float] = {}
    solved_problems = 0

    for pid, p_list in problem_traces.items():
        k = len(p_list)
        c = sum(1 for item in p_list if item.get("is_correct", False))
        rate = c / k if k > 0 else 0.0
        pass_rates[pid] = rate
        if c > 0:
            solved_problems += 1

    # Pass rate distribution buckets: 0%, (0, 25%], (25, 50%], (50, 75%], (75, 100%), 100%
    buckets = {
        "0%": 0,
        "(0, 25%]": 0,
        "(25, 50%]": 0,
        "(50, 75%]": 0,
        "(75, 100%)": 0,
        "100%": 0,
    }
    for rate in pass_rates.values():
        if rate == 0.0:
            buckets["0%"] += 1
        elif rate == 1.0:
            buckets["100%"] += 1
        elif 0.0 < rate <= 0.25:
            buckets["(0, 25%]"] += 1
        elif 0.25 < rate <= 0.50:
            buckets["(25, 50%]"] += 1
        elif 0.50 < rate <= 0.75:
            buckets["(50, 75%]"] += 1
        else:
            buckets["(75, 100%)"] += 1

    return {
        "total_traces": total_traces,
        "total_problems": total_problems,
        "format_compliance_rate": round(compliant_traces / total_traces, 4),
        "format_compliance_percent": round((compliant_traces / total_traces) * 100.0, 2),
        "correct_traces": correct_traces,
        "trace_accuracy": round(correct_traces / total_traces, 4),
        "trace_accuracy_percent": round((correct_traces / total_traces) * 100.0, 2),
        "problems_with_at_least_one_correct": solved_problems,
        "problem_coverage_rate": round(solved_problems / total_problems, 4) if total_problems > 0 else 0.0,
        "pass_rate_distribution": buckets,
    }
