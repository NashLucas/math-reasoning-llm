"""Comprehensive evaluation engine for mathematical reasoning models.

Evaluates any checkpoint x preset x split combination.
Features:
- Enforces the Test-Set Wall: strictly forbids evaluation on the 'test' split without --final.
- Resumable chunked generation via src.generation.run_generation_pipeline.
- Metric calculation via src.evaluation.math_metrics.compute_all_metrics.
- Artifact persistence: saves 'eval_results.json' and 'run_info.json' into stage output folder.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from src.config import ConfigManager
from src.data.formatting import (
    apply_chat_template,
    format_chat_messages,
    format_few_shot_prompt,
    format_zero_shot_prompt,
)
from src.evaluation.math_metrics import compute_all_metrics
from src.generation import (
    append_generation_records,
    load_existing_generations,
    run_generation_pipeline,
)
from src.math_utils.normalization import is_equivalent
from src.run_info import save_run_info

logger = logging.getLogger(__name__)


class TestSetWallError(PermissionError):
    """Raised when an evaluation attempts to access the test split without --final authorization."""

    __test__ = False


class MathEvaluator:
    """Evaluates mathematical reasoning models across splits and decoding presets."""

    VALID_SPLITS = ("val", "val_analysis", "val_gating", "sample", "train", "test")
    VALID_PRESETS = ("greedy", "sampled", "self_consistency")

    def __init__(
        self,
        stage: str = "baseline",
        checkpoint: Optional[str] = None,
        adapter_path: Optional[str] = None,
        preset: str = "greedy",
        split: str = "val",
        limit: Optional[int] = None,
        final: bool = False,
        prompt_format: str = "auto",
        backend: str = "auto",
        batch_size: int = 4,
        output_dir: Optional[Union[str, Path]] = None,
        config_path: Optional[str] = None,
        use_mock: bool = False,
    ) -> None:
        """Initialize MathEvaluator.

        Args:
            stage: Pipeline stage name ('baseline', 'sft', 'targeted_sft', 'grpo', 'final').
            checkpoint: Model ID or checkpoint directory path.
            adapter_path: Optional path to LoRA adapter weights.
            preset: Decoding preset ('greedy', 'sampled', 'self_consistency').
            split: Data split ('val', 'val_analysis', 'val_gating', 'sample', 'train', 'test').
            limit: Optional integer to limit the number of problems evaluated.
            final: Boolean flag required to authorize evaluation on the 'test' split.
            prompt_format: Prompt styling ('auto', 'few_shot', 'zero_shot', 'chat').
            batch_size: Batch size for model inference.
            output_dir: Destination directory for eval_results.json and generations.
            config_path: Path to custom configuration root.
            use_mock: If True, uses deterministic mock generator without loading weights.
        """
        self.stage = stage
        self.adapter_path = adapter_path
        self.preset = preset
        self.split = split
        self.limit = limit
        self.final = final
        self.prompt_format = prompt_format
        self.backend = backend
        self.batch_size = batch_size
        self.use_mock = use_mock

        # Configuration manager & paths
        self.cfg_mgr = ConfigManager()
        self.base_cfg = self.cfg_mgr.base_config
        self.gen_cfg = self.cfg_mgr.load_config("generation.yaml")

        # Resolve checkpoint name
        if checkpoint:
            self.checkpoint = checkpoint
        else:
            # Fallback to dev_model or train_base from base_config
            models = self.base_cfg.get("models", {})
            self.checkpoint = models.get("dev_model", "deepseek-ai/deepseek-math-7b-base")

        # Resolve output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            stage_paths = self.cfg_mgr.get_stage_paths(self.stage)
            self.output_dir = stage_paths.get("outputs", Path(f"outputs/{self.stage}"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def enforce_test_set_wall(self) -> None:
        """Strictly prevent accidental data snooping on the test set."""
        if self.split == "test" and not self.final:
            msg = (
                "TEST-SET WALL VIOLATION: Evaluation on split 'test' is strictly forbidden "
                "during active post-training and development! "
                "The test set may only be touched once per model during final evaluation by "
                "providing the explicit '--final' flag."
            )
            logger.error(msg)
            raise TestSetWallError(msg)

    def resolve_dataset_path(self) -> Path:
        """Resolve the file path for the requested split."""
        root = self.cfg_mgr.project_root

        if self.split == "sample":
            # Prefer sample val set
            sample_p = root / "data" / "sample" / "val_sample.parquet"
            if sample_p.exists():
                return sample_p
            sample_jsonl = root / "data" / "sample" / "val_sample.jsonl"
            if sample_jsonl.exists():
                return sample_jsonl
            raise FileNotFoundError(f"Sample data not found in {root / 'data' / 'sample'}")

        if self.split in ("val", "val_analysis", "val_gating"):
            path = root / "data" / "processed" / "val" / f"{self.split}.parquet"
            if path.exists():
                return path
            fallback_val = root / "data" / "sample" / "val_sample.parquet"
            if fallback_val.exists():
                logger.info(f"Split {path} not found; falling back to sample set {fallback_val}")
                return fallback_val
            raise FileNotFoundError(f"Processed validation split not found: {path}")

        if self.split == "train":
            path = root / "data" / "processed" / "train" / "train.parquet"
            if path.exists():
                return path
            fallback_train = root / "data" / "sample" / "train_sample.parquet"
            if fallback_train.exists():
                logger.info(f"Split {path} not found; falling back to sample set {fallback_train}")
                return fallback_train
            raise FileNotFoundError(f"Processed train split not found: {path}")

        if self.split == "test":
            self.enforce_test_set_wall()
            path = root / "data" / "processed" / "test" / "test.parquet"
            if path.exists():
                return path
            fallback_test = root / "data" / "sample" / "test_sample.parquet"
            if fallback_test.exists():
                logger.info(f"Split {path} not found; falling back to sample set {fallback_test}")
                return fallback_test
            raise FileNotFoundError(f"Processed test split not found: {path}")

        # Check if direct file path was provided as split
        direct_path = Path(self.split)
        if direct_path.exists():
            return direct_path

        raise ValueError(f"Unknown split or path '{self.split}'. Valid: {self.VALID_SPLITS}")

    def load_problems(self) -> List[Dict[str, Any]]:
        """Load dataset split and format into problem dictionaries."""
        data_path = self.resolve_dataset_path()
        logger.info(f"Loading evaluation dataset from: {data_path}")

        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        elif data_path.suffix in (".jsonl", ".json"):
            df = pd.read_json(data_path, lines=(data_path.suffix == ".jsonl"))
        else:
            raise ValueError(f"Unsupported dataset format: {data_path.suffix}")

        if self.limit and self.limit > 0:
            df = df.head(self.limit)

        problems: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            gold = row.get("gold_answer") or row.get("answer") or row.get("gold")
            problems.append(
                {
                    "problem_id": str(row.get("problem_id", f"prob_{len(problems)}")),
                    "question": str(row.get("question", "")).strip(),
                    "gold_answer": str(gold).strip() if gold is not None else None,
                }
            )

        logger.info(f"Loaded {len(problems)} problems for split '{self.split}'.")
        return problems

    def get_prompt_formatter(self, tokenizer: Optional[Any] = None) -> Callable[[Dict[str, Any]], str]:
        """Determine prompt formatting function for the target model."""
        fmt = self.prompt_format.lower()
        model_name = str(self.checkpoint).lower()

        is_chat_model = ("instruct" in model_name) or ("rl" in model_name) or ("chat" in model_name)

        if fmt == "chat" or (fmt == "auto" and is_chat_model):
            def chat_formatter(prob: Dict[str, Any]) -> str:
                msgs = format_chat_messages(prob["question"])
                return apply_chat_template(msgs, tokenizer=tokenizer)

            return chat_formatter

        if fmt == "zero_shot":
            def zero_shot_formatter(prob: Dict[str, Any]) -> str:
                return format_zero_shot_prompt(prob["question"])

            return zero_shot_formatter

        # Default to 8-shot few-shot prompt for base models
        def few_shot_formatter(prob: Dict[str, Any]) -> str:
            return format_few_shot_prompt(prob["question"], num_shots=8)

        return few_shot_formatter

    def _run_mock_generation(
        self,
        problems: List[Dict[str, Any]],
        output_file: Path,
    ) -> List[Dict[str, Any]]:
        """Fast deterministic mock generation for CI and testing."""
        existing = load_existing_generations(output_file)
        already_done = set(existing.keys())
        pending = [p for p in problems if p["problem_id"] not in already_done]

        new_records: List[Dict[str, Any]] = []
        is_sc = (self.preset == "self_consistency")
        num_samples = 4 if is_sc else 1

        for idx, prob in enumerate(pending):
            gold = prob.get("gold_answer", "42")
            pid = prob["problem_id"]

            if not is_sc:
                # Deterministic pattern: 80% correct, 15% fallback, 5% wrong
                mod = idx % 20
                if mod < 16:
                    ans = gold
                    method = "hash"
                    text = f"Step 1: Calculate.\nTherefore the answer is {ans}.\n#### {ans}"
                elif mod < 19:
                    ans = gold
                    method = "boxed"
                    text = f"Step 1: Calculate.\nFinal answer: \\boxed{{{ans}}}"
                else:
                    ans = "9999"
                    method = "hash"
                    text = f"Step 1: Mistake.\n#### {ans}"

                rec = {
                    "problem_id": pid,
                    "question": prob["question"],
                    "gold_answer": gold,
                    "prompt": format_few_shot_prompt(prob["question"], num_shots=1),
                    "generated_text": text,
                    "extracted_answer": ans,
                    "extraction_method": method,
                    "is_correct": is_equivalent(ans, gold),
                    "prompt_tokens": 120,
                    "completion_tokens": 65,
                    "preset": self.preset,
                }
            else:
                # Multi-sample (self-consistency)
                generated_texts: List[str] = []
                extracted_answers: List[Optional[str]] = []
                extraction_methods: List[str] = []
                prompt_tokens: List[int] = []
                completion_tokens: List[int] = []

                for s in range(num_samples):
                    if s < num_samples - 1:
                        ans = gold
                        m = "hash"
                        txt = f"Reasoning path {s}.\n#### {ans}"
                    else:
                        ans = "9999"
                        m = "hash"
                        txt = f"Reasoning path {s}.\n#### {ans}"
                    generated_texts.append(txt)
                    extracted_answers.append(ans)
                    extraction_methods.append(m)
                    prompt_tokens.append(120)
                    completion_tokens.append(70)

                rec = {
                    "problem_id": pid,
                    "question": prob["question"],
                    "gold_answer": gold,
                    "prompt": format_few_shot_prompt(prob["question"], num_shots=1),
                    "generated_texts": generated_texts,
                    "extracted_answers": extracted_answers,
                    "extraction_methods": extraction_methods,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "num_samples": num_samples,
                    "preset": self.preset,
                }

            new_records.append(rec)
            existing[pid] = rec

        if new_records:
            append_generation_records(output_file, new_records)

        return [existing[p["problem_id"]] for p in problems if p["problem_id"] in existing]

    def evaluate(self) -> Dict[str, Any]:
        """Execute the end-to-end evaluation pipeline.

        Returns:
            Dictionary containing evaluation results, metrics, and metadata.
        """
        # 1. Enforce Test-Set Wall
        self.enforce_test_set_wall()

        # 2. Load dataset problems
        problems = self.load_problems()

        # 3. Setup output file paths
        gen_filename = f"generations_{self.split}_{self.preset}.jsonl"
        generations_path = self.output_dir / gen_filename

        eval_filename = f"eval_results_{self.split}_{self.preset}.json"
        eval_path = self.output_dir / eval_filename
        canonical_eval_path = self.output_dir / "eval_results.json"

        # 4. Run generation or mock generation
        if self.use_mock:
            logger.info("Executing evaluation with deterministic mock generation.")
            records = self._run_mock_generation(problems, generations_path)
        else:
            logger.info(
                f"Running model inference: checkpoint={self.checkpoint}, "
                f"preset={self.preset}, batch_size={self.batch_size}"
            )
            preset_cfg = self.gen_cfg.get("presets", {}).get(self.preset, {})

            records = run_generation_pipeline(
                problems=problems,
                output_file=generations_path,
                model_name_or_path=self.checkpoint,
                adapter_path=self.adapter_path,
                backend=self.backend,
                batch_size=self.batch_size,
                preset=self.preset,
                generation_config=preset_cfg,
                prompt_formatter=self.get_prompt_formatter(),
            )

        # 5. Compute full metric suite
        max_tokens_budget = self.gen_cfg.get("presets", {}).get(self.preset, {}).get("max_new_tokens", 512)
        metrics = compute_all_metrics(records, max_new_tokens=max_tokens_budget)

        # 6. Assemble evaluation results payload
        now_utc = datetime.now(timezone.utc).isoformat()
        results: Dict[str, Any] = {
            "stage": self.stage,
            "checkpoint": self.checkpoint,
            "adapter_path": self.adapter_path,
            "preset": self.preset,
            "split": self.split,
            "final": self.final,
            "num_problems": len(problems),
            "generations_file": str(generations_path.resolve()),
            "timestamp_utc": now_utc,
            "metrics": metrics,
        }

        # 7. Persist evaluation results
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(canonical_eval_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved evaluation results to: {eval_path}")
        logger.info(f"Saved canonical evaluation results to: {canonical_eval_path}")

        # 8. Record run provenance info
        run_desc = f"evaluate_{self.stage}_{self.split}_{self.preset}"
        save_run_info(
            stage=self.stage,
            output_dir=self.output_dir,
            config={
                "checkpoint": self.checkpoint,
                "adapter_path": self.adapter_path,
                "preset": self.preset,
                "split": self.split,
                "limit": self.limit,
                "final": self.final,
                "metrics_summary": {
                    "accuracy": metrics.get("accuracy"),
                    "maj_at_k": metrics.get("maj_at_k"),
                    "pass_at_k": metrics.get("pass_at_k"),
                    "format_compliance_rate": metrics.get("format_compliance_rate"),
                },
            },
            description=run_desc,
        )

        return results
