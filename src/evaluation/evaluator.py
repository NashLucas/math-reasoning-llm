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
from src.evaluation.math_metrics import compute_all_metrics, rescore_records
from src.generation import (
    append_generation_records,
    load_existing_generations,
    run_generation_pipeline,
)
from src.math_utils.normalization import is_equivalent
from src.run_info import save_run_info

logger = logging.getLogger(__name__)

PROMPT_FORMATS = ("few_shot", "zero_shot", "chat")


def frozen_test_ids() -> set:
    """Problem IDs of the frozen test split (committed in data/splits/test_ids.json)."""
    from src.data.preprocessing import frozen_splits_dir

    path = frozen_splits_dir() / "test_ids.json"
    return set(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else set()


def run_slug(checkpoint: str, adapter_path: Optional[str] = None) -> str:
    """Folder name for one model (+ adapter): 'deepseek-math-7b-base', 'deepseek-math-7b-base__checkpoint-200'."""
    slug = Path(str(checkpoint).rstrip("/\\")).name or "model"
    if adapter_path:
        adapter = Path(str(adapter_path).rstrip("/\\"))
        slug += f"__{adapter.parent.name}-{adapter.name}" if adapter.name.startswith("checkpoint-") else f"__{adapter.name}"
    return slug


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

        self.seed = int(self.base_cfg.get("seed", 42))

        # Resolve output directory: one folder per model (+ adapter) under the stage outputs, so
        # two models never share a generations file
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            stage_paths = self.cfg_mgr.get_stage_paths(self.stage)
            stage_dir = Path(stage_paths.get("outputs", self.cfg_mgr.outputs_path(self.stage)))
            self.output_dir = stage_dir / run_slug(self.checkpoint, self.adapter_path)
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

    def _processed_or_sample(self, processed: Path, sample: Path) -> Path:
        """Return the processed split; fall back to the 20-problem sample only in mock mode.

        A real run must never silently evaluate the sample set in place of a full split.
        """
        if processed.exists():
            return processed
        if self.use_mock and sample.exists():
            logger.warning(f"Split {processed} not found; mock run falling back to sample set {sample}")
            return sample
        raise FileNotFoundError(
            f"Processed split not found: {processed}. Run scripts/prepare_data.py first."
        )

    def resolve_dataset_path(self) -> Path:
        """Resolve the file path for the requested split.

        Processed splits live in the profile data directory (Google Drive under 'colab');
        sample sets are tracked in the repo under data/sample.
        """
        sample_dir = self.cfg_mgr.project_root / "data" / "sample"
        processed_dir = self.cfg_mgr.data_path("processed")

        if self.split == "sample":
            # Prefer sample val set
            sample_p = sample_dir / "val_sample.parquet"
            if sample_p.exists():
                return sample_p
            sample_jsonl = sample_dir / "val_sample.jsonl"
            if sample_jsonl.exists():
                return sample_jsonl
            raise FileNotFoundError(f"Sample data not found in {sample_dir}")

        if self.split in ("val", "val_analysis", "val_gating"):
            return self._processed_or_sample(
                processed_dir / "val" / f"{self.split}.parquet", sample_dir / "val_sample.parquet"
            )

        if self.split == "train":
            return self._processed_or_sample(
                processed_dir / "train" / "train.parquet", sample_dir / "train_sample.parquet"
            )

        if self.split == "test":
            self.enforce_test_set_wall()
            return self._processed_or_sample(
                processed_dir / "test" / "test.parquet", sample_dir / "test_sample.parquet"
            )

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

        # Test-set wall by content, not by split name: a file path to the test split, a copy of
        # it, or the 20-problem test sample all contain frozen test IDs
        if not self.final and "problem_id" in df.columns:
            leaked = set(df["problem_id"].astype(str)) & frozen_test_ids()
            if leaked:
                msg = (
                    f"TEST-SET WALL VIOLATION: {data_path} contains {len(leaked)} test-split problems. "
                    "Test problems may only be evaluated with --final."
                )
                logger.error(msg)
                raise TestSetWallError(msg)

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

    def resolve_prompt_format(self, tokenizer: Optional[Any] = None) -> str:
        """Prompt format for this run.

        An explicit --prompt_format wins. 'auto' uses the per-model entry in base_config.yaml
        (prompt_formats); a model not listed uses its chat template if its tokenizer has one,
        else the 8-shot prompt.
        """
        fmt = self.prompt_format.lower()
        if fmt in PROMPT_FORMATS:
            return fmt
        if fmt != "auto":
            raise ValueError(f"Unknown prompt_format '{self.prompt_format}'. Valid: auto, {', '.join(PROMPT_FORMATS)}")
        configured = (self.base_cfg.get("prompt_formats") or {}).get(str(self.checkpoint))
        if configured:
            return configured
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            return "chat"
        return "few_shot"

    def needs_tokenizer(self) -> bool:
        """The chat format, and 'auto' for an unlisted model, need the model's tokenizer."""
        fmt = self.prompt_format.lower()
        if fmt == "auto":
            configured = (self.base_cfg.get("prompt_formats") or {}).get(str(self.checkpoint))
            return configured in (None, "chat")
        return fmt == "chat"

    def get_prompt_formatter(self, tokenizer: Optional[Any] = None) -> Callable[[Dict[str, Any]], str]:
        """Prompt formatting function for the target model.

        The chat format renders with the model's own chat template, so it needs the model's
        tokenizer; without one it raises instead of using a generic template.
        """
        fmt = self.resolve_prompt_format(tokenizer)

        if fmt == "chat":
            if tokenizer is None:
                raise ValueError(f"Chat prompt format for {self.checkpoint} needs the model's tokenizer.")

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
            resolved_format = self.resolve_prompt_format()
        else:
            tokenizer = None
            if self.needs_tokenizer():
                from src.model import load_tokenizer

                tokenizer = load_tokenizer(self.checkpoint)
            resolved_format = self.resolve_prompt_format(tokenizer)
            logger.info(
                f"Running model inference: checkpoint={self.checkpoint}, adapter={self.adapter_path}, "
                f"preset={self.preset}, prompt_format={resolved_format}, seed={self.seed}"
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
                prompt_formatter=self.get_prompt_formatter(tokenizer),
                seed=self.seed,
                vllm_chunk_size=int(self.gen_cfg.get("batch_sizes", {}).get("vllm", 128)),
                run_config={
                    "checkpoint": str(self.checkpoint),
                    "adapter_path": str(self.adapter_path) if self.adapter_path else None,
                    "prompt_format": resolved_format,
                    "split": self.split,
                },
            )

        # 5. Re-score with the current parser, then compute the full metric suite
        records, rescored_changes = rescore_records(records)
        if rescored_changes:
            logger.info(f"Current parser changed {rescored_changes} stored answers in {generations_path.name}.")
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
            "prompt_format": resolved_format,
            "seed": self.seed,
            "mock": self.use_mock,
            "rescored_answers_changed": rescored_changes,
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
                "prompt_format": resolved_format,
                "seed": self.seed,
                "mock": self.use_mock,
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
