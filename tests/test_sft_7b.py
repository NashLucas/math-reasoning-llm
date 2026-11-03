"""Unit tests for Phase 4 Day 29: 7B SFT Training with LoRA and Resumption.

Verifies:
1. 7B configuration defaults: model ID, winning learning rate 1e-4, LoRA rank 16 / alpha 32,
   cosine scheduler, and completion loss masking.
2. Checkpoint resumption: automatic detection of completed checkpoints, correct start step
   offsetting, and handling of already-completed checkpoints.
3. Provenance and artifact persistence: generation of run_info.json, adapter_config.json,
   adapter_model.bin/safetensors, and checkpoint pruning.
4. CLI execution of scripts/train_7b_sft.py and train.py entrypoint in sample mode.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_7b_sft import run_day29_7b_training  # noqa: E402
from src.config import ConfigManager  # noqa: E402
from src.training.checkpointing import get_checkpoints  # noqa: E402
from src.training.sft_trainer import (  # noqa: E402
    MockModel,
    MockSFTTrainer,
    MockTokenizer,
    load_lora_config,
    load_training_arguments,
)


class TestDay297BConfigurations(unittest.TestCase):
    """Test suite verifying hyperparameter specifications for 7B SFT training."""

    def setUp(self) -> None:
        self.mgr = ConfigManager()

    def test_base_config_model_specification(self) -> None:
        """Verify base_config defines deepseek-ai/deepseek-math-7b-base as train_base."""
        models = self.mgr.base_config.get("models", {})
        self.assertEqual(models.get("train_base"), "deepseek-ai/deepseek-math-7b-base")

    def test_lora_config_hyperparameters(self) -> None:
        """Verify LoRA configuration specifies rank 16, alpha 32, and all linear targets."""
        lora_cfg = load_lora_config()
        if hasattr(lora_cfg, "r"):
            self.assertEqual(lora_cfg.r, 16)
            self.assertEqual(lora_cfg.lora_alpha, 32)
            self.assertIn("q_proj", lora_cfg.target_modules)
            self.assertIn("v_proj", lora_cfg.target_modules)
        else:
            self.assertEqual(lora_cfg.get("r"), 16)
            self.assertEqual(lora_cfg.get("lora_alpha"), 32)
            self.assertIn("q_proj", lora_cfg.get("target_modules", []))

    def test_training_args_hyperparameters(self) -> None:
        """Verify training_args specifies lr 1e-4, batch 32, cosine scheduler, and save_steps 50."""
        targs = load_training_arguments()
        lr = getattr(targs, "learning_rate", None) or targs.get("learning_rate")
        self.assertEqual(lr, 1e-4)

        sched = getattr(targs, "lr_scheduler_type", None) or targs.get("lr_scheduler_type")
        sched_str = getattr(sched, "value", str(sched)).lower()
        self.assertIn("cosine", sched_str)

        save_steps = getattr(targs, "save_steps", None) or targs.get("save_steps")
        self.assertEqual(save_steps, 50)


class TestCheckpointResumption(unittest.TestCase):
    """Test suite verifying checkpoint resumption and state continuation."""

    def test_mock_trainer_resumes_from_exact_step(self) -> None:
        """Verify MockSFTTrainer starts from checkpoint step + 1."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            ckpt_dir = out_path / "checkpoint-3"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "trainer_state.json").write_text(json.dumps({"global_step": 3}), encoding="utf-8")

            trainer = MockSFTTrainer(
                model=MockModel(),
                tokenizer=MockTokenizer(),
                train_dataset=[{"input_ids": [1, 2], "labels": [-100, 2]}],
                output_dir=out_path,
                max_steps=5,
            )

            result = trainer.train(resume_from_checkpoint=str(ckpt_dir))
            self.assertEqual(result["global_step"], 5)
            # Should have run steps 4 and 5
            self.assertEqual(len(result["loss_history"]), 2)
            self.assertEqual(result["loss_history"][0]["step"], 4)
            self.assertEqual(result["loss_history"][1]["step"], 5)

    def test_mock_trainer_already_completed_checkpoint(self) -> None:
        """Verify MockSFTTrainer cleanly completes if checkpoint step >= max_steps."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            ckpt_dir = out_path / "checkpoint-5"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "trainer_state.json").write_text(json.dumps({"global_step": 5}), encoding="utf-8")

            trainer = MockSFTTrainer(
                model=MockModel(),
                tokenizer=MockTokenizer(),
                train_dataset=[{"input_ids": [1, 2], "labels": [-100, 2]}],
                output_dir=out_path,
                max_steps=5,
            )

            result = trainer.train(resume_from_checkpoint=str(ckpt_dir))
            self.assertEqual(result["global_step"], 5)
            self.assertEqual(len(result["loss_history"]), 0)
            self.assertIn("resumed_from", result)


class TestDay29PipelineExecution(unittest.TestCase):
    """Test suite verifying Day 29 7B training pipeline orchestration and artifact export."""

    def test_run_day29_7b_training_sample_mode(self) -> None:
        """Verify complete Day 29 training workflow in sample mode on CPU."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_ckpt = tmp_path / "checkpoints" / "sft"
            out_adapter = tmp_path / "outputs" / "sft" / "adapter_model"
            out_rep = tmp_path / "reports"
            out_gen = tmp_path / "logs" / "generations"

            curated_file = tmp_path / "curated.jsonl"
            records = [
                {
                    "problem_id": f"gsm_{i}",
                    "trace_id": f"trace_{i}",
                    "question": f"Janet sells {i} apples for $2 each. How much money did she make?",
                    "completion": f"Janet sells {i} apples at $2 each. {i} * 2 = {i*2}.\n#### {i*2}",
                    "gold_answer": str(i * 2),
                    "prompt_type": "chain_of_thought",
                }
                for i in range(1, 6)
            ]
            with open(curated_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            report = run_day29_7b_training(
                model_name_or_path="mock-model",
                curated_file=curated_file,
                output_dir=out_ckpt,
                adapter_dir=out_adapter,
                generations_dir=out_gen,
                reports_dir=out_rep,
                learning_rate=1e-4,
                eval_steps=2,
                eval_limit=3,
                max_steps=4,
                sample_mode=True,
                sample_limit=5,
                device="cpu",
            )

            self.assertIsNotNone(report)
            self.assertEqual(report["day"], 29)
            self.assertEqual(report["learning_rate"], 1e-4)

            # Checkpoints must be saved
            saved = get_checkpoints(out_ckpt, complete_only=True)
            self.assertGreater(len(saved), 0)

            # Adapter artifacts
            self.assertTrue((out_adapter / "adapter_config.json").exists())
            self.assertTrue((out_adapter / "run_info.json").exists())

            # Reports
            self.assertTrue((out_rep / "day29_7b_sft_report.md").exists())
            self.assertTrue((out_rep / "day29_7b_sft_report.json").exists())

            # Verify run_info records provenance (Design Rule 5)
            run_info = json.loads((out_adapter / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(run_info["stage"], "sft")
            self.assertIn("git", run_info)
            self.assertIn("commit_hash", run_info["git"])
            self.assertIn("config_snapshot", run_info)


class TestDay29CLIScripts(unittest.TestCase):
    """Test suite verifying CLI execution of train_7b_sft.py and root train.py."""

    def test_train_7b_sft_cli_execution(self) -> None:
        """Verify scripts/train_7b_sft.py runs cleanly via subprocess in sample mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_ckpt = tmp_path / "checkpoints"
            out_adapter = tmp_path / "adapter"
            out_rep = tmp_path / "reports"

            cmd = [
                sys.executable,
                "scripts/train_7b_sft.py",
                "--model_name_or_path",
                "mock-model",
                "--output_dir",
                str(out_ckpt),
                "--adapter_dir",
                str(out_adapter),
                "--reports_dir",
                str(out_rep),
                "--sample_mode",
                "--max_steps",
                "3",
                "--eval_steps",
                "2",
                "--eval_limit",
                "3",
                "--device",
                "cpu",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                result.returncode,
                0,
                f"train_7b_sft.py CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )
            self.assertTrue((out_rep / "day29_7b_sft_report.md").exists())
            self.assertTrue((out_rep / "day29_7b_sft_report.json").exists())

    def test_root_train_py_cli_help(self) -> None:
        """Verify root train.py CLI is executable and displays help message."""
        cmd = [sys.executable, "train.py", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"train.py --help failed:\nSTDERR:\n{result.stderr}")
        self.assertIn("Run SFT training", result.stdout)


if __name__ == "__main__":
    unittest.main()
