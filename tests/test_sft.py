"""Unit tests for SFT dataset formatting, loss masking, and trainer orchestration (Phase 4 Day 26)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.data.sft_dataset import (
    format_sft_prompt_completion,
    load_sft_dataset,
    prepare_sft_examples_from_curated,
    tokenize_sft_example,
    verify_sft_loss_masking,
)
from src.training.sft_trainer import (
    CompletionOnlyDataCollator,
    MockModel,
    MockTokenizer,
    build_sft_trainer,
    load_lora_config,
    run_sft_training,
)


class TestSFTDatasetFormatting(unittest.TestCase):
    """Test suite for SFT prompt-completion formatting and EOS appending."""

    def test_format_sft_prompt_completion_structure(self) -> None:
        """Prompt uses zero-shot template and completion cleanly ends with EOS."""
        q = "Janet has 3 packs of 10 pencils. How many pencils?"
        comp = "Janet has 3 * 10 = 30 pencils.\n#### 30"
        pair = format_sft_prompt_completion(question=q, completion=comp, eos_token="<|eos|>")

        self.assertIn("Question: Janet has 3 packs of 10 pencils.", pair["prompt"])
        self.assertIn("Solve the problem step by step.", pair["prompt"])
        self.assertTrue(pair["prompt"].endswith("Answer: Let's think step by step.\n"))
        self.assertTrue(pair["completion"].endswith("<|eos|>"))
        self.assertIn("#### 30<|eos|>", pair["completion"])
        self.assertEqual(pair["full_text"], f"{pair['prompt']}{pair['completion']}")

    def test_format_sft_prompt_completion_idempotent_eos(self) -> None:
        """If completion already ends with EOS token, it must not be duplicated."""
        q = "What is 2 + 2?"
        comp = "2 + 2 = 4.\n#### 4<|eos|>"
        pair = format_sft_prompt_completion(question=q, completion=comp, eos_token="<|eos|>")
        self.assertTrue(pair["completion"].endswith("#### 4<|eos|>"))
        self.assertFalse(pair["completion"].endswith("<|eos|><|eos|>"))

    def test_prepare_sft_examples_from_curated(self) -> None:
        """Prepares valid SFT examples from raw curated records."""
        traces = [
            {
                "problem_id": "prob_01",
                "trace_id": "trace_01",
                "question": "What is 7 * 8?",
                "completion": "7 * 8 = 56.\n#### 56",
                "gold_answer": "56",
                "prompt_type": "chain_of_thought",
            }
        ]
        examples = prepare_sft_examples_from_curated(traces, eos_token="[EOS]")
        self.assertEqual(len(examples), 1)
        ex = examples[0]
        self.assertEqual(ex["problem_id"], "prob_01")
        self.assertEqual(ex["gold_answer"], "56")
        self.assertTrue(ex["completion"].endswith("#### 56[EOS]"))

    def test_load_sft_dataset_with_limit(self) -> None:
        """Loads curated dataset file and respects limit."""
        curated_file = Path("data/synthetic/curated/traces_curated_sft.jsonl")
        if curated_file.exists():
            examples = load_sft_dataset(curated_file, limit=5)
            self.assertEqual(len(examples), 5)
            for ex in examples:
                self.assertIn("prompt", ex)
                self.assertIn("completion", ex)
                self.assertIn("text", ex)


class TestSFTTokenizationAndLossMasking(unittest.TestCase):
    """Test suite for tokenization, prompt masking, and EOS supervision."""

    def setUp(self) -> None:
        self.tokenizer = MockTokenizer()

    def test_tokenize_sft_example_masks_prompt_only(self) -> None:
        """Labels must be -100 for all prompt tokens and token IDs for completion."""
        ex = {
            "prompt": "Question: 5 + 5?\nAnswer: Let's think step by step.\n",
            "completion": "5 + 5 = 10.\n#### 10<|eos|>",
        }
        res = tokenize_sft_example(ex, self.tokenizer, max_seq_length=512)

        prompt_len = res["prompt_len"]
        completion_len = res["completion_len"]
        labels = res["labels"]
        input_ids = res["input_ids"]

        self.assertEqual(len(input_ids), prompt_len + completion_len)
        self.assertEqual(len(labels), len(input_ids))

        # Check prompt masked
        for lbl in labels[:prompt_len]:
            self.assertEqual(lbl, -100)

        # Check completion unmasked
        for lbl in labels[prompt_len:]:
            self.assertNotEqual(lbl, -100)

    def test_tokenize_sft_example_eos_not_masked(self) -> None:
        """The final token must be the EOS token and must NOT be masked (-100)."""
        ex = {
            "prompt": "Prompt text here",
            "completion": "Completion steps.\n#### 42<|eos|>",
        }
        res = tokenize_sft_example(ex, self.tokenizer)
        labels = res["labels"]

        self.assertEqual(labels[-1], self.tokenizer.eos_token_id)
        self.assertNotEqual(labels[-1], -100)

    def test_verify_sft_loss_masking_hand_check(self) -> None:
        """Hand-check verification function passes all conditions."""
        prompt = "Question: solve 10 + 20\nAnswer: Let's think step by step.\n"
        completion = "10 + 20 = 30.\n#### 30"
        check = verify_sft_loss_masking(self.tokenizer, prompt=prompt, completion=completion)

        self.assertTrue(check["all_checks_passed"])
        self.assertTrue(check["prompt_masked"])
        self.assertTrue(check["completion_supervised"])
        self.assertTrue(check["eos_present"])
        self.assertTrue(check["eos_unmasked"])
        self.assertEqual(check["last_label"], self.tokenizer.eos_token_id)


class TestLoRAConfigAndTrainingArgs(unittest.TestCase):
    """Test suite validating lora_config.yaml and training_args.json."""

    def test_load_lora_config_parameters(self) -> None:
        """LoRA configuration parameters match project plan specifications."""
        cfg = load_lora_config()
        # Handle dict or peft.LoraConfig
        r = getattr(cfg, "r", None) or cfg.get("r")
        alpha = getattr(cfg, "lora_alpha", None) or cfg.get("lora_alpha")
        dropout = getattr(cfg, "lora_dropout", None) or cfg.get("lora_dropout")
        target_mods = getattr(cfg, "target_modules", None) or cfg.get("target_modules")

        self.assertIn(r, [16, 32], "Rank must be between 16 and 32")
        self.assertEqual(alpha, 2 * r, "Alpha must be twice the rank")
        self.assertEqual(dropout, 0.05, "LoRA dropout must be 0.05")
        for expected_mod in ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            self.assertIn(expected_mod, target_mods, f"Target module {expected_mod} missing")

    def test_load_training_arguments_parameters(self) -> None:
        """Training arguments match project plan specifications."""
        args_file = Path("configs/training_args.json")
        self.assertTrue(args_file.exists())
        data = json.loads(args_file.read_text(encoding="utf-8"))

        self.assertEqual(data["num_train_epochs"], 3)
        self.assertEqual(data["per_device_train_batch_size"], 4)
        self.assertEqual(data["gradient_accumulation_steps"], 8)
        self.assertEqual(data["per_device_train_batch_size"] * data["gradient_accumulation_steps"], 32, "Effective batch must be 32")
        self.assertEqual(data["lr_scheduler_type"], "cosine")
        self.assertEqual(data["max_seq_length"], 512)
        self.assertTrue(data["bf16"])
        self.assertTrue(data["gradient_checkpointing"])


class TestCompletionOnlyDataCollator(unittest.TestCase):
    """Test suite for batch dynamic padding and label preservation."""

    def test_collator_preserves_eos_and_masks_padding(self) -> None:
        """Padding positions receive -100; genuine completion tokens keep their IDs."""
        tokenizer = MockTokenizer()
        collator = CompletionOnlyDataCollator(tokenizer=tokenizer, pad_to_multiple_of=None)

        features = [
            {
                "input_ids": [10, 20, 30, tokenizer.eos_token_id],
                "attention_mask": [1, 1, 1, 1],
                "labels": [-100, -100, 30, tokenizer.eos_token_id],
            },
            {
                "input_ids": [10, 20, tokenizer.eos_token_id],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 20, tokenizer.eos_token_id],
            },
        ]

        batch = collator(features)
        # Sequence 2 must be padded to length 4
        seq2_ids = list(batch["input_ids"][1]) if hasattr(batch["input_ids"], "tolist") else batch["input_ids"][1]
        seq2_labels = list(batch["labels"][1]) if hasattr(batch["labels"], "tolist") else batch["labels"][1]
        seq2_mask = list(batch["attention_mask"][1]) if hasattr(batch["attention_mask"], "tolist") else batch["attention_mask"][1]

        self.assertEqual(len(seq2_ids), 4)
        self.assertEqual(seq2_ids[3], tokenizer.pad_token_id)
        self.assertEqual(seq2_mask[3], 0)
        self.assertEqual(seq2_labels[3], -100)  # Padded position masked
        self.assertEqual(seq2_labels[2], tokenizer.eos_token_id)  # Real EOS preserved


class TestSFTTrainerAndExecution(unittest.TestCase):
    """Test suite for SFT trainer setup and end-to-end dry-run execution."""

    def test_build_sft_trainer_mock(self) -> None:
        """Trainer builds and trains cleanly on mock data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = MockModel()
            tokenizer = MockTokenizer()
            train_data = [
                {
                    "input_ids": [10, 20, 30, 1],
                    "attention_mask": [1, 1, 1, 1],
                    "labels": [-100, -100, 30, 1],
                }
            ]

            trainer = build_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_data,
                output_dir=tmp_dir,
                max_steps=2,
            )

            result = trainer.train()
            self.assertEqual(result["global_step"], 2)
            self.assertIn("training_loss", result)

            trainer.save_model(tmp_dir)
            self.assertTrue((Path(tmp_dir) / "adapter_config.json").exists())

    def test_run_sft_training_end_to_end_cpu(self) -> None:
        """Execute run_sft_training pipeline in sample mode on CPU."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_ckpt = tmp_path / "checkpoints"
            out_adapter = tmp_path / "adapter"

            curated_file = tmp_path / "curated.jsonl"
            records = [
                {
                    "problem_id": f"p_{i}",
                    "trace_id": f"t_{i}",
                    "question": f"What is {i} * 2?",
                    "completion": f"{i} * 2 = {i*2}.\n#### {i*2}",
                    "gold_answer": str(i * 2),
                    "prompt_type": "chain_of_thought",
                }
                for i in range(1, 4)
            ]
            with open(curated_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            trainer, run_info = run_sft_training(
                model_name_or_path="mock-model",
                curated_data_path=curated_file,
                output_dir=out_ckpt,
                final_adapter_dir=out_adapter,
                device="cpu",
                sample_limit=3,
                max_steps=2,
            )

            self.assertIsNotNone(trainer)
            self.assertIn("run_id", run_info)
            self.assertTrue(out_adapter.exists())
            self.assertTrue((out_adapter / "run_info.json").exists())

    def test_save_run_info_serializes_sets_and_namedtuples(self) -> None:
        """Verify save_run_info handles PEFT set fields and HF TrainOutput namedtuples cleanly."""
        from collections import namedtuple
        from src.run_info import save_run_info

        TrainOutput = namedtuple("TrainOutput", ["global_step", "training_loss", "metrics"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            config_with_set = {
                "lora_config": {
                    "r": 16,
                    "target_modules": {"q_proj", "v_proj", "k_proj"},
                },
                "train_result": TrainOutput(global_step=10, training_loss=0.42, metrics={"epoch": 1.0}),
                "path_obj": Path("/tmp/test"),
            }
            out_file = save_run_info(stage="sft", output_dir=out_dir, config=config_with_set)
            self.assertTrue(out_file.exists())
            loaded = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(
                loaded["config_snapshot"]["lora_config"]["target_modules"],
                ["k_proj", "q_proj", "v_proj"],
            )
            self.assertEqual(loaded["config_snapshot"]["train_result"]["global_step"], 10)


class TestCLIScriptsExecution(unittest.TestCase):
    """Test suite for CLI execution of verification and training scripts."""

    def test_cli_verify_sft_tokenization_execution(self) -> None:
        """Run scripts/verify_sft_tokenization.py and assert outputs are written."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reports_dir = Path(tmp_dir) / "reports"
            cmd = [
                sys.executable,
                "scripts/verify_sft_tokenization.py",
                "--reports_dir",
                str(reports_dir),
                "--report_name",
                "test_token_check",
                "--num_examples",
                "2",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                result.returncode,
                0,
                f"verify_sft_tokenization.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )

            self.assertTrue((reports_dir / "test_token_check.md").exists())
            self.assertTrue((reports_dir / "test_token_check.json").exists())


if __name__ == "__main__":
    unittest.main()
