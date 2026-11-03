"""Unit tests for periodic SFT validation evaluation callback (Phase 4 Day 27)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.training.callbacks import (
    ValidationEvalCallback,
    load_validation_problems,
)
from src.training.sft_trainer import (
    MockModel,
    MockSFTTrainer,
    MockTokenizer,
)


class MockTrainerState:
    """Mock HuggingFace TrainerState for testing callback execution."""

    def __init__(self, global_step: int = 0) -> None:
        self.global_step = global_step
        self.log_history: list[dict[str, Any]] = []


class TestValidationEvalCallback(unittest.TestCase):
    """Test suite for periodic validation evaluation callback."""

    def setUp(self) -> None:
        self.sample_problems = [
            {
                "problem_id": f"prob_{i}",
                "question": f"What is {i} + {i}?",
                "gold_answer": str(i * 2),
            }
            for i in range(1, 6)
        ]

    def test_load_validation_problems_limit_and_keys(self) -> None:
        """load_validation_problems returns records with standardized keys and respects limits."""
        probs = load_validation_problems(limit=10)
        self.assertGreater(len(probs), 0)
        self.assertLessEqual(len(probs), 10)
        for p in probs:
            self.assertIn("problem_id", p)
            self.assertIn("question", p)
            self.assertIn("gold_answer", p)

    def test_callback_evaluate_and_log_outputs(self) -> None:
        """evaluate_and_log writes step jsonl and summary json with valid metrics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "generations"
            model = MockModel()
            tokenizer = MockTokenizer()

            cb = ValidationEvalCallback(
                eval_dataset=self.sample_problems,
                tokenizer=tokenizer,
                eval_steps=50,
                output_dir=out_dir,
                verbose=False,
            )

            summary = cb.evaluate_and_log(step=50, model=model, tokenizer=tokenizer)

            self.assertEqual(summary["step"], 50)
            self.assertEqual(summary["total_problems"], 5)
            self.assertEqual(summary["correct_count"], 5)
            self.assertEqual(summary["format_count"], 5)
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["format_compliance_rate"], 1.0)

            # Check written files on disk
            gen_file = out_dir / "step_50.jsonl"
            sum_file = out_dir / "step_50_summary.json"

            self.assertTrue(gen_file.exists(), "step_50.jsonl must exist")
            self.assertTrue(sum_file.exists(), "step_50_summary.json must exist")

            # Verify contents of generation records
            lines = [json.loads(line) for line in gen_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 5)
            for rec in lines:
                self.assertEqual(rec["step"], 50)
                self.assertTrue(rec["is_correct"])
                self.assertTrue(rec["format_compliant"])
                self.assertEqual(rec["extraction_method"], "hash")

    def test_callback_on_step_end_triggers_at_intervals(self) -> None:
        """on_step_end only evaluates when global_step % eval_steps == 0."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "generations"
            model = MockModel()
            tokenizer = MockTokenizer()

            cb = ValidationEvalCallback(
                eval_dataset=self.sample_problems,
                tokenizer=tokenizer,
                eval_steps=100,
                output_dir=out_dir,
                verbose=False,
            )

            # Step 50: should not trigger
            state_50 = MockTrainerState(global_step=50)
            cb.on_step_end(args=None, state=state_50, control=None, model=model, tokenizer=tokenizer)
            self.assertEqual(len(cb.history), 0)
            self.assertFalse((out_dir / "step_50.jsonl").exists())

            # Step 100: should trigger
            state_100 = MockTrainerState(global_step=100)
            cb.on_step_end(args=None, state=state_100, control=None, model=model, tokenizer=tokenizer)
            self.assertEqual(len(cb.history), 1)
            self.assertTrue((out_dir / "step_100.jsonl").exists())
            self.assertEqual(len(state_100.log_history), 1)
            self.assertEqual(state_100.log_history[0]["val_greedy_accuracy"], 1.0)

    def test_mock_sft_trainer_with_callback_integration(self) -> None:
        """MockSFTTrainer executes and triggers registered callbacks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_ckpt = tmp_path / "checkpoints"
            out_gens = tmp_path / "generations"

            model = MockModel()
            tokenizer = MockTokenizer()
            cb = ValidationEvalCallback(
                eval_dataset=self.sample_problems,
                tokenizer=tokenizer,
                eval_steps=2,
                output_dir=out_gens,
                verbose=False,
            )

            train_data = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]}]
            trainer = MockSFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_data,
                output_dir=out_ckpt,
                max_steps=4,
                callbacks=[cb],
            )

            result = trainer.train()
            self.assertEqual(result["global_step"], 4)

            # With max_steps=4 and eval_steps=2, steps 2 and 4 must have fired
            self.assertTrue((out_gens / "step_2.jsonl").exists())
            self.assertTrue((out_gens / "step_4.jsonl").exists())
            self.assertEqual(len(cb.history), 2)


if __name__ == "__main__":
    unittest.main()
