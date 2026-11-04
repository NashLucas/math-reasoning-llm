"""Unit tests for Phase 5 Day 36: Round-2 SFT Training & Automatic Resumption."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_round2_sft import (
    main as cli_main,
)
from scripts.train_round2_sft import (
    run_day36_round2_training,
)
from src.config import ConfigManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRound2Training(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        self.ckpts_dir = self.test_dir / "checkpoints"
        self.adapter_dir = self.test_dir / "adapter_model"
        self.gen_dir = self.test_dir / "generations"
        self.reports_dir = self.test_dir / "reports"

        for d in [self.ckpts_dir, self.adapter_dir, self.gen_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Mock sample dataset
        self.mock_dataset = self.test_dir / "mock_train_round2.jsonl"
        with open(self.mock_dataset, "w", encoding="utf-8") as f:
            for i in range(10):
                record = {
                    "problem_id": f"round2_prob_{i}",
                    "trace_id": f"round2_prob_{i}_trace",
                    "question": f"Question {i}: Solve 2 + {i}?",
                    "completion": f"Step 1: Compute 2 + {i}.\n#### {2 + i}<|eos|>",
                    "gold_answer": str(2 + i),
                    "prompt": f"Question: Solve 2 + {i}?\nAnswer:",
                    "text": f"Question: Solve 2 + {i}?\nAnswer: Step 1: Compute 2 + {i}.\n#### {2 + i}<|eos|>",
                    "source": "targeted_error_mitigation" if i % 2 == 0 else "curated_sft",
                }
                f.write(json.dumps(record) + "\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_stage_paths_resolution(self):
        """Verify ConfigManager resolves sft_targeted stage paths cleanly."""
        mgr = ConfigManager()
        stage_paths = mgr.get_stage_paths("sft_targeted", profile_name="local")
        self.assertIn("checkpoints", stage_paths)
        self.assertIn("outputs", stage_paths)
        self.assertIn("logs", stage_paths)
        self.assertTrue(str(stage_paths["checkpoints"]).endswith("sft_targeted"))

    def test_run_day36_training_mock_execution(self):
        """Verify mock Round-2 SFT training runs end-to-end and produces reports."""
        report = run_day36_round2_training(
            model_name_or_path="mock-model",
            train_file=self.mock_dataset,
            val_file=PROJECT_ROOT / "data" / "sample" / "val_sample.parquet",
            output_dir=self.ckpts_dir,
            adapter_dir=self.adapter_dir,
            generations_dir=self.gen_dir,
            reports_dir=self.reports_dir,
            learning_rate=1e-4,
            eval_steps=2,
            eval_limit=5,
            max_steps=4,
            use_mock=True,
            overwrite=True,
        )

        self.assertIsInstance(report, dict)
        self.assertEqual(report["stage"], "sft_targeted_training")
        self.assertEqual(report["day"], 36)
        self.assertIn("loss_history", report)
        self.assertIn("run_info", report)

        # Check reports written
        report_json = self.reports_dir / "day36_round2_sft_report.json"
        report_md = self.reports_dir / "day36_round2_sft_report.md"
        self.assertTrue(report_json.exists())
        self.assertTrue(report_md.exists())

        # Check adapter directory has mock artifacts and run_info.json
        self.assertTrue((self.adapter_dir / "run_info.json").exists())

    def test_round2_training_resumption(self):
        """Verify automatic checkpoint resumption from checkpoints/sft_targeted."""
        # Create an existing checkpoint
        ckpt_20 = self.ckpts_dir / "checkpoint-20"
        ckpt_20.mkdir(parents=True, exist_ok=True)
        with open(ckpt_20 / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump({"global_step": 20, "log_history": []}, f)

        report = run_day36_round2_training(
            model_name_or_path="mock-model",
            train_file=self.mock_dataset,
            val_file=PROJECT_ROOT / "data" / "sample" / "val_sample.parquet",
            output_dir=self.ckpts_dir,
            adapter_dir=self.adapter_dir,
            generations_dir=self.gen_dir,
            reports_dir=self.reports_dir,
            max_steps=2,
            use_mock=True,
            overwrite=False,
        )

        self.assertIsNotNone(report.get("run_info"))

    def test_cli_runner_sample_mode(self):
        """Verify CLI runner succeeds in sample mode."""
        exit_code = cli_main([
            "--sample_mode",
            "--max_steps",
            "2",
            "--eval_steps",
            "2",
            "--eval_limit",
            "3",
            "--output_dir",
            str(self.ckpts_dir / "cli_ckpts"),
            "--adapter_dir",
            str(self.adapter_dir / "cli_adapter"),
            "--generations_dir",
            str(self.gen_dir / "cli_gen"),
            "--reports_dir",
            str(self.reports_dir / "cli_reports"),
        ])
        self.assertEqual(exit_code, 0)
        self.assertTrue((self.reports_dir / "cli_reports" / "day36_round2_sft_report.json").exists())
        self.assertTrue((self.reports_dir / "cli_reports" / "day36_round2_sft_report.md").exists())


if __name__ == "__main__":
    unittest.main()
