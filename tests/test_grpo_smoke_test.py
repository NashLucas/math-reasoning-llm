"""Unit tests for Phase 6 Day 44: GRPO Local Smoke Test & Verification Pipeline."""

import tempfile
import unittest
from pathlib import Path

from scripts.run_grpo_smoke_test import (
    generate_smoke_test_reports,
    run_full_smoke_test,
    verify_stage1_config,
    verify_stage2_data_and_wall,
    verify_stage6_memory_budget,
)
from scripts.train_grpo import (
    load_grpo_checkpoint,
    load_grpo_prompts,
    run_grpo_training,
    run_grpo_validation,
    save_grpo_checkpoint,
    verify_test_set_wall,
)
from src.data.grpo_dataset import TestSetWallError
from src.rewards.grpo_rewards import CompositeGRPOReward


class TestGRPOSmokeTest(unittest.TestCase):
    """Test suite validating GRPO smoke test components, checkpointing, and pipeline integrity."""

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_path = self.project_root / "configs" / "grpo_args.json"

    def test_config_bounds_valid(self):
        """Verify that default GRPO configuration passes bounds checks."""
        res = verify_stage1_config(self.config_path)
        self.assertEqual(res["status"], "PASSED")
        self.assertEqual(res["group_size"], 4)
        self.assertAlmostEqual(res["kl_beta"], 0.04)
        self.assertAlmostEqual(res["clip_range"], 0.2)

    def test_test_set_wall_enforcement(self):
        """Verify that TestSetWallError is raised when attempting to ingest test split."""
        forbidden_paths = [
            "data/processed/test/test.parquet",
            "data/processed/test/test.jsonl",
            "C:\\project\\data\\processed\\test\\test.parquet",
            "data/splits/test.parquet",
        ]
        for path in forbidden_paths:
            with self.assertRaises(TestSetWallError, msg=f"Failed to raise on: {path}"):
                verify_test_set_wall(path)

    def test_test_set_wall_allows_train_and_val(self):
        """Verify that valid train and val paths pass without error."""
        allowed_paths = [
            "data/processed/grpo/grpo_train.parquet",
            "data/processed/grpo/grpo_val.parquet",
            "data/sample/train_sample.parquet",
        ]
        for path in allowed_paths:
            try:
                verify_test_set_wall(path)
            except TestSetWallError:
                self.fail(f"verify_test_set_wall raised unexpectedly on: {path}")

    def test_prompt_loader_sample_mode(self):
        """Verify prompt loader successfully retrieves prompts in sample mode."""
        prompts = load_grpo_prompts(sample_mode=True, limit=5)
        self.assertGreater(len(prompts), 0)
        self.assertLessEqual(len(prompts), 5)
        first = prompts[0]
        self.assertIn("problem_id", first)
        self.assertIn("prompt", first)
        self.assertIn("gold_answer", first)

    def test_checkpoint_saving_and_loading(self):
        """Verify checkpoint saving produces trainer_state.json and grpo_state.json."""
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            ckpt_dir = Path(tmp_dir_str) / "checkpoint-10"
            history = [{"step": 10, "loss": 0.05, "mean_reward": 1.2}]
            config = {"grpo": {"learning_rate": 1e-6}}

            save_grpo_checkpoint(ckpt_dir, step=10, history=history, config=config)

            self.assertTrue((ckpt_dir / "trainer_state.json").exists())
            self.assertTrue((ckpt_dir / "grpo_state.json").exists())
            self.assertTrue((ckpt_dir / "adapter_config.json").exists())

            loaded_step, _ = load_grpo_checkpoint(ckpt_dir)
            self.assertEqual(loaded_step, 10)

    def test_validation_callback_metrics(self):
        """Verify periodic validation computes accuracy, format, and structure rates."""
        reward_calc = CompositeGRPOReward()
        sample_prompts = [
            {"problem_id": "p1", "prompt": "Solve 1+1", "gold_answer": "2"},
            {"problem_id": "p2", "prompt": "Solve 2+2", "gold_answer": "4"},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            gen_dir = Path(tmp_dir_str) / "generations"
            summary = run_grpo_validation(
                val_prompts=sample_prompts,
                reward_calculator=reward_calc,
                step=2,
                eval_limit=10,
                generations_dir=gen_dir,
            )
            self.assertEqual(summary["step"], 2)
            self.assertEqual(summary["evaluated_count"], 2)
            self.assertGreaterEqual(summary["val_accuracy"], 0.0)
            self.assertGreaterEqual(summary["val_format_compliance"], 0.0)
            self.assertTrue((gen_dir / "step_2.jsonl").exists())

    def test_memory_budget_audit(self):
        """Verify memory budget audit calculates peak VRAM within Colab 16GB ceiling."""
        mem = verify_stage6_memory_budget()
        self.assertEqual(mem["status"], "PASSED")
        self.assertLessEqual(mem["estimated_peak_vram_gb"], 16.0)
        self.assertGreater(mem["vram_headroom_gb"], 0.0)
        self.assertTrue(mem["single_model_peft_sharing"])

    def test_end_to_end_training_smoke_run(self):
        """Verify full end-to-end 3-step training run executes cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            res = run_grpo_training(
                output_dir=tmp_dir / "checkpoints",
                adapter_dir=tmp_dir / "adapter",
                reports_dir=tmp_dir / "reports",
                generations_dir=tmp_dir / "generations",
                max_steps=3,
                eval_steps=2,
                save_steps=2,
                eval_limit=5,
                smoke_test=True,
                sample_mode=True,
                use_mock=True,
                seed=42,
            )
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["final_step"], 3)
            self.assertEqual(res["total_steps_executed"], 3)
            self.assertTrue((tmp_dir / "checkpoints" / "checkpoint-2").exists())
            self.assertTrue((tmp_dir / "adapter" / "adapter_config.json").exists())

    def test_smoke_resumption_continuity(self):
        """Verify resuming from checkpoint executes remaining steps continuously."""
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            out_dir = tmp_dir / "checkpoints"

            # Run steps 1 to 2
            run_grpo_training(
                output_dir=out_dir,
                adapter_dir=tmp_dir / "adapter",
                reports_dir=tmp_dir / "reports",
                generations_dir=tmp_dir / "generations",
                max_steps=2,
                eval_steps=2,
                save_steps=2,
                smoke_test=True,
                sample_mode=True,
                use_mock=True,
                seed=42,
            )
            ckpt_2 = out_dir / "checkpoint-2"
            self.assertTrue(ckpt_2.exists())

            # Resume from step 2 to step 4
            res = run_grpo_training(
                output_dir=out_dir,
                adapter_dir=tmp_dir / "adapter",
                reports_dir=tmp_dir / "reports",
                generations_dir=tmp_dir / "generations",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                smoke_test=True,
                sample_mode=True,
                use_mock=True,
                resume_from_checkpoint=str(ckpt_2),
                seed=42,
            )
            self.assertEqual(res["start_step"], 2)
            self.assertEqual(res["final_step"], 4)
            self.assertEqual(res["total_steps_executed"], 2)
            self.assertTrue((out_dir / "checkpoint-4").exists())

    def test_report_generation(self):
        """Verify report generation emits Markdown and JSON files with all sections."""
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            rep_dir = Path(tmp_dir_str) / "reports"
            stage_results = {
                "stage1_config": verify_stage1_config(self.config_path),
                "stage2_data": verify_stage2_data_and_wall(sample_mode=True),
                "stage3_and_4_training": {
                    "status": "PASSED",
                    "final_step": 5,
                    "initial_loss": 0.05,
                    "final_loss": 0.01,
                    "final_mean_reward": 1.4,
                    "final_val_accuracy": 1.0,
                    "checkpoints_created": ["checkpoint-3", "checkpoint-5"],
                    "generation_logs_count": 2,
                },
                "stage5_resumption": {
                    "status": "PASSED",
                    "resumed_from": "checkpoint-3",
                    "resumed_start_step": 3,
                    "resumed_final_step": 6,
                    "steps_executed_after_resume": 3,
                    "checkpoint_continuity": "EXACT",
                },
                "stage6_memory": verify_stage6_memory_budget(),
            }
            md_path, json_path = generate_smoke_test_reports(stage_results, rep_dir)
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            md_content = md_path.read_text(encoding="utf-8")
            self.assertIn("Phase 6 Day 44", md_content)
            self.assertIn("Stage-by-Stage Verification Results", md_content)
            self.assertIn("VRAM Memory Budget Breakdown", md_content)

    def test_run_full_smoke_test_runner(self):
        """Verify run_full_smoke_test executes all 6 stages cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            rep_dir = Path(tmp_dir_str) / "reports"
            exit_code = run_full_smoke_test(sample_mode=True, reports_dir=rep_dir)
            self.assertEqual(exit_code, 0)
            self.assertTrue((rep_dir / "day44_grpo_smoke_test_report.md").exists())
            self.assertTrue((rep_dir / "day44_grpo_smoke_test_report.json").exists())

    def test_cli_execution_with_flags(self):
        """Verify CLI execution via subprocess supports --config and --report_dir."""
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            rep_dir = Path(tmp_dir_str) / "reports"
            cmd = [
                sys.executable,
                "scripts/run_grpo_smoke_test.py",
                "--sample_mode",
                "--config",
                "configs/grpo_args.json",
                "--report_dir",
                str(rep_dir),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"CLI failed:\nStdout: {proc.stdout}\nStderr: {proc.stderr}")
            self.assertTrue((rep_dir / "day44_grpo_smoke_test_report.md").exists())
            self.assertTrue((rep_dir / "day44_grpo_smoke_test_report.json").exists())


if __name__ == "__main__":
    unittest.main()
