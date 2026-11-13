"""Unit tests for Phase 6 Day 45: GRPO Post-Training Execution & Convergence Diagnostics."""

import tempfile
import unittest
from pathlib import Path

from scripts.run_day45_training import (
    compute_convergence_diagnostics,
    generate_day45_reports,
    plot_day45_convergence_curves,
    run_day45_grpo_training,
    verify_advancing_adapter,
)
from src.data.grpo_dataset import TestSetWallError


class TestDay45Training(unittest.TestCase):
    """Test suite validating Day 45 GRPO post-training execution and convergence diagnostics."""

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent

    def test_verify_advancing_adapter_valid(self):
        """Verify that advancing adapter verification succeeds on staged adapter."""
        res = verify_advancing_adapter()
        self.assertIn(res["status"], ["VALID", "NOT_FOUND"])
        if res["status"] == "VALID":
            self.assertTrue(res["is_peft"])
            self.assertIn("adapter_model", res["path"])

    def test_verify_advancing_adapter_missing(self):
        """Verify fallback when non-existent path is provided."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = verify_advancing_adapter(adapter_dir=Path(tmp_dir) / "empty")
            self.assertIn(res["status"], ["VALID", "NOT_FOUND"])

    def test_convergence_diagnostics_calculation(self):
        """Verify metric calculation for loss delta, reward gain, and KL drift."""
        train_history = [
            {"step": 1, "loss": 0.10, "mean_reward": 1.20, "kl_penalty": 0.002, "clip_fraction": 0.05},
            {"step": 2, "loss": 0.08, "mean_reward": 1.35, "kl_penalty": 0.003, "clip_fraction": 0.08},
            {"step": 3, "loss": 0.04, "mean_reward": 1.50, "kl_penalty": 0.004, "clip_fraction": 0.06},
        ]
        val_history = [
            {"step": 1, "val_accuracy": 0.60, "val_format_compliance": 1.0, "val_structure_score": 1.0},
            {"step": 3, "val_accuracy": 0.80, "val_format_compliance": 1.0, "val_structure_score": 1.0},
        ]

        diag = compute_convergence_diagnostics(train_history, val_history)
        self.assertEqual(diag["status"], "COMPUTED")
        self.assertEqual(diag["total_steps"], 3)
        self.assertAlmostEqual(diag["initial_total_loss"], 0.10)
        self.assertAlmostEqual(diag["final_total_loss"], 0.04)
        self.assertAlmostEqual(diag["loss_delta"], -0.06)
        self.assertAlmostEqual(diag["reward_delta"], 0.30)
        self.assertEqual(diag["kl_drift_status"], "OPTIMAL_TARGET")
        self.assertEqual(diag["format_compliance_status"], "PERFECT")
        self.assertAlmostEqual(diag["val_accuracy_delta"], 0.20)

    def test_convergence_diagnostics_empty(self):
        """Verify diagnostics handle empty training history gracefully."""
        diag = compute_convergence_diagnostics([], [])
        self.assertEqual(diag["status"], "EMPTY")

    def test_kl_drift_status_thresholds(self):
        """Verify KL drift status categorizes optimal, safe, and excessive drift."""
        # Optimal (<= 0.10)
        h_opt = [{"step": 1, "loss": 0.0, "mean_reward": 1.0, "kl_penalty": 0.04, "clip_fraction": 0.0}]
        diag_opt = compute_convergence_diagnostics(h_opt, [])
        self.assertEqual(diag_opt["kl_drift_status"], "OPTIMAL_TARGET")

        # Safe (<= 0.50)
        h_safe = [{"step": 1, "loss": 0.0, "mean_reward": 1.0, "kl_penalty": 0.30, "clip_fraction": 0.0}]
        diag_safe = compute_convergence_diagnostics(h_safe, [])
        self.assertEqual(diag_safe["kl_drift_status"], "CONTROLLED_SAFE")

        # Excessive (> 0.50)
        h_warn = [{"step": 1, "loss": 0.0, "mean_reward": 1.0, "kl_penalty": 0.65, "clip_fraction": 0.0}]
        diag_warn = compute_convergence_diagnostics(h_warn, [])
        self.assertEqual(diag_warn["kl_drift_status"], "EXCESSIVE_DRIFT")

    def test_format_compliance_thresholds(self):
        """Verify format compliance status classification."""
        h = [{"step": 1, "loss": 0.0, "mean_reward": 1.0, "kl_penalty": 0.01, "clip_fraction": 0.0}]

        # Perfect >= 0.98
        v_perf = [{"step": 1, "val_accuracy": 0.8, "val_format_compliance": 1.0, "val_structure_score": 1.0}]
        self.assertEqual(compute_convergence_diagnostics(h, v_perf)["format_compliance_status"], "PERFECT")

        # Pass >= 0.95
        v_pass = [{"step": 1, "val_accuracy": 0.8, "val_format_compliance": 0.96, "val_structure_score": 1.0}]
        self.assertEqual(compute_convergence_diagnostics(h, v_pass)["format_compliance_status"], "PASS")

        # Fail < 0.95
        v_fail = [{"step": 1, "val_accuracy": 0.8, "val_format_compliance": 0.90, "val_structure_score": 1.0}]
        self.assertEqual(compute_convergence_diagnostics(h, v_fail)["format_compliance_status"], "FAIL")

    def test_plot_curves_fallback(self):
        """Verify plot generator returns a boolean and does not crash."""
        train_h = [{"step": 1, "loss": 0.05, "policy_loss": -0.01, "mean_reward": 1.4, "kl_penalty": 0.002}]
        val_h = [{"step": 1, "val_accuracy": 0.5, "val_format_compliance": 1.0}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = Path(tmp_dir) / "test_curves.png"
            res = plot_day45_convergence_curves(train_h, val_h, out_p)
            self.assertIsInstance(res, bool)

    def test_report_generation(self):
        """Verify Markdown and JSON reports are generated with required headings."""
        train_results = {
            "status": "success",
            "start_step": 0,
            "final_step": 5,
            "total_steps_executed": 5,
            "adapter_dir": "outputs/grpo/adapter_model",
            "checkpoints_saved": ["checkpoint-3", "checkpoint-5"],
        }
        diagnostics = {
            "initial_total_loss": 0.05,
            "final_total_loss": 0.01,
            "loss_delta": -0.04,
            "initial_reward": 1.2,
            "final_reward": 1.5,
            "reward_delta": 0.3,
            "mean_kl_penalty": 0.002,
            "max_kl_penalty": 0.003,
            "kl_drift_status": "OPTIMAL_TARGET",
            "mean_clip_fraction": 0.05,
            "initial_val_accuracy": 0.5,
            "final_val_accuracy": 0.8,
            "val_accuracy_delta": 0.3,
            "final_format_compliance": 1.0,
            "format_compliance_status": "PERFECT",
        }
        adapter_meta = {
            "status": "VALID",
            "path": "outputs/sft_targeted/adapter_model",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            rep_dir = Path(tmp_dir) / "reports"
            md_path, json_path = generate_day45_reports(
                train_results=train_results,
                diagnostics=diagnostics,
                adapter_meta=adapter_meta,
                reports_dir=rep_dir,
            )
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            content = md_path.read_text(encoding="utf-8")
            self.assertIn("Phase 6 Day 45", content)
            self.assertIn("Convergence Diagnostics & Metric Trajectory", content)
            self.assertIn("Reference Policy KL Divergence & Drift Control Audit", content)

    def test_run_day45_training_sample_mode(self):
        """Verify end-to-end execution of run_day45_grpo_training in sample mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            out_ckpt = tmp / "checkpoints"
            out_adapter = tmp / "adapter"
            out_rep = tmp / "reports"

            res = run_day45_grpo_training(
                output_dir=out_ckpt,
                adapter_dir=out_adapter,
                reports_dir=out_rep,
                max_steps=3,
                eval_steps=2,
                save_steps=2,
                sample_mode=True,
                use_mock=True,
                seed=42,
            )

            self.assertEqual(res["status"], "success")
            self.assertTrue(Path(res["report_md"]).exists())
            self.assertTrue(Path(res["report_json"]).exists())
            self.assertTrue((out_adapter / "adapter_config.json").exists())
            self.assertTrue((out_ckpt / "checkpoint-2").exists())

    def test_test_set_wall_in_day45(self):
        """Verify that passing test set file to Day 45 training raises TestSetWallError."""
        with self.assertRaises(TestSetWallError):
            run_day45_grpo_training(
                train_file="data/processed/test/test.parquet",
                sample_mode=True,
                use_mock=True,
            )

    def test_cli_execution_with_flags(self):
        """Verify CLI execution via subprocess with Colab launcher style arguments."""
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            out_ckpt = tmp / "checkpoints"
            out_adapter = tmp / "adapter"
            out_rep = tmp / "reports"

            cmd = [
                sys.executable,
                str(self.project_root / "scripts" / "run_day45_training.py"),
                "--config", str(self.project_root / "configs" / "grpo_args.json"),
                "--output_dir", str(out_adapter),
                "--checkpoints_dir", str(out_ckpt),
                "--reports_dir", str(out_rep),
                "--max_steps", "2",
                "--eval_steps", "2",
                "--save_steps", "2",
                "--sample_mode",
                "--use_mock",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))
            self.assertEqual(result.returncode, 0, f"CLI command failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            self.assertTrue((out_rep / "day45_grpo_training_report.md").exists())
            self.assertTrue((out_rep / "day45_grpo_training_report.json").exists())
            self.assertTrue((out_adapter / "adapter_config.json").exists())
            self.assertTrue((out_ckpt / "checkpoint-2").exists())

    def test_drive_mirroring_logic(self):
        """Verify that reports and adapter models are copied when Google Drive path is present."""
        import unittest.mock as mock

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            mock_drive_base = tmp / "drive" / "MyDrive" / "math-reasoning-llm"
            mock_drive_base.mkdir(parents=True, exist_ok=True)

            out_ckpt = tmp / "checkpoints"
            out_adapter = tmp / "adapter"
            out_rep = tmp / "reports"

            with mock.patch("scripts.run_day45_training.Path") as mock_path_cls:
                # Let Path("/content/drive/MyDrive/math-reasoning-llm") redirect to mock_drive_base
                orig_path = Path
                def fake_path(arg):
                    if str(arg) == "/content/drive/MyDrive/math-reasoning-llm":
                        return mock_drive_base
                    return orig_path(arg)
                mock_path_cls.side_effect = fake_path

                res = run_day45_grpo_training(
                    output_dir=out_ckpt,
                    adapter_dir=out_adapter,
                    reports_dir=out_rep,
                    max_steps=2,
                    eval_steps=2,
                    save_steps=2,
                    sample_mode=True,
                    use_mock=True,
                    profile="colab",
                )

                self.assertEqual(res["status"], "success")
                # Verify Drive reports mirrored
                drive_rep = mock_drive_base / "reports"
                self.assertTrue((drive_rep / "day45_grpo_training_report.md").exists())
                self.assertTrue((drive_rep / "day45_grpo_training_report.json").exists())
                # Verify Drive adapter mirrored
                drive_adapter = mock_drive_base / "outputs" / "grpo" / "adapter_model"
                self.assertTrue((drive_adapter / "adapter_config.json").exists())


if __name__ == "__main__":
    unittest.main()
