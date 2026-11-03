"""Unit tests for Day 28 learning-rate sweep on the development model."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.sweep_learning_rates import run_learning_rate_sweep


class TestLearningRateSweep(unittest.TestCase):
    """Test suite validating learning rate sweep orchestration, metrics, and adapter selection."""

    def test_run_learning_rate_sweep_mock(self) -> None:
        """run_learning_rate_sweep executes across 3 LRs, selects winner, and exports reports."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_base = tmp_path / "sweep_out"
            out_reports = tmp_path / "reports"
            best_adapter = tmp_path / "adapter_model_dev"

            curated_file = tmp_path / "sample_curated.jsonl"
            records = [
                {
                    "problem_id": f"p_{i}",
                    "trace_id": f"t_{i}",
                    "question": f"What is {i} * 10?",
                    "completion": f"{i} * 10 = {i * 10}.\n#### {i * 10}",
                    "gold_answer": str(i * 10),
                    "prompt_type": "chain_of_thought",
                }
                for i in range(1, 6)
            ]
            with open(curated_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            val_file = tmp_path / "sample_val.jsonl"
            with open(val_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            sweep_res = run_learning_rate_sweep(
                model_name_or_path="mock-model",
                curated_data_path=str(curated_file),
                val_data_path=str(val_file),
                learning_rates=[5e-5, 1e-4, 2e-4],
                max_steps=4,
                eval_steps=2,
                eval_limit=5,
                output_base_dir=str(out_base),
                reports_dir=str(out_reports),
                best_adapter_dir=str(best_adapter),
                device="cpu",
            )

            self.assertIn("best_learning_rate", sweep_res)
            self.assertEqual(len(sweep_res["all_runs"]), 3)
            self.assertEqual(sweep_res["best_learning_rate"], 1e-4)

            # Assert winning adapter was copied to best_adapter_dir
            self.assertTrue(best_adapter.exists())
            self.assertTrue((best_adapter / "adapter_config.json").exists())

            # Assert reports were written
            rep_json = out_reports / "day28_lr_sweep_report.json"
            rep_md = out_reports / "day28_lr_sweep_report.md"
            self.assertTrue(rep_json.exists())
            self.assertTrue(rep_md.exists())

            rep_data = json.loads(rep_json.read_text(encoding="utf-8"))
            self.assertEqual(len(rep_data["all_runs"]), 3)
            self.assertEqual(rep_data["best_learning_rate"], 1e-4)

    def test_cli_sweep_learning_rates_execution(self) -> None:
        """Execute scripts/sweep_learning_rates.py via CLI in mock mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_base = tmp_path / "sweep_out"
            out_reports = tmp_path / "reports"
            best_adapter = tmp_path / "adapter_model_dev"

            cmd = [
                sys.executable,
                "scripts/sweep_learning_rates.py",
                "--model_name_or_path",
                "mock-model",
                "--learning_rates",
                "5e-5",
                "1e-4",
                "2e-4",
                "--max_steps",
                "2",
                "--eval_steps",
                "2",
                "--eval_limit",
                "3",
                "--sample_mode",
                "--sample_limit",
                "3",
                "--output_base_dir",
                str(out_base),
                "--reports_dir",
                str(out_reports),
                "--best_adapter_dir",
                str(best_adapter),
                "--device",
                "cpu",
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                proc.returncode,
                0,
                f"scripts/sweep_learning_rates.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            )
            self.assertTrue((out_reports / "day28_lr_sweep_report.json").exists())
            self.assertTrue((out_reports / "day28_lr_sweep_report.md").exists())
            self.assertTrue(best_adapter.exists())


if __name__ == "__main__":
    unittest.main()
