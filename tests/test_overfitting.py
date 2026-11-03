"""Unit tests for SFT overfitting sanity test workflow (Phase 4 Day 27)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.test_sft_overfit import run_overfitting_test


class TestSFTOverfitting(unittest.TestCase):
    """Test suite validating SFT overfitting sanity mechanics."""

    def test_overfitting_sanity_check_mock(self) -> None:
        """run_overfitting_test executes cleanly and produces reports with declining loss."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_ckpt = tmp_path / "checkpoints"
            out_rep = tmp_path / "reports"

            curated_file = tmp_path / "sample_curated.jsonl"
            records = [
                {
                    "problem_id": f"p_{i}",
                    "trace_id": f"t_{i}",
                    "question": f"What is {i} * 5?",
                    "completion": f"{i} * 5 = {i * 5}.\n#### {i * 5}",
                    "gold_answer": str(i * 5),
                    "prompt_type": "chain_of_thought",
                }
                for i in range(1, 6)
            ]
            with open(curated_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            res = run_overfitting_test(
                model_name_or_path="mock-model",
                curated_data_path=str(curated_file),
                num_examples=5,
                max_steps=5,
                output_dir=str(out_ckpt),
                reports_dir=str(out_rep),
                device="cpu",
            )

            self.assertTrue(res["test_passed"])
            self.assertLess(res["final_loss"], res["initial_loss"])
            self.assertGreaterEqual(res["loss_reduction_pct"], 50.0)
            self.assertEqual(res["format_compliance_rate"], 1.0)
            self.assertEqual(res["reproduce_accuracy"], 1.0)

            # Check that reports were generated
            rep_json = out_rep / "day27_overfitting_test.json"
            rep_md = out_rep / "day27_overfitting_test.md"
            self.assertTrue(rep_json.exists())
            self.assertTrue(rep_md.exists())

            report_obj = json.loads(rep_json.read_text(encoding="utf-8"))
            self.assertTrue(report_obj["test_passed"])
            self.assertEqual(len(report_obj["results"]), 5)

    def test_cli_test_sft_overfit_execution(self) -> None:
        """Execute scripts/test_sft_overfit.py via CLI in mock mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rep_dir = Path(tmp_dir) / "reports"
            ckpt_dir = Path(tmp_dir) / "checkpoints"

            cmd = [
                sys.executable,
                "scripts/test_sft_overfit.py",
                "--model_name_or_path",
                "mock-model",
                "--num_examples",
                "3",
                "--max_steps",
                "5",
                "--output_dir",
                str(ckpt_dir),
                "--reports_dir",
                str(rep_dir),
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                proc.returncode,
                0,
                f"scripts/test_sft_overfit.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            )
            self.assertTrue((rep_dir / "day27_overfitting_test.json").exists())
            self.assertTrue((rep_dir / "day27_overfitting_test.md").exists())


if __name__ == "__main__":
    unittest.main()
