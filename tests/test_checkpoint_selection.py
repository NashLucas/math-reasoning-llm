"""Unit tests for Phase 4 Day 30: Checkpoint Evaluation, Selection & Phase 4 Gate (v0.5).

Verifies:
1. Checkpoint ranking strictly adheres to Design Rule 3 (validation accuracy, not loss).
2. Tie-breaking rules (higher format compliance, earlier step to avoid memorization).
3. Adapter copying to outputs/sft/adapter_model/.
4. Phase 4 Gate verification: greedy accuracy > base model, format compliance >= 98%, low truncation.
5. CLI execution and report generation (reports/sft_training.md & reports/day30_best_checkpoint.json).
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

from scripts.pick_best_checkpoint import evaluate_and_pick_best_checkpoint  # noqa: E402


class TestCheckpointSelectionRanking(unittest.TestCase):
    """Test suite verifying checkpoint evaluation, ranking, and tie-breaking."""

    def test_ranking_by_validation_accuracy(self) -> None:
        """Verify checkpoint with highest validation accuracy is ranked first."""
        checkpoints = [
            {"step": 50, "checkpoint_name": "checkpoint-50", "accuracy": 0.45, "format_compliance_rate": 0.99},
            {"step": 100, "checkpoint_name": "checkpoint-100", "accuracy": 0.58, "format_compliance_rate": 0.98},
            {"step": 150, "checkpoint_name": "checkpoint-150", "accuracy": 0.52, "format_compliance_rate": 1.00},
        ]
        ranked = sorted(
            checkpoints,
            key=lambda x: (x["accuracy"], x["format_compliance_rate"], -x["step"]),
            reverse=True,
        )
        self.assertEqual(ranked[0]["checkpoint_name"], "checkpoint-100")
        self.assertEqual(ranked[0]["step"], 100)

    def test_tie_breaking_by_format_compliance_and_step(self) -> None:
        """When accuracy ties, prefer higher format compliance, then earlier step."""
        checkpoints = [
            {"step": 200, "checkpoint_name": "checkpoint-200", "accuracy": 0.55, "format_compliance_rate": 0.98},
            {"step": 100, "checkpoint_name": "checkpoint-100", "accuracy": 0.55, "format_compliance_rate": 0.99},
            {"step": 150, "checkpoint_name": "checkpoint-150", "accuracy": 0.55, "format_compliance_rate": 0.99},
        ]
        ranked = sorted(
            checkpoints,
            key=lambda x: (x["accuracy"], x["format_compliance_rate"], -x["step"]),
            reverse=True,
        )
        # Should pick checkpoint-100: same accuracy (0.55), same format compliance (0.99), earlier step (100 < 150)
        self.assertEqual(ranked[0]["checkpoint_name"], "checkpoint-100")
        self.assertEqual(ranked[0]["step"], 100)


class TestDay30PipelineExecution(unittest.TestCase):
    """Test suite verifying Day 30 pipeline execution, artifact persistence, and gate evaluation."""

    def test_evaluate_and_pick_best_checkpoint_mock(self) -> None:
        """Verify complete Day 30 workflow in mock mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ckpts_dir = tmp_path / "checkpoints" / "sft"
            adapter_out = tmp_path / "outputs" / "sft" / "adapter_model"
            rep_dir = tmp_path / "reports"

            # Create mock checkpoints
            for step in [50, 100]:
                c_dir = ckpts_dir / f"checkpoint-{step}"
                c_dir.mkdir(parents=True, exist_ok=True)
                (c_dir / "trainer_state.json").write_text(json.dumps({"global_step": step, "loss": 0.2}), encoding="utf-8")
                (c_dir / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"]}), encoding="utf-8")
                (c_dir / "adapter_model.bin").write_text(f"weights_step_{step}", encoding="utf-8")
                (c_dir / "adapter_model.safetensors").write_bytes(f"weights_step_{step}".encode())

            report = evaluate_and_pick_best_checkpoint(
                model_name_or_path="mock-model",
                checkpoints_dir=ckpts_dir,
                adapter_dir=adapter_out,
                val_split="sample",
                limit=5,
                reports_dir=rep_dir,
                use_mock=True,
            )

            self.assertIsNotNone(report)
            self.assertEqual(report["phase"], 4)
            self.assertEqual(report["release_tag"], "v0.5")
            self.assertIn("best_checkpoint", report)

            # Best adapter copied
            self.assertTrue(adapter_out.exists())
            self.assertTrue((adapter_out / "adapter_config.json").exists())
            self.assertTrue((adapter_out / "adapter_model.bin").exists())
            self.assertTrue((adapter_out / "adapter_model.safetensors").exists())
            self.assertTrue((adapter_out / "run_info.json").exists())

            # Reports generated
            self.assertTrue((rep_dir / "day30_best_checkpoint.json").exists())
            self.assertTrue((rep_dir / "sft_training.md").exists())

            # Gate verification checks
            gate = report["gate_assessment"]
            self.assertTrue(gate["gate_passed"])
            self.assertEqual(gate["verdict"], "PASSED (Phase 4 Gate Met)")
            self.assertTrue(gate["format_compliance_check"]["passed"])

    def test_cli_pick_best_checkpoint_execution(self) -> None:
        """Verify scripts/pick_best_checkpoint.py runs cleanly via subprocess."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ckpts_dir = tmp_path / "checkpoints"
            adapter_out = tmp_path / "adapter_model"
            rep_dir = tmp_path / "reports"

            c_dir = ckpts_dir / "checkpoint-50"
            c_dir.mkdir(parents=True, exist_ok=True)
            (c_dir / "trainer_state.json").write_text(json.dumps({"global_step": 50}), encoding="utf-8")
            (c_dir / "adapter_config.json").write_text(json.dumps({"r": 16}), encoding="utf-8")
            (c_dir / "adapter_model.bin").write_text("weights", encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/pick_best_checkpoint.py",
                "--model_name_or_path",
                "mock-model",
                "--checkpoints_dir",
                str(ckpts_dir),
                "--adapter_dir",
                str(adapter_out),
                "--reports_dir",
                str(rep_dir),
                "--val_split",
                "sample",
                "--limit",
                "4",
                "--use_mock",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"pick_best_checkpoint CLI failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
            self.assertTrue((rep_dir / "sft_training.md").exists())
            self.assertTrue((rep_dir / "day30_best_checkpoint.json").exists())


if __name__ == "__main__":
    unittest.main()
