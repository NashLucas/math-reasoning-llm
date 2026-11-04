"""Unit tests for Phase 6 Day 47: Phase 6 Gate Verifier & Release v0.7 Wrap-Up.

Validates the complete verification suite across all 7 Phase 6 milestones:
1. GRPO Reward Architecture & Advantage Normalization (Day 40).
2. Intermediate Dataset & Test-Set Wall Isolation (Day 41).
3. Reference Model Policy & Schulman k3 Non-Negative KL (Day 42).
4. GRPO Engine & Zero-Advantage Invariance (Day 43).
5. Local Smoke Test & VRAM Feasibility (Day 44).
6. Full GRPO Post-Training Execution & Convergence (Day 45).
7. Post-GRPO Comparative Evaluation & Contract Preservation (Day 46).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.phase6_verifier import (
    GateCriterionStatus,
    Phase6GateSummary,
    run_phase6_wrapup,
    verify_grpo_dataset_and_wall,
    verify_grpo_engine_and_invariance,
    verify_post_grpo_evaluation_and_contract,
    verify_reference_model_and_kl,
    verify_reward_and_advantages,
    verify_smoke_test_and_vram,
    verify_training_execution_and_convergence,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase6Wrapup(unittest.TestCase):
    """Test suite for Phase 6 Gate verification and audit reports."""

    def test_verify_reward_and_advantages(self) -> None:
        """Test verification of Day 40 reward architecture."""
        status = verify_reward_and_advantages(reports_dir=PROJECT_ROOT / "reports")
        self.assertIsInstance(status, GateCriterionStatus)
        self.assertTrue(status.passed)
        self.assertIn("accuracy", status.observed)

    def test_verify_grpo_dataset_and_wall(self) -> None:
        """Test verification of Day 41 intermediate dataset and test wall."""
        status = verify_grpo_dataset_and_wall(dataset_dir=PROJECT_ROOT / "data" / "processed" / "grpo")
        self.assertIsInstance(status, GateCriterionStatus)
        self.assertTrue(status.passed)
        self.assertIn("test_leaks=0", status.observed)

        # Test failure when directory is missing or empty
        with tempfile.TemporaryDirectory() as tmpdir:
            fail_status = verify_grpo_dataset_and_wall(dataset_dir=tmpdir)
            self.assertFalse(fail_status.passed)

    def test_verify_reference_model_and_kl(self) -> None:
        """Test verification of Day 42 reference model policy and k3 KL."""
        status = verify_reference_model_and_kl(reports_dir=PROJECT_ROOT / "reports")
        self.assertIsInstance(status, GateCriterionStatus)
        self.assertTrue(status.passed)
        self.assertIn("Estimator: k3", status.observed)

    def test_verify_grpo_engine_and_invariance(self) -> None:
        """Test verification of Day 43 rollout engine and zero-advantage invariance."""
        status = verify_grpo_engine_and_invariance(reports_dir=PROJECT_ROOT / "reports")
        self.assertIsInstance(status, GateCriterionStatus)
        self.assertTrue(status.passed)
        self.assertIn("Zero-advantage invariance: True", status.observed)

    def test_verify_smoke_test_and_vram(self) -> None:
        """Test verification of Day 44 6-stage smoke test and VRAM budget."""
        status = verify_smoke_test_and_vram(reports_dir=PROJECT_ROOT / "reports")
        self.assertIsInstance(status, GateCriterionStatus)
        self.assertTrue(status.passed)
        self.assertIn("Peak VRAM:", status.observed)

    def test_verify_training_execution_and_convergence(self) -> None:
        """Test verification of Day 45 post-training execution and convergence."""
        status = verify_training_execution_and_convergence(reports_dir=PROJECT_ROOT / "reports")
        self.assertIsInstance(status, GateCriterionStatus)
        self.assertTrue(status.passed)
        self.assertIn("Format: 100.0%", status.observed)

    def test_verify_post_grpo_evaluation_and_contract(self) -> None:
        """Test verification of Day 46 comparative evaluation and contract preservation."""
        status = verify_post_grpo_evaluation_and_contract(reports_dir=PROJECT_ROOT / "reports")
        self.assertIsInstance(status, GateCriterionStatus)
        self.assertTrue(status.passed)
        self.assertIn("Verdict: PASSED", status.observed)

    def test_verify_all_phase6_gates_and_reports(self) -> None:
        """Test aggregate verification suite and report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_md = Path(tmpdir) / "phase6_gate_report.md"
            out_json = Path(tmpdir) / "phase6_summary.json"

            summary = run_phase6_wrapup(
                reports_dir=PROJECT_ROOT / "reports",
                dataset_dir=PROJECT_ROOT / "data" / "processed" / "grpo",
                output_report=out_md,
                summary_json=out_json,
            )

            self.assertIsInstance(summary, Phase6GateSummary)
            self.assertTrue(summary.all_passed)
            self.assertEqual(summary.verdict, "PASSED")
            self.assertEqual(summary.release_tag, "v0.7")
            self.assertEqual(len(summary.criteria), 7)
            self.assertTrue(out_md.exists())
            self.assertTrue(out_json.exists())

            with open(out_json, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                self.assertEqual(data["verdict"], "PASSED")
                self.assertEqual(data["release_tag"], "v0.7")
                self.assertEqual(len(data["criteria"]), 7)

            with open(out_md, "r", encoding="utf-8") as mf:
                content = mf.read()
                self.assertIn("# Phase 6 Gate Verification & Release v0.7 Wrap-Up Report", content)
                self.assertIn("Overall Verdict:** **PASSED**", content)

    def test_cli_runner(self) -> None:
        """Test scripts/verify_phase6_gate.py execution via CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "verify_phase6_gate.py"),
                "--reports_dir",
                str(PROJECT_ROOT / "reports"),
                "--dataset_dir",
                str(PROJECT_ROOT / "data" / "processed" / "grpo"),
                "--output_report",
                str(Path(tmpdir) / "gate_report.md"),
                "--summary_json",
                str(Path(tmpdir) / "summary.json"),
            ]
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"CLI execution failed with code {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )
            self.assertTrue((Path(tmpdir) / "gate_report.md").exists())
            self.assertTrue((Path(tmpdir) / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
