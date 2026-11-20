"""Unit tests for Phase 4 Gate Verifier (scripts/verify_phase4_gate.py)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_phase4_gate import audit_phase4_gate  # noqa: E402


class TestPhase4GateVerifier(unittest.TestCase):
    """Test suite for Phase 4 Gate verification logic."""

    def test_audit_phase4_gate_structure(self) -> None:
        """Verify audit_phase4_gate returns expected keys and 10 checks."""
        audit = audit_phase4_gate()
        self.assertEqual(audit["phase"], 4)
        self.assertEqual(audit["release_tag"], "v0.5")
        self.assertEqual(audit["total_checks"], 10)
        self.assertIn("overall_status", audit)
        self.assertIn(audit["overall_status"], ("PASSED", "MOCK_VERIFIED", "FAILED", "INCOMPLETE"))

        check_ids = [c["id"] for c in audit["checks"]]
        self.assertEqual(len(check_ids), 10)
        self.assertIn("day26_tokenization_hand_check", check_ids)
        self.assertIn("day26_configs", check_ids)
        self.assertIn("day27_template_match", check_ids)
        self.assertIn("day27_overfitting_test", check_ids)
        self.assertIn("day27_callback_logs", check_ids)
        self.assertIn("day28_lr_sweep", check_ids)
        self.assertIn("day29_7b_sft_training", check_ids)
        self.assertIn("day30_checkpoint_selection", check_ids)
        self.assertIn("phase4_gate_criteria", check_ids)
        self.assertIn("phase4_report_deliverables", check_ids)

    def test_cli_execution_verify_phase4_gate(self) -> None:
        """Verify scripts/verify_phase4_gate.py executes cleanly via subprocess."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_json = tmp_path / "phase4_summary.json"
            out_md = tmp_path / "phase4_report.md"

            cmd = [
                sys.executable,
                "scripts/verify_phase4_gate.py",
                "--output_json",
                str(out_json),
                "--output_report",
                str(out_md),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                result.returncode,
                0,
                f"verify_phase4_gate.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )
            self.assertTrue(out_json.exists())
            self.assertTrue(out_md.exists())

            loaded = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(loaded["phase"], 4)
            self.assertEqual(loaded["total_checks"], 10)


if __name__ == "__main__":
    unittest.main()
