"""Unit tests for Phase 7 Day 50: Production Serving Engine, Interactive App & Release v1.0.

Verifies:
1. ServingConfig dataclass and configuration validation.
2. ReasoningServingEngine in mock mode (greedy, sampled, self-consistency maj@k).
3. Answer Contract adherence (\\n#### <number>) and structured prediction output.
4. Batch prediction concurrency and throughput telemetry.
5. Production traffic logging to structured JSONL format.
6. Gradio application smoke test and GSM8K challenge presets.
7. Serving verification runner and operational audit report generation.
8. Phase 7 Gate Verifier auditing all 7 criteria and emitting wrap-up reports.
9. verify_phase7_gate CLI execution.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.app import GSM8K_PRESETS, run_smoke_test
from scripts.verify_serving import run_serving_verification
from src.evaluation.phase7_verifier import (
    Phase7GateSummary,
    generate_phase7_wrapup_report,
    run_phase7_wrapup,
    verify_all_phase7_gates,
)
from src.serving.engine import (
    ReasoningServingEngine,
    ServingConfig,
    ServingPrediction,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestServingConfig(unittest.TestCase):
    """Test ServingConfig creation and serialization."""

    def test_default_config(self):
        cfg = ServingConfig()
        self.assertTrue(cfg.use_mock)
        self.assertEqual(cfg.max_new_tokens, 512)
        self.assertEqual(cfg.default_temperature, 0.7)
        self.assertEqual(cfg.default_top_p, 0.95)
        self.assertEqual(cfg.adapter_path, "outputs/grpo/adapter_model")

    def test_to_dict(self):
        cfg = ServingConfig(adapter_path="outputs/grpo/adapter_model")
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["adapter_path"], "outputs/grpo/adapter_model")
        self.assertTrue(d["use_mock"])


class TestReasoningServingEngine(unittest.TestCase):
    """Test serving engine inference modes and telemetry in mock mode."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = Path(self.temp_dir.name) / "requests.jsonl"
        self.cfg = ServingConfig(
            use_mock=True,
            log_requests=True,
            log_dir=self.temp_dir.name,
        )
        self.engine = ReasoningServingEngine(config=self.cfg)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_greedy_prediction(self):
        prompt = "Janet has 3 apples. She buys 4 more. How many apples does she have?"
        pred = self.engine.predict(prompt, regime="greedy")

        self.assertIsInstance(pred, ServingPrediction)
        self.assertEqual(pred.regime, "greedy")
        self.assertTrue(pred.format_compliant)
        self.assertIsNotNone(pred.extracted_answer)
        self.assertIn("####", pred.completion)
        self.assertGreater(pred.latency_ms, 0.0)
        self.assertGreaterEqual(pred.tokens_per_sec, 0.0)

    def test_sampled_prediction(self):
        prompt = "A train travels at 60 mph for 3 hours. How many miles did it travel?"
        pred = self.engine.predict(prompt, regime="sampled", temperature=0.7)

        self.assertEqual(pred.regime, "sampled")
        self.assertTrue(pred.format_compliant)
        self.assertIsNotNone(pred.extracted_answer)

    def test_self_consistency_majority_vote(self):
        prompt = "There are 5 boxes with 6 pens each. If 4 pens are removed, how many are left?"
        pred = self.engine.predict(prompt, regime="self_consistency", num_samples=4)

        self.assertEqual(pred.regime, "self_consistency")
        self.assertTrue(pred.format_compliant)
        self.assertIsNotNone(pred.extracted_answer)
        self.assertEqual(len(pred.all_extracted), 4)
        self.assertGreaterEqual(pred.majority_vote_confidence, 0.25)

    def test_batch_prediction(self):
        prompts = [
            "What is 10 + 20?",
            "What is 15 * 3?",
            "What is 100 - 45?",
        ]
        preds = self.engine.predict_batch(prompts, regime="greedy")
        self.assertEqual(len(preds), 3)
        for p in preds:
            self.assertTrue(p.format_compliant)
            self.assertIsNotNone(p.extracted_answer)

    def test_traffic_logging(self):
        prompt = "What is 7 * 8?"
        self.engine.predict(prompt, regime="greedy")
        self.engine.predict(prompt, regime="sampled")

        self.assertTrue(self.log_file.exists())
        with open(self.log_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["regime"], "greedy")
        self.assertEqual(lines[1]["regime"], "sampled")
        self.assertTrue(lines[0]["format_compliant"])


class TestInteractiveApp(unittest.TestCase):
    """Test interactive Gradio application components and presets."""

    def test_presets_coverage(self):
        self.assertGreaterEqual(len(GSM8K_PRESETS), 3)
        for name, data in GSM8K_PRESETS.items():
            self.assertIn("prompt", data)
            self.assertIn("tier", data)
            self.assertIn("expected_answer", data)
            self.assertIn("description", data)

    def test_smoke_test_execution(self):
        # Run programmatic smoke test
        exit_code = run_smoke_test(use_mock=True)
        self.assertEqual(exit_code, 0)


class TestServingVerification(unittest.TestCase):
    """Test operational serving verification runner."""

    def test_run_serving_verification(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            report = run_serving_verification(
                output_dir=out_dir,
                use_mock=True,
            )

            self.assertIsInstance(report, dict)
            self.assertEqual(report.get("verdict"), "PASSED")
            self.assertTrue(report.get("all_passed"))

            # Check that report files exist
            json_path = out_dir / "day50_serving_verification.json"
            md_path = out_dir / "day50_serving_verification.md"
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())


class TestPhase7GateVerifier(unittest.TestCase):
    """Test Phase 7 gate verifier auditing all 7 criteria."""

    def test_verify_all_phase7_gates(self):
        summary = verify_all_phase7_gates()
        self.assertIsInstance(summary, Phase7GateSummary)
        self.assertEqual(summary.release_tag, "v1.0")
        self.assertEqual(len(summary.criteria), 7)
        self.assertTrue(summary.all_passed)
        self.assertEqual(summary.verdict, "PASSED")

    def test_generate_phase7_wrapup_report(self):
        summary = verify_all_phase7_gates()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_md = Path(tmp_dir) / "test_gate_report.md"
            report_text = generate_phase7_wrapup_report(summary, out_md)

            self.assertTrue(out_md.exists())
            self.assertIn("GATE PASSED: READY FOR RELEASE v1.0", report_text)
            self.assertIn("deepseek-math-7b-base", report_text)
            self.assertIn("v1.0", report_text)

    def test_run_phase7_wrapup(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            out_md = out_dir / "gate_report.md"
            out_json = out_dir / "summary.json"

            summary = run_phase7_wrapup(
                output_report_path=out_md,
                output_summary_path=out_json,
            )

            self.assertTrue(summary.all_passed)
            self.assertTrue(out_md.exists())
            self.assertTrue(out_json.exists())


class TestCLIRunners(unittest.TestCase):
    """Test CLI commands for serving and gate verification."""

    def test_verify_phase7_gate_cli(self):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "verify_phase7_gate.py"),
            "--quiet",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")

    def test_serve_model_smoke_test_cli(self):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "serve_model.py"),
            "--use_mock",
            "--smoke_test",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
