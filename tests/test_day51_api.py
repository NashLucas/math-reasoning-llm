"""Unit and integration tests for Phase 8 Day 51: Production REST API & OOD Generalization.

Verifies:
1. FastAPI endpoint functions and request/response data contracts.
2. Pydantic request validation rules and constraint enforcement.
3. Prediction responses and Answer Contract adherence (\\n#### <number>).
4. Batch inference concurrency, throughput tracking, and format rates.
5. Operational telemetry reporting and traffic log retrieval.
6. Out-of-Distribution (OOD) problem benchmark coverage (5 reasoning archetypes).
7. OOD evaluator execution and report file emission.
8. CLI runner (scripts/evaluate_ood.py) execution.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from app.api import (
    BatchPredictionRequest,
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
    get_metrics,
    get_recent_logs,
    health_check,
    predict,
    predict_batch,
    root,
    set_engine,
)
from src.evaluation.ood_evaluator import (
    CURATED_OOD_BENCHMARK,
    OODBenchmarkReport,
    OODEvaluator,
    run_ood_evaluation,
)
from src.serving.engine import ReasoningServingEngine, ServingConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestFastAPIEndpoints(unittest.TestCase):
    """Test FastAPI endpoint handlers and contract validation directly."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cfg = ServingConfig(
            use_mock=True,
            adapter_path="outputs/grpo/adapter_model",
            log_dir=self.temp_dir.name,
            log_requests=True,
        )
        self.engine = ReasoningServingEngine(config=self.cfg)
        set_engine(self.engine)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_root_endpoint(self):
        data = root()
        self.assertIsInstance(data, dict)
        self.assertIn("service", data)
        self.assertIn("docs", data)
        self.assertEqual(data["version"], "v1.1")

    def test_health_endpoint(self):
        resp = health_check()
        self.assertIsInstance(resp, HealthResponse)
        self.assertEqual(resp.status, "HEALTHY")
        self.assertEqual(resp.version, "v1.1")
        self.assertGreaterEqual(resp.uptime_seconds, 0.0)
        self.assertIn("Deterministic Mock", resp.backend)

    def test_predict_greedy(self):
        req = PredictionRequest(
            question="Janet has 3 apples. She buys 4 more. How many apples does she have?",
            regime="greedy",
        )
        resp = predict(req)

        self.assertIsInstance(resp, PredictionResponse)
        self.assertEqual(resp.regime, "greedy")
        self.assertTrue(resp.format_compliant)
        self.assertIsNotNone(resp.extracted_answer)
        self.assertIn("####", resp.completion)
        self.assertGreater(resp.latency_ms, 0.0)
        self.assertGreaterEqual(resp.tokens_per_sec, 0.0)

    def test_predict_sampled(self):
        req = PredictionRequest(
            question="A car travels 60 miles in 2 hours. What is its speed in mph?",
            regime="sampled",
            temperature=0.7,
        )
        resp = predict(req)
        self.assertEqual(resp.regime, "sampled")
        self.assertTrue(resp.format_compliant)
        self.assertIsNotNone(resp.extracted_answer)

    def test_predict_self_consistency(self):
        req = PredictionRequest(
            question="There are 5 bags with 4 apples each. How many apples total?",
            regime="self_consistency",
            num_samples=4,
        )
        resp = predict(req)
        self.assertEqual(resp.regime, "self_consistency")
        self.assertTrue(resp.format_compliant)
        self.assertEqual(len(resp.all_extracted), 4)
        self.assertGreaterEqual(resp.majority_vote_confidence, 0.25)

    def test_predict_invalid_regime_validation(self):
        with self.assertRaises(ValidationError):
            PredictionRequest(
                question="What is 1 + 1?",
                regime="invalid_regime_name",
            )

    def test_predict_empty_question_validation(self):
        with self.assertRaises(ValidationError):
            PredictionRequest(
                question="",
                regime="greedy",
            )

    def test_predict_batch(self):
        req = BatchPredictionRequest(
            questions=[
                "What is 10 + 20?",
                "What is 15 * 3?",
                "What is 100 - 45?",
            ],
            regime="greedy",
        )
        resp = predict_batch(req)
        self.assertEqual(resp.total_queries, 3)
        self.assertEqual(len(resp.predictions), 3)
        self.assertGreater(resp.total_elapsed_ms, 0.0)
        self.assertEqual(resp.format_compliance_rate, 1.0)
        for p in resp.predictions:
            self.assertTrue(p.format_compliant)
            self.assertIsNotNone(p.extracted_answer)

    def test_metrics_endpoint(self):
        # Fire two queries first
        predict(PredictionRequest(question="What is 2 + 2?", regime="greedy"))
        predict(PredictionRequest(question="What is 3 * 3?", regime="sampled"))

        resp = get_metrics()
        self.assertIsInstance(resp, MetricsResponse)
        self.assertGreaterEqual(resp.total_requests, 2)
        self.assertEqual(resp.format_compliance_rate, 1.0)
        self.assertIn("greedy", resp.regime_distribution)
        self.assertIn("sampled", resp.regime_distribution)

    def test_logs_endpoint(self):
        predict(PredictionRequest(question="What is 5 * 5?", regime="greedy"))
        logs = get_recent_logs(limit=5)
        self.assertIsInstance(logs, list)
        self.assertGreaterEqual(len(logs), 1)
        self.assertEqual(logs[-1]["regime"], "greedy")


class TestOODGeneralization(unittest.TestCase):
    """Test Out-of-Distribution benchmark suite and evaluator."""

    def test_curated_benchmark_archetypes(self):
        self.assertEqual(len(CURATED_OOD_BENCHMARK), 25)
        categories = {p.category for p in CURATED_OOD_BENCHMARK}
        expected = {
            "multi_rate_work",
            "linguistic_perturbation",
            "compound_finance",
            "algebraic_system",
            "geometric_arithmetic",
        }
        self.assertEqual(categories, expected)

    def test_ood_evaluator_execution(self):
        evaluator = OODEvaluator(use_mock=True)
        report = evaluator.evaluate_suite()

        self.assertIsInstance(report, OODBenchmarkReport)
        self.assertEqual(report.total_problems, 25)
        self.assertEqual(report.overall_greedy_acc, 1.0)
        self.assertEqual(report.overall_format_compliance, 1.0)
        self.assertEqual(len(report.category_metrics), 5)

        for cat, metrics in report.category_metrics.items():
            self.assertEqual(metrics.total_problems, 5)
            self.assertEqual(metrics.accuracy, 1.0)
            self.assertEqual(metrics.format_compliance, 1.0)

    def test_run_ood_evaluation_file_emission(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            out_md = out_dir / "test_ood_report.md"
            out_json = out_dir / "test_ood_summary.json"

            report = run_ood_evaluation(
                output_dir=out_dir,
                output_report_path=out_md,
                output_json_path=out_json,
                use_mock=True,
            )

            self.assertIsInstance(report, OODBenchmarkReport)
            self.assertTrue(out_md.exists())
            self.assertTrue(out_json.exists())

            report_text = out_md.read_text(encoding="utf-8")
            self.assertIn("Out-of-Distribution Mathematical Generalization Report", report_text)
            self.assertIn("Multi Rate Work", report_text)


class TestDay51CLIRunners(unittest.TestCase):
    """Test CLI commands for Day 51."""

    def test_evaluate_ood_cli(self):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_ood.py"),
            "--quiet",
            "--use_mock",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
