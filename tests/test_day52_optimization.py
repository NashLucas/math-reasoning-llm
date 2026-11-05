"""Unit and integration tests for Phase 8 Day 52: Quantization, Monitoring & Load Stress.

Verifies:
1. QuantizationProfile dataclass and default profile configurations.
2. Quantization audit execution, compression ratios, and report emission.
3. TrafficLogRecord and TrafficMonitor metrics computation and percentile tracking.
4. Drift alert triggers (format degradation, latency spike).
5. Load stress benchmarking concurrency scaling and report generation.
6. CLI runner entrypoints for Day 52 scripts.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.benchmark_serving_load import (
    LoadBenchmarkReport,
    run_load_stress_benchmark,
)
from src.serving.monitor import (
    TrafficLogRecord,
    TrafficMonitor,
    run_traffic_monitoring,
)
from src.serving.quantization import (
    QUANTIZATION_PROFILES,
    QuantizationReport,
    run_quantization_audit,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestQuantizationProfiles(unittest.TestCase):
    """Test quantization profile definitions and mathematical properties."""

    def test_default_profiles_present(self):
        self.assertIn("bf16_baseline", QUANTIZATION_PROFILES)
        self.assertIn("bnb_4bit_nf4", QUANTIZATION_PROFILES)
        self.assertIn("bnb_8bit", QUANTIZATION_PROFILES)
        self.assertIn("awq_4bit", QUANTIZATION_PROFILES)
        self.assertIn("gguf_q4_k_m", QUANTIZATION_PROFILES)

    def test_profile_compression_metrics(self):
        baseline = QUANTIZATION_PROFILES["bf16_baseline"]
        awq = QUANTIZATION_PROFILES["awq_4bit"]
        gguf = QUANTIZATION_PROFILES["gguf_q4_k_m"]

        self.assertEqual(baseline.bits, 16)
        self.assertAlmostEqual(baseline.compression_ratio, 1.0, places=2)

        self.assertGreater(awq.compression_ratio, 3.0)
        self.assertLess(awq.estimated_size_gb, baseline.estimated_size_gb)
        self.assertGreater(gguf.compression_ratio, 3.0)
        self.assertLess(gguf.min_vram_gb, baseline.min_vram_gb)

    def test_quantization_audit_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            md_path = tmp_path / "quant_report.md"
            json_path = tmp_path / "quant_report.json"

            report = run_quantization_audit(
                output_report_path=md_path,
                output_json_path=json_path,
                use_mock=True,
            )

            self.assertIsInstance(report, QuantizationReport)
            self.assertEqual(len(report.profiles_evaluated), 5)
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            # Verify JSON payload
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("profiles_evaluated", data)
            profile_ids = [p["profile_id"] for p in data["profiles_evaluated"]]
            self.assertIn("bnb_4bit_nf4", profile_ids)
            for p in data["profiles_evaluated"]:
                self.assertEqual(p["format_compliance"], 1.0)
                self.assertEqual(p["status"], "CERTIFIED")


class TestTrafficMonitorAndDrift(unittest.TestCase):
    """Test traffic monitor log parsing, metric calculations, and drift alert rules."""

    def test_traffic_log_record_from_dict(self):
        data = {
            "problem_id": "gsm8k_001",
            "question": "What is 2+2?",
            "completion": "2+2=4\n#### 4",
            "extracted_answer": "4",
            "format_compliant": True,
            "latency_ms": 12.5,
            "tokens_per_sec": 100.0,
            "regime": "greedy",
            "prompt_tokens": 10,
            "completion_tokens": 25,
            "timestamp": "2026-11-05T00:00:00Z",
        }
        rec = TrafficLogRecord.from_dict(data)
        self.assertEqual(rec.problem_id, "gsm8k_001")
        self.assertTrue(rec.format_compliant)
        self.assertEqual(rec.latency_ms, 12.5)

    def test_empty_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = Path(tmpdir) / "empty.jsonl"
            empty_file.touch()

            monitor = TrafficMonitor(empty_file)
            metrics = monitor.compute_metrics()

            self.assertEqual(metrics.total_records, 0)
            self.assertEqual(metrics.format_compliance_rate, 1.0)
            self.assertFalse(metrics.contract_alert)
            self.assertFalse(metrics.latency_alert)

    def test_normal_traffic_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "normal_traffic.jsonl"
            records = [
                {
                    "problem_id": f"prob_{i:03d}",
                    "question": f"Solve problem {i}",
                    "completion": f"Step by step solution\n#### {i}",
                    "extracted_answer": str(i),
                    "format_compliant": True,
                    "latency_ms": 10.0 + i,
                    "tokens_per_sec": 80.0,
                    "regime": "greedy",
                    "prompt_tokens": 15,
                    "completion_tokens": 40,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                for i in range(20)
            ]
            with open(log_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            monitor = TrafficMonitor(log_file)
            metrics = monitor.compute_metrics()

            self.assertEqual(metrics.total_records, 20)
            self.assertEqual(metrics.format_compliance_rate, 1.0)
            self.assertGreater(metrics.p50_latency_ms, 0.0)
            self.assertFalse(metrics.contract_alert)
            self.assertFalse(metrics.latency_alert)

    def test_format_drift_alert_triggered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "format_drift.jsonl"
            # 10 records, 2 invalid formats -> 80% format compliance (< 98.0%)
            records = []
            for i in range(10):
                valid = i < 8
                records.append({
                    "problem_id": f"prob_{i:03d}",
                    "question": f"Solve problem {i}",
                    "completion": f"Step by step solution\n#### {i}" if valid else "No format here 42",
                    "extracted_answer": str(i) if valid else "42",
                    "format_compliant": valid,
                    "latency_ms": 15.0,
                    "tokens_per_sec": 80.0,
                    "regime": "greedy",
                    "prompt_tokens": 15,
                    "completion_tokens": 40,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            with open(log_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            monitor = TrafficMonitor(log_file)
            metrics = monitor.compute_metrics(min_contract_threshold=0.98)

            self.assertLess(metrics.format_compliance_rate, 0.98)
            self.assertTrue(metrics.contract_alert)
            self.assertTrue(any("Answer Contract Drift Alert" in a for a in metrics.alerts))

    def test_latency_drift_alert_triggered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "latency_drift.jsonl"
            # Latencies with p95 > 500ms
            records = [
                {
                    "problem_id": f"prob_{i:03d}",
                    "question": f"Solve problem {i}",
                    "completion": f"Step by step solution\n#### {i}",
                    "extracted_answer": str(i),
                    "format_compliant": True,
                    "latency_ms": 600.0 if i >= 8 else 50.0,
                    "tokens_per_sec": 80.0,
                    "regime": "greedy",
                    "prompt_tokens": 15,
                    "completion_tokens": 40,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                for i in range(10)
            ]
            with open(log_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            monitor = TrafficMonitor(log_file)
            metrics = monitor.compute_metrics(max_p95_latency_threshold_ms=500.0)

            self.assertGreater(metrics.p95_latency_ms, 500.0)
            self.assertTrue(metrics.latency_alert)
            self.assertTrue(any("Latency Degradation Alert" in a for a in metrics.alerts))

    def test_run_traffic_monitoring_report_emission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_file = tmp_path / "traffic.jsonl"
            md_path = tmp_path / "report.md"
            json_path = tmp_path / "report.json"

            records = [
                {
                    "problem_id": f"prob_{i:03d}",
                    "question": f"Solve problem {i}",
                    "completion": f"Step by step solution\n#### {i}",
                    "extracted_answer": str(i),
                    "format_compliant": True,
                    "latency_ms": 12.0,
                    "tokens_per_sec": 85.0,
                    "regime": "greedy",
                    "prompt_tokens": 15,
                    "completion_tokens": 40,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                for i in range(5)
            ]
            with open(log_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            metrics = run_traffic_monitoring(
                log_file=log_file,
                output_report_path=md_path,
                output_json_path=json_path,
            )

            self.assertEqual(metrics.total_records, 5)
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())


class TestLoadStressBenchmark(unittest.TestCase):
    """Test high-concurrency load stress benchmarking execution and calculations."""

    def test_run_load_stress_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            md_path = tmp_path / "load_report.md"
            json_path = tmp_path / "load_report.json"

            report = run_load_stress_benchmark(
                concurrency_tiers=[1, 2],
                queries_per_tier=4,
                use_mock=True,
                output_report_path=md_path,
                output_json_path=json_path,
            )

            self.assertIsInstance(report, LoadBenchmarkReport)
            self.assertEqual(report.total_queries_tested, 8)
            self.assertEqual(len(report.concurrency_results), 2)
            self.assertEqual(report.overall_format_compliance_rate, 1.0)
            self.assertEqual(report.status, "PASSED")

            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            # Verify contents of generated JSON
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["total_queries_tested"], 8)
            self.assertEqual(len(data["concurrency_results"]), 2)


class TestDay52CLIRunners(unittest.TestCase):
    """Smoke test CLI script entry points."""

    def test_export_quantized_model_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            md_path = tmp_path / "cli_quant.md"
            json_path = tmp_path / "cli_quant.json"

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "export_quantized_model.py"),
                "--output_report",
                str(md_path),
                "--output_json",
                str(json_path),
                "--use_mock",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

    def test_run_traffic_monitor_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_file = tmp_path / "traffic.jsonl"
            md_path = tmp_path / "cli_traffic.md"
            json_path = tmp_path / "cli_traffic.json"

            # Write a single test record
            rec = {
                "problem_id": "prob_cli",
                "question": "CLI test",
                "completion": "Solution\n#### 99",
                "extracted_answer": "99",
                "format_compliant": True,
                "latency_ms": 5.0,
                "tokens_per_sec": 80.0,
                "regime": "greedy",
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_traffic_monitor.py"),
                "--log_file",
                str(log_file),
                "--output_report",
                str(md_path),
                "--output_json",
                str(json_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

    def test_benchmark_serving_load_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            md_path = tmp_path / "cli_load.md"
            json_path = tmp_path / "cli_load.json"

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "benchmark_serving_load.py"),
                "--queries_per_tier",
                "2",
                "--output_report",
                str(md_path),
                "--output_json",
                str(json_path),
                "--use_mock",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())


if __name__ == "__main__":
    unittest.main()
