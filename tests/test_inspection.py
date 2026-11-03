"""Unit tests for reasoning trace inspection, qualitative audit, and diagnostics (Phase 3 Day 24)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.inspect_reasoning_traces import (
    compute_length_statistics,
    select_representative_kept_traces,
    select_representative_rejected_traces,
)


class TestLengthStatistics(unittest.TestCase):
    """Test suite for length statistics and percentiles calculation."""

    def test_empty_lengths(self) -> None:
        """Empty length list should return zeroed stats dictionary."""
        stats = compute_length_statistics([])
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["mean"], 0.0)
        self.assertEqual(stats["median"], 0)
        self.assertEqual(stats["p95"], 0)

    def test_single_length(self) -> None:
        """Single item list should return that item across all percentiles with 0 std."""
        stats = compute_length_statistics([42])
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["mean"], 42.0)
        self.assertEqual(stats["min"], 42)
        self.assertEqual(stats["max"], 42)
        self.assertEqual(stats["median"], 42)
        self.assertEqual(stats["p25"], 42)
        self.assertEqual(stats["p75"], 42)
        self.assertEqual(stats["p95"], 42)
        self.assertEqual(stats["std"], 0.0)

    def test_multiple_lengths_distribution(self) -> None:
        """Test with known sequence 1 to 100."""
        lengths = list(range(1, 101))
        stats = compute_length_statistics(lengths)
        self.assertEqual(stats["count"], 100)
        self.assertEqual(stats["min"], 1)
        self.assertEqual(stats["max"], 100)
        self.assertAlmostEqual(stats["mean"], 50.5, places=1)
        self.assertIn(stats["median"], [50, 51])
        self.assertGreaterEqual(stats["p95"], 95)
        self.assertLessEqual(stats["p25"], 26)


class TestRepresentativeTracesSelection(unittest.TestCase):
    """Test suite for sampling representative kept and rejected traces."""

    def test_select_representative_kept_traces_under_target(self) -> None:
        """If fewer traces than target_count, returns all traces."""
        traces = [
            {"trace_id": "t1", "prompt_type": "chain_of_thought", "score": {"overall_score": 0.9}},
            {"trace_id": "t2", "prompt_type": "alternative_method", "score": {"overall_score": 0.85}},
        ]
        sampled = select_representative_kept_traces(traces, target_count=5)
        self.assertEqual(len(sampled), 2)
        self.assertEqual({t["trace_id"] for t in sampled}, {"t1", "t2"})

    def test_select_representative_kept_traces_balanced(self) -> None:
        """Selects proportionally across prompt strategies."""
        traces = []
        for i in range(20):
            traces.append({
                "trace_id": f"cot_{i}",
                "prompt_type": "chain_of_thought",
                "score": {"overall_score": 0.80 + (i * 0.005)},
            })
            traces.append({
                "trace_id": f"alt_{i}",
                "prompt_type": "alternative_method",
                "score": {"overall_score": 0.82 + (i * 0.005)},
            })

        sampled = select_representative_kept_traces(traces, target_count=10)
        self.assertEqual(len(sampled), 10)
        types = {t["prompt_type"] for t in sampled}
        self.assertIn("chain_of_thought", types)
        self.assertIn("alternative_method", types)

    def test_select_representative_rejected_traces_categories(self) -> None:
        """Rejected traces selection categorizes candidates and ensures target_count."""
        scored_pool = [
            {
                "trace_id": "unsel_1",
                "problem_id": "p1",
                "gold_answer": "42",
                "completion": "To solve this problem, we first perform step 1 which gives 20. Next we add 22 to get 42.\n#### 42",
                "rank_within_problem": 3,
            },
            {
                "trace_id": "bad_answer_1",
                "problem_id": "p2",
                "gold_answer": "10",
                "completion": "Calculation error in intermediate steps leads to the incorrect result 99.\n#### 99",
                "rank_within_problem": 1,
            },
        ]
        curated_ids = {"kept_1"}
        rejected = select_representative_rejected_traces(
            all_scored_traces=scored_pool,
            curated_trace_ids=curated_ids,
            target_count=10,
        )
        self.assertEqual(len(rejected), 10)

        # Check that unselected and filter_rejected categories exist
        cats = [t.get("rejection_category", "") for t in rejected]
        self.assertTrue(any("unselected" in c for c in cats))
        self.assertTrue(any("incorrect_final_answer" in c or "filter_rejected" in c for c in cats))


class TestNotebookAndLauncherIntegrity(unittest.TestCase):
    """Test suite validating notebooks/02_inspect_reasoning_traces.ipynb and 00_colab_launcher.ipynb."""

    def test_inspection_notebook_structure(self) -> None:
        """Verify notebooks/02_inspect_reasoning_traces.ipynb exists, is valid JSON, and has all 6 sections."""
        nb_path = Path("notebooks/02_inspect_reasoning_traces.ipynb")
        self.assertTrue(nb_path.exists(), "Inspection notebook must exist")

        content = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertIn("cells", content)
        self.assertGreater(len(content["cells"]), 10)

        all_sources = "\n".join("".join(cell.get("source", [])) for cell in content["cells"])
        self.assertIn("## 3. Problem-Level Yield & Difficulty Distribution", all_sources)
        self.assertIn("## 4. Response Length Distribution Analysis", all_sources)
        self.assertIn("## 5. Judge Score Spread & Clustering Diagnostics", all_sources)
        self.assertIn("## 6. Qualitative Audit: 50 Kept Traces & 50 Rejected Traces", all_sources)
        self.assertIn("## 7. Systematic Findings & Action Items for Day 25 Freeze", all_sources)

    def test_colab_launcher_contains_step21(self) -> None:
        """Verify notebooks/00_colab_launcher.ipynb contains Step 21 for trace inspection."""
        nb_path = Path("notebooks/00_colab_launcher.ipynb")
        self.assertTrue(nb_path.exists())

        content = json.loads(nb_path.read_text(encoding="utf-8"))
        all_sources = "\n".join("".join(cell.get("source", [])) for cell in content["cells"])
        self.assertIn("Step 21: Phase 3 Day 24 — Reasoning Trace Inspection", all_sources)
        self.assertIn("scripts/inspect_reasoning_traces.py", all_sources)


class TestCLIInspectionExecution(unittest.TestCase):
    """Test suite for CLI execution of scripts/inspect_reasoning_traces.py."""

    def test_cli_execution_generates_reports(self) -> None:
        """Run inspect_reasoning_traces.py on mock curated data and verify output reports."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create mock curated JSONL
            curated_path = tmp_path / "mock_curated.jsonl"
            curated_data = [
                {
                    "trace_id": "curated_01",
                    "problem_id": "prob_01",
                    "question": "What is 5 + 7?",
                    "gold_answer": "12",
                    "completion": "5 + 7 = 12.\n#### 12",
                    "prompt_type": "chain_of_thought",
                    "score": {"overall_score": 0.92, "completeness": 0.95},
                },
                {
                    "trace_id": "curated_02",
                    "problem_id": "prob_01",
                    "question": "What is 5 + 7?",
                    "gold_answer": "12",
                    "completion": "Alternative method: 5 + 5 + 2 = 12.\n#### 12",
                    "prompt_type": "alternative_method",
                    "score": {"overall_score": 0.88, "completeness": 0.90},
                },
            ]
            with open(curated_path, "w", encoding="utf-8") as f:
                for item in curated_data:
                    f.write(json.dumps(item) + "\n")

            # Create mock pass rates JSON
            pass_rates_path = tmp_path / "mock_pass_rates.json"
            pass_rates_data = {
                "prob_01": {
                    "problem_id": "prob_01",
                    "difficulty_bin": "easy",
                    "pass_rate": 1.0,
                }
            }
            pass_rates_path.write_text(json.dumps(pass_rates_data), encoding="utf-8")

            # Output reports dir
            reports_dir = tmp_path / "reports"

            cmd = [
                sys.executable,
                "scripts/inspect_reasoning_traces.py",
                "--curated_file",
                str(curated_path),
                "--scored_file",
                str(curated_path),
                "--pass_rates_file",
                str(pass_rates_path),
                "--reports_dir",
                str(reports_dir),
                "--report_name",
                "test_inspection_report",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                result.returncode,
                0,
                f"Script failed with output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )

            # Check generated outputs
            md_file = reports_dir / "test_inspection_report.md"
            json_file = reports_dir / "test_inspection_report.json"
            samples_file = reports_dir / "day24_inspection_samples.json"

            self.assertTrue(md_file.exists(), "Markdown report must exist")
            self.assertTrue(json_file.exists(), "JSON report must exist")
            self.assertTrue(samples_file.exists(), "Samples JSON must exist")

            json_content = json.loads(json_file.read_text(encoding="utf-8"))
            self.assertIn("yield_summary", json_content)
            self.assertIn("length_statistics", json_content)
            self.assertIn("score_statistics", json_content)
            self.assertEqual(json_content["kept_traces_count"], 2)
            self.assertEqual(json_content["rejected_traces_count"], 50)


if __name__ == "__main__":
    unittest.main()
