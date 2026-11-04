"""Unit tests for Phase 5 Day 33: Error Question Semantic Clustering & Topic Reporting."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_day33_clustering import run_day33_clustering
from src.evaluation.error_analysis import (
    ClusterAnalysisReport,
    ErrorClusterer,
    QuestionEmbedder,
    assign_cluster_topic,
    extract_cluster_keywords,
    generate_cluster_markdown,
    generate_error_markdown_report,
)
from src.evaluation.evaluator import TestSetWallError


class TestQuestionEmbedder(unittest.TestCase):
    """Test QuestionEmbedder feature extraction and fallbacks."""

    def test_embedder_fallback_shapes_and_normalization(self):
        embedder = QuestionEmbedder(use_fallback=True)
        texts = [
            "Janet has 15 apples and buys 12 more.",
            "A car travels 180 miles in 3 hours at 60 mph.",
            "A coat costs $80 and is 25% off on sale.",
        ]
        embs, backend = embedder.embed(texts)
        self.assertEqual(backend, "tfidf_fallback")
        self.assertEqual(embs.shape[0], 3)
        self.assertGreater(embs.shape[1], 0)

        # Check unit normalization (L2 norm approx 1.0)
        norms = np.linalg.norm(embs, axis=1)
        for norm in norms:
            self.assertAlmostEqual(float(norm), 1.0, places=4)

    def test_embedder_empty_inputs(self):
        embedder = QuestionEmbedder(use_fallback=True)
        embs, _ = embedder.embed([])
        self.assertEqual(embs.shape[0], 0)


class TestTopicLabelingAndKeywords(unittest.TestCase):
    """Test keyword extraction and domain topic assignment."""

    def test_extract_cluster_keywords(self):
        cluster_texts = [
            "A car travels 180 miles in 3 hours.",
            "What is the average speed in miles per hour?",
            "The driver drove 60 miles each hour.",
        ]
        all_texts = cluster_texts + [
            "Janet has 15 apples and buys 12 more.",
            "A baker bakes 7 trays of cookies.",
        ]
        keywords = extract_cluster_keywords(cluster_texts, all_texts, top_n=3)
        self.assertTrue(len(keywords) > 0)
        # Should contain speed/miles/hour
        blob = " ".join(keywords).lower()
        self.assertTrue(any(term in blob for term in ["miles", "hour", "speed", "car"]))

    def test_assign_cluster_topic_rates(self):
        texts = [
            "A car travels 180 miles in 3 hours. What is its speed in mph?",
            "If 3 workers paint a fence in 6 hours, what is their rate?",
        ]
        keywords = ["speed", "miles", "hours", "rate"]
        topic = assign_cluster_topic(texts, keywords)
        self.assertIn("Rates, Speed", topic)

    def test_assign_cluster_topic_money(self):
        texts = [
            "Tom had $100 and spent $35 on a shirt and $25 on shoes.",
            "Sam bought 4 notebooks for $5 each. How much money did he spend?",
        ]
        keywords = ["spent", "dollars", "money", "bought"]
        topic = assign_cluster_topic(texts, keywords)
        self.assertIn("Money", topic)

    def test_assign_cluster_topic_percentages(self):
        texts = [
            "A coat normally costs $80 and is on sale for 25% off.",
            "A recipe calls for a ratio of 2 cups to 5 cups.",
        ]
        keywords = ["percent", "discount", "sale", "ratio"]
        topic = assign_cluster_topic(texts, keywords)
        self.assertIn("Percentages", topic)


class TestErrorClusterer(unittest.TestCase):
    """Test ErrorClusterer grouping, representative example selection, and serialization."""

    def setUp(self):
        self.sample_errors = [
            {
                "problem_id": "err_01",
                "question": "A car travels 180 miles in 3 hours. What is its average speed in miles per hour?",
                "gold_answer": "60",
                "generated_text": "180 * 3 = 540 mph.\n#### 540",
                "extracted_answer": "540",
                "category": "wrong_setup",
                "is_error": True,
            },
            {
                "problem_id": "err_02",
                "question": "If a train goes 60 miles each hour for 4 hours, how far does it go?",
                "gold_answer": "240",
                "generated_text": "60 * 4 =\n#### 240",
                "extracted_answer": "240",
                "category": "truncation",
                "is_error": True,
            },
            {
                "problem_id": "err_03",
                "question": "Tom had $100 and spent $35 on a shirt. How much money is left?",
                "gold_answer": "65",
                "generated_text": "100 - 35 = 75.\n#### 75",
                "extracted_answer": "75",
                "category": "arithmetic_slip",
                "is_error": True,
            },
            {
                "problem_id": "err_04",
                "question": "A coat costs $80 and is on sale for 25% off. What is the discount?",
                "gold_answer": "20",
                "generated_text": "80 + 25 = 105.\n#### 105",
                "extracted_answer": "105",
                "category": "wrong_setup",
                "is_error": True,
            },
            {
                "problem_id": "corr_01",
                "question": "What is 2 + 2?",
                "gold_answer": "4",
                "generated_text": "2 + 2 = 4.\n#### 4",
                "extracted_answer": "4",
                "is_correct": True,
                "is_error": False,
            },
        ]

    def test_cluster_filters_errors_and_produces_clusters(self):
        clusterer = ErrorClusterer(embedder=QuestionEmbedder(use_fallback=True), num_clusters=2)
        report: ClusterAnalysisReport = clusterer.cluster(self.sample_errors, num_clusters=2)

        # Correct solution should be filtered out -> exactly 4 error questions
        self.assertEqual(report.total_error_questions, 4)
        self.assertEqual(report.num_clusters, 2)
        self.assertEqual(len(report.clusters), 2)

        total_clustered = sum(c.size for c in report.clusters)
        self.assertEqual(total_clustered, 4)

        for c in report.clusters:
            self.assertGreater(c.size, 0)
            self.assertGreater(len(c.top_keywords), 0)
            self.assertLessEqual(len(c.representative_examples), 2)
            self.assertGreater(len(c.representative_examples), 0)

            # Check representative example structure
            ex = c.representative_examples[0]
            self.assertIn("problem_id", ex)
            self.assertIn("question", ex)
            self.assertIn("gold_answer", ex)
            self.assertIn("category", ex)
            self.assertIn("distance_to_centroid", ex)

    def test_cluster_markdown_generation(self):
        clusterer = ErrorClusterer(embedder=QuestionEmbedder(use_fallback=True), num_clusters=2)
        report = clusterer.cluster(self.sample_errors, num_clusters=2)
        md = generate_cluster_markdown(report)

        self.assertIn("## 4. Semantic Topic Clusters of Reasoning Errors", md)
        self.assertIn("| Cluster | Topic / Mathematical Domain |", md)
        self.assertIn("Representative Examples", md)
        self.assertIn("Example", md)

    def test_generate_error_markdown_report_includes_clusters(self):
        clusterer = ErrorClusterer(embedder=QuestionEmbedder(use_fallback=True), num_clusters=2)
        report = clusterer.cluster(self.sample_errors, num_clusters=2)

        summary = {
            "total_evaluated": 10,
            "correct_count": 6,
            "error_count": 4,
            "accuracy": 60.0,
            "error_rate": 40.0,
            "category_counts": {"wrong_setup": 2, "arithmetic_slip": 1, "truncation": 1},
            "category_percentages": {"wrong_setup": 50.0, "arithmetic_slip": 25.0, "truncation": 25.0},
            "rule_based_count": 1,
            "rule_based_percentage": 25.0,
            "semantic_count": 3,
            "semantic_percentage": 75.0,
        }
        full_md = generate_error_markdown_report(
            summary=summary,
            classifications=[],
            stage_name="SFT Test Run",
            cluster_report=report,
        )
        self.assertIn("# Error Analysis Report", full_md)
        self.assertIn("## 2. Error Category Breakdown", full_md)
        self.assertIn("## 4. Semantic Topic Clusters of Reasoning Errors", full_md)


class TestDay33PipelineExecution(unittest.TestCase):
    """Test full Day 33 clustering CLI execution and test-set wall safety."""

    def test_test_set_wall_enforcement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            test_file = tmp_path / "test.parquet"
            test_file.touch()

            with self.assertRaises(TestSetWallError):
                run_day33_clustering(
                    input_file=test_file,
                    output_clusters_file=tmp_path / "clusters.json",
                    report_file=tmp_path / "report.md",
                    use_mock=True,
                )

    def test_run_day33_pipeline_cli_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_json = tmp_path / "error_clusters.json"
            out_md = tmp_path / "error_analysis_sft.md"

            cmd = [
                sys.executable,
                "scripts/run_day33_clustering.py",
                "--sample_mode",
                "--input_file",
                "data/sample/labeled_30_errors.jsonl",
                "--output_clusters_file",
                str(out_json),
                "--report_file",
                str(out_md),
                "--num_clusters",
                "4",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI error:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

            self.assertTrue(out_json.exists())
            self.assertTrue(out_md.exists())

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["num_clusters"], 4)
            self.assertEqual(payload["total_errors"], 30)

            md_text = out_md.read_text(encoding="utf-8")
            self.assertIn("## 4. Semantic Topic Clusters of Reasoning Errors", md_text)
            self.assertIn("Representative Examples", md_text)


if __name__ == "__main__":
    unittest.main()
