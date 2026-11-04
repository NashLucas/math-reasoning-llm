"""Unit tests for GRPO dataset preparation and difficulty filtering (Phase 6 Day 41)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.grpo_dataset import (
    DifficultyTier,
    TestSetWallError,
    compute_sha256,
    count_reasoning_steps,
    format_grpo_prompt,
    prepare_grpo_dataset,
    profile_problem_difficulty,
)


class TestGRPODataset(unittest.TestCase):
    """Test suite for GRPO dataset filtering and preparation."""

    def test_count_reasoning_steps(self):
        """Test step counter accurately separates derivation lines from answer markers."""
        self.assertEqual(count_reasoning_steps(""), 0)
        self.assertEqual(count_reasoning_steps("   \n\n  "), 0)

        sol_2 = "Step 1: 5 + 5 = 10.\nStep 2: 10 * 2 = 20.\n#### 20"
        self.assertEqual(count_reasoning_steps(sol_2), 2)

        sol_4 = (
            "Line 1\n"
            "Line 2\n"
            "Line 3\n"
            "Line 4\n"
            "#### 100"
        )
        self.assertEqual(count_reasoning_steps(sol_4), 4)

        sol_no_marker = "Line 1\nLine 2\nLine 3"
        self.assertEqual(count_reasoning_steps(sol_no_marker), 3)

    def test_profile_empirical_pass_rate(self):
        """Test empirical pass-rate boundaries (0.20 <= p <= 0.80)."""
        # Intermediate difficulty (0.50) -> Medium, accepted
        p_med = profile_problem_difficulty(
            problem_id="p1",
            question="What is 2+2?",
            gold_answer="4",
            cleaned_solution="2+2=4\n#### 4",
            pass_rate=0.50,
        )
        self.assertTrue(p_med.is_grpo_candidate)
        self.assertEqual(p_med.difficulty_tier, DifficultyTier.MEDIUM.value)
        self.assertIn("optimal_pass_rate", p_med.selection_reason)

        # Trivial problem (1.00) -> Easy, rejected to avoid zero-variance
        p_easy = profile_problem_difficulty(
            problem_id="p2",
            question="What is 1+1?",
            gold_answer="2",
            cleaned_solution="1+1=2\n#### 2",
            pass_rate=1.00,
        )
        self.assertFalse(p_easy.is_grpo_candidate)
        self.assertEqual(p_easy.difficulty_tier, DifficultyTier.EASY.value)
        self.assertIn("pass_rate_too_high", p_easy.selection_reason)

        # Impossible problem (0.00) -> Hard, rejected to avoid zero-variance
        p_hard = profile_problem_difficulty(
            problem_id="p3",
            question="Impossible question?",
            gold_answer="0",
            cleaned_solution="Derivation\n#### 0",
            pass_rate=0.00,
        )
        self.assertFalse(p_hard.is_grpo_candidate)
        self.assertEqual(p_hard.difficulty_tier, DifficultyTier.HARD.value)
        self.assertIn("pass_rate_too_low", p_hard.selection_reason)

    def test_profile_step_complexity_fallback(self):
        """Test step-count complexity boundaries when empirical pass rate is absent."""
        # 2 steps -> Easy, rejected
        p_2 = profile_problem_difficulty(
            problem_id="p1",
            question="Short question?",
            gold_answer="10",
            cleaned_solution="Step 1: 5+5=10\nStep 2: done\n#### 10",
            pass_rate=None,
            min_steps=3,
            max_steps=6,
        )
        self.assertFalse(p_2.is_grpo_candidate)
        self.assertEqual(p_2.difficulty_tier, DifficultyTier.EASY.value)

        # 4 steps -> Medium, accepted
        p_4 = profile_problem_difficulty(
            problem_id="p2",
            question="Medium question?",
            gold_answer="20",
            cleaned_solution="Step 1\nStep 2\nStep 3\nStep 4\n#### 20",
            pass_rate=None,
            min_steps=3,
            max_steps=6,
        )
        self.assertTrue(p_4.is_grpo_candidate)
        self.assertEqual(p_4.difficulty_tier, DifficultyTier.MEDIUM.value)

        # 8 steps -> Hard, rejected
        p_8 = profile_problem_difficulty(
            problem_id="p3",
            question="Long question?",
            gold_answer="80",
            cleaned_solution="\n".join([f"Step {i}" for i in range(1, 9)]) + "\n#### 80",
            pass_rate=None,
            min_steps=3,
            max_steps=6,
        )
        self.assertFalse(p_8.is_grpo_candidate)
        self.assertEqual(p_8.difficulty_tier, DifficultyTier.HARD.value)

    def test_max_prompt_chars_ceiling(self):
        """Test overly long prompt rejection."""
        long_q = "A" * 1500
        p_long = profile_problem_difficulty(
            problem_id="p_long",
            question=long_q,
            gold_answer="1",
            cleaned_solution="Step 1\nStep 2\nStep 3\n#### 1",
            max_prompt_chars=1000,
        )
        self.assertFalse(p_long.is_grpo_candidate)
        self.assertEqual(p_long.difficulty_tier, DifficultyTier.HARD.value)
        self.assertIn("question_exceeds_max_chars", p_long.selection_reason)

    def test_filtering_modes(self):
        """Test pass_rate_only, steps_only, and hybrid filtering modes."""
        # pass_rate_only with None -> unobserved, rejected
        p_pro = profile_problem_difficulty(
            problem_id="p",
            question="Q",
            gold_answer="1",
            cleaned_solution="Step 1\nStep 2\nStep 3\n#### 1",
            pass_rate=None,
            filter_mode="pass_rate_only",
        )
        self.assertFalse(p_pro.is_grpo_candidate)
        self.assertEqual(p_pro.difficulty_tier, DifficultyTier.UNOBSERVED.value)

        # steps_only ignores pass_rate
        p_so = profile_problem_difficulty(
            problem_id="p",
            question="Q",
            gold_answer="1",
            cleaned_solution="Step 1\nStep 2\nStep 3\n#### 1",
            pass_rate=1.0,  # would fail pass_rate check
            filter_mode="steps_only",
        )
        self.assertTrue(p_so.is_grpo_candidate)
        self.assertEqual(p_so.difficulty_tier, DifficultyTier.MEDIUM.value)

    def test_format_grpo_prompt(self):
        """Test canonical zero-shot prompt formatting."""
        formatted = format_grpo_prompt("If Mary has 5 apples, how many does she have?")
        self.assertIn("Question: If Mary has 5 apples, how many does she have?", formatted)
        self.assertIn("#### <number>", formatted)
        self.assertIn("Answer: Let's think step by step.", formatted)

    def test_test_set_wall_leak_detection(self):
        """Test that leaking a test problem ID into training raises TestSetWallError."""
        train_df = pd.DataFrame([
            {"problem_id": "gsm8k_train_1", "question": "Q1", "answer": "Sol 1\n#### 1"},
            {"problem_id": "gsm8k_test_leak", "question": "Q_leak", "answer": "Sol 2\n#### 2"},
        ])
        val_df = pd.DataFrame([
            {"problem_id": "gsm8k_val_1", "question": "Q_val", "answer": "Sol V\n#### 3"},
        ])
        test_df = pd.DataFrame([
            {"problem_id": "gsm8k_test_leak", "question": "Q_leak", "answer": "Sol T\n#### 2"},
        ])

        with self.assertRaises(TestSetWallError):
            prepare_grpo_dataset(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
            )

    def test_dataset_preparation_pipeline(self):
        """Test full preparation pipeline, outputs, schemas, and manifest."""
        train_rows = []
        for i in range(20):
            # 10 medium (3 steps), 5 easy (2 steps), 5 hard (7 steps)
            if i < 10:
                sol = "S1\nS2\nS3\n#### 42"
            elif i < 15:
                sol = "S1\nS2\n#### 10"
            else:
                sol = "\n".join([f"S{j}" for j in range(1, 8)]) + "\n#### 99"
            train_rows.append({
                "problem_id": f"gsm8k_train_{i:03d}",
                "question": f"Question {i} about math?",
                "cleaned_solution": sol,
                "gold_answer": "42",
            })
        train_df = pd.DataFrame(train_rows)

        val_rows = [
            {
                "problem_id": f"gsm8k_val_{i:03d}",
                "question": f"Val Question {i}?",
                "cleaned_solution": "V1\nV2\nV3\n#### 100",
                "gold_answer": "100",
            }
            for i in range(5)
        ]
        val_df = pd.DataFrame(val_rows)

        test_rows = [
            {
                "problem_id": f"gsm8k_test_{i:03d}",
                "question": f"Test Question {i}?",
                "cleaned_solution": "T1\nT2\n#### 50",
                "gold_answer": "50",
            }
            for i in range(5)
        ]
        test_df = pd.DataFrame(test_rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            result = prepare_grpo_dataset(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                output_dir=out_path,
                config={
                    "filtering": {"min_steps": 3, "max_steps": 6, "mode": "hybrid"},
                    "sampling": {"target_train_size": 5, "val_size": 3, "seed": 42},
                },
                seed=42,
            )

            manifest = result["manifest"]
            self.assertEqual(manifest["statistics"]["total_profiled"], 20)
            self.assertEqual(manifest["statistics"]["eligible_candidates"], 10)
            self.assertEqual(manifest["statistics"]["sampled_train_count"], 5)
            self.assertEqual(manifest["statistics"]["sampled_val_count"], 3)
            self.assertTrue(manifest["test_set_wall"]["passed"])

            # Verify saved files
            self.assertTrue(result["train_parquet_path"].exists())
            self.assertTrue(result["train_jsonl_path"].exists())
            self.assertTrue(result["val_parquet_path"].exists())
            self.assertTrue(result["val_jsonl_path"].exists())
            self.assertTrue(result["manifest_path"].exists())

            # Verify SHA256 integrity
            calc_sha = compute_sha256(result["train_parquet_path"])
            self.assertEqual(calc_sha, manifest["files"]["grpo_train_parquet"]["sha256"])

            # Verify dataframe contents
            df_loaded = pd.read_parquet(result["train_parquet_path"])
            self.assertEqual(len(df_loaded), 5)
            self.assertIn("prompt", df_loaded.columns)
            self.assertIn("gold_answer", df_loaded.columns)
            self.assertIn("difficulty_tier", df_loaded.columns)

    def test_deterministic_sampling(self):
        """Test identical seed produces identical problem IDs."""
        train_rows = [
            {
                "problem_id": f"p_{i:03d}",
                "question": f"Q {i}",
                "cleaned_solution": "S1\nS2\nS3\n#### 1",
                "gold_answer": "1",
            }
            for i in range(30)
        ]
        train_df = pd.DataFrame(train_rows)
        val_df = pd.DataFrame([{"problem_id": "v_1", "question": "VQ", "cleaned_solution": "S1\n#### 1", "gold_answer": "1"}])
        test_df = pd.DataFrame([{"problem_id": "t_1", "question": "TQ", "cleaned_solution": "S1\n#### 1", "gold_answer": "1"}])

        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            res1 = prepare_grpo_dataset(train_df, val_df, test_df, output_dir=tmp1, seed=123, config={"sampling": {"target_train_size": 10, "val_size": 1}})
            res2 = prepare_grpo_dataset(train_df, val_df, test_df, output_dir=tmp2, seed=123, config={"sampling": {"target_train_size": 10, "val_size": 1}})

            df1 = pd.read_parquet(res1["train_parquet_path"])
            df2 = pd.read_parquet(res2["train_parquet_path"])
            self.assertEqual(df1["problem_id"].tolist(), df2["problem_id"].tolist())

    def test_cli_execution(self):
        """Test standalone CLI execution via subprocess in sample mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                "scripts/prepare_grpo_dataset.py",
                "--sample-mode",
                "--output-dir",
                str(Path(tmpdir) / "grpo"),
                "--report-dir",
                str(Path(tmpdir) / "reports"),
                "--target-train-size",
                "10",
                "--val-size",
                "5",
                "--seed",
                "42",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"CLI failed:\nStdout: {proc.stdout}\nStderr: {proc.stderr}")

            out_manifest = Path(tmpdir) / "grpo" / "manifest.json"
            self.assertTrue(out_manifest.exists())
            with open(out_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertLessEqual(data["statistics"]["sampled_train_count"], 10)
                self.assertLessEqual(data["statistics"]["sampled_val_count"], 5)


if __name__ == "__main__":
    unittest.main()
