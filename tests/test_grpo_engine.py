"""Unit tests for GRPO Rollout Generation & Policy Update Engine (Phase 6 Day 43)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training.grpo_engine import (
    GRPOLossMetrics,
    GRPORolloutGroup,
    GRPORolloutSampler,
    GRPOTrainerEngine,
    compute_grpo_loss,
    load_grpo_engine_config,
    verify_zero_advantage_invariance,
)
from grpo_trainer import (
    MathGRPOTrainer,
    accuracy_reward_func,
    build_peft_config,
    composite_reward_func,
    format_reward_func,
    reasoning_structure_reward_func,
    resolve_starting_adapter,
)


class TestGRPOEngine(unittest.TestCase):
    """Test suite for GRPO rollout sampling, loss computation, clipping, and training steps."""

    def test_grpo_loss_exact_ratio(self):
        """Test that identical policy and rollout distributions produce -E[A] = 0 loss."""
        # 4 sequences, length 4 tokens
        cur_lp = np.zeros((4, 4))
        old_lp = np.zeros((4, 4))
        advantages = [1.0, 1.0, -1.0, -1.0]
        mask = np.ones((4, 4))

        loss, metrics = compute_grpo_loss(
            current_logprobs=cur_lp,
            old_logprobs=old_lp,
            advantages=advantages,
            completion_mask=mask,
            clip_range=0.2,
            beta=0.0,
        )

        self.assertAlmostEqual(loss, 0.0, places=6)
        self.assertAlmostEqual(metrics.policy_loss, 0.0, places=6)
        self.assertAlmostEqual(metrics.clip_fraction, 0.0, places=6)
        self.assertAlmostEqual(metrics.mean_ratio, 1.0, places=6)

    def test_grpo_loss_positive_advantage_reinforcement(self):
        """Test that increased probability on positive-advantage candidates decreases loss."""
        # Policy increases probability: cur_lp > old_lp -> ratio > 1.0
        cur_lp = np.log(1.1) * np.ones((2, 4))
        old_lp = np.zeros((2, 4))
        advantages = [1.0, 1.0]  # Both positive
        mask = np.ones((2, 4))

        loss, metrics = compute_grpo_loss(
            current_logprobs=cur_lp,
            old_logprobs=old_lp,
            advantages=advantages,
            completion_mask=mask,
            clip_range=0.2,
            beta=0.0,
        )

        # Loss = - E[r * A] = - (1.1 * 1.0) = -1.1
        self.assertAlmostEqual(loss, -1.1, places=5)
        self.assertLess(loss, 0.0)

    def test_grpo_loss_negative_advantage_penalty(self):
        """Test that increased probability on negative-advantage candidates increases loss."""
        # Policy increases probability on bad completion: ratio = 1.1, adv = -1.0
        cur_lp = np.log(1.1) * np.ones((2, 4))
        old_lp = np.zeros((2, 4))
        advantages = [-1.0, -1.0]  # Both negative
        mask = np.ones((2, 4))

        loss, metrics = compute_grpo_loss(
            current_logprobs=cur_lp,
            old_logprobs=old_lp,
            advantages=advantages,
            completion_mask=mask,
            clip_range=0.2,
            beta=0.0,
        )

        # Loss = - E[r * A] = - (1.1 * -1.0) = +1.1
        self.assertAlmostEqual(loss, 1.1, places=5)
        self.assertGreater(loss, 0.0)

    def test_grpo_loss_clipping_bounds(self):
        """Test that ratios exceeding 1 + eps are clipped to 1 + eps."""
        # ratio = 2.0, clip_range = 0.2 -> clipped to 1.2
        cur_lp = np.log(2.0) * np.ones((2, 4))
        old_lp = np.zeros((2, 4))
        advantages = [1.0, 1.0]
        mask = np.ones((2, 4))

        loss, metrics = compute_grpo_loss(
            current_logprobs=cur_lp,
            old_logprobs=old_lp,
            advantages=advantages,
            completion_mask=mask,
            clip_range=0.2,
            beta=0.0,
        )

        # Clipped surrogate objective is 1.2 * 1.0 = 1.2 -> loss = -1.2
        self.assertAlmostEqual(loss, -1.2, places=5)
        self.assertEqual(metrics.clip_fraction, 1.0)

    def test_grpo_loss_zero_variance_advantages(self):
        """Test zero-variance groups (all advantages 0.0) produce 0.0 policy gradient loss."""
        cur_lp = np.log(1.5) * np.ones((4, 4))
        old_lp = np.zeros((4, 4))
        advantages = [0.0, 0.0, 0.0, 0.0]
        mask = np.ones((4, 4))

        loss, metrics = compute_grpo_loss(
            current_logprobs=cur_lp,
            old_logprobs=old_lp,
            advantages=advantages,
            completion_mask=mask,
            clip_range=0.2,
            beta=0.0,
        )

        self.assertAlmostEqual(loss, 0.0, places=6)
        self.assertAlmostEqual(metrics.policy_loss, 0.0, places=6)

    def test_grpo_loss_kl_penalty_integration(self):
        """Test that reference logprobs and positive beta add a positive KL penalty."""
        cur_lp = np.zeros((2, 4))
        old_lp = np.zeros((2, 4))
        ref_lp = -1.0 * np.ones((2, 4))  # delta = +1.0
        advantages = [0.0, 0.0]
        mask = np.ones((2, 4))

        loss, metrics = compute_grpo_loss(
            current_logprobs=cur_lp,
            old_logprobs=old_lp,
            advantages=advantages,
            completion_mask=mask,
            clip_range=0.2,
            ref_logprobs=ref_lp,
            beta=0.04,
        )

        # Policy loss is 0.0 because advantages are 0.0
        self.assertAlmostEqual(metrics.policy_loss, 0.0, places=6)
        # KL penalty must be strictly positive
        self.assertGreater(metrics.kl_penalty, 0.0)
        self.assertAlmostEqual(loss, metrics.kl_penalty, places=6)

    def test_completion_mask_enforcement(self):
        """Test that prompt tokens (mask=0) do not contribute to loss."""
        # 1 sequence of 4 tokens: first 2 are prompt (mask 0), last 2 are completion (mask 1)
        # Prompt tokens have huge delta (100.0), completion tokens have delta 0.0
        cur_lp = np.array([[100.0, 100.0, 0.0, 0.0]])
        old_lp = np.zeros((1, 4))
        advantages = [1.0]
        mask = np.array([[0.0, 0.0, 1.0, 1.0]])

        loss, metrics = compute_grpo_loss(
            current_logprobs=cur_lp,
            old_logprobs=old_lp,
            advantages=advantages,
            completion_mask=mask,
            clip_range=0.2,
            beta=0.0,
        )

        # Completion tokens have ratio = 1.0, adv = 1.0 -> loss = -1.0
        self.assertAlmostEqual(loss, -1.0, places=5)
        self.assertEqual(metrics.num_valid_tokens, 2)

    def test_rollout_sampler_generation(self):
        """Test GRPORolloutSampler generates G candidates with normalized advantages."""
        sampler = GRPORolloutSampler(group_size=4, mock_mode=True)
        group = sampler.sample_rollout_group(
            prompt_id="prob_test",
            prompt_text="Question: What is 2+2?\nAnswer:",
            gold_answer="4",
        )

        self.assertIsInstance(group, GRPORolloutGroup)
        self.assertEqual(len(group.completions), 4)
        self.assertEqual(len(group.rewards), 4)
        self.assertEqual(len(group.advantages), 4)
        # Advantages in heterogeneous group should sum to ~0
        self.assertAlmostEqual(float(np.mean(group.advantages)), 0.0, places=5)
        # Verify masks
        self.assertEqual(group.input_ids.shape, (4, 16))
        self.assertEqual(group.completion_mask.shape, (4, 16))

    def test_trainer_engine_step(self):
        """Test GRPOTrainerEngine executes a train step and updates history."""
        engine = GRPOTrainerEngine(mock_mode=True)
        metrics = engine.train_step(
            prompt_id="step_test_1",
            prompt_text="Solve 10 * 10.\n#### 100",
            gold_answer="100",
        )

        self.assertIsInstance(metrics, GRPOLossMetrics)
        self.assertEqual(engine.step_count, 1)
        self.assertEqual(len(engine.history), 1)
        self.assertEqual(engine.history[0]["prompt_id"], "step_test_1")
        self.assertTrue(np.isfinite(metrics.total_loss))

    def test_load_grpo_engine_config(self):
        """Test configuration loader loads grpo and rewards settings."""
        cfg = load_grpo_engine_config()
        self.assertIn("grpo", cfg)
        self.assertIn("rewards", cfg)
        self.assertIn("kl", cfg)

    def test_cli_execution(self):
        """Test standalone CLI execution via subprocess."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                "scripts/verify_grpo_engine.py",
                "--output-dir",
                str(Path(tmpdir)),
                "--num-steps",
                "3",
                "--seed",
                "42",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"CLI failed:\nStdout: {proc.stdout}\nStderr: {proc.stderr}")

            md_path = Path(tmpdir) / "day43_grpo_engine_report.md"
            json_path = Path(tmpdir) / "day43_grpo_engine_report.json"
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertIn("rollout_benchmark", data)
                self.assertIn("clipped_loss", data)
                self.assertIn("trajectory_simulation", data)
                self.assertEqual(data["trajectory_simulation"]["final_step"], 3)

    def test_cli_execution_with_config_and_report_dir(self):
        """Test Step 40 CLI signature with --config and --report_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                "scripts/verify_grpo_engine.py",
                "--config",
                "configs/grpo_args.json",
                "--report_dir",
                str(Path(tmpdir)),
                "--num_steps",
                "2",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"CLI failed:\nStdout: {proc.stdout}\nStderr: {proc.stderr}")

            md_path = Path(tmpdir) / "day43_grpo_engine_report.md"
            json_path = Path(tmpdir) / "day43_grpo_engine_report.json"
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

    def test_zero_advantage_invariance_function(self):
        """Test verify_zero_advantage_invariance identity."""
        res = verify_zero_advantage_invariance(group_size=4, seq_len=6, clip_range=0.2)
        self.assertTrue(res["invariance_verified"])
        self.assertTrue(res["policy_loss_zero"])
        self.assertAlmostEqual(res["numpy_loss"], 0.0, places=10)

    def test_grpo_trainer_module_components(self):
        """Test root grpo_trainer.py components and reward functions."""
        # Test reward adapters
        prompts = ["What is 2+2?", "Solve 5*3"]
        comps = ["Let's think.\n#### 4", "Wrong.\n#### 999"]
        golds = ["4", "15"]

        acc_rewards = accuracy_reward_func(prompts, comps, gold_answer=golds)
        self.assertEqual(len(acc_rewards), 2)
        self.assertEqual(acc_rewards[0], 1.0)
        self.assertEqual(acc_rewards[1], 0.0)

        fmt_rewards = format_reward_func(prompts, comps)
        self.assertEqual(len(fmt_rewards), 2)
        self.assertEqual(fmt_rewards[0], 1.0)
        self.assertEqual(fmt_rewards[1], 1.0)

        struct_rewards = reasoning_structure_reward_func(prompts, comps)
        self.assertEqual(len(struct_rewards), 2)

        comp_rewards = composite_reward_func(prompts, comps, gold_answer=golds)
        self.assertEqual(len(comp_rewards), 2)
        self.assertGreater(comp_rewards[0], comp_rewards[1])

        # Test PEFT config builder
        peft_cfg = build_peft_config(r=16, lora_alpha=32)
        self.assertEqual(peft_cfg["r"], 16)
        self.assertEqual(peft_cfg["peft_type"], "LORA")

        # Test adapter resolver
        resolved = resolve_starting_adapter()
        # Even if none exists on disk, function should not crash and return Path or None
        self.assertTrue(resolved is None or isinstance(resolved, Path))

    def test_math_grpo_trainer_execution(self):
        """Test MathGRPOTrainer execution in mock/smoke mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_dir = tmp_path / "checkpoints"
            adapter_dir = tmp_path / "adapter_model"
            rep_dir = tmp_path / "reports"

            trainer = MathGRPOTrainer(use_mock=True)
            res = trainer.train(
                output_dir=out_dir,
                adapter_dir=adapter_dir,
                reports_dir=rep_dir,
                max_steps=2,
                eval_steps=1,
                save_steps=2,
                sample_mode=True,
                smoke_test=True,
            )

            self.assertEqual(res["status"], "success")
            self.assertEqual(res["final_step"], 2)
            self.assertTrue(adapter_dir.exists())


if __name__ == "__main__":
    unittest.main()
