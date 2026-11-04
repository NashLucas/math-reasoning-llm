"""Unit tests for Phase 6 Day 40: GRPO Reward Functions & Group Advantage Estimation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.verify_reward_functions import main as cli_main
from src.rewards.grpo_rewards import (
    CompositeGRPOReward,
    RewardBreakdown,
    accuracy_reward,
    compute_group_relative_advantages,
    format_reward,
    length_penalty,
    load_reward_config,
    reasoning_structure_reward,
    trl_accuracy_reward_fn,
    trl_format_reward_fn,
    trl_reasoning_structure_reward_fn,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestGRPORewardFunctions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_accuracy_reward_correctness(self):
        completions = [
            "Step 1: 5 * 6 = 30.\n#### 30",
            "The calculation gives 15.5.\n#### 15.5",
            "Step 1: 30 / 10 = 3.\n#### 3",
            "I calculate 99.\n#### 99",
            "",
        ]
        golds = ["30", "15.5", "3.0", "100", "42"]
        scores = accuracy_reward(completions, golds)
        self.assertEqual(scores, [1.0, 1.0, 1.0, 0.0, 0.0])

    def test_format_reward_canonical_contract(self):
        valid = [
            "Step 1: calculate.\n#### 42",
            "Step 1: calculate.\n#### -15",
            "Step 1: calculate.\n#### 3.1415",
            "Step 1: calculate.\n#### 42   ",
        ]
        invalid = [
            "Step 1: calculate.\nFinal answer: \\boxed{42}",
            "The answer is 42.",
            "Step 1: calculate.\n#### 42 but there is text after it",
            "Step 1: calculate.\n####",
            "",
        ]
        valid_scores = format_reward(valid)
        invalid_scores = format_reward(invalid)

        self.assertEqual(valid_scores, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(invalid_scores, [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_reasoning_structure_reward(self):
        # Good derivations with steps and math operators
        well_reasoned = [
            "Step 1: John has 5 apples. Step 2: He buys 7 more. 5 + 7 = 12.\n#### 12",
            "First, calculate the base rate of 20 dollars per hour times 4 hours = 80.\n#### 80",
        ]
        # Guess-hacking (no derivation) or empty
        hacked_or_empty = [
            "#### 12",
            "The answer is\n#### 12",
            "",
        ]
        self.assertEqual(reasoning_structure_reward(well_reasoned), [1.0, 1.0])
        self.assertEqual(reasoning_structure_reward(hacked_or_empty), [0.0, 0.0, 0.0])

    def test_length_penalty_repetition(self):
        normal_text = ["Step 1: 2 + 2 = 4.\n#### 4"]
        pen_normal = length_penalty(normal_text)
        self.assertEqual(pen_normal, [0.0])

        looping_text = ["We add 5 and 5 to obtain 10. " * 20 + "\n#### 10"]
        pen_looping = length_penalty(looping_text)
        self.assertGreater(pen_looping[0], 0.0)

    def test_compute_group_relative_advantages(self):
        # Group of 4 with mixed rewards
        rewards = [1.7, 0.7, 1.2, 0.0]
        advs = compute_group_relative_advantages(rewards, group_size=4)
        self.assertEqual(len(advs), 4)
        self.assertGreater(advs[0], 0.0)  # Best reward has positive advantage
        self.assertLess(advs[3], 0.0)  # Worst reward has negative advantage
        self.assertAlmostEqual(sum(advs), 0.0, places=5)

        # Zero variance group (all equal rewards)
        all_equal = [1.7, 1.7, 1.7, 1.7]
        equal_advs = compute_group_relative_advantages(all_equal, group_size=4)
        self.assertEqual(equal_advs, [0.0, 0.0, 0.0, 0.0])

        # All zeros group
        all_zeros = [0.0, 0.0, 0.0, 0.0]
        zero_advs = compute_group_relative_advantages(all_zeros, group_size=4)
        self.assertEqual(zero_advs, [0.0, 0.0, 0.0, 0.0])

        # Multi-group batch (2 groups of 4)
        batch_rewards = [1.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 2.0]
        batch_advs = compute_group_relative_advantages(batch_rewards, group_size=4)
        self.assertEqual(len(batch_advs), 8)
        self.assertAlmostEqual(sum(batch_advs[:4]), 0.0, places=5)
        self.assertEqual(batch_advs[4:], [0.0, 0.0, 0.0, 0.0])

        # Empty rewards list
        self.assertEqual(compute_group_relative_advantages([]), [])

    def test_composite_grpo_reward_breakdown(self):
        composite = CompositeGRPOReward()
        comp = "Step 1: 10 + 10 = 20.\nStep 2: 20 * 2 = 40.\n#### 40"
        gold = "40"
        breakdown = composite.score_single(completion=comp, gold_answer=gold, problem_id="test_p1")

        self.assertIsInstance(breakdown, RewardBreakdown)
        self.assertEqual(breakdown.accuracy_reward, 1.0)
        self.assertEqual(breakdown.format_reward, 1.0)
        self.assertEqual(breakdown.reasoning_structure_reward, 1.0)
        self.assertEqual(breakdown.length_penalty, 0.0)
        self.assertAlmostEqual(breakdown.total_reward, 1.7, places=2)

    def test_composite_grpo_reward_batch_and_callable(self):
        composite = CompositeGRPOReward()
        completions = [
            "Step 1: calculate 5 + 5 = 10.\n#### 10",
            "Wrong answer step.\n#### 99",
            "#### 10",
            "Broken output",
        ]
        golds = ["10", "10", "10", "10"]

        # Batch scoring with advantages
        breakdowns = composite.score_batch(completions, golds, group_size=4)
        self.assertEqual(len(breakdowns), 4)
        self.assertIsNotNone(breakdowns[0].group_advantage)
        self.assertGreater(breakdowns[0].group_advantage, 0.0)

        # TRL Callable interface
        scores = composite(prompts=["Q1"] * 4, completions=completions, gold_answer=golds)
        self.assertEqual(len(scores), 4)
        self.assertAlmostEqual(scores[0], 1.7, places=2)

    def test_trl_standalone_functions(self):
        prompts = ["Q1", "Q2"]
        completions = [
            "Step 1: 2 + 2 = 4.\n#### 4",
            "I guess\n#### 99",
        ]
        golds = ["4", "4"]

        acc = trl_accuracy_reward_fn(prompts, completions, gold_answer=golds)
        self.assertEqual(acc, [1.0, 0.0])

        fmt = trl_format_reward_fn(prompts, completions)
        self.assertEqual(fmt, [1.0, 1.0])

        struct = trl_reasoning_structure_reward_fn(prompts, completions)
        self.assertEqual(struct, [1.0, 0.0])

    def test_load_reward_config_fallbacks(self):
        # Default config load
        cfg = load_reward_config()
        self.assertIn("weights", cfg)
        self.assertEqual(cfg["weights"]["accuracy"], 1.0)

        # Missing path fallback
        missing_cfg = load_reward_config(self.test_dir / "nonexistent.yaml")
        self.assertIn("weights", missing_cfg)
        self.assertEqual(missing_cfg["weights"]["format"], 0.5)

    def test_cli_runner(self):
        out_dir = self.test_dir / "reports"
        cli_args = [
            "scripts/verify_reward_functions.py",
            "--output_dir",
            str(out_dir),
            "--quiet",
        ]
        with patch("sys.argv", cli_args):
            exit_code = cli_main()
            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "day40_reward_design.json").exists())
            self.assertTrue((out_dir / "day40_reward_design.md").exists())


if __name__ == "__main__":
    unittest.main()
