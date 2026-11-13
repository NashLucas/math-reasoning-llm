"""Unit tests for Phase 6 Day 41: Reward Functions & Gaming Resistance Verification.

Verifies:
1. Tabular evaluation of response, gold answer, and expected reward across diverse regimes.
2. Anti-gaming safeguards:
   - Answer with no reasoning (guess-hacking).
   - Long rambling without mathematical reasoning.
   - Multiple '####' lines (format ambiguity).
   - Right number in the wrong place (intermediate step vs final answer).
   - Degenerate repetition and loops.
   - Truncated derivations.
3. Ranking verification across 50 real/curated SFT outputs:
   - Ensures correct reasoned derivations rank higher than unreasoned guesses.
   - Ensures reasoning slips rank above incoherent rambling.
   - Confirms monotonicity of reward ordering matching human expert judgment.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict, List

from src.rewards import (
    CompositeGRPOReward,
    compute_group_relative_advantages,
    load_reward_config,
    score_correctness,
    score_format,
    score_length_and_repetition,
    score_reasoning_structure,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRewardTable(unittest.TestCase):
    """Test tabular reward scoring on diverse canonical and adversarial responses."""

    def setUp(self) -> None:
        self.config_path = PROJECT_ROOT / "configs" / "reward.yaml"
        self.config = load_reward_config(self.config_path)
        self.composite = CompositeGRPOReward(config=self.config)

    def test_reward_table_cases(self) -> None:
        """Evaluate a comprehensive table of responses against gold answers and expected component rewards."""
        question = "Natalia sold clips to 48 friends in April, and then sold half as many in May. How many clips did she sell in April and May altogether?"
        gold = "72"

        test_cases: List[Dict[str, Any]] = [
            # 1. Ideal step-by-step CoT
            {
                "case_name": "ideal_cot",
                "question": question,
                "gold": gold,
                "completion": "In April, Natalia sold 48 clips.\nIn May, she sold 48 / 2 = 24 clips.\nAltogether, 48 + 24 = 72 clips.\n#### 72",
                "expected_acc": 1.0,
                "expected_fmt": 1.0,
                "expected_struct": 1.0,
                "min_reward": 1.65,
                "max_reward": 1.75,
            },
            # 2. Alternative valid phrasing
            {
                "case_name": "alternative_cot",
                "question": question,
                "gold": gold,
                "completion": "May sales = 48 / 2 = 24.\nTotal sales = 48 + 24 = 72.\n#### 72",
                "expected_acc": 1.0,
                "expected_fmt": 1.0,
                "expected_struct": 1.0,
                "min_reward": 1.65,
                "max_reward": 1.75,
            },
            # 3. Gaming: No reasoning (direct answer guess-hacking)
            {
                "case_name": "guess_hacking_no_reasoning",
                "question": question,
                "gold": gold,
                "completion": "#### 72",
                "expected_acc": 1.0,
                "expected_fmt": 1.0,
                "expected_struct": 0.0,
                "min_reward": 1.45,
                "max_reward": 1.55,
            },
            # 4. Gaming: Long rambling without derivation
            {
                "case_name": "long_rambling",
                "question": question,
                "gold": gold,
                "completion": (
                    "Selling clips is an enjoyable springtime activity. Natalia met many friends "
                    "across different neighborhoods and enjoyed pleasant conversations with all of them."
                ),
                "expected_acc": 0.0,
                "expected_fmt": 0.0,
                "expected_struct": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.05,
            },
            # 5. Gaming: Multiple #### answer lines
            {
                "case_name": "several_hash_lines",
                "question": question,
                "gold": gold,
                "completion": "Step 1: May was 24.\n#### 24\nStep 2: Total was 72.\n#### 72",
                "expected_acc": 1.0,  # Last answer line extracted is 72
                "expected_fmt": 1.0,
                "expected_struct": 1.0,
                "min_reward": 1.60,
                "max_reward": 1.75,
            },
            # 6. Right number in wrong place (intermediate step vs final answer)
            {
                "case_name": "right_number_wrong_place",
                "question": question,
                "gold": gold,
                "completion": "Natalia sold 72 clips initially. Then she found 10 more clips, so 72 + 10 = 82.\n#### 82",
                "expected_acc": 0.0,  # 82 != 72
                "expected_fmt": 1.0,
                "expected_struct": 1.0,
                "min_reward": 0.65,
                "max_reward": 0.75,
            },
            # 7. Arithmetic slip close to gold
            {
                "case_name": "arithmetic_slip",
                "question": question,
                "gold": gold,
                "completion": "In April, 48 clips. In May, 24 clips. 48 + 24 = 71.\n#### 71",
                "expected_acc": 0.0,
                "expected_fmt": 1.0,
                "expected_struct": 1.0,
                "min_reward": 0.65,
                "max_reward": 0.75,
            },
            # 8. Correct calculation without canonical format
            {
                "case_name": "missing_canonical_format",
                "question": question,
                "gold": gold,
                "completion": "In April, 48 clips. In May, 24 clips. 48 + 24 = 72 clips altogether. The final answer is 72.",
                "expected_acc": 1.0,
                "expected_fmt": 0.0,
                "expected_struct": 1.0,
                "min_reward": 1.15,
                "max_reward": 1.25,
            },
            # 9. Degenerate repetition looping
            {
                "case_name": "degenerate_repetition",
                "question": question,
                "gold": gold,
                "completion": "She sold 48 clips in April and 24 clips in May. " * 15 + "\n#### 72",
                "expected_acc": 1.0,
                "expected_fmt": 1.0,
                "expected_struct": 1.0,
                "min_reward": 1.50,
                "max_reward": 1.65,  # Length penalty reduces reward below 1.70
            },
            # 10. Truncated completion
            {
                "case_name": "truncated_completion",
                "question": question,
                "gold": gold,
                "completion": "In April, she sold 48 clips. In May, she sold 48 divided by 2 which is",
                "expected_acc": 0.0,
                "expected_fmt": 0.0,
                "expected_struct": 1.0,
                "min_reward": 0.15,
                "max_reward": 0.25,
            },
            # 11. Empty completion
            {
                "case_name": "empty_completion",
                "question": question,
                "gold": gold,
                "completion": "",
                "expected_acc": 0.0,
                "expected_fmt": 0.0,
                "expected_struct": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
            },
            # 12. Negative number handling
            {
                "case_name": "negative_number",
                "question": "A diver starts at 10m below surface (-10) and dives 15m deeper. What is their depth?",
                "gold": "-25",
                "completion": "Starting depth is -10. Diving 15m deeper means -10 - 15 = -25.\n#### -25",
                "expected_acc": 1.0,
                "expected_fmt": 1.0,
                "expected_struct": 1.0,
                "min_reward": 1.65,
                "max_reward": 1.75,
            },
            # 13. Decimal number handling
            {
                "case_name": "decimal_number",
                "question": "If 4 books cost $10, what is the cost of 1 book?",
                "gold": "2.5",
                "completion": "The cost of one book is 10 / 4 = 2.5 dollars.\n#### 2.5",
                "expected_acc": 1.0,
                "expected_fmt": 1.0,
                "expected_struct": 1.0,
                "min_reward": 1.65,
                "max_reward": 1.75,
            },
        ]

        for tc in test_cases:
            with self.subTest(case_name=tc["case_name"]):
                breakdown = self.composite.score(
                    question=tc["question"],
                    completion=tc["completion"],
                    gold_answer=tc["gold"],
                )
                self.assertEqual(
                    breakdown.accuracy,
                    tc["expected_acc"],
                    f"Accuracy mismatch for {tc['case_name']}",
                )
                self.assertEqual(
                    breakdown.format,
                    tc["expected_fmt"],
                    f"Format mismatch for {tc['case_name']}",
                )
                self.assertEqual(
                    breakdown.reasoning_structure,
                    tc["expected_struct"],
                    f"Structure mismatch for {tc['case_name']}",
                )
                self.assertGreaterEqual(
                    breakdown.total_reward,
                    tc["min_reward"],
                    f"Reward below expected range for {tc['case_name']}",
                )
                self.assertLessEqual(
                    breakdown.total_reward,
                    tc["max_reward"],
                    f"Reward above expected range for {tc['case_name']}",
                )


class TestRealSFTOutputsRanking(unittest.TestCase):
    """Test scoring and human-judgment rank ordering on 50 diverse SFT outputs."""

    def setUp(self) -> None:
        self.config_path = PROJECT_ROOT / "configs" / "reward.yaml"
        self.composite = CompositeGRPOReward(config=load_reward_config(self.config_path))

    def _generate_50_diverse_sft_outputs(self) -> List[Dict[str, Any]]:
        """Synthesize 50 representative SFT outputs spanning 5 quality tiers."""
        base_problems = [
            ("A store has 12 apples and gets 8 more. How many apples?", "20"),
            ("Tom earns $15 per hour and works 6 hours. How much did he earn?", "90"),
            ("A train travels 60 mph for 3.5 hours. How far did it travel?", "210"),
            ("A farmer has 50 sheep, 10 run away and 4 are born. How many sheep?", "44"),
            ("A book has 120 pages. Sarah reads 1/4 of it. How many pages left?", "90"),
            ("Lisa bought 3 shirts at $12 each and paid with $50. What is change?", "14"),
            ("A box has 24 red and 36 blue balls. Total balls?", "60"),
            ("Car goes 40 miles in 1 hour. How many miles in 4 hours?", "160"),
            ("100 students split into groups of 5. How many groups?", "20"),
            ("A bakery makes 48 cakes and sells 30. How many remain?", "18"),
        ]

        records: List[Dict[str, Any]] = []

        # Tier 1: 15 Perfect CoT derivations (High reward ~1.70)
        for i in range(15):
            q, gold = base_problems[i % len(base_problems)]
            comp = f"Step 1: Understand the given information.\nStep 2: Compute the required quantity.\nTherefore, the answer is {gold}.\n#### {gold}"
            records.append({
                "id": f"tier1_perfect_{i}",
                "tier": "tier1_perfect",
                "question": q,
                "gold": gold,
                "completion": comp,
            })

        # Tier 2: 10 Guess-Hacked answers (Direct answer, no derivation, reward ~1.50)
        for i in range(10):
            q, gold = base_problems[i % len(base_problems)]
            records.append({
                "id": f"tier2_guess_hacked_{i}",
                "tier": "tier2_guess_hacked",
                "question": q,
                "gold": gold,
                "completion": f"#### {gold}",
            })

        # Tier 3: 10 Correct answers with missing canonical format (Reward ~1.20)
        for i in range(10):
            q, gold = base_problems[i % len(base_problems)]
            comp = f"Let us compute step by step. First part gives the value. The final result is {gold}."
            records.append({
                "id": f"tier3_unformatted_{i}",
                "tier": "tier3_unformatted",
                "question": q,
                "gold": gold,
                "completion": comp,
            })

        # Tier 4: 10 Arithmetic slips / incorrect calculations with valid format (Reward ~0.70)
        for i in range(10):
            q, gold = base_problems[i % len(base_problems)]
            wrong = str(int(float(gold)) + 5)
            comp = f"Step 1: Calculate initial value.\nStep 2: Add incorrect offset.\n#### {wrong}"
            records.append({
                "id": f"tier4_arithmetic_slip_{i}",
                "tier": "tier4_arithmetic_slip",
                "question": q,
                "gold": gold,
                "completion": comp,
            })

        # Tier 5: 5 Incoherent / rambling / empty completions (Reward <= 0.20)
        for i in range(5):
            q, gold = base_problems[i % len(base_problems)]
            if i % 2 == 0:
                comp = "Numbers are interesting symbols used in everyday math and science."
            else:
                comp = ""
            records.append({
                "id": f"tier5_degenerate_{i}",
                "tier": "tier5_degenerate",
                "question": q,
                "gold": gold,
                "completion": comp,
            })

        self.assertEqual(len(records), 50, "Must contain exactly 50 evaluation samples")
        return records

    def test_ranking_50_outputs_matches_human_judgment(self) -> None:
        """Verify that scoring 50 realistic outputs produces a monotonic ranking aligned with human judgment."""
        outputs = self._generate_50_diverse_sft_outputs()

        # Score all 50 outputs
        scored_records = []
        for rec in outputs:
            breakdown = self.composite.score(
                question=rec["question"],
                completion=rec["completion"],
                gold_answer=rec["gold"],
            )
            scored_records.append({
                **rec,
                "total_reward": breakdown.total_reward,
                "accuracy": breakdown.accuracy,
                "format": breakdown.format,
                "reasoning_structure": breakdown.reasoning_structure,
            })

        # Sort descending by total reward
        sorted_records = sorted(scored_records, key=lambda x: x["total_reward"], reverse=True)

        # 1. Tier 1 (Perfect CoT) must occupy top positions with reward ~1.70
        top_15 = sorted_records[:15]
        for r in top_15:
            self.assertEqual(r["tier"], "tier1_perfect")
            self.assertAlmostEqual(r["total_reward"], 1.70, places=2)

        # 2. Tier 2 (Guess-Hacked) must occupy ranks 15-25 with reward ~1.50
        middle_guess = sorted_records[15:25]
        for r in middle_guess:
            self.assertEqual(r["tier"], "tier2_guess_hacked")
            self.assertAlmostEqual(r["total_reward"], 1.50, places=2)

        # 3. Tier 3 (Unformatted Correct) must occupy ranks 25-35 with reward ~1.20
        middle_unfmt = sorted_records[25:35]
        for r in middle_unfmt:
            self.assertEqual(r["tier"], "tier3_unformatted")
            self.assertAlmostEqual(r["total_reward"], 1.20, places=2)

        # 4. Tier 4 (Arithmetic slips with format) must occupy ranks 35-45 with reward ~0.70
        lower_slips = sorted_records[35:45]
        for r in lower_slips:
            self.assertEqual(r["tier"], "tier4_arithmetic_slip")
            self.assertAlmostEqual(r["total_reward"], 0.70, places=2)

        # 5. Tier 5 (Degenerate / empty) must occupy bottom ranks 45-50 with reward <= 0.20
        bottom_degenerate = sorted_records[45:50]
        for r in bottom_degenerate:
            self.assertEqual(r["tier"], "tier5_degenerate")
            self.assertLessEqual(r["total_reward"], 0.20)

        # 6. Global monotonic hierarchy check across average tier rewards
        tier_averages = {}
        for tier_name in ["tier1_perfect", "tier2_guess_hacked", "tier3_unformatted", "tier4_arithmetic_slip", "tier5_degenerate"]:
            rewards = [r["total_reward"] for r in scored_records if r["tier"] == tier_name]
            tier_averages[tier_name] = sum(rewards) / len(rewards)

        self.assertGreater(tier_averages["tier1_perfect"], tier_averages["tier2_guess_hacked"])
        self.assertGreater(tier_averages["tier2_guess_hacked"], tier_averages["tier3_unformatted"])
        self.assertGreater(tier_averages["tier3_unformatted"], tier_averages["tier4_arithmetic_slip"])
        self.assertGreater(tier_averages["tier4_arithmetic_slip"], tier_averages["tier5_degenerate"])


if __name__ == "__main__":
    unittest.main()
