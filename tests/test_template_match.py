"""Unit tests verifying that the SFT training template matches the inference template exactly (Phase 4 Day 27).

Design Rule 1: One answer format, one parser.
If the training prompt format and the evaluation inference prompt format diverge by even
a single whitespace character, the post-trained model is evaluated out-of-distribution.
"""

from __future__ import annotations

import unittest

from src.data.formatting import (
    ANSWER_FORMAT_INSTRUCTION,
    format_zero_shot_prompt,
)
from src.data.sft_dataset import format_sft_prompt_completion


class TestTemplateMatch(unittest.TestCase):
    """Test suite ensuring exact bit-for-bit equivalence between SFT training and inference prompts."""

    def setUp(self) -> None:
        self.sample_questions = [
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with 4 every day. How many eggs does she have left at the end of the day?",
            "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "Josh decides to try flipping a house. He buys a house for $80,000 and spends $50,000 repairing it. He sells it for $150,000. How much profit did he make?",
            "   Leading and trailing whitespace question?   ",
            "Question with multiple\nlines and special symbols: $100, 25%, and 3/4.",
        ]

    def test_sft_prompt_matches_zero_shot_inference_prompt_exactly(self) -> None:
        """SFT training prompt must match format_zero_shot_prompt exactly."""
        for q in self.sample_questions:
            inference_prompt = format_zero_shot_prompt(q)
            pair = format_sft_prompt_completion(
                question=q,
                completion="Step 1: Calculate result.\n#### 42",
                gold_answer="42",
            )
            training_prompt = pair["prompt"]

            self.assertEqual(
                training_prompt,
                inference_prompt,
                f"Mismatch between training prompt and inference prompt for question:\n{q!r}",
            )
            # Ensure character-by-character identity
            self.assertEqual(len(training_prompt), len(inference_prompt))

    def test_prompt_contains_exact_answer_format_instruction(self) -> None:
        """Both templates must explicitly enforce the canonical '#### <number>' contract."""
        for q in self.sample_questions:
            pair = format_sft_prompt_completion(
                question=q,
                completion="Step 1: Calculate result.\n#### 42",
                gold_answer="42",
            )
            prompt = pair["prompt"]

            self.assertIn(ANSWER_FORMAT_INSTRUCTION, prompt)
            self.assertIn("#### <number>", prompt)
            self.assertTrue(prompt.startswith(f"Question: {q.strip()}\n"))
            self.assertTrue(prompt.endswith("Answer: Let's think step by step.\n"))

    def test_full_text_concatenation_structure(self) -> None:
        """The full_text field must be the exact concatenation of prompt and formatted completion."""
        q = "What is 7 * 6?"
        comp = "7 * 6 = 42.\n#### 42"
        eos = "<|eos|>"

        pair = format_sft_prompt_completion(question=q, completion=comp, eos_token=eos)
        expected_comp = f"7 * 6 = 42.\n#### 42{eos}"

        self.assertEqual(pair["completion"], expected_comp)
        self.assertEqual(pair["full_text"], f"{pair['prompt']}{expected_comp}")
        self.assertTrue(pair["full_text"].endswith(eos))


if __name__ == "__main__":
    unittest.main()
