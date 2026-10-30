"""Unit tests for prompt formatting and chat template rendering."""

import unittest

from src.data.formatting import (
    FEW_SHOT_EXEMPLARS,
    apply_chat_template,
    format_chat_messages,
    format_few_shot_prompt,
    format_zero_shot_prompt,
)


class TestFormatting(unittest.TestCase):
    def test_few_shot_exemplars_count_and_format(self):
        self.assertEqual(len(FEW_SHOT_EXEMPLARS), 8)
        for ex in FEW_SHOT_EXEMPLARS:
            self.assertIn("id", ex)
            self.assertIn("question", ex)
            self.assertIn("solution", ex)
            self.assertIn("####", ex["solution"])
            self.assertTrue(ex["id"].startswith("gsm8k_"))

    def test_format_few_shot_prompt(self):
        q = "A store has 15 red shirts and 20 blue shirts. How many shirts in total?"
        prompt = format_few_shot_prompt(q, num_shots=8)

        # Check all 8 exemplars appear in prompt
        for ex in FEW_SHOT_EXEMPLARS:
            self.assertIn(ex["question"], prompt)

        # Check prompt terminates with the target question and 'Answer:'
        self.assertTrue(prompt.endswith(f"Question: {q}\nAnswer:"))

    def test_format_zero_shot_prompt(self):
        q = "Calculate 15 * 4."
        prompt = format_zero_shot_prompt(q)
        self.assertIn(q, prompt)
        self.assertIn("Let's think step by step", prompt)

    def test_format_chat_messages_zero_shot(self):
        q = "How many days in 3 weeks?"
        messages = format_chat_messages(q, few_shot=False)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], q)

    def test_format_chat_messages_few_shot(self):
        q = "How many days in 3 weeks?"
        messages = format_chat_messages(q, few_shot=True, num_shots=8)
        # 1 system + 8 * 2 (user/assistant) + 1 target user = 18 messages
        self.assertEqual(len(messages), 18)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], q)

    def test_apply_chat_template_fallback(self):
        messages = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        rendered = apply_chat_template(messages, tokenizer=None)
        self.assertIn("<|System|>", rendered)
        self.assertIn("<|User|>", rendered)
        self.assertIn("<|Assistant|>", rendered)


if __name__ == "__main__":
    unittest.main()
