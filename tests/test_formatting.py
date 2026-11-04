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
        self.assertTrue(prompt.endswith("Answer: Let's think step by step.\n"))

    def test_every_template_asks_for_hash_answer_line(self):
        """Plan, Day 9: every template asks for the final line '#### <number>'."""
        q = "How many days in 3 weeks?"
        self.assertIn("#### <number>", format_zero_shot_prompt(q))
        self.assertIn("#### ", format_chat_messages(q)[0]["content"])
        few_shot = format_few_shot_prompt(q)
        self.assertEqual(few_shot.count("\n#### "), len(FEW_SHOT_EXEMPLARS))

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

    def test_apply_chat_template_has_no_generic_fallback(self):
        """Phase 2 review: without a tokenizer the instruct models got a made-up template."""
        messages = [{"role": "user", "content": "What is 2+2?"}]
        with self.assertRaises(ValueError):
            apply_chat_template(messages, tokenizer=None)
        with self.assertRaises(ValueError):
            apply_chat_template(messages, tokenizer=FakeTokenizer(chat_template=None))

    def test_apply_chat_template_uses_model_template_with_single_bos(self):
        """DeepSeek's template starts with BOS and the tokenizer adds BOS too: keep only one."""
        tok = FakeTokenizer(chat_template=DEEPSEEK_CHAT_TEMPLATE, adds_bos=True)
        rendered = apply_chat_template(format_chat_messages("What is 2+2?"), tokenizer=tok)
        self.assertFalse(rendered.startswith("<s>"))
        self.assertTrue(rendered.startswith("You are an expert mathematical problem solver."))
        self.assertIn("User: What is 2+2?\n\n", rendered)
        self.assertTrue(rendered.endswith("Assistant:"))

        # A tokenizer that does not add BOS itself keeps the template's BOS
        tok_no_bos = FakeTokenizer(chat_template=DEEPSEEK_CHAT_TEMPLATE, adds_bos=False)
        self.assertTrue(apply_chat_template([{"role": "user", "content": "hi"}], tokenizer=tok_no_bos).startswith("<s>"))


# Chat template of deepseek-math-7b-instruct / -rl (tokenizer_config.json), with BOS shown as <s>
DEEPSEEK_CHAT_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}"
    "{{ 'User: ' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}"
    "{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}"
    "{{ message['content'] + '\\n\\n' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)


class FakeTokenizer:
    """Minimal tokenizer: renders a Jinja chat template and optionally prepends BOS on encode."""

    bos_token, bos_token_id, eos_token, name_or_path = "<s>", 1, "</s>", "fake"

    def __init__(self, chat_template, adds_bos=True):
        self.chat_template = chat_template
        self.adds_bos = adds_bos

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": ([1] if (self.adds_bos and add_special_tokens) else []) + [5]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Same output as DEEPSEEK_CHAT_TEMPLATE, rendered without jinja2 (not installed in CI)."""
        out = self.bos_token
        for m in messages:
            if m["role"] == "user":
                out += "User: " + m["content"] + "\n\n"
            elif m["role"] == "assistant":
                out += "Assistant: " + m["content"] + self.eos_token
            elif m["role"] == "system":
                out += m["content"] + "\n\n"
        return out + ("Assistant:" if add_generation_prompt else "")


if __name__ == "__main__":
    unittest.main()
