"""Unit tests for reasoning prompt templates and Day 16 verification (Phase 3 Day 16)."""

from __future__ import annotations

import unittest
from typing import Dict, List
from unittest.mock import MagicMock, patch

from scripts.test_reasoning_prompts import (
    day16_verdict,
    evaluate_prompts_on_samples,
    text_after_hash_line,
)
from src.reasoning.prompting import (
    CANONICAL_LINE_INSTRUCTION,
    ReasoningPromptConfig,
    ReasoningPromptType,
    build_reasoning_prompt_messages,
    format_reasoning_prompt_text,
    get_reasoning_prompt_config,
    get_supported_prompt_types,
    normalize_prompt_type,
    render_reasoning_prompt,
)


class TestReasoningPrompting(unittest.TestCase):
    """Test suite for reasoning prompt builders and templates."""

    def test_prompt_types_enum(self) -> None:
        """Verify prompt types are correctly defined."""
        self.assertEqual(ReasoningPromptType.CHAIN_OF_THOUGHT.value, "chain_of_thought")
        self.assertEqual(ReasoningPromptType.SOLVE_THEN_VERIFY.value, "solve_then_verify")
        self.assertEqual(ReasoningPromptType.ALTERNATIVE_METHOD.value, "alternative_method")

    def test_normalize_prompt_type(self) -> None:
        """Verify prompt type normalization including aliases and error handling."""
        # Exact enum
        self.assertEqual(
            normalize_prompt_type(ReasoningPromptType.CHAIN_OF_THOUGHT),
            ReasoningPromptType.CHAIN_OF_THOUGHT,
        )

        # Canonical string
        self.assertEqual(
            normalize_prompt_type("chain_of_thought"),
            ReasoningPromptType.CHAIN_OF_THOUGHT,
        )
        self.assertEqual(
            normalize_prompt_type("solve_then_verify"),
            ReasoningPromptType.SOLVE_THEN_VERIFY,
        )
        self.assertEqual(
            normalize_prompt_type("alternative_method"),
            ReasoningPromptType.ALTERNATIVE_METHOD,
        )

        # Case-insensitive
        self.assertEqual(
            normalize_prompt_type("CHAIN_OF_THOUGHT"),
            ReasoningPromptType.CHAIN_OF_THOUGHT,
        )

        # Aliases
        self.assertEqual(normalize_prompt_type("cot"), ReasoningPromptType.CHAIN_OF_THOUGHT)
        self.assertEqual(normalize_prompt_type("verify"), ReasoningPromptType.SOLVE_THEN_VERIFY)
        self.assertEqual(normalize_prompt_type("solve-then-verify"), ReasoningPromptType.SOLVE_THEN_VERIFY)
        self.assertEqual(normalize_prompt_type("alt"), ReasoningPromptType.ALTERNATIVE_METHOD)
        self.assertEqual(normalize_prompt_type("alternative"), ReasoningPromptType.ALTERNATIVE_METHOD)
        self.assertEqual(normalize_prompt_type("alternative-method"), ReasoningPromptType.ALTERNATIVE_METHOD)

        # Invalid type
        with self.assertRaises(ValueError) as ctx:
            normalize_prompt_type("invalid_prompt_type")
        self.assertIn("Unknown reasoning prompt type", str(ctx.exception))

    def test_get_supported_prompt_types(self) -> None:
        """Verify supported prompt types listing."""
        types = get_supported_prompt_types()
        self.assertIn("chain_of_thought", types)
        self.assertIn("solve_then_verify", types)
        self.assertIn("alternative_method", types)
        self.assertEqual(len(types), 3)

    def test_all_configs_state_canonical_format(self) -> None:
        """Every prompt configuration must explicitly require the '#### <number>' format."""
        for ptype in ReasoningPromptType:
            config = get_reasoning_prompt_config(ptype)
            self.assertIsInstance(config, ReasoningPromptConfig)
            self.assertEqual(config.prompt_type, ptype)
            self.assertTrue(config.name)
            self.assertTrue(config.description)

            # Check canonical instruction presence
            self.assertIn("####", config.system_prompt)
            self.assertIn("#### <number>", config.user_template)
            self.assertIn("#### <number>", config.plain_text_template)

    def test_build_reasoning_prompt_messages(self) -> None:
        """Verify chat message construction across all three prompt types."""
        question = "Janet had 16 apples. She gave half to her sister and 3 to her mom. How many are left?"

        for ptype in ReasoningPromptType:
            messages = build_reasoning_prompt_messages(question, prompt_type=ptype)
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[1]["role"], "user")

            # System prompt requirements
            self.assertIn("####", messages[0]["content"])

            # User prompt requirements
            self.assertIn(question, messages[1]["content"])
            self.assertIn(CANONICAL_LINE_INSTRUCTION, messages[1]["content"])

    def test_format_reasoning_prompt_text(self) -> None:
        """Verify plain text prompt formatting across all three prompt types."""
        question = "A car travels at 60 mph for 3 hours. How far did it travel?"

        for ptype in ReasoningPromptType:
            prompt_text = format_reasoning_prompt_text(question, prompt_type=ptype)
            self.assertIn(f"Question: {question}", prompt_text)
            self.assertIn(CANONICAL_LINE_INSTRUCTION, prompt_text)
            self.assertTrue(
                prompt_text.endswith("Answer: Let's think step by step.\n")
                or prompt_text.endswith("Answer: Stage 1 - Initial Solution:\n")
                or prompt_text.endswith("Answer: Alternative Method:\n")
            )

    def test_render_reasoning_prompt_without_tokenizer(self) -> None:
        """Verify rendering without tokenizer falls back to plain text."""
        question = "What is 2 + 2?"
        rendered = render_reasoning_prompt(question, prompt_type="cot", tokenizer=None)
        expected = format_reasoning_prompt_text(question, prompt_type="cot")
        self.assertEqual(rendered, expected)

    def test_render_reasoning_prompt_with_chat_template(self) -> None:
        """Verify rendering with a tokenizer that provides a chat template."""
        question = "Find x if 3x + 6 = 21."

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.bos_token_id = 1

        def mock_apply_chat_template(
            messages: List[Dict[str, str]],
            tokenize: bool = False,
            add_generation_prompt: bool = True,
        ) -> str:
            rendered = "<s>" + "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            if add_generation_prompt:
                rendered += "\nassistant:"
            return rendered

        mock_tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
        # Tokenizer adds BOS when encoding
        mock_tokenizer.side_effect = lambda text, **kwargs: {"input_ids": [1, 42]}

        rendered = render_reasoning_prompt(question, prompt_type="verify", tokenizer=mock_tokenizer)
        self.assertIn("user:", rendered)
        self.assertIn("system:", rendered)
        self.assertIn(question, rendered)
        # Should strip leading <s> to prevent double BOS
        self.assertFalse(rendered.startswith("<s>"))

    def test_text_after_hash_line_and_day16_verdict(self) -> None:
        """Verify detection of trailing text after '####' and mock vs real verdict logic."""
        self.assertEqual(text_after_hash_line("Step 1: 2 + 2 = 4\n#### 4"), "")
        self.assertEqual(
            text_after_hash_line("Step 1: 2 + 2 = 4\n#### 4\nTherefore the answer is 4."),
            "Therefore the answer is 4.",
        )

        compliant_results = {
            "chain_of_thought": {"format_compliance_rate": 1.0},
            "solve_then_verify": {"format_compliance_rate": 1.0},
            "alternative_method": {"format_compliance_rate": 1.0},
        }
        # Mock runs must never claim PASSED
        self.assertEqual(day16_verdict(compliant_results, use_real_model=False), "MOCK")
        self.assertEqual(day16_verdict(compliant_results, use_real_model=True), "PASSED")

        non_compliant_results = {
            "chain_of_thought": {"format_compliance_rate": 1.0},
            "solve_then_verify": {"format_compliance_rate": 0.9},
            "alternative_method": {"format_compliance_rate": 1.0},
        }
        self.assertEqual(
            day16_verdict(non_compliant_results, use_real_model=True),
            "NEEDS WORDING ADJUSTMENT",
        )

    def test_real_model_path_batches_and_does_not_stop_on_hash(self) -> None:
        """Regression test: Day 16 real-model path must batch prompts and never pass '####' in stop_sequences."""
        problems = [
            {"problem_id": "p1", "question": "What is 2 + 2?", "gold_answer": "4"},
            {"problem_id": "p2", "question": "What is 3 + 5?", "gold_answer": "8"},
        ]

        def fake_generate_batch_hf(
            model,
            tokenizer,
            prompts,
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            do_sample=False,
            stop_sequences=None,
            num_return_sequences=1,
        ):
            # Must not stop on '####' before the answer is emitted
            if stop_sequences is not None:
                self.assertNotIn("####", stop_sequences)
            return [
                [
                    {
                        "completion_raw": "Step 1\n#### 4",
                        "completion": "Step 1\n#### 4",
                        "extracted_answer": "4",
                        "extraction_method": "hash",
                        "prompt_tokens": 12,
                        "completion_tokens": 6,
                        "finish_reason": "stop",
                    }
                ]
                for _ in prompts
            ]

        with patch("src.model.load_model_and_tokenizer", autospec=True, return_value=(object(), None)), \
                patch("src.generation.generate_batch_hf", autospec=True, side_effect=fake_generate_batch_hf) as mock_gen:
            res = evaluate_prompts_on_samples(
                sample_problems=problems,
                use_real_model=True,
                batch_size=10,
            )
            # 3 prompt types, 2 problems batched into 1 call per prompt type = 3 calls
            self.assertEqual(mock_gen.call_count, 3)
            for ptype_res in res.values():
                self.assertEqual(ptype_res["format_compliance_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
