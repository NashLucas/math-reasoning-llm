"""Unit tests for generation engine, stop rules, and resumable chunking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.generation import (
    append_generation_records,
    generate_batch_vllm,
    load_existing_generations,
    resolve_generation_backend,
    run_generation_pipeline,
    truncate_at_stop_sequences,
)


class TestGenerationEngine(unittest.TestCase):
    """Test suite for generation stop rules, JSONL persistence, and batching."""

    def test_truncate_at_stop_sequence_question(self) -> None:
        """Verify that generation truncates when model hallucinates a new 'Question:' prompt."""
        text = "Step 1: 5 * 2 = 10.\n#### 10\n\nQuestion: A boy had 3 dogs..."
        truncated = truncate_at_stop_sequences(text, stop_sequences=["Question:", "\n\n\n"])
        self.assertEqual(truncated, "Step 1: 5 * 2 = 10.\n#### 10")

    def test_truncate_at_stop_sequence_triple_newline(self) -> None:
        """Verify that generation truncates at triple newline."""
        text = "First calculate the cost: 20.\n#### 20\n\n\nSome unrelated rambling"
        truncated = truncate_at_stop_sequences(text, stop_sequences=["Question:", "\n\n\n"])
        self.assertEqual(truncated, "First calculate the cost: 20.\n#### 20")

    def test_truncate_line_after_hash_answer(self) -> None:
        """Verify stop rule: truncate text on the line immediately following #### <number>."""
        text = "First compute the total.\n#### 42\nNote that 42 is the answer. Further comments..."
        truncated = truncate_at_stop_sequences(text)
        self.assertEqual(truncated, "First compute the total.\n#### 42")

    def test_truncate_hash_answer_without_trailing_newline(self) -> None:
        """Verify that #### answer at the end of text without trailing newline is kept intact."""
        text = "Reasoning step 1.\nReasoning step 2.\n#### 100"
        truncated = truncate_at_stop_sequences(text)
        self.assertEqual(truncated, "Reasoning step 1.\nReasoning step 2.\n#### 100")

    def test_truncate_hash_with_commas(self) -> None:
        """Verify stop rule preserves thousands separators in #### 1,000."""
        text = "The total cars produced is 1000.\n#### 1,000\nExtra restated text."
        truncated = truncate_at_stop_sequences(text)
        self.assertEqual(truncated, "The total cars produced is 1000.\n#### 1,000")

    def test_truncate_empty_and_whitespace(self) -> None:
        """Verify handling of empty or whitespace strings."""
        self.assertEqual(truncate_at_stop_sequences(""), "")
        self.assertEqual(truncate_at_stop_sequences("   \n\t  "), "")

    def test_load_and_append_generation_records(self) -> None:
        """Verify JSONL record serialization, append, and index loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_file = Path(tmp_dir) / "generations.jsonl"

            records_batch_1 = [
                {"problem_id": "gsm8k_001", "question": "Q1", "extracted_answer": "10"},
                {"problem_id": "gsm8k_002", "question": "Q2", "extracted_answer": "20"},
            ]
            append_generation_records(jsonl_file, records_batch_1)

            # Read back
            loaded = load_existing_generations(jsonl_file)
            self.assertEqual(len(loaded), 2)
            self.assertIn("gsm8k_001", loaded)
            self.assertIn("gsm8k_002", loaded)
            self.assertEqual(loaded["gsm8k_001"]["extracted_answer"], "10")

            # Append second batch
            records_batch_2 = [
                {"problem_id": "gsm8k_003", "question": "Q3", "extracted_answer": "30"},
            ]
            append_generation_records(jsonl_file, records_batch_2)

            loaded_again = load_existing_generations(jsonl_file)
            self.assertEqual(len(loaded_again), 3)
            self.assertIn("gsm8k_003", loaded_again)

    @patch("src.generation.generate_batch_hf")
    def test_run_generation_pipeline_skips_existing(self, mock_generate_batch: MagicMock) -> None:
        """Verify that running generation pipeline skips already completed problems on rerun."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_file = Path(tmp_dir) / "output.jsonl"

            # Pre-populate with problem 1
            existing = [
                {
                    "problem_id": "p1",
                    "question": "Q1",
                    "gold_answer": "10",
                    "generated_text": "#### 10",
                    "extracted_answer": "10",
                    "is_correct": True,
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                }
            ]
            append_generation_records(jsonl_file, existing)

            # Problems to generate: p1 and p2
            problems = [
                {"problem_id": "p1", "question": "Q1", "gold_answer": "10"},
                {"problem_id": "p2", "question": "Q2", "gold_answer": "20"},
            ]

            # Mock model batch generation for the remaining p2 problem
            mock_generate_batch.return_value = [
                [
                    {
                        "completion_raw": "Reasoning #### 20",
                        "completion": "Reasoning #### 20",
                        "extracted_answer": "20",
                        "prompt_tokens": 25,
                        "completion_tokens": 15,
                    }
                ]
            ]

            mock_model = MagicMock()
            mock_tok = MagicMock()

            results = run_generation_pipeline(
                problems=problems,
                output_file=jsonl_file,
                model=mock_model,
                tokenizer=mock_tok,
                batch_size=2,
            )

            # Verify that generate_batch_hf was called ONLY for the pending problem p2
            self.assertEqual(mock_generate_batch.call_count, 1)
            call_prompts = mock_generate_batch.call_args[1]["prompts"]
            self.assertEqual(len(call_prompts), 1)

            # Verify total results contains both p1 and p2
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["problem_id"], "p1")
            self.assertEqual(results[1]["problem_id"], "p2")

    @patch("src.generation.generate_batch_hf")
    def test_run_generation_pipeline_all_completed(self, mock_generate_batch: MagicMock) -> None:
        """Verify that running generation pipeline when all problems are done skips generation entirely."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_file = Path(tmp_dir) / "output.jsonl"

            existing = [
                {"problem_id": "p1", "question": "Q1", "gold_answer": "10"},
                {"problem_id": "p2", "question": "Q2", "gold_answer": "20"},
            ]
            append_generation_records(jsonl_file, existing)

            mock_model = MagicMock()
            mock_tok = MagicMock()

            results = run_generation_pipeline(
                problems=existing,
                output_file=jsonl_file,
                model=mock_model,
                tokenizer=mock_tok,
            )

            self.assertEqual(mock_generate_batch.call_count, 0)
            self.assertEqual(len(results), 2)


    def test_resolve_generation_backend(self) -> None:
        """Verify backend resolution and graceful fallback to 'hf'."""
        # Explicit HF backend always returns 'hf'
        self.assertEqual(resolve_generation_backend("hf"), "hf")

        # When vLLM is not installed/mocked as False, 'vllm' or 'auto' resolves to 'hf'
        with patch("src.generation.is_vllm_available", return_value=False):
            self.assertEqual(resolve_generation_backend("auto"), "hf")
            self.assertEqual(resolve_generation_backend("vllm"), "hf")

        # When vLLM is available, 'auto' and 'vllm' resolve to 'vllm'
        with patch("src.generation.is_vllm_available", return_value=True):
            self.assertEqual(resolve_generation_backend("auto"), "vllm")
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            with patch("src.generation.HAS_VLLM", True), patch("src.generation.HAS_TORCH", True), patch("src.generation.torch", mock_torch):
                self.assertEqual(resolve_generation_backend("vllm"), "vllm")

    def test_generate_batch_vllm_mocked(self) -> None:
        """Verify generate_batch_vllm correctly maps vLLM outputs to standardized completion dicts."""
        mock_llm = MagicMock()
        mock_output_item = MagicMock()
        mock_output_item.prompt_token_ids = [101, 102, 103]  # length 3

        mock_completion = MagicMock()
        mock_completion.text = "Step 1: Calculate.\n#### 42\nExtra text after answer."
        mock_completion.token_ids = [201, 202, 203, 204]  # length 4
        mock_output_item.outputs = [mock_completion]

        mock_llm.generate.return_value = [mock_output_item]

        with patch("src.generation.HAS_VLLM", True), patch("src.generation.SamplingParams", MagicMock()):
            results = generate_batch_vllm(
                llm=mock_llm,
                prompts=["Solve this:"],
                max_new_tokens=256,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 1)
        comp = results[0][0]
        self.assertEqual(comp["extracted_answer"], "42")
        self.assertEqual(comp["extraction_method"], "hash")
        # Truncation rule stopped at line after #### 42
        self.assertEqual(comp["completion"], "Step 1: Calculate.\n#### 42")
        self.assertEqual(comp["prompt_tokens"], 3)
        self.assertEqual(comp["completion_tokens"], 4)

    @patch("src.generation.generate_batch_vllm")
    def test_run_generation_pipeline_vllm_backend(self, mock_generate_vllm: MagicMock) -> None:
        """Verify run_generation_pipeline routes through vLLM backend when requested."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_file = Path(tmp_dir) / "output_vllm.jsonl"
            problems = [{"problem_id": "p1", "question": "What is 2+2?", "gold_answer": "4"}]

            mock_generate_vllm.return_value = [
                [
                    {
                        "completion_raw": "4\n#### 4",
                        "completion": "4\n#### 4",
                        "extracted_answer": "4",
                        "extraction_method": "hash",
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                    }
                ]
            ]

            mock_llm = MagicMock()
            with patch("src.generation.resolve_generation_backend", return_value="vllm"):
                results = run_generation_pipeline(
                    problems=problems,
                    output_file=jsonl_file,
                    model=mock_llm,
                    backend="vllm",
                )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["problem_id"], "p1")
            self.assertEqual(results[0]["extracted_answer"], "4")
            self.assertTrue(results[0]["is_correct"])
            self.assertEqual(mock_generate_vllm.call_count, 1)

if __name__ == "__main__":
    unittest.main()
