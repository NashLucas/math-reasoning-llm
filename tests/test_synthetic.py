"""Unit tests for reasoning trace sampling and synthetic data generation (Phase 3 Day 17)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.config import ConfigManager
from src.data.synthetic import (
    ReasoningTrace,
    TraceSampler,
    build_trace_id,
    compute_trace_yield,
    load_synthetic_problems,
    load_synthetic_subset_ids,
    load_synthetic_traces,
    select_synthetic_subset_ids,
)
from tests.test_generation_hf import HAS_TORCH, make_tiny_lm


class TestSyntheticDataGeneration(unittest.TestCase):
    """Test suite for reasoning trace sampler, data loaders, and yield calculations."""

    def test_reasoning_trace_dataclass(self) -> None:
        """Verify ReasoningTrace creation, dictionary conversion, and reconstruction."""
        trace = ReasoningTrace(
            trace_id="gsm8k_123_chain_of_thought_s0",
            problem_id="gsm8k_123",
            question="What is 10 + 20?",
            gold_answer="30",
            prompt_type="chain_of_thought",
            sample_idx=0,
            teacher_model="Qwen/Qwen2.5-1.5B-Instruct",
            prompt="Solve: 10 + 20",
            completion="10 + 20 = 30.\n#### 30",
            extracted_answer="30",
            extraction_method="hash",
            is_correct=True,
            format_compliant=True,
            prompt_tokens=10,
            completion_tokens=12,
            finish_reason="stop",
            timestamp_utc="2026-11-02T12:00:00Z",
        )

        data = trace.to_dict()
        self.assertEqual(data["trace_id"], "gsm8k_123_chain_of_thought_s0")
        self.assertTrue(data["is_correct"])
        self.assertTrue(data["format_compliant"])

        reconstructed = ReasoningTrace.from_dict(data)
        self.assertEqual(reconstructed.trace_id, trace.trace_id)
        self.assertEqual(reconstructed.extracted_answer, "30")

    def test_build_trace_id(self) -> None:
        """Verify deterministic trace ID formatting and prompt type normalization."""
        tid1 = build_trace_id("prob_001", "chain_of_thought", 0)
        self.assertEqual(tid1, "prob_001_chain_of_thought_s0")

        # Alias normalization
        tid2 = build_trace_id("prob_001", "cot", 0)
        self.assertEqual(tid2, "prob_001_chain_of_thought_s0")

        tid3 = build_trace_id("prob_002", "verify", 2)
        self.assertEqual(tid3, "prob_002_solve_then_verify_s2")

    def test_synthetic_subset_ids_reproducible_from_seed(self) -> None:
        """Verify the committed 3,000 synthetic subset IDs are reproduced exactly from train_ids with seed 42."""
        root = ConfigManager().project_root
        train_ids = json.loads((root / "data" / "splits" / "train_ids.json").read_text(encoding="utf-8"))
        subset_ids = load_synthetic_subset_ids()

        self.assertEqual(len(subset_ids), 3000)
        self.assertEqual(len(set(subset_ids)), 3000)
        reproduced = select_synthetic_subset_ids(train_ids, subset_size=3000, seed=42)
        self.assertEqual(subset_ids, reproduced)

    def test_load_synthetic_problems_and_contamination_protection(self) -> None:
        """Verify loading synthetic problems, zero test/val overlap, and no silent fallback in real mode."""
        problems = load_synthetic_problems(limit=5, allow_sample_fallback=True)
        self.assertEqual(len(problems), 5)

        for prob in problems:
            self.assertIn("problem_id", prob)
            self.assertIn("question", prob)
            self.assertIn("gold_answer", prob)

        # Missing split file without allow_sample_fallback must raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_synthetic_problems(
                split_path="nonexistent_train.parquet",
                allow_sample_fallback=False,
            )

    def test_trace_sampler_resumability_and_chunking(self) -> None:
        """Verify chunked generation, output formatting, and resumption."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_problems = [
                {"problem_id": "prob_1", "question": "What is 2 + 2?", "gold_answer": "4"},
                {"problem_id": "prob_2", "question": "What is 3 * 5?", "gold_answer": "15"},
            ]

            sampler = TraceSampler(
                model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
                output_dir=tmp_dir,
                chunk_size=2,
                use_mock=True,
            )

            # First run: 2 samples per problem = 4 total traces
            out_file = sampler.sample_traces(
                problems=sample_problems,
                prompt_types=["chain_of_thought"],
                samples_per_problem=2,
            )

            self.assertTrue(out_file.exists())
            records = load_synthetic_traces(out_file)
            self.assertEqual(len(records), 4)

            # Second run: rerun exact same config -> should detect all 4 already exist and generate 0 new
            out_file_2 = sampler.sample_traces(
                problems=sample_problems,
                prompt_types=["chain_of_thought"],
                samples_per_problem=2,
            )
            records_2 = load_synthetic_traces(out_file_2)
            self.assertEqual(len(records_2), 4)

            # Third run: rerun with different config and overwrite=True -> replaces old file safely
            out_file_3 = sampler.sample_traces(
                problems=sample_problems,
                prompt_types=["chain_of_thought"],
                samples_per_problem=1,
                overwrite=True,
            )
            records_3 = load_synthetic_traces(out_file_3)
            self.assertEqual(len(records_3), 2)

            sampler.close()

    def test_real_vllm_backend_signatures_batching_and_stop_rules(self) -> None:
        """Regression test (with autospec): vLLM path batches prompts, enables sampling, and does not stop on '####'."""
        problems = [
            {"problem_id": "p1", "question": "What is 6 * 7?", "gold_answer": "42"},
            {"problem_id": "p2", "question": "What is 10 + 5?", "gold_answer": "15"},
        ]

        fake_vllm = MagicMock()
        fake_vllm.get_tokenizer.return_value = None

        def fake_gen_vllm(
            llm,
            prompts,
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            do_sample=False,
            stop_sequences=None,
            num_return_sequences=1,
            seed=None,
            lora_request=None,
        ):
            self.assertIs(llm, fake_vllm)
            self.assertTrue(do_sample)
            self.assertEqual(temperature, 0.7)
            self.assertEqual(top_p, 0.95)
            self.assertEqual(top_k, -1)
            self.assertEqual(num_return_sequences, 3)
            self.assertIsNotNone(seed)
            self.assertNotIn("####", stop_sequences or [])
            return [
                [
                    {
                        "completion_raw": f"Step {s}\n#### 42",
                        "completion": f"Step {s}\n#### 42",
                        "extracted_answer": "42",
                        "extraction_method": "hash",
                        "prompt_tokens": 15,
                        "completion_tokens": 8,
                        "finish_reason": "stop",
                    }
                    for s in range(num_return_sequences)
                ]
                for _ in prompts
            ]

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch("src.data.synthetic.resolve_generation_backend", autospec=True, return_value="vllm") as mock_res, \
                patch("src.data.synthetic.load_vllm_model", autospec=True, return_value=fake_vllm), \
                patch("src.data.synthetic.generate_batch_vllm", autospec=True, side_effect=fake_gen_vllm) as mock_gen, \
                patch("src.data.synthetic.shutdown_model", autospec=True) as mock_shut:
            sampler = TraceSampler(
                model_name_or_path="deepseek-ai/deepseek-math-7b-instruct",
                backend="vllm",
                output_dir=tmp_dir,
                chunk_size=10,
                batch_size=8,
                seed=42,
                use_mock=False,
            )
            out_file = sampler.sample_traces(
                problems=problems,
                prompt_types=["chain_of_thought"],
                samples_per_problem=3,
                temperature=0.7,
                top_p=0.95,
            )
            sampler.close()

            mock_res.assert_called_once_with("vllm")
            # Both problems batched in a single vLLM call returning 3 samples each
            self.assertEqual(mock_gen.call_count, 1)
            mock_shut.assert_called_once_with(fake_vllm, "vllm")

            records = load_synthetic_traces(out_file)
            self.assertEqual(len(records), 6)
            self.assertTrue(all(r["format_compliant"] for r in records))
            self.assertEqual(sum(1 for r in records if r["is_correct"]), 3)  # p1 (42) right, p2 (15) wrong

    def test_real_hf_backend_signatures_batching_and_stop_rules(self) -> None:
        """Regression test (with autospec): HF path batches prompts, enables sampling, and does not stop on '####'."""
        problems = [
            {"problem_id": "p1", "question": "What is 2 + 2?", "gold_answer": "4"},
            {"problem_id": "p2", "question": "What is 3 + 3?", "gold_answer": "6"},
        ]

        fake_hf_model = object()

        def fake_gen_hf(
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
            self.assertIs(model, fake_hf_model)
            self.assertTrue(do_sample)
            self.assertEqual(num_return_sequences, 2)
            self.assertNotIn("####", stop_sequences or [])
            return [
                [
                    {
                        "completion_raw": "2 + 2 = 4\n#### 4",
                        "completion": "2 + 2 = 4\n#### 4",
                        "extracted_answer": "4",
                        "extraction_method": "hash",
                        "prompt_tokens": 10,
                        "completion_tokens": 6,
                        "finish_reason": "stop",
                    }
                    for _ in range(num_return_sequences)
                ]
                for _ in prompts
            ]

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch("src.model.load_model_and_tokenizer", autospec=True, return_value=(fake_hf_model, None)), \
                patch("src.data.synthetic.generate_batch_hf", autospec=True, side_effect=fake_gen_hf) as mock_gen, \
                patch("src.data.synthetic.shutdown_model", autospec=True) as mock_shut, \
                patch("transformers.set_seed"):
            sampler = TraceSampler(
                model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
                backend="hf",
                output_dir=tmp_dir,
                chunk_size=10,
                batch_size=4,
                seed=42,
                use_mock=False,
            )
            out_file = sampler.sample_traces(
                problems=problems,
                prompt_types=["chain_of_thought"],
                samples_per_problem=2,
                temperature=0.7,
            )
            sampler.close()

            self.assertEqual(mock_gen.call_count, 1)
            mock_shut.assert_called_once_with(fake_hf_model, "hf")
            records = load_synthetic_traces(out_file)
            self.assertEqual(len(records), 4)

    @unittest.skipUnless(HAS_TORCH, "needs torch")
    def test_trace_sampler_on_tiny_hf_model_samples_diverse_reproducible_traces(self) -> None:
        """End-to-end HF test on tiny GPT-2: samples_per_problem > 1 produces distinct traces and is reproducible."""
        model, tokenizer = make_tiny_lm(seed=0)
        problems = [
            {"problem_id": "p1", "question": "w01 w02 w03", "gold_answer": "42"},
            {"problem_id": "p2", "question": "w04 w05 w06", "gold_answer": "42"},
        ]
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            for tmp_dir in (tmp_a, tmp_b):
                sampler = TraceSampler(
                    model_name_or_path="tiny-gpt2",
                    backend="hf",
                    output_dir=tmp_dir,
                    chunk_size=2,
                    batch_size=2,
                    seed=123,
                    use_mock=False,
                    model=model,
                    tokenizer=tokenizer,
                )
                sampler.sample_traces(
                    problems=problems,
                    prompt_types=["chain_of_thought"],
                    samples_per_problem=3,
                    temperature=0.7,
                    max_new_tokens=8,
                )
                sampler.close()

            recs_a = load_synthetic_traces(Path(tmp_a) / "traces_tiny-gpt2_chain_of_thought.jsonl")
            recs_b = load_synthetic_traces(Path(tmp_b) / "traces_tiny-gpt2_chain_of_thought.jsonl")
            self.assertEqual(len(recs_a), 6)
            self.assertEqual([r["completion"] for r in recs_a], [r["completion"] for r in recs_b])
            p1_completions = [r["completion"] for r in recs_a if r["problem_id"] == "p1"]
            self.assertGreater(len(set(p1_completions)), 1, "Sampled completions for a problem must not all be identical")

    def test_compute_trace_yield(self) -> None:
        """Verify yield metrics and pass rate distribution calculation."""
        mock_traces = [
            {"problem_id": "p1", "is_correct": True, "format_compliant": True},
            {"problem_id": "p1", "is_correct": True, "format_compliant": True},
            {"problem_id": "p2", "is_correct": True, "format_compliant": True},
            {"problem_id": "p2", "is_correct": False, "format_compliant": True},
            {"problem_id": "p3", "is_correct": False, "format_compliant": False},
            {"problem_id": "p3", "is_correct": False, "format_compliant": True},
        ]

        metrics = compute_trace_yield(mock_traces)
        self.assertEqual(metrics["total_traces"], 6)
        self.assertEqual(metrics["total_problems"], 3)
        self.assertEqual(metrics["correct_traces"], 3)
        self.assertEqual(metrics["trace_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["format_compliance_rate"], 5 / 6, places=3)
        self.assertEqual(metrics["problems_with_at_least_one_correct"], 2)
        self.assertEqual(metrics["pass_rate_distribution"]["100%"], 1)  # p1: 2/2 = 1.0
        self.assertEqual(metrics["pass_rate_distribution"]["(25, 50%]"], 1)  # p2: 1/2 = 0.5
        self.assertEqual(metrics["pass_rate_distribution"]["0%"], 1)  # p3: 0/2 = 0.0


if __name__ == "__main__":
    unittest.main()
