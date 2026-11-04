"""Phase 2 review regressions in the evaluator and the vLLM path (no GPU needed)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.preprocessing import frozen_splits_dir
from src.evaluation.evaluator import MathEvaluator, TestSetWallError, run_slug
from src.evaluation.math_metrics import compute_all_metrics, rescore_records
from src.generation import run_generation_pipeline, vllm_max_lora_rank


def first_test_ids(n=2):
    return json.loads((frozen_splits_dir() / "test_ids.json").read_text(encoding="utf-8"))[:n]


class TestTestSetWallByContent(unittest.TestCase):
    def _evaluator(self, tmp, path, final=False):
        return MathEvaluator(split=str(path), final=final, use_mock=True, output_dir=Path(tmp) / "out")

    def test_test_problems_by_file_path_need_final(self):
        """Phase 2 review: --split <path to test file> loaded the test set without --final."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "copy.parquet"
            pd.DataFrame({"problem_id": first_test_ids(), "question": ["q1", "q2"], "gold_answer": ["1", "2"]}).to_parquet(path)
            with self.assertRaises(TestSetWallError):
                self._evaluator(tmp, path).load_problems()
            self.assertEqual(len(self._evaluator(tmp, path, final=True).load_problems()), 2)

    def test_committed_test_sample_is_walled(self):
        sample = Path(__file__).resolve().parent.parent / "data" / "sample" / "test_sample.parquet"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TestSetWallError):
                self._evaluator(tmp, sample).load_problems()

    def test_non_test_file_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mine.parquet"
            pd.DataFrame({"problem_id": ["custom_1"], "question": ["q"], "gold_answer": ["1"]}).to_parquet(path)
            self.assertEqual(len(self._evaluator(tmp, path).load_problems()), 1)


class TestPromptFormatAndOutputs(unittest.TestCase):
    def _ev(self, tmp, checkpoint, fmt="auto"):
        return MathEvaluator(checkpoint=checkpoint, prompt_format=fmt, output_dir=tmp)

    def test_formats_come_from_config_not_the_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._ev(tmp, "deepseek-ai/deepseek-math-7b-base").resolve_prompt_format(), "few_shot")
            self.assertEqual(self._ev(tmp, "deepseek-ai/deepseek-math-7b-instruct").resolve_prompt_format(), "chat")
            self.assertEqual(self._ev(tmp, "deepseek-ai/deepseek-math-7b-rl").resolve_prompt_format(), "chat")
            # Unlisted model: name is ignored ('early' contains 'rl'); no chat template -> few-shot
            self.assertEqual(self._ev(tmp, "/ckpt/early_stop_model").resolve_prompt_format(tokenizer=None), "few_shot")
            tok = MagicMock(chat_template="{{ x }}")
            self.assertEqual(self._ev(tmp, "/ckpt/early_stop_model").resolve_prompt_format(tokenizer=tok), "chat")
            self.assertEqual(self._ev(tmp, "deepseek-ai/deepseek-math-7b-instruct", "zero_shot").resolve_prompt_format(), "zero_shot")

    def test_chat_format_requires_the_model_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._ev(tmp, "deepseek-ai/deepseek-math-7b-instruct")
            self.assertTrue(ev.needs_tokenizer())
            with self.assertRaises(ValueError):
                ev.get_prompt_formatter(tokenizer=None)
            self.assertFalse(self._ev(tmp, "deepseek-ai/deepseek-math-7b-base").needs_tokenizer())

    def test_default_output_dir_is_per_model_and_adapter(self):
        self.assertEqual(run_slug("deepseek-ai/deepseek-math-7b-base"), "deepseek-math-7b-base")
        self.assertEqual(
            run_slug("deepseek-ai/deepseek-math-7b-base", "/drive/checkpoints/sft/checkpoint-200"),
            "deepseek-math-7b-base__sft-checkpoint-200",
        )
        ev_a = MathEvaluator(checkpoint="org/model-a", use_mock=True)
        ev_b = MathEvaluator(checkpoint="org/model-b", use_mock=True)
        self.assertNotEqual(ev_a.output_dir, ev_b.output_dir)
        for ev in (ev_a, ev_b):
            ev.output_dir.rmdir()


class TestRescoringAndTruncation(unittest.TestCase):
    def test_stored_answers_are_rescored_with_current_parser(self):
        """A parser fix must change the metrics of existing generations."""
        records = [{
            "problem_id": "p1", "gold_answer": "85", "generated_text": "#### 60 + 25 = 85",
            "extracted_answer": "60", "extraction_method": "hash", "is_correct": False,
        }]
        self.assertEqual(compute_all_metrics(records)["accuracy"], 0.0)
        rescored, changed = rescore_records(records)
        self.assertEqual(changed, 1)
        self.assertEqual(rescored[0]["extracted_answer"], "60 + 25 = 85")
        self.assertEqual(compute_all_metrics(rescored)["accuracy"], 1.0)

        stale = [{"problem_id": "p2", "gold_answer": "12", "generated_text": "The answer is 3 x 4 = 12.",
                  "extracted_answer": "3", "extraction_method": "last_number", "is_correct": False}]
        rescored, changed = rescore_records(stale)
        self.assertEqual(changed, 1)
        self.assertTrue(rescored[0]["is_correct"])

    def test_truncation_uses_finish_reason(self):
        records = [
            {"gold_answer": "1", "extracted_answer": "1", "extraction_method": "hash", "completion_tokens": 512, "finish_reason": "stop"},
            {"gold_answer": "1", "extracted_answer": None, "extraction_method": "none", "completion_tokens": 512, "finish_reason": "length"},
        ]
        self.assertEqual(compute_all_metrics(records, max_new_tokens=512)["truncated_count"], 1)
        multi = [{"gold_answer": "1", "extracted_answers": ["1", "1"], "extraction_methods": ["hash", "hash"],
                  "completion_tokens": [512, 30], "finish_reasons": ["length", "stop"]}]
        self.assertEqual(compute_all_metrics(multi, max_new_tokens=512)["truncated_count"], 1)


class TestVllmPath(unittest.TestCase):
    def test_lora_rank_rounding(self):
        self.assertEqual(vllm_max_lora_rank(8), 8)
        self.assertEqual(vllm_max_lora_rank(12), 16)
        self.assertEqual(vllm_max_lora_rank(64), 64)
        with self.assertRaises(ValueError):
            vllm_max_lora_rank(1024)

    def test_adapter_is_applied_and_prompts_go_in_one_call(self):
        """Phase 2 review: vLLM ignored adapter_path, so adapters were evaluated as the base model."""
        llm_cls, lora_cls, sampling_cls = MagicMock(name="LLM"), MagicMock(name="LoRARequest"), MagicMock(name="SamplingParams")
        completion = MagicMock(text="x\n#### 4", token_ids=[1, 2, 3], finish_reason="stop")
        request_output = MagicMock(prompt_token_ids=[1, 2], outputs=[completion])
        llm_cls.return_value.generate.side_effect = lambda prompts, params, **kw: [request_output] * len(prompts)

        problems = [{"problem_id": f"p{i}", "question": f"Q{i}", "gold_answer": "4"} for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "checkpoint-100"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(json.dumps({"r": 16}))
            with patch("src.generation.HAS_VLLM", True), patch("src.generation.LLM", llm_cls), \
                    patch("src.generation.LoRARequest", lora_cls), patch("src.generation.SamplingParams", sampling_cls), \
                    patch("src.generation.resolve_generation_backend", return_value="vllm"), \
                    patch("src.generation.shutdown_model") as shutdown:
                records = run_generation_pipeline(
                    problems, Path(tmp) / "gen.jsonl", model_name_or_path="base-model", adapter_path=str(adapter),
                    backend="vllm", prompt_formatter=lambda p: p["question"], vllm_chunk_size=128,
                )

        llm_kwargs = llm_cls.call_args.kwargs
        self.assertTrue(llm_kwargs["enable_lora"])
        self.assertEqual(llm_kwargs["max_lora_rank"], 16)
        lora_cls.assert_called_once_with("adapter", 1, str(adapter))
        generate = llm_cls.return_value.generate
        self.assertEqual(generate.call_count, 1, "all 5 prompts should go to vLLM in one chunk")
        self.assertIs(generate.call_args.kwargs["lora_request"], lora_cls.return_value)
        self.assertEqual(len(records), 5)
        self.assertTrue(all(r["is_correct"] and r["finish_reason"] == "stop" for r in records))
        shutdown.assert_called_once()  # the engine loaded here is released for the next run


if __name__ == "__main__":
    unittest.main()
