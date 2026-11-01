"""Generation engine tests on a real (tiny, random) model, not mocks.

The Phase 2 review found bugs that mock-only tests could not catch: batch padding counted as
generated tokens, stop strings applied only after generating to the token limit, and
truncation inferred from those counts. These tests run the Hugging Face path end to end.
"""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

from src.generation import (
    GenerationRunMismatchError,
    append_generation_records,
    generate_batch_hf,
    run_generation_pipeline,
)


def make_tiny_lm(seed: int = 0):
    """Random 2-layer GPT-2 with a small byte-level BPE tokenizer (like real models use, so
    Hugging Face stop strings work)."""
    from tokenizers import decoders, trainers

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320, special_tokens=["[PAD]", "[EOS]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    )
    corpus = [" ".join(f"w{i:02d}" for i in range(60))] * 3 + ["Question: what is 12 + 30?\n#### 42"] * 3
    tk.train_from_iterator(corpus, trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", eos_token="[EOS]", padding_side="left")
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=tk.get_vocab_size(), n_positions=256, n_embd=32, n_layer=2, n_head=2,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config).eval()
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


def make_scripted_lm(tokenizer, chain: dict):
    """GPT-2 whose greedy next token depends only on the last token: chain[current] -> next.

    One-hot embeddings, zeroed attention/MLP outputs and a lm_head that maps token i to
    chain[i] (EOS for tokens not in the chain), so generation is exact and known in advance.
    """
    vocab = len(tokenizer)
    config = GPT2Config(
        vocab_size=vocab, n_positions=256, n_embd=vocab, n_layer=1, n_head=1, tie_word_embeddings=False,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config).eval()
    with torch.no_grad():
        model.transformer.wte.weight.copy_(torch.eye(vocab) * 10)
        model.transformer.wpe.weight.zero_()
        for block in model.transformer.h:
            for proj in (block.attn.c_proj, block.mlp.c_proj):
                proj.weight.zero_()
                proj.bias.zero_()
        head = torch.zeros(vocab, vocab)
        for i in range(vocab):
            head[chain.get(i, tokenizer.eos_token_id), i] = 1.0
        model.lm_head.weight.copy_(head)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model


def letter_chain(tokenizer, letters: str, end_with_eos: bool = True) -> dict:
    ids = [tokenizer.convert_tokens_to_ids(c) for c in letters]
    chain = {a: b for a, b in zip(ids, ids[1:])}
    if not end_with_eos:
        chain[ids[-1]] = ids[0]  # cycle forever
    return chain


@unittest.skipUnless(HAS_TORCH, "needs torch")
class TestHfGenerationOnTinyModel(unittest.TestCase):
    def setUp(self):
        self.model, self.tokenizer = make_tiny_lm()

    def test_padding_is_not_counted_and_finish_reason_is_per_sequence(self):
        """One sequence ends at EOS after 1 token while the other runs to the limit."""
        model = make_scripted_lm(self.tokenizer, letter_chain(self.tokenizer, "bcdefghijk", end_with_eos=False))
        # 'x' is not in the chain, so it is followed by EOS; 'b' cycles through the letters forever
        out = generate_batch_hf(model, self.tokenizer, ["x", "b"], max_new_tokens=12)
        self.assertEqual(out[0][0]["completion_tokens"], 1)
        self.assertEqual(out[0][0]["finish_reason"], "stop")
        self.assertEqual(out[1][0]["completion_tokens"], 12)
        self.assertEqual(out[1][0]["finish_reason"], "length")

    def test_eos_ends_generation_and_is_counted_once(self):
        model = make_scripted_lm(self.tokenizer, letter_chain(self.tokenizer, "bcdefghijk"))
        comp = generate_batch_hf(model, self.tokenizer, ["b"], max_new_tokens=30)[0][0]
        self.assertEqual(comp["completion"], "cdefghijk")
        self.assertEqual(comp["completion_tokens"], 10)  # 9 letters + EOS
        self.assertEqual(comp["finish_reason"], "stop")

    def test_stop_string_ends_generation(self):
        """Stop strings end generation itself, instead of generating to the limit and cutting the text."""
        model = make_scripted_lm(self.tokenizer, letter_chain(self.tokenizer, "bcdefghijk", end_with_eos=False))
        comp = generate_batch_hf(model, self.tokenizer, ["b"], max_new_tokens=30, stop_sequences=["fg"])[0][0]
        self.assertEqual(comp["completion_tokens"], 5)  # c d e f g
        self.assertEqual(comp["finish_reason"], "stop")
        self.assertEqual(comp["completion"], "cde")

    def test_left_padded_batch_matches_single_prompt(self):
        """Plan pitfall: with right padding a short prompt would end in padding and go wrong."""
        model = make_scripted_lm(self.tokenizer, letter_chain(self.tokenizer, "bcdefghijk"))
        alone = generate_batch_hf(model, self.tokenizer, ["b"], max_new_tokens=20)[0][0]
        batched = generate_batch_hf(model, self.tokenizer, ["xyz qrs tuv", "b"], max_new_tokens=20)[1][0]
        self.assertEqual(alone["completion"], "cdefghijk")
        self.assertEqual(batched["completion"], alone["completion"])
        self.assertEqual(batched["completion_tokens"], alone["completion_tokens"])

    def test_sampled_runs_are_reproducible_with_seed(self):
        problems = [{"problem_id": f"p{i}", "question": f"w0{i} w11", "gold_answer": "1"} for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            texts = []
            for name in ("a", "b"):
                recs = run_generation_pipeline(
                    problems, Path(tmp) / f"{name}.jsonl", model=self.model, tokenizer=self.tokenizer,
                    preset="self_consistency", generation_config={"max_new_tokens": 6, "num_samples": 3},
                    prompt_formatter=lambda p: p["question"], stop_sequences=["zzz"], seed=7, batch_size=2,
                )
                texts.append([r["generated_texts"] for r in recs])
            self.assertEqual(texts[0], texts[1])
            self.assertEqual(len(texts[0][0]), 3)

            other = run_generation_pipeline(
                problems, Path(tmp) / "c.jsonl", model=self.model, tokenizer=self.tokenizer,
                preset="self_consistency", generation_config={"max_new_tokens": 6, "num_samples": 3},
                prompt_formatter=lambda p: p["question"], stop_sequences=["zzz"], seed=8, batch_size=2,
            )
            self.assertNotEqual([r["generated_texts"] for r in other], texts[0], "a different seed should sample differently")


class TestGenerationRunIdentity(unittest.TestCase):
    """A generations file belongs to one run; resuming with another run must not reuse it."""

    def _run(self, path, run_config, formatter=lambda p: p["question"]):
        return run_generation_pipeline(
            [{"problem_id": "p1", "question": "Q1", "gold_answer": "1"}], path,
            model=object(), tokenizer=object(), run_config=run_config, prompt_formatter=formatter, backend="hf",
        )

    def test_different_model_in_same_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen.jsonl"
            cfg_a = {"checkpoint": "model-a", "adapter_path": None, "prompt_format": "few_shot", "split": "val"}
            # First run: write meta, then pretend the generation finished
            from src.generation import check_or_write_run_meta

            check_or_write_run_meta(path, dict(cfg_a), has_records=False)
            meta = json.loads((Path(tmp) / "gen.meta.json").read_text())
            append_generation_records(path, [{"problem_id": "p1", "prompt": "Q1", "generated_text": "#### 1"}])

            # Same run config resumes; another model raises
            check_or_write_run_meta(path, dict(cfg_a), has_records=True)
            with self.assertRaises(GenerationRunMismatchError):
                check_or_write_run_meta(path, dict(cfg_a, checkpoint="model-b"), has_records=True)
            self.assertEqual(meta["run_config"]["checkpoint"], "model-a")

    def test_existing_file_without_meta_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen.jsonl"
            append_generation_records(path, [{"problem_id": "p1", "prompt": "Q1", "generated_text": "#### 1"}])
            with self.assertRaises(GenerationRunMismatchError):
                self._run(path, {"checkpoint": "m", "adapter_path": None, "prompt_format": "few_shot", "split": "val"})

    def test_stored_prompt_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen.jsonl"
            append_generation_records(path, [{"problem_id": "p1", "prompt": "OLD PROMPT", "generated_text": "#### 1"}])
            with self.assertRaises(GenerationRunMismatchError):
                self._run(path, None)


if __name__ == "__main__":
    unittest.main()
