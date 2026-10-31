"""Verification script for Day 11 Generation Engine.

Verifies:
1. Batched left-padded generation pipeline.
2. Chunked JSONL output persistence.
3. Resumability: rerunning skips already completed chunks.
4. Stop rules: line after '####' and 'Question:'.
5. Answer extraction and gold comparison on the 20 sample problems.

Supports both:
- Mock mode (default, for fast CI/CPU verification without downloading weights)
- Real model mode (--use_real_model, for Colab T4/G4 GPU verification with Qwen2.5-1.5B-Instruct)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.generation import (  # noqa: E402
    load_existing_generations,
    run_generation_pipeline,
    truncate_at_stop_sequences,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_mock_reasoning_generator(sample_problems: List[Dict[str, Any]]) -> MagicMock:
    """Create a mock generation function simulating realistic model responses with stop conditions."""
    def mock_generate_batch(
        model: Any,
        tokenizer: Any,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = False,
        stop_sequences: Any = None,
        num_return_sequences: int = 1,
    ) -> List[List[Dict[str, Any]]]:
        batch_results = []
        for prompt in prompts:
            simulated_completion = (
                "Let's think step by step.\n"
                "First, we calculate the initial total: 10 + 10 = 20.\n"
                "Then we add the remaining amount: 20 + 22 = 42.\n"
                "#### 42\n"
                "Now suppose someone asked another question...\n"
                "Question: Next problem"
            )

            clean_text = truncate_at_stop_sequences(simulated_completion, stop_sequences=stop_sequences)
            extracted = "42"

            completions = []
            for _ in range(num_return_sequences):
                completions.append({
                    "completion_raw": simulated_completion,
                    "completion": clean_text,
                    "extracted_answer": extracted,
                    "extraction_method": "hash",
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(simulated_completion.split()),
                })
            batch_results.append(completions)
        return batch_results

    return mock_generate_batch


def verify_with_real_model(
    problems: List[Dict[str, Any]],
    output_jsonl: Path,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    batch_size: int = 4,
) -> bool:
    """Run generation with real development model on GPU and test resumption."""
    from src.model import load_model_and_tokenizer

    logger.info(f"Loading real development model: '{model_name}'...")
    start_load = time.time()
    model, tokenizer = load_model_and_tokenizer(model_name_or_path=model_name)
    logger.info(f"Model and tokenizer loaded in {time.time() - start_load:.2f}s")

    # Run 1: Generate all 20 problems
    logger.info("Starting Run 1: Generating 20 problems with real model...")
    t0 = time.time()
    results_run1 = run_generation_pipeline(
        problems=problems,
        output_file=output_jsonl,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        preset="greedy",
    )
    duration_run1 = time.time() - t0
    logger.info(f"Run 1 completed in {duration_run1:.2f}s. Saved {len(results_run1)} problems to {output_jsonl}")

    # Inspect results
    correct_count = sum(1 for r in results_run1 if r.get("is_correct"))
    accuracy = (correct_count / len(results_run1)) * 100
    print("\n--- SAMPLE GENERATION RESULTS (Dev Model: Qwen2.5-1.5B-Instruct) ---")
    for idx, r in enumerate(results_run1[:3], start=1):
        print(f"\n[Problem {idx}] ID: {r['problem_id']}")
        print(f"Question: {r['question'][:120]}...")
        print(f"Gold: {r['gold_answer']} | Extracted: {r['extracted_answer']} | Correct: {r['is_correct']}")
        print(f"Generated text preview:\n{r['generated_text'][:180]}...")
    print(f"\nSample Accuracy (20 problems): {correct_count}/{len(results_run1)} ({accuracy:.1f}%)")

    # Run 2: Verify resumption skips finished chunks
    logger.info("Starting Run 2: Verifying chunk skipping on rerun...")
    t1 = time.time()
    results_run2 = run_generation_pipeline(
        problems=problems,
        output_file=output_jsonl,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        preset="greedy",
    )
    duration_run2 = time.time() - t1
    logger.info(f"Run 2 completed in {duration_run2:.4f}s.")

    assert len(results_run2) == len(problems), "Rerun results length mismatch"
    assert duration_run2 < 1.0, f"Rerun took {duration_run2:.2f}s; expected < 1s (must skip finished chunks)"
    logger.info("Rerun successfully skipped all 20 finished chunks!")

    print("\n" + "=" * 65)
    print("DAY 11 REAL-MODEL GPU VERIFICATION: ALL CHECKS PASSED")
    print(f"Model: {model_name}")
    print(f"Problems processed: {len(problems)}")
    print(f"Run 1 duration: {duration_run1:.2f}s")
    print(f"Run 2 duration (resumed): {duration_run2:.4f}s (skipped 20/20)")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Output file: {output_jsonl}")
    print("=" * 65)
    return True


def verify_with_mock_model(
    problems: List[Dict[str, Any]],
    output_jsonl: Path,
    batch_size: int = 4,
) -> bool:
    """Run generation with mock reasoning generator (for fast local/CI testing)."""
    # 1. Test Stop Rule Truncation
    test_ramble = "5 * 5 = 25\n#### 25\nThis line should be truncated!\nQuestion: Next?"
    truncated = truncate_at_stop_sequences(test_ramble)
    assert truncated == "5 * 5 = 25\n#### 25", f"Stop rule failed: '{truncated}'"
    logger.info("Stop rule verification: PASSED (rambling after #### was cleanly truncated)")

    # 2. First Run: Generate all 20 problems
    logger.info("Starting Run 1: Generating 20 problems...")
    mock_generator = create_mock_reasoning_generator(problems)

    with unittest_mock_generate_batch(mock_generator):
        results_run1 = run_generation_pipeline(
            problems=problems,
            output_file=output_jsonl,
            model=MagicMock(),
            tokenizer=MagicMock(),
            batch_size=batch_size,
            preset="greedy",
        )

    assert len(results_run1) == len(problems), f"Expected {len(problems)} results, got {len(results_run1)}"
    assert output_jsonl.exists(), "Output JSONL file was not created"

    # Verify JSONL lines on disk
    saved_records = load_existing_generations(output_jsonl)
    assert len(saved_records) == len(problems), f"Expected {len(problems)} saved in JSONL, got {len(saved_records)}"
    logger.info(f"Run 1 completed: {len(results_run1)} problems saved to {output_jsonl}")

    # 3. Second Run (Resumption): Rerun on the same file
    logger.info("Starting Run 2: Verifying chunk skipping on rerun...")
    mock_generator_run2 = MagicMock()

    with unittest_mock_generate_batch(mock_generator_run2):
        results_run2 = run_generation_pipeline(
            problems=problems,
            output_file=output_jsonl,
            model=MagicMock(),
            tokenizer=MagicMock(),
            batch_size=batch_size,
            preset="greedy",
        )

    assert mock_generator_run2.call_count == 0, "Generator was called on rerun despite all problems being completed!"
    assert len(results_run2) == len(problems), f"Expected {len(problems)} results on rerun, got {len(results_run2)}"
    logger.info("Run 2 completed: 0 problems re-generated. Rerun successfully skipped all finished chunks!")

    # 4. Partial Resumption Test: Delete last 5 records and verify it generates only the missing 5
    logger.info("Starting Run 3: Testing partial resumption...")
    all_lines = output_jsonl.read_text(encoding="utf-8").strip().split("\n")
    output_jsonl.write_text("\n".join(all_lines[:15]) + "\n", encoding="utf-8")
    assert len(load_existing_generations(output_jsonl)) == 15

    call_count = {"count": 0}

    def counting_generator(*args: Any, **kwargs: Any) -> Any:
        call_count["count"] += len(kwargs.get("prompts", []))
        return mock_generator(*args, **kwargs)

    with unittest_mock_generate_batch(counting_generator):
        results_run3 = run_generation_pipeline(
            problems=problems,
            output_file=output_jsonl,
            model=MagicMock(),
            tokenizer=MagicMock(),
            batch_size=batch_size,
            preset="greedy",
        )

    assert call_count["count"] == 5, f"Expected exactly 5 missing problems to be generated, got {call_count['count']}"
    assert len(results_run3) == 20
    assert len(load_existing_generations(output_jsonl)) == 20
    logger.info("Run 3 completed: Exactly 5 missing problems were resumed and appended!")

    print("\n" + "=" * 65)
    print("DAY 11 GENERATION ENGINE MOCK VERIFICATION: ALL CHECKS PASSED")
    print(f"Sample problems processed: {len(problems)}")
    print(f"Output file: {output_jsonl}")
    print("Resumable chunked JSONL: Verified (Run 2 skipped 20/20, Run 3 skipped 15/20)")
    print("=" * 65)
    return True


def verify_generation_engine(
    sample_file: str = "data/sample/val_sample.parquet",
    output_dir: str = "outputs/dry_run_generation",
    use_real_model: bool = False,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    batch_size: int = 4,
) -> bool:
    """Run verification of generation pipeline and resumption on 20 sample problems."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = out_dir / "generations.jsonl"

    if output_jsonl.exists():
        output_jsonl.unlink()

    sample_path = Path(sample_file)
    if not sample_path.exists():
        logger.error(f"Sample file {sample_path} not found. Run scripts/create_sample_sets.py first.")
        return False

    df_sample = pd.read_parquet(sample_path)
    problems = df_sample.to_dict(orient="records")
    logger.info(f"Loaded {len(problems)} sample problems from {sample_path}")

    if use_real_model:
        return verify_with_real_model(
            problems=problems,
            output_jsonl=output_jsonl,
            model_name=model_name,
            batch_size=batch_size,
        )
    else:
        return verify_with_mock_model(
            problems=problems,
            output_jsonl=output_jsonl,
            batch_size=batch_size,
        )


class unittest_mock_generate_batch:
    """Context manager to patch generate_batch_hf."""
    def __init__(self, target_fn: Any) -> None:
        self.target_fn = target_fn
        self.patcher = None

    def __enter__(self) -> Any:
        from unittest.mock import patch
        self.patcher = patch("src.generation.generate_batch_hf", side_effect=self.target_fn)
        return self.patcher.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.patcher:
            self.patcher.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test generation engine and resumability")
    parser.add_argument(
        "--sample_file",
        type=str,
        default="data/sample/val_sample.parquet",
        help="Path to sample problems file (default: data/sample/val_sample.parquet)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/dry_run_generation",
        help="Output directory (default: outputs/dry_run_generation)",
    )
    parser.add_argument(
        "--use_real_model",
        action="store_true",
        help="Run using real development model on GPU (e.g. on Google Colab T4)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model ID to use with --use_real_model (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation (default: 4)",
    )
    args = parser.parse_args()

    success = verify_generation_engine(
        sample_file=args.sample_file,
        output_dir=args.output_dir,
        use_real_model=args.use_real_model,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
