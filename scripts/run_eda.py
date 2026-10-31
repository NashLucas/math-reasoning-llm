"""
Exploratory Data Analysis (EDA) on GSM8K dataset splits with DeepSeek tokenizer.

Computes:
1. Token lengths for questions, solutions, and combined sequences.
2. Percentiles (min, median, 90th, 95th, 99th, max) for sequence budgeting.
3. Distribution of solution reasoning step counts.
4. Answer type distributions (integer, decimal, negative) and magnitude ranges.
5. 8-shot few-shot prompt length analysis.
6. Recommended limits for `max_seq_length` and `max_new_tokens`.

Outputs summary to stdout and writes JSON report to reports/eda_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src.data.formatting import format_few_shot_prompt, format_zero_shot_prompt  # noqa: E402
from src.data.preprocessing import (  # noqa: E402
    PROCESSED_DATA_DIR,
    extract_gold_answer,
    prepare_all_data,
)
from src.math_utils.normalization import parse_numeric_value  # noqa: E402


def compute_distribution_stats(values: List[int | float]) -> Dict[str, float]:
    """Compute summary statistics and percentiles for a numerical list."""
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "mean": round(float(np.mean(arr)), 2),
        "std": round(float(np.std(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p75": round(float(np.percentile(arr, 75)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "max": float(np.max(arr)),
    }


def get_solution_text(row: pd.Series) -> str:
    """Extract solution text from row, preferring cleaned_solution over answer/solution."""
    for col in ["cleaned_solution", "solution", "answer"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return ""


def analyze_answers(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze answer types: integers, decimals, negatives, and value distributions."""
    answers: List[float] = []
    int_count = 0
    decimal_count = 0
    negative_count = 0

    for _, row in df.iterrows():
        # Use gold_answer if pre-extracted, else extract from solution/answer
        if "gold_answer" in row and pd.notna(row["gold_answer"]):
            parsed = str(row["gold_answer"]).strip()
        else:
            sol = get_solution_text(row)
            parsed = extract_gold_answer(sol)

        if parsed is not None:
            num = parse_numeric_value(parsed)
            if num is not None:
                answers.append(num)
                if num < 0:
                    negative_count += 1
                if num.is_integer():
                    int_count += 1
                else:
                    decimal_count += 1

    total = len(answers)
    stats = compute_distribution_stats(answers) if answers else {}

    return {
        "total_parsed": total,
        "integer_count": int_count,
        "integer_pct": round(int_count / total * 100, 2) if total else 0.0,
        "decimal_count": decimal_count,
        "decimal_pct": round(decimal_count / total * 100, 2) if total else 0.0,
        "negative_count": negative_count,
        "negative_pct": round(negative_count / total * 100, 2) if total else 0.0,
        "value_stats": stats,
    }


def analyze_steps(df: pd.DataFrame) -> Dict[str, Any]:
    """Count reasoning steps in solutions (lines before #### final answer)."""
    step_counts: List[int] = []

    for _, row in df.iterrows():
        sol = get_solution_text(row)
        # Split before #### delimiter
        reasoning = sol.split("####")[0].strip()
        lines = [line.strip() for line in reasoning.split("\n") if line.strip()]
        step_counts.append(len(lines))

    return compute_distribution_stats(step_counts)


def compute_tokens_by_split(
    splits: Dict[str, pd.DataFrame],
    tokenizer: Any,
) -> Dict[str, Dict[str, Any]]:
    """Compute token lengths across splits efficiently without redundant tokenization."""
    results: Dict[str, Dict[str, Any]] = {}
    all_q: List[int] = []
    all_s: List[int] = []
    all_c: List[int] = []

    for split_name in ["train", "val", "test"]:
        df = splits[split_name]
        q_lens: List[int] = []
        s_lens: List[int] = []
        c_lens: List[int] = []

        for _, row in df.iterrows():
            q_text = str(row["question"])
            s_text = get_solution_text(row)

            q_tok = len(tokenizer.encode(q_text, add_special_tokens=False))
            s_tok = len(tokenizer.encode(s_text, add_special_tokens=False))

            zero_shot = format_zero_shot_prompt(q_text)
            full_text = f"{zero_shot}\n{s_text}"
            c_tok = len(tokenizer.encode(full_text, add_special_tokens=False))

            q_lens.append(q_tok)
            s_lens.append(s_tok)
            c_lens.append(c_tok)

        results[split_name] = {
            "question_tokens": compute_distribution_stats(q_lens),
            "solution_tokens": compute_distribution_stats(s_lens),
            "combined_tokens": compute_distribution_stats(c_lens),
        }
        all_q.extend(q_lens)
        all_s.extend(s_lens)
        all_c.extend(c_lens)

    results["all"] = {
        "question_tokens": compute_distribution_stats(all_q),
        "solution_tokens": compute_distribution_stats(all_s),
        "combined_tokens": compute_distribution_stats(all_c),
    }
    return results


def analyze_few_shot_prompt_length(df: pd.DataFrame, tokenizer: Any, num_samples: int = 50) -> Dict[str, Any]:
    """Analyze token lengths for the 8-shot few-shot prompt."""
    few_shot_lengths: List[int] = []

    for i in range(min(num_samples, len(df))):
        q_text = str(df.iloc[i]["question"])
        few_shot_text = format_few_shot_prompt(q_text)
        length = len(tokenizer.encode(few_shot_text, add_special_tokens=False))
        few_shot_lengths.append(length)

    return compute_distribution_stats(few_shot_lengths)


def run_eda(
    model_name: str = "deepseek-ai/deepseek-math-7b-base",
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """Execute complete EDA pipeline across all GSM8K splits."""
    print("=" * 70)
    print("STARTING GSM8K EXPLORATORY DATA ANALYSIS (EDA)")
    print(f"Tokenizer: {model_name}")
    print("=" * 70)

    # 1. Load splits
    processed_dir = PROCESSED_DATA_DIR
    train_file = processed_dir / "train" / "train.parquet"
    if not train_file.exists():
        print("Processed splits not found on disk. Preparing splits...")
        prepare_all_data(seed=42)

    print(f"Loading processed splits from {processed_dir}...")
    splits = {
        "train": pd.read_parquet(processed_dir / "train" / "train.parquet"),
        "val_analysis": pd.read_parquet(processed_dir / "val" / "val_analysis.parquet"),
        "val_gating": pd.read_parquet(processed_dir / "val" / "val_gating.parquet"),
        "val": pd.read_parquet(processed_dir / "val" / "val.parquet"),
        "test": pd.read_parquet(processed_dir / "test" / "test.parquet"),
    }

    all_data = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    print(f"Total problems: {len(all_data)} (Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])})")

    # 2. Load tokenizer
    print(f"Loading tokenizer '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Step counts
    print("\n[1/4] Analyzing reasoning step counts...")
    step_stats_all = analyze_steps(all_data)
    step_stats_train = analyze_steps(splits["train"])
    step_stats_test = analyze_steps(splits["test"])

    # 4. Answer types
    print("[2/4] Analyzing answer types...")
    answer_stats_all = analyze_answers(all_data)
    answer_stats_train = analyze_answers(splits["train"])
    answer_stats_test = analyze_answers(splits["test"])

    # 5. Token distributions
    print("[3/4] Analyzing token length distributions (Question, Solution, Combined)...")
    token_stats_by_split = compute_tokens_by_split(splits, tokenizer)
    token_stats_all = token_stats_by_split["all"]
    token_stats_train = token_stats_by_split["train"]
    token_stats_test = token_stats_by_split["test"]

    # 6. Few-shot prompt length
    print("[4/4] Analyzing 8-shot few-shot prompt token lengths...")
    few_shot_stats = analyze_few_shot_prompt_length(splits["test"], tokenizer, num_samples=50)

    # Sequence recommendations
    # SFT sequence limit needs to cover 99% of zero-shot prompt + solution
    # Generation max_new_tokens needs to cover 99% of solution tokens
    comb_p99 = token_stats_all["combined_tokens"]["p99"]
    sol_p99 = token_stats_all["solution_tokens"]["p99"]
    q_p99 = token_stats_all["question_tokens"]["p99"]

    # Recommended powers-of-two or standard LLM context boundaries:
    recommended_max_seq_length = 512 if comb_p99 <= 512 else 1024
    recommended_max_new_tokens = 512 if sol_p99 <= 512 else 768

    summary: Dict[str, Any] = {
        "dataset": "openai/gsm8k",
        "tokenizer": model_name,
        "counts": {
            "total": len(all_data),
            "train": len(splits["train"]),
            "val_total": len(splits["val"]),
            "val_analysis": len(splits["val_analysis"]),
            "val_gating": len(splits["val_gating"]),
            "test": len(splits["test"]),
        },
        "reasoning_steps": {
            "all": step_stats_all,
            "train": step_stats_train,
            "test": step_stats_test,
        },
        "answer_types": {
            "all": answer_stats_all,
            "train": answer_stats_train,
            "test": answer_stats_test,
        },
        "token_lengths": {
            "all": token_stats_all,
            "train": token_stats_train,
            "test": token_stats_test,
        },
        "few_shot_prompt_tokens": few_shot_stats,
        "recommendations": {
            "question_p99": q_p99,
            "solution_p99": sol_p99,
            "combined_p99": comb_p99,
            "max_seq_length": recommended_max_seq_length,
            "max_new_tokens": recommended_max_new_tokens,
            "reasoning": (
                f"99% of GSM8K solutions fit within {int(sol_p99)} tokens. "
                f"99% of zero-shot prompt+solution pairs fit within {int(comb_p99)} tokens. "
                f"Setting max_seq_length={recommended_max_seq_length} and "
                f"max_new_tokens={recommended_max_new_tokens} provides safety margins "
                f"while fitting comfortably within memory budgets on T4/A10/G4 runtimes."
            ),
        },
    }

    # Save summary
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_file = out_path / "eda_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReport successfully written to {summary_file}")

    # Print summary tables
    print("\n" + "=" * 70)
    print("EDA SUMMARY FINDINGS")
    print("=" * 70)
    print("\n--- TOKEN LENGTHS (All 8,792 Problems) ---")
    print(f"{'Category':<18} | {'Min':<6} | {'Median':<6} | {'P90':<6} | {'P95':<6} | {'P99':<6} | {'Max':<6}")
    print("-" * 70)
    for cat, key in [
        ("Question", "question_tokens"),
        ("Solution", "solution_tokens"),
        ("Prompt+Solution", "combined_tokens"),
    ]:
        s = token_stats_all[key]
        print(f"{cat:<18} | {int(s['min']):<6} | {int(s['p50']):<6} | {int(s['p90']):<6} | {int(s['p95']):<6} | {int(s['p99']):<6} | {int(s['max']):<6}")

    print("\n--- 8-SHOT FEW-SHOT PROMPT TOKENS ---")
    fs = few_shot_stats
    print(f"Few-Shot (8-shot) | Min: {int(fs['min'])}, Median: {int(fs['p50'])}, P95: {int(fs['p95'])}, Max: {int(fs['max'])}")

    print("\n--- ANSWER TYPE BREAKDOWN (All 8,792 Problems) ---")
    ans = answer_stats_all
    print(f"Total Parsed: {ans['total_parsed']}")
    print(f"Integers:     {ans['integer_count']:<5} ({ans['integer_pct']}%)")
    print(f"Decimals:     {ans['decimal_count']:<5} ({ans['decimal_pct']}%)")
    print(f"Negative:     {ans['negative_count']:<5} ({ans['negative_pct']}%)")

    print("\n--- REASONING STEPS (Lines before '####') ---")
    st = step_stats_all
    print(f"Steps: Min: {int(st['min'])}, Mean: {st['mean']}, Median: {int(st['p50'])}, P95: {int(st['p95'])}, Max: {int(st['max'])}")

    print("\n--- SEQUENCE BUDGET RECOMMENDATIONS ---")
    print(f"max_seq_length: {recommended_max_seq_length}")
    print(f"max_new_tokens: {recommended_max_new_tokens}")
    print(f"Reasoning: {summary['recommendations']['reasoning']}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run_eda()
