"""Verification script for Day 8.

Tests extract_answer and is_equivalent across all 8,792 raw gold solutions
(7,473 train + 1,319 test) to verify 100% extraction accuracy:
'Done when: the parser is correct on 8,792 of 8,792 gold solutions.'
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_raw_gsm8k  # noqa: E402
from src.math_utils.answer_extraction import extract_answer  # noqa: E402
from src.math_utils.normalization import is_equivalent  # noqa: E402


def verify_gold_solutions() -> None:
    print("Loading raw GSM8K datasets...")
    train_df = load_raw_gsm8k("train")
    test_df = load_raw_gsm8k("test")

    total_problems = len(train_df) + len(test_df)
    assert total_problems == 8792, f"Expected 8,792 total problems, found {total_problems}"

    print(f"Testing answer parser on {total_problems} gold solutions...")

    failures = []
    methods_count = {}

    for split_name, df in [("train", train_df), ("test", test_df)]:
        for idx, row in df.iterrows():
            solution = row["answer"]
            problem_id = row["problem_id"]

            extracted_str, method = extract_answer(solution)
            methods_count[method] = methods_count.get(method, 0) + 1

            # Expected gold answer from last line of solution after '####'
            gold_line = solution.split("####")[-1].strip()

            # Compare extracted vs gold
            if not is_equivalent(extracted_str, gold_line):
                failures.append(
                    {
                        "split": split_name,
                        "problem_id": problem_id,
                        "extracted": extracted_str,
                        "gold": gold_line,
                        "solution": solution,
                        "method": method,
                    }
                )

    print("\nExtraction Method Distribution:")
    for m, count in sorted(methods_count.items(), key=lambda x: -x[1]):
        pct = (count / total_problems) * 100
        print(f"  {m:<14}: {count:>5} / {total_problems} ({pct:.2f}%)")

    if failures:
        print(f"\nFAILED: {len(failures)} extraction failures found!")
        for f in failures[:5]:
            print(f"[{f['problem_id']}] Extracted: '{f['extracted']}' | Gold: '{f['gold']}'")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DAY 8 GATE MET: 100% ACCURACY ON GOLD SOLUTIONS!")
    print(f"Correct: {total_problems} / {total_problems} (100.00%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    verify_gold_solutions()
