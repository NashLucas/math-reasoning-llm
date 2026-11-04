"""Phase 1 parser gate: run the answer parser on all 8,792 GSM8K gold solutions.

Gate (must be 8,792 / 8,792):
  1. every gold answer after '####' is a plain number and parses to that number;
  2. the parser, on the full solution, uses the '####' line and returns that number
     (compared numerically, so an unparseable answer cannot pass by string equality).

Benchmark (recorded, regression floor in src/math_utils/gold_check.py):
  3. the parser on each solution with the '####' line removed, i.e. the fallback tiers on
     real reasoning text.

Writes reports/parser_gold_check.json. Exit code 1 if the gate fails or the benchmark drops
below its floor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.loader import load_raw_gsm8k  # noqa: E402
from src.math_utils.gold_check import check_gold_solutions  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "reports" / "parser_gold_check.json"


def main() -> int:
    raw = pd.concat([load_raw_gsm8k("train"), load_raw_gsm8k("test")], ignore_index=True)
    if len(raw) != 8792:
        print(f"Expected 8,792 gold solutions, found {len(raw)}")
        return 1

    result = check_gold_solutions(raw["answer"], raw["problem_id"])
    bench = result["no_hash_benchmark"]

    print(f"Gold solutions:                 {result['total']}")
    print(f"Gold answers not a plain number: {result['gold_format_failures']}")
    print(f"Parser wrong on full solution:   {result['parser_failures']}")
    print(f"No-#### benchmark:               {bench['correct']}/{bench['total']} ({bench['accuracy']:.2%}), floor {bench['floor']:.0%}")
    for method, c in bench["by_method"].items():
        print(f"  {method:<12} {c['correct']}/{c['total']}")
    for kind, examples in result["examples"].items():
        for ex in examples:
            print(f"  [{kind}] {ex}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {REPORT_PATH}")

    passed = result["gate_passed"] and bench["accuracy"] >= bench["floor"]
    print(f"PHASE 1 PARSER GATE: {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
