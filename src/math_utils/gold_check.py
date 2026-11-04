"""Phase 1 parser gate on the GSM8K gold solutions.

The gate has two parts that are checked independently of the parser's own string handling:

1. Every gold answer (the text after the last '####') is a plain number and parses to it.
2. The parser, run on the full gold solution, finds the '####' line and returns that number.

A third, harder measurement is recorded alongside: the parser on each gold solution with the
'####' line removed. That exercises the fallback tiers ('boxed', 'answer_is', 'last_number')
on real reasoning text, which is what model outputs without a clean '####' line go through.
It is a benchmark, not a 100% gate: some solutions end on a sentence whose last number is not
the answer ("... 5 hours to read 120 pages."), and no parser can recover those.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List

from src.math_utils.answer_extraction import extract_answer
from src.math_utils.normalization import is_equivalent, parse_numeric_value

# A gold answer must look like a number: optional minus, digits with optional thousands commas,
# optional decimal part. Anything else ("5 dollars", "1/2") is a data problem, not a parse problem.
GOLD_FORMAT = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")

# Regression floor for the no-'####' benchmark (97.6% after the Phase 1 review fixes).
NO_HASH_BENCHMARK_FLOOR = 0.97


def gold_answer_text(solution: str) -> str:
    return solution.split("####")[-1].strip()


def check_gold_solutions(solutions: Iterable[str], problem_ids: Iterable[str]) -> Dict[str, Any]:
    """Run the gate and the no-'####' benchmark. Returns counts and the first failures."""
    format_failures: List[Dict[str, Any]] = []
    parse_failures: List[Dict[str, Any]] = []
    benchmark_ok: Counter = Counter()
    benchmark_n: Counter = Counter()
    total = 0

    for pid, solution in zip(problem_ids, solutions):
        total += 1
        gold_text = gold_answer_text(solution)
        gold_value = parse_numeric_value(gold_text)

        # 1. The gold answer itself is a plain number
        if solution.count("####") != 1 or not GOLD_FORMAT.fullmatch(gold_text) or gold_value is None:
            format_failures.append({"problem_id": pid, "gold": gold_text})
            continue
        expected = float(gold_text.replace(",", ""))
        if gold_value != expected:
            format_failures.append({"problem_id": pid, "gold": gold_text, "parsed": gold_value})
            continue

        # 2. The parser finds the '####' line and the same number (numeric equality, no string fallback)
        extracted, method = extract_answer(solution)
        value = parse_numeric_value(extracted)
        if method != "hash" or value is None or value != expected:
            parse_failures.append({"problem_id": pid, "gold": gold_text, "extracted": extracted, "method": method})

        # 3. Benchmark: the same solution without its '####' line
        body = solution.split("####")[0]
        bench_answer, bench_method = extract_answer(body)
        benchmark_n[bench_method] += 1
        benchmark_ok[bench_method] += int(is_equivalent(bench_answer, expected))

    bench_correct = sum(benchmark_ok.values())
    bench_total = sum(benchmark_n.values())
    return {
        "total": total,
        "gold_format_failures": len(format_failures),
        "parser_failures": len(parse_failures),
        "gate_passed": total > 0 and not format_failures and not parse_failures,
        "no_hash_benchmark": {
            "correct": bench_correct,
            "total": bench_total,
            "accuracy": round(bench_correct / bench_total, 4) if bench_total else 0.0,
            "floor": NO_HASH_BENCHMARK_FLOOR,
            "by_method": {m: {"correct": benchmark_ok[m], "total": n} for m, n in sorted(benchmark_n.items())},
        },
        "examples": {"gold_format": format_failures[:5], "parser": parse_failures[:5]},
    }
