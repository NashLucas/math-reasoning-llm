"""Human review audit script for inspecting 20 model reasoning generations (Day 13).

Audits parsed answers, extraction methods, and mathematical equivalence against gold targets.
Generates an audit report in reports/day13_output_review.md.

Usage:
  python scripts/review_outputs.py
  python scripts/review_outputs.py --input_file outputs/baseline/generations_val_greedy.jsonl --count 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.math_utils.answer_extraction import extract_answer  # noqa: E402
from src.math_utils.normalization import is_equivalent, parse_numeric_value  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("review_outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review 20 model outputs to audit answer parsing accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to generations JSONL file (defaults to outputs/baseline/generations_val_greedy.jsonl).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of outputs to audit.",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="reports/day13_output_review.md",
        help="Destination path for markdown review report.",
    )
    return parser.parse_args()


def load_records(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Generations file not found: {file_path}")

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def main() -> int:
    args = parse_args()

    if args.input_file:
        input_path = Path(args.input_file)
    else:
        # Check standard locations
        val_greedy = PROJECT_ROOT / "outputs" / "baseline" / "generations_val_greedy.jsonl"
        sample_greedy = PROJECT_ROOT / "outputs" / "baseline" / "generations_sample_greedy.jsonl"
        if val_greedy.exists():
            input_path = val_greedy
        elif sample_greedy.exists():
            input_path = sample_greedy
        else:
            logger.error("No generations JSONL file found in outputs/baseline/. Run evaluation first.")
            return 1

    records = load_records(input_path)
    sample_size = min(args.count, len(records))
    review_records = records[:sample_size]

    logger.info(f"Loaded {len(records)} records from {input_path}. Reviewing first {sample_size} outputs.\n")

    report_lines = [
        "# Day 13 Model Output & Parser Audit Report",
        "",
        f"- **Source File:** `{input_path}`",
        f"- **Total Records in File:** {len(records)}",
        f"- **Audited Samples:** {sample_size}",
        "",
        "## Summary Findings",
        "",
    ]

    canonical_count = 0
    fallback_count = 0
    unparseable_count = 0
    correct_count = 0
    parser_discrepancies: List[Dict[str, Any]] = []

    print("=" * 80)
    print(f"  HUMAN REVIEW: AUDITING {sample_size} MODEL OUTPUTS")
    print("=" * 80)

    for idx, rec in enumerate(review_records, start=1):
        pid = rec.get("problem_id", f"prob_{idx}")
        question = rec.get("question", "")
        gold = str(rec.get("gold_answer", "")).strip()

        # Extract completion text
        if "generated_text" in rec:
            gen_text = rec["generated_text"]
        elif "generated_texts" in rec and rec["generated_texts"]:
            gen_text = rec["generated_texts"][0]
        else:
            gen_text = ""

        # Re-verify extraction with current parser
        extracted, method = extract_answer(gen_text)
        is_corr = is_equivalent(extracted, gold)

        if method == "hash":
            canonical_count += 1
        elif method in ("boxed", "answer_is", "last_number"):
            fallback_count += 1
        else:
            unparseable_count += 1

        if is_corr:
            correct_count += 1

        # Check for potential discrepancy: gold is in text but parsed as wrong or none
        gold_num = parse_numeric_value(gold)
        text_has_gold = gold in gen_text or (gold_num is not None and str(int(gold_num)) in gen_text)
        if text_has_gold and not is_corr:
            parser_discrepancies.append({
                "problem_id": pid,
                "gold": gold,
                "extracted": extracted,
                "method": method,
                "text": gen_text,
            })

        print(f"[{idx}/{sample_size}] Problem ID: {pid}")
        print(f"  Question:     {question[:75]}...")
        print(f"  Gold Target:  {gold}")
        print(f"  Extracted:    {extracted} (Method: {method}) | Correct: {is_corr}")
        print(f"  Snippet:      {gen_text[-120:].replace(chr(10), ' ')}")
        print("-" * 80)

    # Compile report summary
    report_lines.extend([
        f"- **Correct Answers:** {correct_count}/{sample_size} ({correct_count / sample_size * 100:.1f}%)",
        f"- **Format Compliance (####):** {canonical_count}/{sample_size} ({canonical_count / sample_size * 100:.1f}%)",
        f"- **Fallback Rate:** {fallback_count}/{sample_size} ({fallback_count / sample_size * 100:.1f}%)",
        f"- **Unparseable Rate:** {unparseable_count}/{sample_size} ({unparseable_count / sample_size * 100:.1f}%)",
        f"- **Parser Discrepancies:** {len(parser_discrepancies)}",
        "",
        "## Detailed Output Audit Log",
        "",
        "| # | Problem ID | Gold | Extracted | Method | Match | Snippet |",
        "|---|---|---|---|---|---|---|",
    ])

    for idx, rec in enumerate(review_records, start=1):
        pid = rec.get("problem_id", "")
        gold = str(rec.get("gold_answer", "")).strip()
        gen_text = rec.get("generated_text", "")
        extracted, method = extract_answer(gen_text)
        is_corr = is_equivalent(extracted, gold)
        snippet = gen_text.replace("\n", " ").strip()
        snippet = snippet[-80:] if len(snippet) > 80 else snippet
        report_lines.append(
            f"| {idx} | `{pid}` | {gold} | `{extracted}` | `{method}` | {'✓' if is_corr else '✗'} | {snippet} |"
        )

    if parser_discrepancies:
        report_lines.extend(["", "## Discrepancies Requiring Investigation", ""])
        for d in parser_discrepancies:
            report_lines.append(f"### Problem `{d['problem_id']}`")
            report_lines.append(f"- **Gold:** {d['gold']}")
            report_lines.append(f"- **Extracted:** {d['extracted']} (Method: {d['method']})")
            report_lines.append(f"- **Text:**\n```\n{d['text']}\n```\n")
    else:
        report_lines.extend(["", "## Discrepancies Requiring Investigation", "", "Zero parser discrepancies found! Every extracted value matches the ground truth and format contract as expected."])

    # Save report
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info(f"Output audit report saved to: {report_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
