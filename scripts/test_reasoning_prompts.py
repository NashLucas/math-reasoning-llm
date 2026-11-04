"""Verification script for Day 16: Reasoning Prompts (Phase 3).

Evaluates three reasoning prompt strategies on 10 sample problems with the development model:
1. Chain-of-Thought (CoT)
2. Solve-then-Verify
3. Alternative-Method

Verifies that all three prompt types produce the canonical GSM8K format ('#### <number>').
Supports:
- Real model generation via --use_real_model (e.g. Qwen/Qwen2.5-1.5B-Instruct on a Colab GPU; CPU works, slowly)
- Mock generation (default) to check the plumbing only. The mock writes '#### <gold>' itself, so a
  mock run says nothing about the prompts and is never reported as passing the Day 16 check.

Outputs:
- reports/day16_prompt_verification.json
- reports/day16_reasoning_prompts.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.math_utils.answer_extraction import extract_answer  # noqa: E402
from src.math_utils.normalization import is_equivalent  # noqa: E402
from src.reasoning.prompting import (  # noqa: E402
    ReasoningPromptType,
    get_reasoning_prompt_config,
    render_reasoning_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_PROMPT_TYPES = [
    ReasoningPromptType.CHAIN_OF_THOUGHT,
    ReasoningPromptType.SOLVE_THEN_VERIFY,
    ReasoningPromptType.ALTERNATIVE_METHOD,
]


def generate_mock_reasoning_completion(
    question: str,
    gold_answer: str,
    prompt_type: ReasoningPromptType,
) -> str:
    """Simulate model output adhering strictly to each prompt strategy (plumbing check only)."""
    gold_str = str(gold_answer).strip()

    if prompt_type == ReasoningPromptType.CHAIN_OF_THOUGHT:
        return (
            "Let's solve this step by step.\n"
            f"Given the problem: {question[:80]}...\n"
            "Step 1: Identify given quantities and relationships.\n"
            "Step 2: Compute the intermediate operations.\n"
            f"Step 3: Calculating final total yields {gold_str}.\n"
            f"#### {gold_str}"
        )
    elif prompt_type == ReasoningPromptType.SOLVE_THEN_VERIFY:
        return (
            "Stage 1 - Initial Solution:\n"
            f"Working through the problem step by step gives an initial result of {gold_str}.\n\n"
            "Stage 2 - Verification:\n"
            "Let us check this by reversing the arithmetic and verifying constraints.\n"
            "All conditions check out and confirm the initial calculation.\n"
            f"#### {gold_str}"
        )
    elif prompt_type == ReasoningPromptType.ALTERNATIVE_METHOD:
        return (
            "Alternative Strategy (Algebraic Formulation / Working Backwards):\n"
            "Instead of sequential arithmetic, let x represent the unknown quantity.\n"
            "Formulating the equation according to the problem constraints:\n"
            f"Solving for x directly gives x = {gold_str}.\n"
            f"#### {gold_str}"
        )
    else:
        return f"Step by step solution yields {gold_str}.\n#### {gold_str}"


def text_after_hash_line(raw_text: str) -> str:
    """Text the model wrote after its last '#### <answer>' line (empty if none).

    The generation engine cuts the completion right after the '####' line, so this is only
    visible in the raw output. Restating or explaining after the answer line is the main way
    a prompt fails the "on the last line" contract.
    """
    if not raw_text or "####" not in raw_text:
        return ""
    tail = raw_text[raw_text.rfind("####"):]
    newline = tail.find("\n")
    return tail[newline + 1:].strip() if newline != -1 else ""


def _resolve_dtype(dtype: str) -> str:
    if dtype != "auto":
        return dtype
    try:
        import torch

        return "bfloat16" if torch.cuda.is_available() else "float32"
    except ImportError:
        return "float32"


def evaluate_prompts_on_samples(
    sample_problems: List[Dict[str, Any]],
    use_real_model: bool = False,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    batch_size: int = 10,
    device: Optional[str] = None,
    dtype: str = "auto",
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Test all reasoning prompt types on sample problems and verify format compliance."""
    model = None
    tokenizer = None

    if use_real_model:
        resolved_dtype = _resolve_dtype(dtype)
        logger.info(f"Loading real development model: {model_name} ({resolved_dtype})")
        from src.model import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path=model_name,
            torch_dtype=resolved_dtype,
            device_map=device or "auto",
        )
        if not getattr(tokenizer, "chat_template", None):
            logger.warning(f"{model_name} has no chat template; the plain-text prompts are being tested.")

    results_by_type: Dict[str, Any] = {}

    for ptype in ALL_PROMPT_TYPES:
        config = get_reasoning_prompt_config(ptype)
        logger.info(f"--- Testing Prompt Strategy: {config.name} ({ptype.value}) ---")

        prompts = [
            render_reasoning_prompt(question=p["question"], prompt_type=ptype, tokenizer=tokenizer)
            for p in sample_problems
        ]

        outputs: List[Dict[str, Any]] = []
        if use_real_model:
            from src.generation import generate_batch_hf

            start = time.time()
            for i in range(0, len(prompts), batch_size):
                # Default stop rules: "Question:", "\n\n\n", and a cut after the '#### <number>' line.
                # '####' itself must NOT be a stop string: generation would end before the answer line.
                batch_out = generate_batch_hf(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts[i:i + batch_size],
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                )
                outputs.extend(item[0] for item in batch_out)
            logger.info(f"Generated {len(prompts)} completions in {time.time() - start:.1f}s")
        else:
            for prob in sample_problems:
                text = generate_mock_reasoning_completion(prob["question"], str(prob.get("gold_answer", "")), ptype)
                outputs.append({"completion": text, "completion_raw": text, "completion_tokens": None,
                                "finish_reason": "mock"})

        type_records = []
        compliant_count = 0
        correct_count = 0
        truncated_count = 0
        trailing_text_count = 0

        for prob, prompt_str, out in zip(sample_problems, prompts, outputs):
            gold = str(prob.get("gold_answer", ""))
            completion_text = out["completion"]
            raw_text = out.get("completion_raw", completion_text)

            parsed_answer, extraction_method = extract_answer(completion_text)
            is_compliant = extraction_method == "hash"
            is_match = is_equivalent(parsed_answer, gold) if parsed_answer else False
            trailing = text_after_hash_line(raw_text)

            compliant_count += int(is_compliant)
            correct_count += int(is_match)
            truncated_count += int(out.get("finish_reason") == "length")
            trailing_text_count += int(bool(trailing))

            type_records.append({
                "problem_id": prob.get("problem_id", "sample"),
                "question": prob["question"],
                "gold_answer": gold,
                "rendered_prompt": prompt_str,
                "completion": completion_text,
                "completion_raw": raw_text,
                "text_after_answer_line": trailing,
                "completion_tokens": out.get("completion_tokens"),
                "finish_reason": out.get("finish_reason"),
                "extracted_answer": parsed_answer,
                "extraction_method": extraction_method,
                "format_compliant": is_compliant,
                "is_correct": is_match,
            })

        total = len(sample_problems)
        compliance_rate = compliant_count / total if total > 0 else 0.0
        accuracy = correct_count / total if total > 0 else 0.0

        logger.info(
            f"Strategy [{config.name}]: Compliance = {compliant_count}/{total} "
            f"({compliance_rate * 100:.1f}%), Accuracy = {correct_count}/{total} ({accuracy * 100:.1f}%), "
            f"truncated = {truncated_count}, text after answer line = {trailing_text_count}"
        )

        results_by_type[ptype.value] = {
            "name": config.name,
            "description": config.description,
            "total_problems": total,
            "compliant_count": compliant_count,
            "format_compliance_rate": round(compliance_rate, 4),
            "format_compliance_percent": round(compliance_rate * 100.0, 2),
            "correct_count": correct_count,
            "accuracy": round(accuracy, 4),
            "accuracy_percent": round(accuracy * 100.0, 2),
            "truncated_count": truncated_count,
            "text_after_answer_line_count": trailing_text_count,
            "samples": type_records,
        }

    return results_by_type


def day16_verdict(results_by_type: Dict[str, Any], use_real_model: bool) -> str:
    """'PASSED', 'NEEDS WORDING ADJUSTMENT', or 'MOCK' (a mock run never passes the check)."""
    if not use_real_model:
        return "MOCK"
    all_compliant = all(r["format_compliance_rate"] >= 1.0 for r in results_by_type.values())
    return "PASSED" if all_compliant else "NEEDS WORDING ADJUSTMENT"


def generate_markdown_report(
    results_by_type: Dict[str, Any],
    model_name: str,
    use_real_model: bool,
    output_path: Path,
    device_label: str = "",
) -> str:
    """Generate markdown summary table and sample trace inspection for Day 16."""
    verdict = day16_verdict(results_by_type, use_real_model)
    mode = f"Real model{f' on {device_label}' if device_label else ''}" if use_real_model else "MOCK / simulation"
    lines = [
        "# Day 16 Reasoning Prompts Verification Report",
        "",
        f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- **Model Evaluated:** `{model_name}` ({mode})",
        "- **Decoding:** greedy, engine default stop rules",
        "- **Phase:** Phase 3 (Reasoning Data)",
        "- **Milestone:** Day 16 — Reasoning Prompts Verification",
        "",
    ]
    if not use_real_model:
        lines.extend([
            "> [!WARNING]",
            "> **Mock run.** The mock generator writes `#### <gold answer>` itself, so the numbers below are",
            "> 100% by construction and say nothing about the prompts. Rerun with `--use_real_model`.",
            "",
        ])

    lines.extend([
        "## Summary Results Table",
        "",
        "| Prompt Strategy | Type Key | Problems | Format Compliance | Accuracy | Truncated | Text after `####` |",
        "|---|---|---|---|---|---|---|",
    ])
    for key, data in results_by_type.items():
        lines.append(
            f"| {data['name']} | `{key}` | {data['total_problems']} | {data['format_compliance_percent']:.1f}% "
            f"| {data['accuracy_percent']:.1f}% | {data.get('truncated_count', 0)} "
            f"| {data.get('text_after_answer_line_count', 0)} |"
        )

    lines.extend([
        "",
        "Format compliance means the parser found the answer on a `####` line. Accuracy is only a sanity",
        "check here; the Day 16 check is about the format. \"Text after `####`\" counts completions where",
        "the model kept writing after the answer line (cut off by the engine, but a sign the prompt is not",
        "followed literally).",
        "",
        "## Non-compliant and Notable Completions",
        "",
    ])

    notable = 0
    for key, data in results_by_type.items():
        for s in data["samples"]:
            if s["format_compliant"] and not s["text_after_answer_line"] and s["finish_reason"] != "length":
                continue
            notable += 1
            lines.extend([
                f"### {data['name']}: `{s['problem_id']}`",
                f"- **Gold:** `{s['gold_answer']}` | **Extracted:** `{s['extracted_answer']}` "
                f"(`{s['extraction_method']}`) | **Finish:** `{s['finish_reason']}` "
                f"| **Tokens:** {s['completion_tokens']}",
                "",
                "```text",
                s["completion_raw"],
                "```",
                "",
            ])
    if notable == 0:
        lines.extend(["None: every completion ended on a `####` line with nothing after it.", ""])

    lines.extend(["## One Completion per Strategy", ""])
    for key, data in results_by_type.items():
        sample = data["samples"][0]
        lines.extend([
            f"### Strategy: {data['name']}",
            f"- **Problem ID:** `{sample['problem_id']}`",
            f"- **Question:** {sample['question']}",
            f"- **Gold Answer:** `{sample['gold_answer']}`",
            f"- **Extracted Answer:** `{sample['extracted_answer']}` (Method: `{sample['extraction_method']}`)",
            f"- **Compliant:** `{sample['format_compliant']}` | **Match:** `{sample['is_correct']}`",
            "",
            "```text",
            sample["completion"],
            "```",
            "",
        ])

    lines.extend(["## Day 16 Check", ""])
    if verdict == "PASSED":
        lines.extend([
            "**Verdict:** PASSED",
            "- All three reasoning prompt types produce the canonical `#### <number>` line on the sample problems.",
        ])
    elif verdict == "MOCK":
        lines.extend([
            "**Verdict:** NOT VERIFIED (mock run)",
            "- Day 16's done-when needs real development-model outputs.",
        ])
    else:
        lines.extend([
            "**Verdict:** NEEDS WORDING ADJUSTMENT",
            "- At least one prompt type missed the canonical `#### <number>` line; see the completions above.",
        ])

    content = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    logger.info(f"Saved Markdown report to: {output_path}")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Test reasoning prompt strategies on sample problems")
    parser.add_argument(
        "--sample_file",
        type=str,
        default="data/sample/train_sample.parquet",
        help="Path to sample problems file (default: data/sample/train_sample.parquet)",
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=10,
        help="Number of sample problems to test (default: 10)",
    )
    parser.add_argument(
        "--use_real_model",
        action="store_true",
        help="Run the real development model (e.g. Qwen/Qwen2.5-1.5B-Instruct); without it, a mock run",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model ID to evaluate (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype: auto (bfloat16 on GPU, float32 on CPU), bfloat16, float16 or float32",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Prompts per generate call (default: 10)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens per completion (default: 512, from the EDA)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Directory to save verification artifacts (default: reports)",
    )
    args = parser.parse_args()

    sample_path = Path(args.sample_file)
    if not sample_path.is_absolute():
        sample_path = PROJECT_ROOT / sample_path
    if not sample_path.exists():
        logger.error(f"Sample file {sample_path} not found.")
        sys.exit(1)

    df_sample = pd.read_parquet(sample_path)
    problems = df_sample.head(args.num_problems).to_dict(orient="records")
    logger.info(f"Loaded {len(problems)} sample problems from {sample_path}")

    device_label = ""
    if args.use_real_model:
        import torch

        device_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    results = evaluate_prompts_on_samples(
        sample_problems=problems,
        use_real_model=args.use_real_model,
        model_name=args.model_name,
        batch_size=args.batch_size,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    verdict = day16_verdict(results, args.use_real_model)
    json_path = out_dir / "day16_prompt_verification.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "mode": "real" if args.use_real_model else "mock",
            "device": device_label or None,
            "verdict": verdict,
            "results": results,
        }, f, indent=2)
    logger.info(f"Saved JSON verification results to: {json_path}")

    md_path = out_dir / "day16_reasoning_prompts.md"
    generate_markdown_report(
        results_by_type=results,
        model_name=args.model_name,
        use_real_model=args.use_real_model,
        output_path=md_path,
        device_label=device_label,
    )

    if verdict == "MOCK":
        logger.warning("Mock run: plumbing works, but the prompts were NOT verified. Use --use_real_model.")
    elif verdict == "PASSED":
        logger.info("Day 16 reasoning prompts verification passed.")
    else:
        logger.error("Some prompt types missed the canonical '#### <number>' line; adjust the wording.")
        sys.exit(1)


if __name__ == "__main__":
    main()


evaluate_prompts_on_samples.__test__ = False
