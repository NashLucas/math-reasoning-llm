"""Safety evaluation with Llama Guard 3 8B on the data/safety/ prompt set (Day 15, reused in Phase 7).

Two passes, so only one model is on the GPU at a time:
  1. Generate a greedy response to every safety prompt with the model under test, then unload it.
  2. Load Llama Guard and screen every prompt and every (prompt, response) pair.

Reports per label (harmful / benign):
  - prompt_flagged_rate: share of prompts Llama Guard flags (a sanity check on the classifier)
  - unsafe_response_rate: share of responses Llama Guard flags
  - refusal_rate: share of responses that refuse (benign refusal rate is the over-refusal metric)

Memory: Llama Guard 3 8B needs ~16 GB in bf16, so use an L4/G4/A100, not a T4.

Usage:
  # Development model on a cheap GPU (Day 15):
  python scripts/run_safety_eval.py

  # A 7B reference model or trained adapter (Phase 7):
  python scripts/run_safety_eval.py --model deepseek-ai/deepseek-math-7b-base --prompt_format zero_shot
  python scripts/run_safety_eval.py --model deepseek-ai/deepseek-math-7b-base \
      --adapter_path <outputs_dir>/grpo/adapter_model --stage final_eval

Outputs go to the stage's outputs folder (Google Drive under the colab profile).

  # CPU / CI (no weights):
  python scripts/run_safety_eval.py --use_mock
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager  # noqa: E402
from src.data.formatting import apply_chat_template, format_chat_messages, format_zero_shot_prompt  # noqa: E402
from src.data.safety_data import load_safety_prompt_set  # noqa: E402
from src.evaluation.safety import (  # noqa: E402
    LlamaGuardAuditor,
    SafetyAssessment,
    compute_safety_metrics,
    is_refusal,
)
from src.run_info import save_run_info, set_seed  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("safety_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate responses to the safety prompt set and screen them with Llama Guard 3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=None, help="Model under test (defaults to dev_model).")
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional LoRA adapter.")
    parser.add_argument("--guard_model", type=str, default=None, help="Safety classifier (defaults to models.safety).")
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="auto",
        choices=["auto", "chat", "zero_shot"],
        help="'auto' uses the chat template when the tokenizer has one, otherwise zero-shot.",
    )
    parser.add_argument(
        "--safety_data",
        type=str,
        default=str(ConfigManager().data_path("safety", "safety_prompts.jsonl")),
        help="Safety prompt set from scripts/prepare_safety_data.py.",
    )
    parser.add_argument("--stage", type=str, default="baseline", help="Stage folder under outputs/.")
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to outputs/<stage>/safety/<model_slug>.")
    parser.add_argument("--limit", type=int, default=None, help="Optional prompt limit for dry runs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation and screening.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Response token budget.")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Load models in 4-bit NF4 quantization.")
    parser.add_argument(
        "--strict_guard",
        action="store_true",
        default=False,
        help="Disable heuristic fallback when Llama Guard 3 8B is gated/inaccessible on Hugging Face.",
    )
    parser.add_argument("--use_mock", action="store_true", default=False, help="Mock generation and mock classifier.")
    return parser.parse_args()


def get_model_slug(model_name: str, adapter_path: Optional[str]) -> str:
    slug = model_name.rstrip("/").split("/")[-1]
    if adapter_path:
        slug = f"{slug}+{Path(adapter_path).parent.name or Path(adapter_path).name}"
    return slug


def mock_response(record: Dict[str, Any]) -> str:
    """Canned responses: refuse harmful prompts, solve benign ones."""
    if record["label"] == "harmful":
        return "I'm sorry, but I can't help with that request."
    gold = record.get("gold_answer") or "0"
    return f"Let's solve this step by step.\nThe result is {gold}.\n#### {gold}"


def generate_responses(
    records: List[Dict[str, Any]],
    model_name: str,
    adapter_path: Optional[str],
    prompt_format: str,
    batch_size: int,
    max_new_tokens: int,
    stop_sequences: List[str],
    hf_cache_dir: Optional[str],
    load_in_4bit: bool = False,
    cache_path: Optional[Path] = None,
) -> List[str]:
    """Greedy-generate one response per safety prompt (resuming from cache_path if present), then free the model."""
    cached_by_id: Dict[str, str] = {}
    if cache_path and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "safety_id" in item and "response" in item:
                        cached_by_id[item["safety_id"]] = item["response"]
        if all(r["safety_id"] in cached_by_id for r in records):
            logger.info(f"Loaded all {len(records)} cached safety responses from {cache_path}")
            return [cached_by_id[r["safety_id"]] for r in records]

    import gc

    import torch

    from src.generation import generate_batch_hf
    from src.model import load_model_and_tokenizer

    load_kwargs: Dict[str, Any] = {}
    if load_in_4bit:
        load_kwargs["load_in_4bit"] = True

    model, tokenizer = load_model_and_tokenizer(
        model_name, adapter_path=adapter_path, hf_cache_dir=hf_cache_dir, **load_kwargs
    )

    use_chat = prompt_format == "chat" or (
        prompt_format == "auto" and getattr(tokenizer, "chat_template", None)
    )
    logger.info(f"Prompt format: {'chat template' if use_chat else 'zero-shot'}")

    def render(prompt: str) -> str:
        if use_chat:
            return apply_chat_template(format_chat_messages(prompt), tokenizer)
        return format_zero_shot_prompt(prompt)

    pending = [r for r in records if r["safety_id"] not in cached_by_id]
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        outputs = generate_batch_hf(
            model,
            tokenizer,
            [render(r["prompt"]) for r in chunk],
            max_new_tokens=max_new_tokens,
            stop_sequences=None if use_chat else stop_sequences,
        )
        for rec, out in zip(chunk, outputs):
            resp_text = out[0]["completion"]
            cached_by_id[rec["safety_id"]] = resp_text
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"safety_id": rec["safety_id"], "response": resp_text}, ensure_ascii=False) + "\n")
        logger.info(f"Generated {len(cached_by_id)}/{len(records)} responses")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [cached_by_id[r["safety_id"]] for r in records]


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"count": 0}
    prompt_metrics = compute_safety_metrics([r["prompt_assessment"] for r in rows])
    response_metrics = compute_safety_metrics([r["response_assessment"] for r in rows])
    refusals = sum(1 for r in rows if r["refused"])
    return {
        "count": n,
        "prompt_flagged_rate": prompt_metrics["unsafe_rate"],
        "prompt_unknown_rate": prompt_metrics["unknown_rate"],
        "unsafe_response_rate": response_metrics["unsafe_rate"],
        "response_unknown_rate": response_metrics["unknown_rate"],
        "refusal_rate": round(refusals / n, 4),
        "response_category_breakdown": response_metrics["category_breakdown"],
    }


def main() -> int:
    args = parse_args()
    set_seed(42)

    cfg = ConfigManager()
    models_cfg = cfg.base_config.get("models", {})
    model_name = args.model or models_cfg.get("dev_model", "Qwen/Qwen2.5-1.5B-Instruct")
    guard_name = args.guard_model or models_cfg.get("safety", "meta-llama/Llama-Guard-3-8B")
    hf_cache_dir = str(cfg.get_profile().get("hf_cache_dir")) if not args.use_mock else None
    stop_sequences = cfg.load_config("generation.yaml").get("stop_sequences", [])

    data_path = Path(args.safety_data)
    if not data_path.exists():
        local_safety = PROJECT_ROOT / "data" / "safety" / "safety_prompts.jsonl"
        if local_safety.exists():
            data_path = local_safety
        else:
            logger.error(f"Safety prompt set not found: {data_path}. Run scripts/prepare_safety_data.py first.")
            return 1
    records = load_safety_prompt_set(data_path)
    if args.limit:
        # Keep both labels represented in dry runs
        harmful = [r for r in records if r["label"] == "harmful"][: (args.limit + 1) // 2]
        benign = [r for r in records if r["label"] == "benign"][: args.limit // 2]
        records = harmful + benign

    slug = get_model_slug(model_name, args.adapter_path)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else cfg.get_stage_paths(args.stage)["outputs"] / "safety" / slug
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info(f"Safety evaluation | model={model_name} adapter={args.adapter_path} guard={guard_name}")
    logger.info(f"Prompts: {len(records)} | mock={args.use_mock} | output={out_dir}")
    logger.info("=" * 72)

    # Pass 1: responses from the model under test
    if args.use_mock:
        responses = [mock_response(r) for r in records]
    else:
        responses = generate_responses(
            records,
            model_name,
            args.adapter_path,
            args.prompt_format,
            args.batch_size,
            args.max_new_tokens,
            stop_sequences,
            hf_cache_dir,
            load_in_4bit=args.load_in_4bit,
            cache_path=out_dir / "safety_generations.jsonl",
        )

    # Pass 2: Llama Guard screening
    auditor = LlamaGuardAuditor(
        model_name_or_path=guard_name,
        use_mock=args.use_mock,
        batch_size=args.batch_size,
        hf_cache_dir=hf_cache_dir,
        load_in_4bit=args.load_in_4bit,
        allow_heuristic_fallback=not args.strict_guard,
    )
    prompt_assessments: List[SafetyAssessment] = auditor.assess_prompt_batch([r["prompt"] for r in records])
    response_assessments: List[SafetyAssessment] = auditor.assess_response_batch(
        [(r["prompt"], resp) for r, resp in zip(records, responses)]
    )
    guard_mode = (
        "mock"
        if args.use_mock
        else ("heuristic_fallback_gated_model" if auditor.fallback_used else "llama_guard")
    )
    auditor.unload()

    rows: List[Dict[str, Any]] = []
    for rec, resp, pa, ra in zip(records, responses, prompt_assessments, response_assessments):
        rows.append({
            "safety_id": rec["safety_id"],
            "label": rec["label"],
            "category": rec["category"],
            "prompt": rec["prompt"],
            "response": resp,
            "refused": is_refusal(resp),
            "prompt_assessment": pa.to_dict(),
            "response_assessment": ra.to_dict(),
        })

    assessments_path = out_dir / "safety_assessments.jsonl"
    with open(assessments_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    results = {
        "model": model_name,
        "adapter_path": args.adapter_path,
        "guard_model": guard_name,
        "guard_mode": guard_mode,
        "safety_data": str(data_path),
        "num_prompts": len(rows),
        "mock": args.use_mock,
        "harmful": summarize_group([r for r in rows if r["label"] == "harmful"]),
        "benign": summarize_group([r for r in rows if r["label"] == "benign"]),
        "all_responses": compute_safety_metrics([r["response_assessment"] for r in rows]),
    }
    results_path = out_dir / "safety_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    save_run_info(
        stage=args.stage,
        output_dir=out_dir,
        config={k: v for k, v in results.items() if k not in ("harmful", "benign", "all_responses")},
        description=f"safety_eval_{slug}",
    )

    # Mirror safety results between Drive and local repository workspace when not in mock mode
    if not args.use_mock and args.output_dir is None:
        local_safety_dir = PROJECT_ROOT / "outputs" / args.stage / "safety" / slug
        drive_root = Path("/content/drive/MyDrive/math-reasoning-llm")
        drive_safety_dir = drive_root / "outputs" / args.stage / "safety" / slug if drive_root.exists() else None
        run_info_file = out_dir / "run_info.json"
        for target_dir in (local_safety_dir, drive_safety_dir):
            if target_dir is None:
                continue
            try:
                if target_dir.resolve() == out_dir.resolve():
                    continue
                target_dir.mkdir(parents=True, exist_ok=True)
                for artifact in (results_path, assessments_path, run_info_file):
                    if artifact.exists():
                        shutil.copy2(artifact, target_dir / artifact.name)
                logger.info(f"Mirrored safety evaluation artifacts to: {target_dir}")
            except Exception as exc:
                logger.warning(f"Could not mirror safety artifacts to {target_dir}: {exc}")

    h, b = results["harmful"], results["benign"]
    print("\n" + "=" * 72)
    print(f"  SAFETY RESULTS: {slug}{'  (MOCK)' if args.use_mock else ''}")
    print("=" * 72)
    print(f"  Harmful prompts ({h.get('count', 0)}): unsafe responses {h.get('unsafe_response_rate', 0):.1%}, "
          f"refusals {h.get('refusal_rate', 0):.1%}, prompts flagged {h.get('prompt_flagged_rate', 0):.1%}")
    print(f"  Benign prompts  ({b.get('count', 0)}): unsafe responses {b.get('unsafe_response_rate', 0):.1%}, "
          f"refusals {b.get('refusal_rate', 0):.1%}, prompts flagged {b.get('prompt_flagged_rate', 0):.1%}")
    unknown = results["all_responses"]["unknown_count"]
    if unknown:
        print(f"  WARNING: {unknown} response classifications failed or were unparseable (counted as unknown).")
    print(f"  Results: {out_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
