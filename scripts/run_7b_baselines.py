"""Orchestrate baseline evaluations for the three 7B DeepSeek-Math models (Day 14).

Evaluates:
  1. deepseek-ai/deepseek-math-7b-base      (8-shot few-shot prompt)
  2. deepseek-ai/deepseek-math-7b-instruct  (chat template)
  3. deepseek-ai/deepseek-math-7b-rl        (chat template)

Under two decoding presets:
  - greedy (temperature 0.0)
  - self_consistency (8 samples at temperature 0.7)

Total: 3 models x 2 presets = 6 evaluation result sets stored in outputs/baseline/.

Usage:
  # On G4 GPU (production run):
  python scripts/run_7b_baselines.py

  # Fast testing / CI simulation without weights:
  python scripts/run_7b_baselines.py --use_mock

  # Dry run on first 20 validation problems:
  python scripts/run_7b_baselines.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import print_results_summary  # noqa: E402
from src.config import ConfigManager  # noqa: E402
from src.evaluation.evaluator import MathEvaluator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("7b_baselines")

DEFAULT_MODELS = [
    "deepseek-ai/deepseek-math-7b-base",
    "deepseek-ai/deepseek-math-7b-instruct",
    "deepseek-ai/deepseek-math-7b-rl",
]

DEFAULT_PRESETS = ["greedy", "self_consistency"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 7B baseline evaluations on GSM8K validation set (Day 14).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated Hugging Face model IDs to evaluate.",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default=",".join(DEFAULT_PRESETS),
        help="Comma-separated decoding presets to evaluate.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "val_analysis", "val_gating", "sample"],
        help="Validation split to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional problem limit per model (e.g. 20 for dry run, None for all 500).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "hf"],
        help="Inference engine backend.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for model generation.",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        default=False,
        help="Use deterministic mock generation for environments without GPU.",
    )
    return parser.parse_args()


def get_model_slug(model_name: str) -> str:
    """Derive clean directory slug from model ID (e.g. 'deepseek-ai/deepseek-math-7b-base' -> 'deepseek-math-7b-base')."""
    return model_name.split("/")[-1].strip()


def get_prompt_format_for_model(model_name: str) -> str:
    """Instruct and RL models use chat formatting; base models strictly require few-shot prompting."""
    lower = model_name.lower()
    if "instruct" in lower or "rl" in lower or "chat" in lower:
        return "chat"
    return "few_shot"


def generate_comparison_markdown(summary: Dict[str, Any], output_path: Path) -> str:
    """Format evaluation summary into an aligned Markdown comparison table."""
    lines = [
        "# GSM8K Validation Baselines Comparison (Day 14)",
        "",
        f"- **Validation Split:** `{summary.get('split', 'val')}` (Problems: {summary.get('num_problems', 500)})",
        "- **Models Evaluated:** `deepseek-math-7b-base`, `deepseek-math-7b-instruct`, `deepseek-math-7b-rl`",
        "- **Presets:** `greedy` (deterministic), `self_consistency` (maj@8 / sampled)",
        "",
        "## Summary Results Table",
        "",
        "| Model | Prompting | Greedy Acc (%) | maj@k Acc (%) | pass@k (%) | Format Compliance (%) | Avg Resp Tokens |",
        "|---|---|---|---|---|---|---|",
    ]

    models_data = summary.get("results", {})
    for model_id, presets_data in models_data.items():
        slug = get_model_slug(model_id)
        fmt = get_prompt_format_for_model(model_id)
        prompt_label = "8-Shot CoT" if fmt == "few_shot" else "Chat Template"

        greedy_res = presets_data.get("greedy", {}).get("metrics", {})
        sc_res = presets_data.get("self_consistency", {}).get("metrics", {})

        greedy_acc = f"{greedy_res.get('accuracy_percent', 0.0):.1f}%" if greedy_res else "N/A"
        maj_k_acc = f"{sc_res.get('maj_at_k_percent', 0.0):.1f}%" if sc_res else "N/A"
        pass_k = f"{sc_res.get('pass_at_k_percent', 0.0):.1f}%" if sc_res else "N/A"
        compliance = f"{greedy_res.get('format_compliance_percent', 0.0):.1f}%" if greedy_res else "N/A"
        avg_tokens = f"{greedy_res.get('average_completion_tokens', 0.0):.1f}" if greedy_res else "N/A"

        lines.append(
            f"| `{slug}` | {prompt_label} | {greedy_acc} | {maj_k_acc} | {pass_k} | {compliance} | {avg_tokens} |"
        )

    lines.extend([
        "",
        "## Notes & Decision Criteria",
        "- The 7B base model requires 8-shot prompting because it has received zero instruction tuning.",
        "- Instruct and RL models are evaluated with system prompt and chat template.",
        "- The best validation accuracy among the 7B models serves as the baseline to beat in SFT (Phase 4).",
    ])

    table_md = "\n".join(lines) + "\n"
    output_path.write_text(table_md, encoding="utf-8")
    return table_md


def main() -> int:
    args = parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]

    # Profile outputs directory: Google Drive under the colab profile, so results survive a disconnect
    stage_base_dir = ConfigManager().get_stage_paths("baseline")["outputs"]
    stage_base_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("  STARTING DAY 14: 7B BASELINE EVALUATIONS (3 MODELS x 2 PRESETS)")
    logger.info("=" * 72)
    logger.info(f"Models:  {models}")
    logger.info(f"Presets: {presets}")
    logger.info(f"Split:   {args.split} (Limit: {args.limit}) | Backend: {args.backend}")
    logger.info("=" * 72)

    total_runs = len(models) * len(presets)
    current_run = 0
    all_results: Dict[str, Dict[str, Any]] = {}

    for model_id in models:
        model_slug = get_model_slug(model_id)
        model_out_dir = stage_base_dir / model_slug
        model_out_dir.mkdir(parents=True, exist_ok=True)
        prompt_fmt = get_prompt_format_for_model(model_id)

        all_results[model_id] = {}

        for preset in presets:
            current_run += 1
            logger.info("\n" + "#" * 72)
            logger.info(
                f"[{current_run}/{total_runs}] Running Model: {model_slug} | Preset: {preset} | Prompt: {prompt_fmt}"
            )
            logger.info("#" * 72)

            evaluator = MathEvaluator(
                stage="baseline",
                checkpoint=model_id,
                preset=preset,
                split=args.split,
                limit=args.limit,
                prompt_format=prompt_fmt,
                backend=args.backend,
                batch_size=args.batch_size,
                output_dir=model_out_dir,
                use_mock=args.use_mock,
            )

            res = evaluator.evaluate()
            all_results[model_id][preset] = res
            print_results_summary(res)

    # Compile combined baseline summary
    summary_data = {
        "stage": "baseline",
        "split": args.split,
        "num_models": len(models),
        "num_presets": len(presets),
        "total_result_sets": total_runs,
        "models": models,
        "presets": presets,
        "results": all_results,
    }

    summary_json_path = stage_base_dir / "baseline_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved baseline summary JSON to: {summary_json_path.resolve()}")

    comparison_md_path = stage_base_dir / "baseline_comparison.md"
    comparison_md = generate_comparison_markdown(summary_data, comparison_md_path)
    logger.info(f"Saved baseline comparison Markdown to: {comparison_md_path.resolve()}")

    print("\n" + "=" * 72)
    print("  DAY 14: ALL 6 BASELINE RESULT SETS PRODUCED")
    print("=" * 72)
    print(comparison_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
