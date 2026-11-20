"""Verify and display Day 13 Development Model Baseline results.

Reads and compares greedy and self-consistency evaluation metrics for the development model.

Usage:
  python scripts/verify_dev_baseline.py
  python scripts/verify_dev_baseline.py --model_slug Qwen2.5-1.5B-Instruct --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("verify_dev_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Day 13 development model baseline metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_slug",
        type=str,
        default="Qwen2.5-1.5B-Instruct",
        help="Model directory slug inside outputs/baseline/.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Evaluation split name.",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        default=False,
        help="Return 0 if at least one preset (greedy or self_consistency) exists.",
    )
    return parser.parse_args()


def resolve_baseline_dir(cfg_mgr: ConfigManager, model_slug: str) -> Path:
    candidates = [
        cfg_mgr.outputs_path("baseline") / model_slug,
        PROJECT_ROOT / "outputs" / "baseline" / model_slug,
        Path("/content/drive/MyDrive/math-reasoning-llm/outputs/baseline") / model_slug,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def main() -> int:
    args = parse_args()
    cfg_mgr = ConfigManager()
    out_dir = resolve_baseline_dir(cfg_mgr, args.model_slug)

    if not out_dir.exists():
        logger.error(f"Baseline directory not found: {out_dir}")
        return 1

    greedy_path = out_dir / f"eval_results_{args.split}_greedy.json"
    sc_path = out_dir / f"eval_results_{args.split}_self_consistency.json"

    print("=" * 70)
    print(f"DAY 13 BASELINE VERIFICATION: {args.model_slug} ({args.split})")
    print("=" * 70)

    has_greedy = greedy_path.exists()
    has_sc = sc_path.exists()

    if not has_greedy:
        print(f"[!] Greedy evaluation file missing: {greedy_path}")
        print(f"    Run: python scripts/run_dev_baseline.py --split {args.split} --presets greedy")
    else:
        try:
            greedy_data = json.loads(greedy_path.read_text(encoding="utf-8"))
            gm = greedy_data.get("metrics", {})
            mode_tag = "MOCK" if greedy_data.get("mock") else "REAL GPU"
            print(f"--- Greedy Evaluation (N={greedy_data.get('num_problems', 'N/A')}, mode={mode_tag}) ---")
            print("  Accuracy:                 {:.2f}%".format(gm.get("accuracy_percent", 0.0)))
            print("  Format Compliance (####): {:.2f}%".format(gm.get("format_compliance_percent", 0.0)))
            print("  Fallback Rate:            {:.2f}%".format(gm.get("fallback_percent", 0.0)))
            print("  Unparseable Rate:         {:.2f}%".format(gm.get("unparseable_percent", 0.0)))
            print("  Avg Completion Tokens:    {:.2f}".format(gm.get("average_completion_tokens", 0.0)))
            print("  Truncation Rate:          {:.2f}%".format(gm.get("truncation_percent", 0.0)))
        except Exception as e:
            print(f"[!] Error reading greedy results: {e}")

    print("-" * 70)

    if not has_sc:
        print(f"[!] Self-consistency evaluation file missing: {sc_path}")
        print(f"    Run: python scripts/run_dev_baseline.py --split {args.split} --presets self_consistency")
    else:
        try:
            sc_data = json.loads(sc_path.read_text(encoding="utf-8"))
            sm = sc_data.get("metrics", {})
            mode_tag = "MOCK" if sc_data.get("mock") else "REAL GPU"
            print(f"--- Self-Consistency Evaluation (N={sc_data.get('num_problems', 'N/A')}, mode={mode_tag}) ---")
            print("  maj@k Accuracy:           {:.2f}%".format(sm.get("maj_at_k_percent", 0.0)))
            print("  pass@k Accuracy:          {:.2f}%".format(sm.get("pass_at_k_percent", 0.0)))
            print("  Format Compliance (####): {:.2f}%".format(sm.get("format_compliance_percent", 0.0)))
            print("  Fallback Rate:            {:.2f}%".format(sm.get("fallback_percent", 0.0)))
            print("  Unparseable Rate:         {:.2f}%".format(sm.get("unparseable_percent", 0.0)))
            print("  Avg Completion Tokens:    {:.2f}".format(sm.get("average_completion_tokens", 0.0)))
            print("  Truncation Rate:          {:.2f}%".format(sm.get("truncation_percent", 0.0)))
        except Exception as e:
            print(f"[!] Error reading self-consistency results: {e}")

    print("=" * 70)
    if args.allow_partial:
        return 0 if (has_greedy or has_sc) else 1
    return 0 if (has_greedy and has_sc) else 1


if __name__ == "__main__":
    sys.exit(main())
