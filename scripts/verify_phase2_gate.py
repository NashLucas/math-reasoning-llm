"""Verify Phase 2 Gate (v0.3) deliverables against Implementation plan.md.

Checks:
  1. Development model (Qwen2.5-1.5B-Instruct) greedy & self-consistency validation runs (real GPU, 500 problems).
  2. Day 13 manual output review (reports/day13_output_review.md, 0 parser discrepancies).
  3. Three DeepSeek-Math 7B baseline models (base, instruct, rl) x 2 presets (greedy, self_consistency) on 500 validation problems (real GPU).
  4. Day 15 manual output reviews (30 outputs per 7B model: reports/day15_review_deepseek-math-7b-{base,instruct,rl}.md).
  5. Safety prompt set (data/safety/safety_prompts.jsonl + manifest.json) and development model safety screening (outputs/baseline/safety/Qwen2.5-1.5B-Instruct/safety_results.json).

Usage:
  python scripts/verify_phase2_gate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager, get_default_profile  # noqa: E402

MODELS_7B = (
    "deepseek-math-7b-base",
    "deepseek-math-7b-instruct",
    "deepseek-math-7b-rl",
)
PRESETS = ("greedy", "self_consistency")


def find_file(rel_path: Path, roots: List[Path]) -> Optional[Path]:
    for root in roots:
        cand = root / rel_path
        if cand.exists():
            return cand
    return None


def check_eval_file(
    slug: str,
    preset: str,
    split: str,
    roots: List[Path],
    expected_problems: int = 500,
) -> Tuple[str, str]:
    rel = Path(slug) / f"eval_results_{split}_{preset}.json"
    path = find_file(rel, roots)
    if path is None:
        return "MISSING", f"{rel} not found"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return "ERROR", f"{rel} unreadable: {exc}"

    is_mock = bool(data.get("mock", False))
    n_probs = int(data.get("num_problems", 0))
    metrics = data.get("metrics", {})
    acc = (
        metrics.get("maj_at_k_percent", 0.0)
        if preset == "self_consistency"
        else metrics.get("accuracy_percent", 0.0)
    )
    fmt = metrics.get("format_compliance_percent", 0.0)
    trunc = metrics.get("truncation_percent", 0.0)
    summary = f"N={n_probs}, acc={acc:.2f}%, format={fmt:.2f}%, trunc={trunc:.2f}%"

    if is_mock:
        return "MOCK", f"{summary} (mock=True)"
    if n_probs < expected_problems:
        return "PARTIAL", f"{summary} (expected {expected_problems} problems)"
    return "PASSED", summary


def check_review_report(report_path: Path) -> Tuple[str, str]:
    if not report_path.exists():
        return "MISSING", f"{report_path.relative_to(PROJECT_ROOT)} not found"
    text = report_path.read_text(encoding="utf-8")
    if "MOCK DATA" in text:
        return "MOCK", "Report marked as MOCK DATA"
    if "**Parser Discrepancies:** 0" in text:
        return "PASSED", "0 parser discrepancies"
    return "NEEDS_REVIEW", "Non-zero parser discrepancies reported"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Phase 2 Gate deliverables.")
    parser.add_argument("--split", type=str, default="val", help="Validation split name.")
    parser.add_argument("--expected_problems", type=int, default=500, help="Expected validation problem count.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["local", "colab"],
        help="Runtime profile (defaults to auto-detected profile).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ConfigManager()
    active_profile = args.profile or get_default_profile()

    profile_outputs = cfg.get_profile_dir("outputs_dir", profile_name=active_profile)
    profile_data = cfg.get_profile_dir("data_dir", profile_name=active_profile)
    drive_reports = profile_outputs.parent / "reports"

    baseline_roots: List[Path] = [
        profile_outputs / "baseline",
        PROJECT_ROOT / "outputs" / "baseline",
        Path("/content/drive/MyDrive/math-reasoning-llm/outputs/baseline"),
    ]
    report_roots: List[Path] = (
        [drive_reports, PROJECT_ROOT / "reports", Path("/content/drive/MyDrive/math-reasoning-llm/reports")]
        if active_profile == "colab"
        else [PROJECT_ROOT / "reports", drive_reports, Path("/content/drive/MyDrive/math-reasoning-llm/reports")]
    )

    checks: List[Dict[str, Any]] = []

    # 1. Dev model (Qwen2.5-1.5B-Instruct)
    for preset in PRESETS:
        status, detail = check_eval_file(
            "Qwen2.5-1.5B-Instruct", preset, args.split, baseline_roots, args.expected_problems
        )
        checks.append({"item": f"Day 13 Dev Model ({preset})", "status": status, "detail": detail})

    # 2. Day 13 output review
    day13_report = find_file(Path("day13_output_review.md"), report_roots) or (PROJECT_ROOT / "reports" / "day13_output_review.md")
    status, detail = check_review_report(day13_report)
    checks.append({"item": "Day 13 20-Output Review", "status": status, "detail": detail})

    # 3. 7B Baselines (3 models x 2 presets)
    for slug in MODELS_7B:
        for preset in PRESETS:
            status, detail = check_eval_file(
                slug, preset, args.split, baseline_roots, args.expected_problems
            )
            checks.append({"item": f"Day 14 {slug} ({preset})", "status": status, "detail": detail})

    # 4. Day 15 30-output reviews per 7B model
    for slug in MODELS_7B:
        rep_name = Path(f"day15_review_{slug}.md")
        rep_path = find_file(rep_name, report_roots) or (PROJECT_ROOT / "reports" / rep_name)
        status, detail = check_review_report(rep_path)
        checks.append({"item": f"Day 15 30-Output Review ({slug})", "status": status, "detail": detail})

    # 5. Safety dataset & safety evaluation
    safety_data_roots = [
        profile_data / "safety",
        PROJECT_ROOT / "data" / "safety",
        Path("/content/drive/MyDrive/math-reasoning-llm/data/safety"),
    ]
    safety_jsonl = find_file(Path("safety_prompts.jsonl"), safety_data_roots)
    safety_manifest = find_file(Path("manifest.json"), safety_data_roots)
    if safety_jsonl and safety_manifest:
        manifest_data = json.loads(safety_manifest.read_text(encoding="utf-8"))
        checks.append({
            "item": "Day 15 Safety Prompt Set (100 prompts)",
            "status": "PASSED" if manifest_data.get("total") == 100 else "PARTIAL",
            "detail": f"total={manifest_data.get('total')}, harmful={manifest_data.get('harmful')}, benign={manifest_data.get('benign')}",
        })
    else:
        checks.append({
            "item": "Day 15 Safety Prompt Set (100 prompts)",
            "status": "MISSING",
            "detail": "data/safety/safety_prompts.jsonl or manifest.json not found",
        })

    safety_res_path = find_file(Path("safety/Qwen2.5-1.5B-Instruct/safety_results.json"), baseline_roots)
    if safety_res_path is None:
        checks.append({
            "item": "Day 15 Dev Model Safety Screening",
            "status": "MISSING",
            "detail": "safety/Qwen2.5-1.5B-Instruct/safety_results.json not found",
        })
    else:
        sdata = json.loads(safety_res_path.read_text(encoding="utf-8"))
        is_mock = bool(sdata.get("mock", False))
        n_p = int(sdata.get("num_prompts", 0))
        guard_mode = sdata.get("guard_mode", "llama_guard")
        h_ref = sdata.get("harmful", {}).get("refusal_rate", 0.0)
        b_ref = sdata.get("benign", {}).get("refusal_rate", 0.0)
        checks.append({
            "item": "Day 15 Dev Model Safety Screening",
            "status": "MOCK" if is_mock else ("PASSED" if n_p >= 100 else "PARTIAL"),
            "detail": f"N={n_p}, guard_mode={guard_mode}, harmful_refusal={h_ref:.1%}, benign_refusal={b_ref:.1%}",
        })

    print("=" * 88)
    print("  PHASE 2 GATE VERIFICATION (v0.3)")
    print("=" * 88)
    passed_count = 0
    for c in checks:
        tag = f"[{c['status']}]"
        if c["status"] == "PASSED":
            passed_count += 1
        print(f"  {tag:<14} {c['item']:<42} | {c['detail']}")
    print("-" * 88)
    all_passed = passed_count == len(checks)
    verdict = "PASSED (All Phase 2 deliverables verified on real GPU)" if all_passed else f"INCOMPLETE ({passed_count}/{len(checks)} items verified; run remaining GPU steps in 00_colab_launcher.ipynb)"
    print(f"  VERDICT: {verdict}")
    print("=" * 88)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
