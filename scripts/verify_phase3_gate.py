#!/usr/bin/env python3
"""Phase 3 Gate Verifier (`v0.4` milestone).

Audits all Phase 3 deliverables (Days 16-25) across local and Colab/Drive paths,
distinguishing real teacher GPU runs (`PASSED`) from mock or gold-fallback
dev runs (`MOCK` / `PARTIAL` / `MISSING`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager, get_default_profile  # noqa: E402
from src.data.curation import (  # noqa: E402
    assert_no_contamination,
    compute_file_sha256,
    is_mock_trace,
    load_curated_dataset,
)


def _find_file(rel_path: str, cfg: ConfigManager, active_profile: str) -> Path:
    """Locate a file across local repo and profile Drive directory (preferring Drive/larger/newer)."""
    local_p = PROJECT_ROOT / rel_path
    p_dir = cfg.get_profile_dir("data_dir", profile_name=active_profile)
    drive_p: Path | None = None

    if rel_path.startswith("data/"):
        sub = rel_path[len("data/") :]
        drive_p = p_dir / sub
    elif rel_path.startswith("reports/"):
        name = Path(rel_path).name
        drive_p = p_dir.parent / "reports" / name

    if drive_p is not None and drive_p.exists() and local_p.exists():
        if drive_p.resolve() == local_p.resolve():
            return local_p
        if rel_path.startswith("data/"):
            return drive_p if drive_p.stat().st_size >= local_p.stat().st_size else local_p
        # For reports, prefer the more recently written file (or Drive on tie)
        return drive_p if drive_p.stat().st_mtime >= local_p.stat().st_mtime else local_p
    if drive_p is not None and drive_p.exists():
        return drive_p
    return local_p


def audit_phase3_gate(profile: str | None = None) -> Dict[str, Any]:
    """Run all Phase 3 gate checks and return structured audit summary."""
    cfg = ConfigManager()
    active_profile = profile or get_default_profile()
    checks: List[Dict[str, Any]] = []

    # 1. Day 16: Reasoning Prompts
    d16_path = _find_file("reports/day16_prompt_verification.json", cfg, active_profile)
    if d16_path.exists():
        d16 = json.loads(d16_path.read_text(encoding="utf-8"))
        mode = d16.get("mode", "mock")
        verdict = d16.get("verdict", "MOCK")
        status = "PASSED" if (mode == "real" and verdict == "PASSED") else ("MOCK" if mode == "mock" else "FAILED")
        checks.append(
            {
                "id": "day16_prompts",
                "name": "Day 16: Reasoning Prompts Verification (3 strategies)",
                "status": status,
                "detail": f"mode={mode}, verdict={verdict}, model={d16.get('model_name')}",
            }
        )
    else:
        checks.append(
            {
                "id": "day16_prompts",
                "name": "Day 16: Reasoning Prompts Verification (3 strategies)",
                "status": "MISSING",
                "detail": "reports/day16_prompt_verification.json not found",
            }
        )

    # 2. Day 17: 3,000-Problem Subset & Contamination Isolation
    subset_path = PROJECT_ROOT / "data" / "splits" / "synthetic_subset_ids.json"
    if subset_path.exists():
        subset_ids = json.loads(subset_path.read_text(encoding="utf-8"))
        try:
            assert_no_contamination(subset_ids)
            contam_ok = True
        except Exception:
            contam_ok = False
        status = "PASSED" if (len(subset_ids) == 3000 and len(set(subset_ids)) == 3000 and contam_ok) else "FAILED"
        checks.append(
            {
                "id": "day17_subset",
                "name": "Day 17: 3,000-Problem Synthetic Subset & Zero Contamination",
                "status": status,
                "detail": f"{len(subset_ids)} unique training IDs, contamination_free={contam_ok}",
            }
        )
    else:
        checks.append(
            {
                "id": "day17_subset",
                "name": "Day 17: 3,000-Problem Synthetic Subset & Zero Contamination",
                "status": "MISSING",
                "detail": "data/splits/synthetic_subset_ids.json not found",
            }
        )

    # 3. Day 17: 100-Problem Dev Trace Check
    d17_path = _find_file("reports/day17_synthetic_generation_report.json", cfg, active_profile)
    if d17_path.exists():
        d17 = json.loads(d17_path.read_text(encoding="utf-8"))
        mode = d17.get("mode", "mock")
        ym = d17.get("yield_metrics", {})
        status = "PASSED" if mode == "real" else "MOCK"
        checks.append(
            {
                "id": "day17_dev_sampling",
                "name": "Day 17: 100-Problem Development Trace Run",
                "status": status,
                "detail": f"mode={mode}, traces={ym.get('total_traces', 0)}, acc={ym.get('trace_accuracy_percent', 0)}%",
            }
        )
    else:
        checks.append(
            {
                "id": "day17_dev_sampling",
                "name": "Day 17: 100-Problem Development Trace Run",
                "status": "MISSING",
                "detail": "reports/day17_synthetic_generation_report.json not found",
            }
        )

    # 4. Day 18: 7B Teacher CoT Sampling (3,000 problems x 4 samples)
    d18_path = _find_file("reports/day18_cot_sampling_report.json", cfg, active_profile)
    if d18_path.exists():
        d18 = json.loads(d18_path.read_text(encoding="utf-8"))
        mode = d18.get("mode", "mock")
        ym = d18.get("yield_metrics", {})
        tot_probs = ym.get("total_problems", 0)
        status = "PASSED" if (mode == "real" and tot_probs >= 2500) else ("MOCK" if mode == "mock" else "PARTIAL")
        checks.append(
            {
                "id": "day18_cot_7b",
                "name": "Day 18: 7B Teacher CoT Sampling (3,000 problems x 4)",
                "status": status,
                "detail": f"mode={mode}, problems={tot_probs}, traces={ym.get('total_traces', 0)}",
            }
        )
    else:
        checks.append(
            {
                "id": "day18_cot_7b",
                "name": "Day 18: 7B Teacher CoT Sampling (3,000 problems x 4)",
                "status": "MISSING",
                "detail": "reports/day18_cot_sampling_report.json not found",
            }
        )

    # 5. Day 19: Yield Analysis & Per-Problem Pass Rates
    pr_path = _find_file("data/synthetic/generated/problem_pass_rates.json", cfg, active_profile)
    if pr_path.exists():
        pr_data = json.loads(pr_path.read_text(encoding="utf-8"))
        num_pr = len(pr_data)
        bins = {v.get("difficulty_bin", "zero") for v in pr_data.values() if isinstance(v, dict)}
        is_real_full = num_pr >= 2500 and len(bins) > 1
        status = "PASSED" if is_real_full else "MOCK"
        checks.append(
            {
                "id": "day19_pass_rates",
                "name": "Day 19: Per-Problem Pass Rates (`problem_pass_rates.json`)",
                "status": status,
                "detail": f"{num_pr} problems indexed, difficulty_bins={sorted(bins)}",
            }
        )
    else:
        checks.append(
            {
                "id": "day19_pass_rates",
                "name": "Day 19: Per-Problem Pass Rates (`problem_pass_rates.json`)",
                "status": "MISSING",
                "detail": "data/synthetic/generated/problem_pass_rates.json not found",
            }
        )

    # 6. Day 20: Solve-then-Verify & Alternative-Method Traces
    d20_path = _find_file("reports/day20_multi_strategy_traces_report.json", cfg, active_profile)
    if d20_path.exists():
        d20 = json.loads(d20_path.read_text(encoding="utf-8"))
        mode = d20.get("mode", "mock")
        ym = d20.get("yield_metrics", {})
        flt = d20.get("curation_filter") or d20.get("filter_statistics") or {}
        status = "PASSED" if mode == "real" else "MOCK"
        checks.append(
            {
                "id": "day20_multi_strategy",
                "name": "Day 20: Verification & Alternative-Method Traces",
                "status": status,
                "detail": f"mode={mode}, traces={ym.get('total_traces', 0)}, kept={flt.get('kept_count', 0)}",
            }
        )
    else:
        checks.append(
            {
                "id": "day20_multi_strategy",
                "name": "Day 20: Verification & Alternative-Method Traces",
                "status": "MISSING",
                "detail": "reports/day20_multi_strategy_traces_report.json not found",
            }
        )

    # 7. Day 21: LLM Judge Calibration (r >= 0.60 on 20-trace benchmark)
    d21_path = _find_file("reports/day21_judge_calibration_report.json", cfg, active_profile)
    if d21_path.exists():
        d21 = json.loads(d21_path.read_text(encoding="utf-8"))
        mode = d21.get("mode", "mock")
        cm = d21.get("calibration_metrics", {})
        r_val = float(cm.get("pearson_r", 0.0))
        is_cal = bool(cm.get("is_calibrated", False))
        status = "PASSED" if (mode == "real" and is_cal) else ("MOCK" if (mode == "mock" and is_cal) else "FAILED")
        checks.append(
            {
                "id": "day21_judge_calib",
                "name": "Day 21: LLM Judge Calibration (Pearson r >= 0.60)",
                "status": status,
                "detail": f"mode={mode}, pearson_r={r_val:.4f}, mae={cm.get('mae', 0.0):.4f}",
            }
        )
    else:
        checks.append(
            {
                "id": "day21_judge_calib",
                "name": "Day 21: LLM Judge Calibration (Pearson r >= 0.60)",
                "status": "MISSING",
                "detail": "reports/day21_judge_calibration_report.json not found",
            }
        )

    # 8. Day 22: Full-Scale Judge Scoring
    d22_path = _find_file("reports/day22_judge_scoring_report.json", cfg, active_profile)
    if d22_path.exists():
        d22 = json.loads(d22_path.read_text(encoding="utf-8"))
        mode = d22.get("mode", "mock")
        tot_scored = int(d22.get("total_scored", 0))
        status = "PASSED" if (mode == "real" and tot_scored >= 2500) else ("MOCK" if mode == "mock" else "PARTIAL")
        checks.append(
            {
                "id": "day22_judge_scoring",
                "name": "Day 22: Full-Scale LLM Judge Scoring (`traces_scored.jsonl`)",
                "status": status,
                "detail": f"mode={mode}, total_scored={tot_scored}, problems={d22.get('total_problems', 0)}",
            }
        )
    else:
        checks.append(
            {
                "id": "day22_judge_scoring",
                "name": "Day 22: Full-Scale LLM Judge Scoring (`traces_scored.jsonl`)",
                "status": "MISSING",
                "detail": "reports/day22_judge_scoring_report.json not found",
            }
        )

    # 9. Day 24: 100-Trace Qualitative Inspection (50 kept + 50 rejected)
    d24_samples = _find_file("reports/day24_inspection_samples.json", cfg, active_profile)
    if d24_samples.exists():
        s_data = json.loads(d24_samples.read_text(encoding="utf-8"))
        kept_n = len(s_data.get("kept_traces_50", []))
        rej_n = len(s_data.get("rejected_traces_50", []))
        status = "PASSED" if (kept_n == 50 and rej_n == 50) else "PARTIAL"
        checks.append(
            {
                "id": "day24_inspection",
                "name": "Day 24: 100-Trace Qualitative Audit (50 kept + 50 rejected)",
                "status": status,
                "detail": f"kept={kept_n}, rejected={rej_n}",
            }
        )
    else:
        checks.append(
            {
                "id": "day24_inspection",
                "name": "Day 24: 100-Trace Qualitative Audit (50 kept + 50 rejected)",
                "status": "MISSING",
                "detail": "reports/day24_inspection_samples.json not found",
            }
        )

    # 10. Days 23 & 25: Frozen Curated Dataset & Phase 3 Gate (`traces_curated_sft.jsonl`)
    curated_path = _find_file("data/synthetic/curated/traces_curated_sft.jsonl", cfg, active_profile)
    manifest_path = _find_file("data/synthetic/curated/traces_curated_sft_manifest.json", cfg, active_profile)
    if curated_path.exists() and manifest_path.exists():
        traces = load_curated_dataset(curated_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_sha = compute_file_sha256(curated_path)
        sha_ok = actual_sha == manifest.get("sha256")
        pids = [str(t.get("problem_id", "")) for t in traces]
        unique_probs = len(set(pids))
        try:
            assert_no_contamination(pids)
            no_contam = True
        except Exception:
            no_contam = False

        gold_n = sum(1 for t in traces if t.get("is_gold_fallback") or t.get("source") == "gold_clean")
        mock_n = sum(
            1
            for t in traces
            if not (t.get("is_gold_fallback") or t.get("source") == "gold_clean") and is_mock_trace(t)
        )
        real_n = len(traces) - gold_n - mock_n

        if unique_probs >= 2500 and len(traces) >= 4000 and real_n >= 1000 and sha_ok and no_contam:
            status = "PASSED"
        elif unique_probs >= 2500 and sha_ok and no_contam:
            status = "MOCK" if mock_n > 0 or real_n == 0 else "PARTIAL"
        else:
            status = "FAILED"

        checks.append(
            {
                "id": "day25_frozen_dataset",
                "name": "Day 25: Frozen Curated Dataset (>= 4,000 traces, >= 2,500 problems, SHA-256)",
                "status": status,
                "detail": (
                    f"traces={len(traces)} (real={real_n}, mock={mock_n}, gold_fallback={gold_n}), "
                    f"problems={unique_probs}, sha256_match={sha_ok}, zero_contamination={no_contam}"
                ),
            }
        )
    else:
        checks.append(
            {
                "id": "day25_frozen_dataset",
                "name": "Day 25: Frozen Curated Dataset (>= 4,000 traces, >= 2,500 problems, SHA-256)",
                "status": "MISSING",
                "detail": "traces_curated_sft.jsonl or manifest missing",
            }
        )

    passed_count = sum(1 for c in checks if c["status"] == "PASSED")
    mock_count = sum(1 for c in checks if c["status"] == "MOCK")
    overall_passed = passed_count == len(checks)

    return {
        "overall_status": "PASSED" if overall_passed else ("PARTIAL_OR_MOCK" if (passed_count + mock_count) > 0 else "FAILED"),
        "passed_count": passed_count,
        "mock_count": mock_count,
        "total_checks": len(checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 3 Gate (v0.4) deliverables.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["local", "colab"],
        help="Runtime profile (defaults to auto-detected profile).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 unless all checks are PASSED (not MOCK).",
    )
    args = parser.parse_args()

    summary = audit_phase3_gate(profile=args.profile)
    print("=" * 80)
    print(f"PHASE 3 GATE VERIFICATION (v0.4) — Overall: {summary['overall_status']}")
    print(f"Passed (Real GPU): {summary['passed_count']}/{summary['total_checks']} | Mock/Dev: {summary['mock_count']}/{summary['total_checks']}")
    print("=" * 80)
    for c in summary["checks"]:
        print(f"[{c['status']:^7}] {c['name']}")
        print(f"          {c['detail']}")
    print("=" * 80)

    if args.strict and summary["overall_status"] != "PASSED":
        sys.exit(1)


if __name__ == "__main__":
    main()
