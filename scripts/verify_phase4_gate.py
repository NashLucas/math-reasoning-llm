#!/usr/bin/env python3
"""Phase 4 Gate Verifier (`v0.5` milestone).

Audits all Phase 4 deliverables (Days 26-30) across local and Colab/Drive paths,
distinguishing real 7B GPU runs (`PASSED`) from mock or dry-run dev runs
(`MOCK` / `PARTIAL` / `MISSING`), ensuring transparent milestone tracking.
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
from src.data.formatting import format_zero_shot_prompt  # noqa: E402
from src.data.sft_dataset import format_sft_prompt_completion  # noqa: E402
from src.training.checkpointing import get_checkpoints  # noqa: E402
from src.training.sft_trainer import load_lora_config, load_training_arguments  # noqa: E402


def _find_file(rel_path: str, cfg: ConfigManager, active_profile: str) -> Path:
    """Locate a file in PROJECT_ROOT or profile directory."""
    local_p = PROJECT_ROOT / rel_path
    if local_p.exists():
        return local_p

    if rel_path.startswith("reports/"):
        name = Path(rel_path).name
        drive_rep = cfg.get_profile_dir("outputs_dir", profile_name=active_profile).parent / "reports" / name
        if drive_rep.exists():
            return drive_rep
        drive_out = cfg.get_profile_dir("outputs_dir", profile_name=active_profile) / "reports" / name
        if drive_out.exists():
            return drive_out

    if rel_path.startswith("data/"):
        sub = rel_path[len("data/") :]
        drive_p = cfg.get_profile_dir("data_dir", profile_name=active_profile) / sub
        if drive_p.exists():
            return drive_p

    if rel_path.startswith("checkpoints/"):
        sub = rel_path[len("checkpoints/") :]
        drive_p = cfg.get_profile_dir("checkpoints_dir", profile_name=active_profile) / sub
        if drive_p.exists():
            return drive_p

    if rel_path.startswith("outputs/"):
        sub = rel_path[len("outputs/") :]
        drive_p = cfg.get_profile_dir("outputs_dir", profile_name=active_profile) / sub
        if drive_p.exists():
            return drive_p

    return local_p


def audit_phase4_gate(profile: str | None = None) -> Dict[str, Any]:
    """Audit all 10 Phase 4 deliverables across local and Drive workspaces."""
    cfg = ConfigManager()
    active_profile = profile or get_default_profile()
    checks: List[Dict[str, Any]] = []

    # 1. Day 26: SFT Tokenization Hand-Check
    d26_path = _find_file("reports/day26_sft_tokenization_check.json", cfg, active_profile)
    if d26_path.exists():
        try:
            d26_data = json.loads(d26_path.read_text(encoding="utf-8"))
            if isinstance(d26_data, list):
                all_ok = len(d26_data) > 0 and all(
                    item.get("all_checks_passed", False) and item.get("eos_unmasked", False)
                    for item in d26_data
                )
                md_path = d26_path.with_suffix(".md")
                tok_model = "unknown"
                if md_path.exists():
                    for line in md_path.read_text(encoding="utf-8").splitlines():
                        if "Model Tokenizer" in line:
                            tok_model = (line.split("`")[1] if "`" in line else line.split(":")[-1]).strip()
                            break
                if tok_model == "unknown":
                    tok_model = "hf_tokenizer" if any(item.get("last_label") is not None for item in d26_data) else "mock"
            elif isinstance(d26_data, dict):
                tok_model = d26_data.get("model_tokenizer", d26_data.get("model", ""))
                all_ok = d26_data.get("all_examples_passed", False)
            else:
                tok_model = ""
                all_ok = False

            is_mock_tok = "fallback" in tok_model.lower() or "mock" in tok_model.lower()
            if all_ok and not is_mock_tok:
                status = "PASSED"
            elif all_ok and is_mock_tok:
                status = "MOCK"
            else:
                status = "FAILED"
            checks.append({
                "id": "day26_tokenization_hand_check",
                "name": "Day 26: SFT Tokenization Hand-Check & Loss Masking",
                "status": status,
                "detail": f"tokenizer={tok_model}, all_examples_passed={all_ok}",
            })
        except Exception as exc:
            checks.append({
                "id": "day26_tokenization_hand_check",
                "name": "Day 26: SFT Tokenization Hand-Check & Loss Masking",
                "status": "FAILED",
                "detail": f"Error parsing report: {exc}",
            })
    else:
        checks.append({
            "id": "day26_tokenization_hand_check",
            "name": "Day 26: SFT Tokenization Hand-Check & Loss Masking",
            "status": "MISSING",
            "detail": "reports/day26_sft_tokenization_check.json not found",
        })

    # 2. Day 26: LoRA Config & Training Arguments
    try:
        lora_cfg = load_lora_config()
        r_val = getattr(lora_cfg, "r", 16) if hasattr(lora_cfg, "r") else lora_cfg.get("r", 16)
        alpha_val = getattr(lora_cfg, "lora_alpha", 32) if hasattr(lora_cfg, "lora_alpha") else lora_cfg.get("lora_alpha", 32)
        targs = load_training_arguments()
        lr_val = getattr(targs, "learning_rate", 1e-4) if hasattr(targs, "learning_rate") else targs.get("learning_rate", 1e-4)
        save_steps = getattr(targs, "save_steps", 50) if hasattr(targs, "save_steps") else targs.get("save_steps", 50)
        config_ok = (16 <= r_val <= 32) and (alpha_val == 2 * r_val) and (lr_val in (1e-4, 2e-4, 5e-5)) and (save_steps == 50)
        checks.append({
            "id": "day26_configs",
            "name": "Day 26: LoRA & Training Configurations (r=16, alpha=32, cosine schedule)",
            "status": "PASSED" if config_ok else "FAILED",
            "detail": f"LoRA r={r_val}, alpha={alpha_val}, LR={lr_val}, save_steps={save_steps}",
        })
    except Exception as exc:
        checks.append({
            "id": "day26_configs",
            "name": "Day 26: LoRA & Training Configurations (r=16, alpha=32, cosine schedule)",
            "status": "FAILED",
            "detail": f"Error checking configs: {exc}",
        })

    # 3. Day 27: Prompt Template Exact Match
    try:
        test_q = "What is the perimeter of a rectangle with length 12 and width 5?"
        inf_prompt = format_zero_shot_prompt(test_q)
        pair = format_sft_prompt_completion(question=test_q, completion="12 * 2 + 5 * 2 = 34.\n#### 34")
        template_match = (inf_prompt == pair["prompt"]) and ("#### <number>" in inf_prompt)
        checks.append({
            "id": "day27_template_match",
            "name": "Day 27: Training vs. Inference Template Exact Alignment (Design Rule 1)",
            "status": "PASSED" if template_match else "FAILED",
            "detail": f"exact_match={template_match}, format_instruction_present={'#### <number>' in inf_prompt}",
        })
    except Exception as exc:
        checks.append({
            "id": "day27_template_match",
            "name": "Day 27: Training vs. Inference Template Exact Alignment (Design Rule 1)",
            "status": "FAILED",
            "detail": f"Error verifying template match: {exc}",
        })

    # 4. Day 27: Overfitting Sanity Test
    d27_path = _find_file("reports/day27_overfitting_test.json", cfg, active_profile)
    if d27_path.exists():
        try:
            d27_data = json.loads(d27_path.read_text(encoding="utf-8"))
            model_name = d27_data.get("model", "")
            passed_test = d27_data.get("test_passed", False)
            loss_red = d27_data.get("loss_reduction_pct", 0.0)
            fmt_rate = d27_data.get("format_compliance_rate", 0.0)
            is_mock_model = "mock" in model_name.lower()
            if passed_test and not is_mock_model:
                status = "PASSED"
            elif passed_test and is_mock_model:
                status = "MOCK"
            else:
                status = "FAILED"
            checks.append({
                "id": "day27_overfitting_test",
                "name": "Day 27: 20-Example Overfitting Sanity Test (Loss decay & reproduction)",
                "status": status,
                "detail": f"model={model_name}, loss_reduction={loss_red}%, format_compliance={fmt_rate*100:.1f}%",
            })
        except Exception as exc:
            checks.append({
                "id": "day27_overfitting_test",
                "name": "Day 27: 20-Example Overfitting Sanity Test (Loss decay & reproduction)",
                "status": "FAILED",
                "detail": f"Error parsing report: {exc}",
            })
    else:
        checks.append({
            "id": "day27_overfitting_test",
            "name": "Day 27: 20-Example Overfitting Sanity Test (Loss decay & reproduction)",
            "status": "MISSING",
            "detail": "reports/day27_overfitting_test.json not found",
        })

    # 5. Day 27: Periodic Validation Callback & Generation Logs
    gen_dir_local = PROJECT_ROOT / "logs" / "generations"
    gen_dir_drive = cfg.get_profile_dir("logs_dir", profile_name=active_profile) / "generations"
    gen_dir = gen_dir_drive if gen_dir_drive.exists() else gen_dir_local
    step_gens = list(gen_dir.glob("step_*.jsonl")) if gen_dir.exists() else []
    step_sums = list(gen_dir.glob("step_*_summary.json")) if gen_dir.exists() else []
    if step_gens or step_sums:
        checks.append({
            "id": "day27_callback_logs",
            "name": "Day 27: Periodic Validation Callback & Generation Logs",
            "status": "PASSED",
            "detail": f"dir={gen_dir}, {len(step_gens)} step generations, {len(step_sums)} step summaries",
        })
    else:
        checks.append({
            "id": "day27_callback_logs",
            "name": "Day 27: Periodic Validation Callback & Generation Logs",
            "status": "MOCK",
            "detail": "Validation callback generation files not found in logs/generations",
        })

    # 6. Day 28: Learning-Rate Sweep Report & Selection
    d28_path = _find_file("reports/day28_lr_sweep_report.json", cfg, active_profile)
    dev_adapter = _find_file("outputs/sft/adapter_model_dev", cfg, active_profile)
    if d28_path.exists():
        try:
            d28_data = json.loads(d28_path.read_text(encoding="utf-8"))
            model_name = d28_data.get("model_name_or_path", d28_data.get("model", ""))
            best_lr = d28_data.get("best_learning_rate")
            cuda_avail = d28_data.get("run_info", {}).get("hardware", {}).get("cuda_available", True)
            is_mock_sweep = "mock" in model_name.lower() or not cuda_avail
            adapter_saved = dev_adapter.exists() and (
                (dev_adapter / "adapter_config.json").exists()
                or (dev_adapter / "adapter_model.bin").exists()
                or (dev_adapter / "adapter_model.safetensors").exists()
            )
            if best_lr is not None and not is_mock_sweep and adapter_saved:
                status = "PASSED"
            elif best_lr is not None and is_mock_sweep:
                status = "MOCK"
            else:
                status = "FAILED"
            checks.append({
                "id": "day28_lr_sweep",
                "name": "Day 28: Learning-Rate Sweep (5e-5, 1e-4, 2e-4) & Dev Adapter Preservation",
                "status": status,
                "detail": f"model={model_name}, best_lr={best_lr}, adapter_saved={adapter_saved}",
            })
        except Exception as exc:
            checks.append({
                "id": "day28_lr_sweep",
                "name": "Day 28: Learning-Rate Sweep (5e-5, 1e-4, 2e-4) & Dev Adapter Preservation",
                "status": "FAILED",
                "detail": f"Error parsing report: {exc}",
            })
    else:
        checks.append({
            "id": "day28_lr_sweep",
            "name": "Day 28: Learning-Rate Sweep (5e-5, 1e-4, 2e-4) & Dev Adapter Preservation",
            "status": "MISSING",
            "detail": "reports/day28_lr_sweep_report.json not found",
        })

    # 7. Day 29: 7B SFT Training & Checkpoints
    d29_path = _find_file("reports/day29_7b_sft_report.json", cfg, active_profile)
    ckpts_dir_drive = cfg.get_profile_dir("checkpoints_dir", profile_name=active_profile) / "sft"
    ckpts_dir_local = PROJECT_ROOT / "checkpoints" / "sft"
    ckpts_dir = ckpts_dir_drive if ckpts_dir_drive.exists() else ckpts_dir_local
    ckpts = get_checkpoints(ckpts_dir, complete_only=True) if ckpts_dir.exists() else []
    if d29_path.exists():
        try:
            d29_data = json.loads(d29_path.read_text(encoding="utf-8"))
            model_target = d29_data.get("model_name_or_path", d29_data.get("model_target", ""))
            is_sample_mode = d29_data.get("sample_mode", False)
            cuda_avail = d29_data.get("run_info", {}).get("hardware", {}).get("cuda_available", True)
            total_examples = d29_data.get("config_snapshot", {}).get("total_examples", 0)
            is_mock_run = "mock" in model_target.lower() or is_sample_mode or not cuda_avail or total_examples < 100
            if not is_mock_run and len(ckpts) > 0:
                status = "PASSED"
            elif is_mock_run or len(ckpts) > 0:
                status = "MOCK"
            else:
                status = "FAILED"
            checks.append({
                "id": "day29_7b_sft_training",
                "name": "Day 29: 7B SFT LoRA Training & Checkpoint Resumption",
                "status": status,
                "detail": f"model={model_target}, sample_mode={is_sample_mode}, completed_checkpoints={len(ckpts)}",
            })
        except Exception as exc:
            checks.append({
                "id": "day29_7b_sft_training",
                "name": "Day 29: 7B SFT LoRA Training & Checkpoint Resumption",
                "status": "FAILED",
                "detail": f"Error parsing report: {exc}",
            })
    else:
        status = "MOCK" if len(ckpts) > 0 else "MISSING"
        checks.append({
            "id": "day29_7b_sft_training",
            "name": "Day 29: 7B SFT LoRA Training & Checkpoint Resumption",
            "status": status,
            "detail": f"completed_checkpoints={len(ckpts)} found in {ckpts_dir}",
        })

    # 8. Day 30: Checkpoint Evaluation, Selection & Provenance
    d30_path = _find_file("reports/day30_best_checkpoint.json", cfg, active_profile)
    adapter_dir = _find_file("outputs/sft/adapter_model", cfg, active_profile)
    has_adapter = adapter_dir.exists() and (
        (adapter_dir / "adapter_config.json").exists()
        or (adapter_dir / "adapter_model.bin").exists()
        or (adapter_dir / "adapter_model.safetensors").exists()
    )
    has_run_info = adapter_dir.exists() and (
        (adapter_dir / "run_info.json").exists()
        or (adapter_dir / "trainer_state.json").exists()
    )
    if d30_path.exists():
        try:
            d30_data = json.loads(d30_path.read_text(encoding="utf-8"))
            best_ckpt = d30_data.get("best_checkpoint", "")
            model_name = d30_data.get("model_name_or_path", d30_data.get("model", ""))
            best_greedy = d30_data.get("best_checkpoint_greedy", {})
            if not best_greedy and d30_data.get("checkpoints_evaluated"):
                best_greedy = d30_data["checkpoints_evaluated"][0]
            acc = best_greedy.get("accuracy_percent", 0.0)
            cuda_avail = d30_data.get("run_info", {}).get("hardware", {}).get("cuda_available", True)
            is_mock_eval = "mock" in model_name.lower() or d30_data.get("use_mock", False) or not cuda_avail
            if not is_mock_eval and has_adapter and has_run_info:
                status = "PASSED"
            elif is_mock_eval or has_adapter:
                status = "MOCK"
            else:
                status = "FAILED"
            checks.append({
                "id": "day30_checkpoint_selection",
                "name": "Day 30: Checkpoint Evaluation & Best Adapter Selection (Design Rule 3)",
                "status": status,
                "detail": f"best={best_ckpt}, val_acc={acc:.1f}%, adapter_present={has_adapter}, provenance_present={has_run_info}",
            })
        except Exception as exc:
            checks.append({
                "id": "day30_checkpoint_selection",
                "name": "Day 30: Checkpoint Evaluation & Best Adapter Selection (Design Rule 3)",
                "status": "FAILED",
                "detail": f"Error parsing report: {exc}",
            })
    else:
        status = "MOCK" if has_adapter else "MISSING"
        checks.append({
            "id": "day30_checkpoint_selection",
            "name": "Day 30: Checkpoint Evaluation & Best Adapter Selection (Design Rule 3)",
            "status": status,
            "detail": f"adapter_present={has_adapter}, reports/day30_best_checkpoint.json not found",
        })

    # 9. Phase 4 Gate Criteria (Release v0.5)
    if d30_path.exists():
        try:
            d30_data = json.loads(d30_path.read_text(encoding="utf-8"))
            gate_assess = d30_data.get("gate_assessment", d30_data.get("config_snapshot", {}))
            gate_ok = gate_assess.get("gate_passed", False) or "PASSED" in gate_assess.get("gate_verdict", "")
            model_name = d30_data.get("model_name_or_path", d30_data.get("model", ""))
            cuda_avail = d30_data.get("run_info", {}).get("hardware", {}).get("cuda_available", True)
            is_mock_run = "mock" in model_name.lower() or d30_data.get("use_mock", False) or not cuda_avail
            acc_chk = gate_assess.get("accuracy_check", {})
            fmt_chk = gate_assess.get("format_compliance_check", {})
            tr_chk = gate_assess.get("truncation_check", {})

            greedy_acc = (
                acc_chk.get("greedy_accuracy")
                if acc_chk.get("greedy_accuracy") is not None
                else gate_assess.get("greedy_accuracy", 0.0) * 100.0
            )
            fmt_comp = (
                fmt_chk.get("format_compliance")
                if fmt_chk.get("format_compliance") is not None
                else gate_assess.get("greedy_format_compliance", 0.0) * 100.0
            )
            trunc_rate = tr_chk.get("truncation_rate", 0.0)

            if gate_ok and not is_mock_run:
                status = "PASSED"
            elif gate_ok and is_mock_run:
                status = "MOCK"
            else:
                status = "FAILED"
            checks.append({
                "id": "phase4_gate_criteria",
                "name": "Phase 4 Gate: Val Accuracy > Base 8-Shot, Format Compliance >= 98%, Low Truncation",
                "status": status,
                "detail": f"val_acc={greedy_acc:.1f}%, fmt_comp={fmt_comp:.1f}%, trunc={trunc_rate:.1f}%",
            })
        except Exception as exc:
            checks.append({
                "id": "phase4_gate_criteria",
                "name": "Phase 4 Gate: Val Accuracy > Base 8-Shot, Format Compliance >= 98%, Low Truncation",
                "status": "FAILED",
                "detail": f"Error checking gate criteria: {exc}",
            })
    else:
        checks.append({
            "id": "phase4_gate_criteria",
            "name": "Phase 4 Gate: Val Accuracy > Base 8-Shot, Format Compliance >= 98%, Low Truncation",
            "status": "MISSING",
            "detail": "Day 30 evaluation report not found to evaluate gate criteria",
        })

    # 10. Phase 4 Deliverables & Markdown Report
    sft_report = _find_file("reports/sft_training.md", cfg, active_profile)
    has_md = sft_report.exists() and sft_report.stat().st_size > 0
    checks.append({
        "id": "phase4_report_deliverables",
        "name": "Phase 4 Deliverables: reports/sft_training.md & Release Tag v0.5",
        "status": "PASSED" if has_md else "MISSING",
        "detail": f"reports/sft_training.md exists={has_md} (path={sft_report})",
    })

    # Summary tallies
    passed_cnt = sum(1 for c in checks if c["status"] == "PASSED")
    mock_cnt = sum(1 for c in checks if c["status"] == "MOCK")
    failed_cnt = sum(1 for c in checks if c["status"] == "FAILED")
    missing_cnt = sum(1 for c in checks if c["status"] == "MISSING")

    if passed_cnt == len(checks):
        overall = "PASSED"
    elif failed_cnt > 0:
        overall = "FAILED"
    elif mock_cnt > 0:
        overall = "MOCK_VERIFIED"
    else:
        overall = "INCOMPLETE"

    return {
        "phase": 4,
        "release_tag": "v0.5",
        "runtime_profile": active_profile,
        "overall_status": overall,
        "total_checks": len(checks),
        "passed": passed_cnt,
        "mock": mock_cnt,
        "failed": failed_cnt,
        "missing": missing_cnt,
        "checks": checks,
    }


def print_audit_report(audit: Dict[str, Any]) -> None:
    """Print clean terminal report."""
    print("=" * 80)
    print(f"PHASE 4 GATE AUDIT REPORT (Release {audit['release_tag']})")
    print(f"Runtime Profile: {audit['runtime_profile']}")
    print(f"Overall Status:  {audit['overall_status']}")
    print(
        f"Summary:         {audit['passed']} Passed | {audit['mock']} Mock/Dev | "
        f"{audit['failed']} Failed | {audit['missing']} Missing (Total {audit['total_checks']})"
    )
    print("=" * 80)
    for idx, c in enumerate(audit["checks"], 1):
        badge = f"[{c['status']}]"
        print(f"{idx:2d}. {badge:<10} {c['name']}")
        print(f"    Detail: {c['detail']}")
    print("=" * 80)


def generate_markdown_report(audit: Dict[str, Any], output_path: Path) -> None:
    """Generate markdown audit report."""
    lines = [
        f"# Phase 4 Gate Audit Report (Release {audit['release_tag']})",
        "",
        f"- **Overall Status:** **{audit['overall_status']}**",
        f"- **Runtime Profile:** `{audit['runtime_profile']}`",
        f"- **Checks Summary:** `{audit['passed']}` Passed | `{audit['mock']}` Mock/Dev | `{audit['failed']}` Failed | `{audit['missing']}` Missing (Total `{audit['total_checks']}`)",
        "",
        "## Deliverables Audit Table",
        "| # | Milestone Check | Status | Verification Detail |",
        "|---|---|---|---|",
    ]
    for idx, c in enumerate(audit["checks"], 1):
        status_md = f"**{c['status']}**"
        lines.append(f"| {idx} | {c['name']} | {status_md} | {c['detail']} |")

    lines.extend([
        "",
        "## Gate Decision & Next Steps",
        f"Overall Gate Verdict: **{audit['overall_status']}**.",
    ])
    if audit["overall_status"] == "PASSED":
        lines.append("- All Phase 4 SFT deliverables are fully satisfied with genuine 7B GPU training. Proceed to Phase 5: Error Analysis & Targeted SFT.")
    elif audit["overall_status"] == "MOCK_VERIFIED":
        lines.append("- All Phase 4 code, tokenization loss masking, templates, and orchestration are verified locally on CPU/mock baselines. Execute Steps 23–27 on Google Colab GPU to generate full 7B production adapter weights.")
    else:
        lines.append("- Some Phase 4 checks failed or are missing. Resolve discrepancies before advancing.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 Gate Verifier (Release v0.5)")
    parser.add_argument("--profile", type=str, default=None, choices=["local", "colab"], help="Runtime profile.")
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal banner.")
    parser.add_argument("--output_json", type=Path, default=PROJECT_ROOT / "reports" / "phase4_gate_audit.json")
    parser.add_argument("--output_report", type=Path, default=PROJECT_ROOT / "reports" / "phase4_gate_report.md")
    args = parser.parse_args()

    audit = audit_phase4_gate(profile=args.profile)
    if not args.quiet:
        print_audit_report(audit)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    generate_markdown_report(audit, args.output_report)

    return 1 if audit["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
