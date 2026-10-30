# Experiment Log

This file tracks every experimental run, training phase, baseline evaluation, and pipeline milestone in chronological order (newest runs first).

---

## Run Entry: Phase 0 Setup & Infrastructure Gate (v0.1)

- **Date:** 2026-10-30
- **Run ID:** `setup-20261030_213500-phase0_infrastructure_gate`
- **Git Commit:** `d5054a8`
- **Branch:** `phase-0-setup`
- **Hardware / Platform:** Local workstation / Colab Runtime (PyTorch, CPU/CUDA)
- **Hours & Compute Units:** ~0.0 G4 hours (CPU & dry-run simulation only)

### Configurations & Environment Snapshot
- Profiles: `local` (workstation development) and `colab` (Google Drive mount at `/content/drive/MyDrive/math-reasoning-llm`)
- Hugging Face Cache: Configured strictly on local SSD disk (`/content/hf_cache`), never mounted on Drive.
- Answer Contract: Strict `#### <number>` final line format with regex `####\s*(-?[\d,]+(?:\.\d+)?)`.

### Key Metrics & Verifications
- **Unit Test Suite:** 19/19 passing tests (`tests/test_config.py`, `tests/test_run_info.py`, `tests/test_checkpointing.py`).
- **Path Resolution:** All 8 stages (`baseline`, `reasoning_data`, `sft`, `error_analysis`, `sft_targeted`, `grpo`, `final_eval`, `serve`) resolve properly across both profiles.
- **Run Provenance:** `run_info.py` successfully records git hash, detects uncommitted changes, logs GPU/CPU hardware and library versions, and persists structured JSON metadata.
- **Checkpoint Resume Gate:** Forced interruption test (`scripts/test_resume_simulation.py`) verified that a stopped training run properly resumes from `checkpoint-30` instead of restarting from step 0.
- **vLLM Verification Note:** Tested dependency configuration. If `vllm` installation encounters PyTorch binary ABI mismatches on a given Colab environment, `generation.py` is architected to gracefully fall back to native Hugging Face batched generation.

### Decisions Taken & Status
- **Phase 0 Gate:** PASSED.
- Repository tagged: `v0.1`.
- **Next Phase:** Phase 1: Data and Answer Contract (GSM8K dataset download, cleaning, 500-sample validation split carving, 8,792 gold solution parser testing, prompt templates, and EDA).
