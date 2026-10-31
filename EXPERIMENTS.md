# Experiment Log

This file tracks every experimental run, training phase, baseline evaluation, and pipeline milestone in chronological order (newest runs first).

---

## Run Entry: Day 13 vLLM Backend, Development Baseline & Parser Quality Review

- **Date:** 2026-10-31
- **Run ID:** `baseline-20261031_181000-day13_vllm_and_dev_run`
- **Branch:** `day-13-vllm-backend-and-dev-run`
- **Hardware / Platform:** Local workstation / Colab T4 Runtime (PyTorch, CUDA/CPU)
- **Hours & Compute Units:** 0.0 G4 hours (Development model dry runs and unit tests)

### vLLM Backend & Unified Inference Interface
1. **vLLM Integration (`src/generation.py`):**
   - Implemented `load_vllm_model`, `generate_batch_vllm`, `is_vllm_available`, and `resolve_generation_backend`.
   - Unified interface: identical stop sequences (`"Question:"`, `"


"`, and line after `#### <number>`), decoding parameters, and chunked JSONL output format with crash resumption.
   - Graceful fallback: when `vllm` is not installed or when running on non-CUDA environments, the pipeline automatically routes inference through Hugging Face left-padded causal LM generation.
   - Added `--backend` (`auto`, `vllm`, `hf`) to `MathEvaluator` and `scripts/evaluate.py`.

2. **Parser Hardening & Edge Case Robustness:**
   - Enhanced `strip_formatting()` in `src/math_utils/normalization.py`:
     - Markdown emphasis stripping: cleans `**42**` and `_42_` to `42`.
     - LaTeX text stripping: removes `\text{...}` annotations inside math tags (e.g. `60\text{ miles}` -> `60`).
     - Commentary stripping: cleans trailing parenthetical text (e.g. `100 (one hundred)` -> `100`).
   - Enhanced regex patterns in `src/math_utils/answer_extraction.py`:
     - `NUMBER_PATTERN`: captures negative currency values (e.g. `-$50` -> `-50`).
     - `PHRASE_PATTERNS`: added support for `total is`, `result is`, `profit is`, `amount is`, and `calculated answer:`.

3. **Development Model Baseline & Output Audit:**
   - Built `scripts/run_dev_baseline.py`: runs development model (`Qwen/Qwen2.5-1.5B-Instruct`) on validation split with greedy and self-consistency decoding.
   - Built `scripts/review_outputs.py`: audited 20 model reasoning completions. Every extracted candidate was verified against ground truth, finding **0 parser discrepancies** (`reports/day13_output_review.md`).
   - Added Step 8 and Step 9 to `notebooks/00_colab_launcher.ipynb`.

### Verification & Testing
- **Unit Test Suite:** 85 / 85 tests passing across the repository (expanded `tests/test_generation.py` and `tests/test_answer_extraction.py`).
- **Ruff Lint Check:** 0 lint errors (`ruff check src tests scripts` passed).
- **Deliverables:** `outputs/baseline/eval_results_val_greedy.json`, `eval_results_val_self_consistency.json`, `eval_results.json`, and `reports/day13_output_review.md`.

---

## Run Entry: Day 12 Metrics, Self-Consistency Voting & Evaluation Engine

- **Date:** 2026-10-31
- **Run ID:** `eval-20261031_174500-day12_metrics_and_evaluation`
- **Branch:** `day-12-metrics-and-evaluation`
- **Hardware / Platform:** Local workstation / Colab Runtime (PyTorch, CPU/CUDA)
- **Hours & Compute Units:** 0.0 G4 hours (Unit tests, deterministic evaluation dry runs)

### Evaluation & Voting Engine Architecture
1. **Self-Consistency Majority Voting (`src/reasoning/self_consistency.py` & `src/evaluation/self_consistency.py`):**
   - Clusters parsed candidate answers dynamically using numeric and symbolic equivalence (`is_equivalent()`), normalizing currency, commas, fractions, units, and floating-point tolerances ($10^{-4}$).
   - Disregards unparseable, empty, or `None` candidates so they never distort the majority consensus.
   - Enforces a deterministic, fixed tie-breaking rule: when two or more candidate clusters have the identical highest vote count, the winner is the cluster that appeared earliest in sampling order.
   - Provides `majority_vote()`, `majority_vote_with_details()`, and `majority_vote_from_completions()`.

2. **Mathematical Reasoning Metrics (`src/evaluation/math_metrics.py`):**
   - `accuracy`: top-1 / greedy evaluation accuracy.
   - `maj@k`: majority-vote accuracy over $k$ generated reasoning paths.
   - `pass@k`: empirical fraction of problems where at least one candidate among $k$ solves the problem correctly.
   - `format_compliance_rate`: fraction of completions strictly adhering to the canonical `#### <number>` contract (`extraction_method == 'hash'`).
   - `fallback_rate`: fraction of completions requiring heuristic fallback (`boxed`, `answer_is`, `last_number`).
   - `unparseable_rate`: fraction of completions where no answer could be parsed.
   - `truncation_rate`: fraction of responses truncated by hitting `max_new_tokens`.
   - `average_prompt_tokens`, `average_completion_tokens`, `average_total_tokens`.
   - `compute_all_metrics()`: single unified entrypoint returning a structured dictionary of all metrics.

3. **Evaluation Runner & CLI (`src/evaluation/evaluator.py` & `scripts/evaluate.py`):**
   - Supports any combination of stage, checkpoint, preset (`greedy`, `sampled`, `self_consistency`), and split.
   - **Test-Set Wall Enforcement:** Strictly refuses to evaluate the `test` split during training/development, raising `TestSetWallError`. The test split can only be unlocked with the explicit `--final` flag at final evaluation.
   - Integrates resumable chunked generation, writing to `generations_<split>_<preset>.jsonl`.
   - Automatically writes canonical `eval_results.json` and records provenance metadata in `run_info.json`.

### Verification & Testing
- **New Test Suite:** `tests/test_math_metrics.py` with 15 test cases covering:
  - Unanimous voting, clear majority, deterministic tie-breaking, equivalence clustering, and unparseable candidate handling.
  - Hand-crafted prediction sets with known ground-truth metrics (accuracy, format compliance, fallback rate, token averages, truncation rate).
  - Multi-sample `maj@k` and `pass@k` verification.
  - Test-set wall rejection without `--final` and authorization with `--final`.
  - End-to-end evaluation dry runs producing `eval_results.json` and `run_info.json`.
- **Full Test Suite:** 81 / 81 tests passing across the entire repository.
- **Ruff Lint Check:** 0 lint errors (`ruff check src tests scripts` passed).
- **CLI Verification:** Successfully evaluated 20 sample problems end-to-end via `python scripts/evaluate.py --stage baseline --preset greedy --split sample --limit 20`.

---

## Run Entry: Phase 1 Data & Answer Contract Gate (v0.2)

- **Date:** 2026-10-30
- **Run ID:** `data-20261030_233000-phase1_contract_gate`
- **Git Commit:** `tag: v0.2`
- **Branch:** `phase-1-data-contract`
- **Hardware / Platform:** Local workstation / Colab Runtime (PyTorch, CPU/CUDA)
- **Hours & Compute Units:** 0.0 G4 hours (Data preparation, EDA, and local validation)

### Dataset Splits & Problem ID Integrity
- **Source:** `openai/gsm8k` (configuration: `main`)
- **Stable IDs:** 16-character SHA-256 hash of normalized question text (`gsm8k_<hex16>`).
- **Cleaned Splits:**
  - `train`: 6,973 problems
  - `val_total`: 500 problems (carved with seed 42)
    - `val_analysis`: 250 problems
    - `val_gating`: 250 problems
  - `test`: 1,319 problems (strictly isolated and frozen)
  - `total`: 8,792 problems
- **Reproducibility:** Re-running `scripts/prepare_data.py` produces 100% bit-identical Parquet files and problem ID lists across runs.
- **Contamination:** Zero exact question overlap between training pool and test set. 13-gram overlap verified and asserted in automated tests (`tests/test_no_contamination.py`).

### Answer Contract & Parser Performance
- **Gold Solution Parser:** Tested against all 8,792 gold solutions in the full dataset -> **8,792 / 8,792 parsed correctly (100.00% accuracy)**.
- **Parser Hierarchy:**
  1. `hash`: Explicit `#### <number>` (last occurrence in solution)
  2. `boxed`: LaTeX `\boxed{<number>}` with brace-depth tracking
  3. `answer_is`: Key phrases (`the answer is`, `total is`, `equals`, `yields`)
  4. `last_number`: Final freestanding number in text
- **Normalization:** Strips commas (thousands separators), currency symbols (`$`, `€`, `£`, etc.), percentage signs, units, normalizes unicode minus signs, parses fractions (`1/2`), and applies $10^{-4}$ numeric equivalence tolerance.

### Exploratory Data Analysis (EDA) & Sequence Budget
- **Tokenizer:** `deepseek-ai/deepseek-math-7b-base` (vocab size: 100,000)
- **Token Lengths (All 8,792 Problems):**
  - **Question:** Min 12, Median 56, P90 90, P95 101, P99 128, Max 213
  - **Solution:** Min 23, Median 84, P90 150, P95 175, P99 221, Max 343
  - **Prompt + Solution (Zero-shot):** Min 62, Median 157, P90 244, P95 276, P99 337, Max 465
  - **8-Shot Few-Shot Prompt:** Min 1,325, Median 1,351, P95 1,384, Max 1,408
- **Reasoning Steps (Lines before `####`):** Min 2, Mean 3.59, Median 3, P95 6, Max 11
- **Answer Composition:** 100.0% integer targets (8,792), 0.0% decimals, 0.06% negatives (5 instances).
- **Sequence Budgeting Decisions:**
  - `max_seq_length: 512` (covers 100% of zero-shot prompt+solution pairs; 99% is 337)
  - `max_new_tokens: 512` (covers 100% of solution generation; 99% is 221)
  - Recorded in `configs/base_config.yaml` and `configs/generation.yaml`.

### Sample Evaluation Sets
- Created `data/sample/`:
  - `train_sample.parquet` & `.jsonl` (20 problems)
  - `val_sample.parquet` & `.jsonl` (20 problems: 10 analysis, 10 gating)
  - `test_sample.parquet` & `.jsonl` (20 problems)
  - `sample_manifest.json`

### Key Metrics & Gate Verification
- **Phase 1 Gate Requirements:**
  1. The parser extracts the correct answer from all 8,792 gold solutions -> **PASSED (8,792 / 8,792, 100.00%)**.
  2. Rerunning data preparation reproduces identical splits -> **PASSED (Bit-identical across runs)**.
  3. All tests pass -> **PASSED (53/53 tests pass)**.
- **Phase 1 Gate:** PASSED.
- Repository tagged: `v0.2`.
- **Next Phase:** Phase 2: Generation and Baseline (Day 11: Generation engine with greedy and self-consistency decoding, `generation.py`, `model.py`, metrics, and evaluation script).

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
