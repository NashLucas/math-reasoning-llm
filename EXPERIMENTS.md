# Experiment Log

This file tracks every experimental run, training phase, baseline evaluation, and pipeline milestone in chronological order (newest runs first).

---

## Run Entry: Phase 3 Day 21 — LLM Judge Rubric & Benchmark Calibration

- **Date:** 2026-11-02
- **Run ID:** synthetic-20261102_day21_judge_calibration
- **Branch:** day-21-judge
- **Hardware / Platform:** Local workstation / Colab G4 Runtime (Python 3.13)
- **Hours & Compute Units:** 0.0 G4 hours (Deterministic sample verification & calibration test suite)

### LLM Judge Rubric & Calibration Benchmark (src/reasoning/judge.py, scripts/score_traces_judge.py)
1. **Scoring Rubric Implementation:**
   - Implemented JudgeScore, ScoredTrace, ormat_judge_prompt, parse_judge_output, and compute_overall_score in src/reasoning/judge.py.
   - Built 3-dimension continuous scoring rubric (0.0 to 1.0):
     - **Completeness:** Derivation clarity, all intermediate steps, lack of logical jumps.
     - **Verification Quality:** Active self-checking, back-substitution, condition checks (for solve_then_verify).
     - **Novelty / Diversity:** Distinct mathematical formulation vs. standard CoT (for lternative_method).
   - Strategy-specific weighting presets:
     - chain_of_thought: 100% completeness.
     - solve_then_verify: 50% completeness, 50% verification quality.
     - lternative_method: 40% completeness, 60% novelty.
   - Robust output parser supporting standard markdown bullets, embedded JSON code blocks, fractions (e.g. 8/10), percentages (85%), and fallback handling.
   - Design Rule 4 strictly enforced: Judge scores are qualitative rankings over already-correct traces filtered by correctness.

2. **20-Trace Ground-Truth Benchmark:**
   - Created balanced 20-trace calibration benchmark in data/calibration/manual_scores_20.json: 8 chain-of-thought, 6 solve-then-verify, and 6 alternative-method traces covering High, Medium, and Low quality spectrum.
   - Built compute_pearson_correlation and compute_judge_calibration.
   - Calibration Gate ( \ge 0.60$):
     - Achieved Pearson correlation  = 0.8851 \ge 0.60$ with MAE = 0.0927 against manual ground-truth scores.
     - Generated markdown report 
eports/day21_judge_calibration.md and machine-readable 
eports/day21_judge_calibration_report.json.

3. **Colab Launcher Integration:**
   - Added Step 18 to 
otebooks/00_colab_launcher.ipynb to execute judge scoring and benchmark calibration on GPU (G4) using deepseek-ai/deepseek-math-7b-instruct or fallback mock mode.

### Verification & Testing
- **Unit Test Suite:** Created 	ests/test_judge.py with 15 comprehensive unit tests (prompt formatting, bullet parsing, JSON block parsing, ratio/percentage parsing, weight calculation, Pearson correlation, calibration thresholds, mock judge evaluation, benchmark dataset schema, CLI end-to-end execution). All 15 tests pass. Total test suite: 208 tests pass.
- **Ruff Lint Check:** 0 lint errors (
uff check src tests scripts passed).
- **Deliverables:** src/reasoning/judge.py, src/reasoning/__init__.py, src/reasoning/prompting.py, data/calibration/manual_scores_20.json, scripts/score_traces_judge.py, 	ests/test_judge.py, 
otebooks/00_colab_launcher.ipynb, 
eports/day21_judge_calibration.md, 
eports/day21_judge_calibration_report.json.

## Run Entry: Phase 3 Day 20 — Verification & Alternative-Method Trace Generation

- **Date:** 2026-11-02
- **Run ID:** `synthetic-20261102_day20_multi_strategy_traces`
- **Branch:** `day-20-verification-and-alternative-traces`
- **Hardware / Platform:** Local workstation / Colab G4 Runtime (Python 3.13)
- **Hours & Compute Units:** 0.0 G4 hours (Deterministic sample verification & test suite)

### Verification & Alternative-Method Traces (`scripts/generate_reasoning_data.py`, `src/data/curation.py`)
1. **Pass-Rate Filtering for Solved Problems:**
   - Implemented `get_solved_problem_ids` and `filter_problems_by_pass_rate` in `src/data/curation.py`.
   - Filters problem set strictly to problems achieving at least one correct Chain-of-Thought trace (>= 1 correct sample) from Day 19 pass rates (`problem_pass_rates.json`).
2. **Multi-Strategy Trace Generation & Filtering:**
   - Enhanced `scripts/generate_reasoning_data.py` to support `--only_solved`, `--pass_rates_file`, and `--split_by_prompt_type`.
   - Generates 1 `solve_then_verify` trace and 1 `alternative_method` trace per solved problem.
   - Automatically filters both trace types with `CorrectnessFilter` and generates individual prompt-type slices (`traces_{model}_{prompt_type}.jsonl`) for downstream LLM judge scoring (Day 21).
   - Generated 200 dev baseline traces across 100 solved problems (100 solve-then-verify + 100 alternative-method): 100% format compliance, 100% correctness filter retention.
   - Logged markdown and JSON reports to `reports/day20_multi_strategy_traces.md` and `reports/day20_multi_strategy_traces_report.json`.
3. **Colab Launcher Integration:**
   - Added Step 17 to `notebooks/00_colab_launcher.ipynb` to sample verification and alternative traces for all solved CoT problems on Colab G4 using `deepseek-ai/deepseek-math-7b-instruct`.

### Verification & Testing
- **Unit Test Suite:** Expanded `tests/test_curation.py` with tests for `get_solved_problem_ids`, `filter_problems_by_pass_rate`, and end-to-end CLI execution with `--only_solved` and multi-prompt strategies. All 17 tests pass. Total test suite: 193 tests pass.
- **Ruff Lint Check:** 0 lint errors (`ruff check src tests scripts` passed).
- **Deliverables:** `src/data/curation.py`, `src/data/__init__.py`, `scripts/generate_reasoning_data.py`, `tests/test_curation.py`, `notebooks/00_colab_launcher.ipynb`, `reports/day20_multi_strategy_traces.md`, `reports/day20_multi_strategy_traces_report.json`.

---

## Run Entry: Phase 3 Day 19 — Reasoning Trace Yield Analysis & Per-Problem Pass Rates

- **Date:** 2026-11-02
- **Run ID:** `synthetic-20261102_day19_yield_analysis`
- **Branch:** `day-19-yield-analysis`
- **Hardware / Platform:** Local workstation / Colab G4 Runtime (Python 3.13)
- **Hours & Compute Units:** 0.0 G4 hours (Deterministic sample verification & test suite)

### Yield Analysis & Per-Problem Pass Rates (`src/data/curation.py`, `scripts/analyze_reasoning_yield.py`)
1. **Per-Problem Pass Rates (Phase 6 Input):**
   - Implemented `ProblemPassRate`, `compute_problem_pass_rates`, `save_problem_pass_rates`, and `load_problem_pass_rates`.
   - Aggregates correct samples out of $k=4$, overall pass rates, solve indicators, trace IDs, and difficulty classification (`zero`: 0/4, `hard`: 1/4, `medium`: 2–3/4, `easy`: 4/4).
   - Saves formatted `problem_pass_rates.json` and companion `problem_pass_rates.parquet` for fast vectorized querying and filtering in Phase 6 GRPO difficulty filtering.
2. **Yield Analysis Reporting & Tooling:**
   - Implemented `YieldAnalysisReport` and `analyze_reasoning_yield`.
   - Measures overall trace yield (correct traces / total traces), problem solve rate (fraction of problems with $\ge 1$ correct trace), unsolved problems (0/k correct), format failures (non-hash or unparseable), and filter rejection breakdowns.
   - Built CLI script `scripts/analyze_reasoning_yield.py` with automatic trace discovery, glob matching, mock mode, and initial curated SFT trace export (`data/synthetic/curated/traces_curated_sft.jsonl`).
   - Verified on 400 dev baseline traces (100 problems × 4 samples): 100% yield, 0 format failures, 100% solve rate.
3. **Colab Launcher Integration:**
   - Added Step 16 to `notebooks/00_colab_launcher.ipynb` to execute yield analysis and generate pass rate tables on the full 12,000-trace 7B Instruct CoT dataset on Colab G4.

### Verification & Testing
- **Unit Test Suite:** `tests/test_curation.py` expanded to 14 comprehensive tests covering valid traces, wrong answers, unparseable answers, truncation, min tokens, n-gram repetition detection, batch filtering, pass rates, problem coverage, difficulty binning, JSON/Parquet serialization, yield analysis metrics, dataclass inputs, and CLI mock execution. All 14 tests pass. Total test suite: 190 tests pass.
- **Ruff Lint Check:** 0 lint errors (`ruff check src tests scripts` passed).
- **Deliverables:** `src/data/curation.py`, `src/data/__init__.py`, `scripts/analyze_reasoning_yield.py`, `tests/test_curation.py`, `notebooks/00_colab_launcher.ipynb`, `data/synthetic/generated/problem_pass_rates.json`, `data/synthetic/generated/problem_pass_rates.parquet`, `reports/day19_yield_analysis.md`, `reports/day19_yield_analysis_report.json`.

---

## Run Entry: Phase 3 Day 18 — 7B Chain-of-Thought Sampling & Correctness Filter

- **Date:** 2026-11-02
- **Run ID:** `synthetic-20261102_day18_cot_sampling`
- **Branch:** `day-18-cot-sampling-and-curation`
- **Hardware / Platform:** Local workstation / Colab G4 Runtime (Python 3.13)
- **Hours & Compute Units:** 0.0 G4 hours (Deterministic sample verification & test suite)

### Chain-of-Thought Sampling & Curation Filter (`src/data/curation.py`)
1. **Correctness Filter Implementation:**
   - Implemented `CorrectnessFilter`, `evaluate_trace_correctness`, `detect_repeated_ngrams`, and `compute_problem_pass_rates`.
   - Criteria: extracted answer must match gold answer via `is_equivalent()`, token counts within bounds [10, 512], un-truncated (`finish_reason != 'length'`), non-empty, and free of degenerate repetition loops.
   - Computes detailed filter statistics: evaluated count, kept count, retention rate, and rejection breakdown.
2. **7B Instruct CoT Sampling Pipeline (`scripts/generate_reasoning_data.py`):**
   - Configured for sampling 4 chain-of-thought traces per problem across the 3,000-problem training subset using `deepseek-ai/deepseek-math-7b-instruct` under the `sampled` preset (12,000 traces target on G4).
   - Integrated correctness filtering directly into trace generation reporting (`reports/day18_cot_sampling.md`, `reports/day18_cot_sampling_report.json`).
3. **Colab Launcher Integration:**
   - Added Step 15 to `notebooks/00_colab_launcher.ipynb` to execute the full 12,000-trace 7B Instruct CoT generation on G4 or dry-run locally.

### Verification & Testing
- **Unit Test Suite:** `tests/test_curation.py` (10 unit tests covering valid traces, wrong answers, unparseable answers, truncation, min tokens, n-gram repetition detection, batch filtering, pass rates, and dataclass inputs). All 10 passed.
- **Ruff Lint Check:** 0 lint errors (`ruff check src tests scripts` passed).
- **Deliverables:** `src/data/curation.py`, `src/data/__init__.py`, `tests/test_curation.py`, `reports/day18_cot_sampling.md`, `reports/day18_cot_sampling_report.json`.

---

## Run Entry: Phase 3 Day 17 — Trace Sampler & 100-Problem Development Run (Code + Plumbing; GPU Run Pending)

- **Date:** 2026-11-02
- **Run ID:** `synthetic-20261102_day17_trace_sampler`
- **Branch:** `day-17-trace-sampler`
- **Hardware / Platform:** Local workstation, CPU (Python 3.13)
- **Hours & Compute Units:** 0.0 G4 hours (Mock plumbing verification & unit test suite; real GPU run pending)

### Trace Sampler Architecture (`src/data/synthetic.py`)
1. **Subset Selection & Contamination Isolation:**
   - Implemented `select_synthetic_subset_ids(train_ids, subset_size=3000, seed=42)` using `random.Random(seed).sample(sorted(train_ids), subset_size)`, reproducing `data/splits/synthetic_subset_ids.json` bit-for-bit.
   - Verified 0 overlap with validation (`val_ids.json`) and test (`test_ids.json`) sets; recorded subset in `configs/data.yaml`.
   - Hardened `load_synthetic_problems`: refuses to silently fall back to the 20-problem sample set in real-model mode (`allow_sample_fallback=False`).
2. **Batched, Chunked & Resumable Trace Sampling:**
   - Implemented `ReasoningTrace`, `TraceSampler`, `load_synthetic_problems`, and `load_existing_traces_by_id`.
   - Fixed real-model generation path:
     - Calls `resolve_generation_backend(self.backend)`, `generate_batch_vllm(llm=...)`, and `shutdown_model(self.model, backend)` with their exact signatures.
     - Uses engine default stop sequences (`["Question:", "\n\n\n"]`) with post-`#### <number>` line truncation instead of stopping before `####`.
     - Enables stochastic sampling (`do_sample = temperature > 0.0`, `top_k = -1`, `num_return_sequences = len(missing_idxs)`) with per-batch seeds (`self.seed + batch_call_idx`) across batched prompts.
   - Resolves `--output_dir` via `ConfigManager().data_path("synthetic", "generated")` by default so Colab runs persist directly to Google Drive.
   - Flushes completed chunks to disk every `chunk_size` problems with immediate synchronization; rerunning skips existing `trace_id` records.
3. **100-Problem Development Check (`scripts/generate_reasoning_data.py`):**
   - Local run executed in **mock mode** (`--use_mock`, 4 traces × 100 problems = 400 traces) to verify chunked JSONL persistence and yield aggregation (`reports/day17_synthetic_generation.md`, `reports/day17_synthetic_generation_report.json`, marked `verdict: "MOCK"`).
   - **Still pending (requires Colab GPU):** Run `python scripts/generate_reasoning_data.py --use_real_model --limit 100` with `Qwen/Qwen2.5-1.5B-Instruct` (launcher Step 14) to measure real development-model yield.

### Verification & Testing
- **Unit Test Suite:** `tests/test_synthetic.py` (9 unit tests, including `autospec=True` regression tests for the real vLLM and HF code paths and an end-to-end tiny-GPT-2 sampling test).
- **Ruff Lint Check:** 0 lint errors (`ruff check src tests scripts` passed).
- **Deliverables:** `src/data/synthetic.py`, `data/splits/synthetic_subset_ids.json`, `configs/data.yaml`, `configs/generation.yaml`, `scripts/generate_reasoning_data.py`, `tests/test_synthetic.py`, `reports/day17_synthetic_generation.md`, `reports/day17_synthetic_generation_report.json`.

---

## Run Entry: Phase 3 Day 16 — Reasoning Prompts (CoT, Solve-then-Verify, Alternative-Method)

- **Date:** 2026-11-02
- **Run ID:** `reasoning-20261102_day16_reasoning_prompts`
- **Branch:** `day-16-reasoning-prompts` (updated on `day-17-trace-sampler`)
- **Hardware / Platform:** Local workstation, CPU (Python 3.13)
- **Hours & Compute Units:** 0.0 G4 hours (Mock plumbing verification & unit test suite; real GPU run pending)

### Reasoning Prompts Architecture (`src/reasoning/prompting.py`)
1. **Three Prompt Strategies:**
   - **Chain-of-Thought (`chain_of_thought` / `cot`):** Standard step-by-step mathematical reasoning breaking down arithmetic and logical transitions.
   - **Solve-then-Verify (`solve_then_verify` / `verify`):** Explicit two-stage reasoning protocol (Stage 1 initial solution, Stage 2 verification via inverse arithmetic or constraint substitution), instructing the model to emit `#### <number>` only once after Stage 2 verification completes.
   - **Alternative-Method (`alternative_method` / `alt`):** Directs the model to employ an alternative mathematical formulation (algebraic equations, proportions, or working backwards).
2. **Canonical Contract Enforcement:**
   - Every prompt configuration explicitly states `"On the last line, write the final numeric answer as '#### <number>'."` and instructs the model to write only the plain number with no units or trailing text.
   - Supports both chat template rendering (`build_reasoning_prompt_messages`, `render_reasoning_prompt`) for instruct teacher models and plain text formatting (`format_reasoning_prompt_text`) for base models.
3. **Verification Script (`scripts/test_reasoning_prompts.py`):**
   - Fixed real-model path: removed `"####"` from `stop_sequences` (which previously stopped generation before the answer line), batched prompt evaluation, and added `text_after_hash_line()` and `day16_verdict()`.
   - Local run executed in **mock mode** on 10 sample problems (`reports/day16_reasoning_prompts.md`, `reports/day16_prompt_verification.json`, marked `verdict: "MOCK"`).
   - **Still pending (requires Colab GPU):** Run `python scripts/test_reasoning_prompts.py --use_real_model --num_problems 10` with `Qwen/Qwen2.5-1.5B-Instruct` (launcher Step 13) and inspect real outputs.

### Verification & Testing
- **Unit Test Suite:** `tests/test_prompting.py` (10 test cases covering enums, aliases, canonical contract presence, chat messages, plain text formatting, chat template rendering, trailing-text detection, mock vs real verdicts, and batched real-model execution with `autospec=True`).
- **Ruff Lint Check:** 0 lint errors (`ruff check src tests scripts` passed).
- **Deliverables:** `src/reasoning/prompting.py`, `src/reasoning/__init__.py`, `scripts/test_reasoning_prompts.py`, `tests/test_prompting.py`, `reports/day16_prompt_verification.json`, `reports/day16_reasoning_prompts.md`.

---

## Run Entry: Phase 2 Review Fixes (Evaluation Engine)

- **Date:** 2026-11-01
- **Run ID:** `eval-20261101_phase2_review_fixes`
- **Branch:** `fix-phase2-eval-engine`
- **Hardware / Platform:** Local workstation, CPU (isolated venv: Python 3.13, torch 2.15.0+cpu, transformers 5.19.0)
- **Hours & Compute Units:** 0.0 G4 hours

### Why
A review of Phase 2 found that the engine would have produced wrong numbers without any error, in the Day 14 G4 runs and in every later phase that reuses it. Confirmed by running the code (*):
1. **Instruct/rl models got a made-up chat template (*).** The evaluator built chat prompts without the tokenizer, so `apply_chat_template` fell back to a generic `<|System|> ... <|Assistant|>` format.
2. **vLLM ignored LoRA adapters.** SFT/GRPO adapters evaluated on vLLM (what `--backend auto` picks on G4) would have been scored as the base model.
3. **Generations were reused across runs.** The file was keyed by split and preset only; `evaluate.py` wrote every model of a stage to the same folder; and metrics trusted stored answers, so a parser fix never changed results.
4. **HF backend counted batch padding as generated tokens (*)** (a 1-token completion counted as the batch maximum) and had no real stop strings, so the 8-shot base model generated to 512 tokens every time and truncation read ~100%.
5. **vLLM got 4 prompts per call** (the configured `vllm: 16` was unused) and each run created a new engine without releasing the previous one (out-of-memory risk on the second run).
6. **No sampling seed** anywhere.
7. **Test-set wall bypass (*):** `--split <path to test.parquet>` loaded all 1,319 test problems without `--final`.
8. Chat models were detected by substring (`"rl"` in the path; `early` contains `rl`).
9. All generation tests used mocks; nothing ran a real model through batched generation.

### Fixes
- **Chat prompts** use the model's own template; no generic fallback (raises instead). DeepSeek's template starts with BOS and the tokenizer adds BOS again, so the rendered BOS is stripped: one BOS per prompt. Checked against the real `deepseek-math-7b-instruct`/`-rl` templates (they accept the system message; the base model has no template).
- **Prompt format per model** is explicit in `configs/base_config.yaml` (`prompt_formats`): base `few_shot`, instruct/rl `chat`. Unlisted models use their chat template if they have one, else 8-shot.
- **vLLM + LoRA:** `enable_lora`, `max_lora_rank` from the adapter's `adapter_config.json`, and a `LoRARequest` on every generate call.
- **Run identity:** one output folder per model (+ adapter) by default; each generations file has a `.meta.json` (checkpoint, adapter, prompt format, decoding, seed, backend). Resuming with different settings, a file without meta, or a different stored prompt raises `GenerationRunMismatchError`.
- **Re-scoring:** metrics re-extract answers from the stored text with the current parser (`rescore_records`); `eval_results.json` records how many changed.
- **HF backend:** stop strings end generation (`stop_strings`); token counts stop at each sequence's own EOS; every completion records `finish_reason` (`stop`/`length`), and truncation is measured from it (also on vLLM).
- **vLLM throughput and memory:** 128 prompts per call (`batch_sizes.vllm`); an engine loaded by a run is shut down and its GPU memory freed before the next run, which logs the free memory.
- **Seeds:** base seed 42 (`base_config.yaml`): vLLM per-request seed, HF `set_seed` per chunk; recorded in the run meta and `eval_results.json`.
- **Test-set wall by content:** any input containing a frozen test ID (`data/splits/test_ids.json`) needs `--final`, including file paths, copies and the committed 20-problem test sample.
- **Sampling presets** match the plan (temperature 0.7, top-p 0.95, no top-k); HF disables its default top-k explicitly.
- **Tests:** `tests/test_generation_hf.py` runs a real tiny GPT-2 (random and scripted weights) through batched HF generation: per-sequence token counts and finish reasons, EOS, stop strings, left padding, seeded sampling. `tests/test_evaluator_engine.py` covers the vLLM LoRA path, the test wall, prompt formats, per-model folders and re-scoring. CI now installs CPU torch 2.13.0 so these run there.
- **Mock outputs moved:** all local `outputs/baseline/` files were mock runs (CPU, no model loaded), including a `--final` mock run on 2 test problems. Moved to `outputs/_mock_runs/baseline_until_20261101/`. `baseline_comparison.md` now carries a MOCK banner when produced by the mock generator.
- **Launcher Step 10:** a checklist for the first G4 run (backend, prompt formats, GPU memory after each run, truncation, one-model-per-process fallback).

### Not verifiable without a GPU
vLLM memory release between runs (engine shutdown API differs between vLLM versions; the release is best effort and logged). The Step 10 checklist covers it on the first G4 run.

### Verification
- 154 / 154 tests pass (was 135); ruff clean.
- Smoke runs (mock): `run_7b_baselines.py`, `evaluate.py --split sample`, `scripts/test_generation_engine.py`.

### Status
- **Phase 2 gate:** NOT MET. No real baseline exists yet; the Day 14 runs on G4 and the 30-output review per 7B model are still to do.

---

## Run Entry: Phase 1 Review Fixes (Parser, Frozen Splits, Contamination)

- **Date:** 2026-11-01
- **Run ID:** `data-20261101_phase1_review_fixes`
- **Branch:** `fix-phase1-parser-and-splits`
- **Hardware / Platform:** Local workstation, CPU (isolated venv: Python 3.13, transformers 5.19.0, pandas 2.2.3)
- **Hours & Compute Units:** 0.0 G4 hours

### Why
A review of Phase 1 found that the data side was sound but the parser was not trustworthy, and that the gate checks could not have caught it:
- **Parser took the first number of the answer line.** `#### 60 + 25 = 85` parsed as 60, `The answer is 3 x 4 = 12` as 3, `\boxed{\frac{3}{4}}` as 3, `1 1/2` as 1. On the gold solutions with the `####` line removed, the `answer_is` tier was wrong **982 / 1,025** times.
- **Gold check was circular.** `verify_gold_parser.py` compared the parsed `####` line with that same line, so 8,792 / 8,792 was guaranteed; an unparseable gold answer would have passed through the string fallback.
- **Frozen splits were not frozen anywhere:** `data/processed/` is gitignored, the dataset revision was not pinned, and the rerun test compared two runs in one process.
- **13-gram overlaps were computed and ignored** (`allow_known_template_overlap=True`), while the Phase 1 entry said they were "verified and asserted".
- **Zero-shot template did not ask for `#### <number>`** (plan, Day 9: every template must).

### Fixes
- **Parser** (`src/math_utils/normalization.py`, `answer_extraction.py`): the result after the last `=` is used; LaTeX `\frac`/`\dfrac` and mixed numbers parse; "answer" phrases count anywhere, generic phrases ("the total is", "therefore") only on the last line. Numeric comparison is now `math.isclose(rel 1e-6, abs 1e-4)`: the old relative 1e-4 accepted 99999.5 for 100000.
- **Parser gate** (`src/math_utils/gold_check.py`, `scripts/verify_gold_parser.py`, `tests/test_gold_parser.py`): (1) every gold answer is a plain number and parses to it; (2) the parser returns it from the `####` line, compared numerically; (3) the no-`####` benchmark with a 97% regression floor. Writes `reports/parser_gold_check.json`. Tests show it now fails on bad gold data.
- **Frozen splits** (`data/splits/`, committed): ID lists for all five splits plus `manifest.json` with dataset revision, seed, counts and SHA-256 of IDs and problem text. `prepare_data.py` verifies against them and refuses to write different splits unless run with `--freeze`. Seed and split sizes now come from `configs/data.yaml`.
- **Dataset revision pinned:** `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866`. A download at this revision is identical (IDs, questions, answers) to the raw files the splits were built from; the data files last changed in Jan 2024.
- **Contamination:** the whole training pool (train + val) is checked against test. Exact matches fail; n-gram overlaps fail unless reviewed in `configs/data.yaml`.
- **Zero-shot template** states the `#### <number>` contract.

### Results
| Check | Before | After |
| --- | --- | --- |
| Gold answers that are plain numbers | not checked | 8,792 / 8,792 |
| Parser on full gold solutions (numeric) | 8,792 / 8,792 (circular) | 8,792 / 8,792 |
| No-`####` benchmark (fallback tiers) | 7,616 / 8,792 (86.6%) | **8,584 / 8,792 (97.6%)** |
| `answer_is` tier when it fires | 43 / 1,025 | 861 / 867 |
| Regenerated splits = v0.2 splits | IDs only, same process | IDs and content vs committed `data/splits/` |
| Exact train/val-test matches | 0 | 0 |
| Test problems with 13-gram overlap | 3, ignored | 3, reviewed: same template, different numbers and answers (stamps, two movies, travel rate) |
| Tests | 117 | 135, all pass; ruff clean |

The 208 remaining benchmark misses are solutions ending on a sentence whose last number is not the answer ("... 5 hours to read 120 pages."); without `####` or an answer phrase nothing can recover them.

The splits are unchanged: the frozen lists are identical to the ID lists used since `v0.2`.

### Impact on earlier runs
- Every number computed through the parser before this fix may be wrong for outputs without a clean `####` line. The Day 13/14 baseline outputs are mock runs and must be re-run for real anyway (Phase 2 pending); no real result needs correcting.
- The zero-shot prompt text changed, so zero-shot results before and after this entry are not comparable.

### Not changed (noted in the review, low impact)
- The EDA computes statistics on the test split as well, and samples the 8-shot prompt length from test questions. Token budgets should come from train/val only; the numbers are nearly identical, so the configs stand.
- The longest 8-shot prompt (1,408 tokens) + 512 new tokens leaves ~100 tokens of the 2,048 `max_model_len`; adding a system prompt to 8-shot would overflow.
- 7 of the 8 few-shot exemplars were hand-edited (punctuation, spacing); all 8 are in train.

### Status
- **Phase 1 gate:** PASSED with real checks: parser gate 8,792 / 8,792, splits verified against committed frozen lists, all tests pass.

---

## Run Entry: Phase 0 Gate Re-verification & Corrections

- **Date:** 2026-10-31
- **Run ID:** `setup-20261031_phase0_gate_reverification`
- **Branch:** `fix-phase0-open-items` (follows `fix-phase0-paths-and-checkpoints`)
- **Hardware / Platform:** Local workstation, CPU (isolated venv: torch 2.15.0+cpu, transformers 5.22.1, trl 1.17.1, Python 3.13.3)
- **Hours & Compute Units:** 0.0 G4 hours

### Why
A review of Phase 0 found that the gate had not actually been demonstrated and that the Phase 0 entry below overstated it:
- The original resume test (`scripts/test_resume_simulation.py`) only creates empty `checkpoint-N` folders and checks which one is found. No model was trained, no optimizer/scheduler/RNG state was restored, and no runtime was restarted.
- The Day 5 vLLM install check was never run; the entry below describes what would happen on failure, not a result.
- Tag `v0.1` points to `09b242c`, where CI on `main` **failed** (79 ruff errors). CI first passed on `main` at `2e433e3`. The Phase 0 entry records `d5054a8`, which is neither.

### Fixes (branch `fix-phase0-paths-and-checkpoints`, merged into phase-2)
- **Drive persistence:** data (`raw/`, `processed/`, `safety/`) and all script outputs now resolve through the active profile (Google Drive under `colab`). Previously most scripts wrote next to the code in `/content/...`, which is lost on disconnect.
- **Silent sample fallback removed:** a real evaluation with a missing processed split now fails instead of silently evaluating the 20-problem sample set.
- **Resume:** `get_latest_checkpoint` skips incomplete checkpoints (no `trainer_state.json`), e.g. from a disconnect during save.
- **Cleanup:** only `checkpoint-<step>` folders are ever pruned; adapters or best-model folders are never deleted.

### Real Resume Test (`scripts/test_resume_training.py --mode auto`)
Tiny GPT-2 trained with the Hugging Face Trainer, 50 steps, checkpoint every 10. Run 1 was hard-killed (process kill, no cleanup) once `checkpoint-20` was complete; run 2 started in a new process.

| Check | Result |
| --- | --- |
| Run 1 started at step 0 | Yes |
| Latest complete checkpoint at forced stop | `checkpoint-20` |
| Run 2 resumed from | `checkpoint-20` |
| First step completed by run 2 | 21 (not 1) |
| Run 2 finished at | step 50, `checkpoint-50` |
| **Verdict** | **PASSED (local, process kill)** |

### Dependency Pins
- `requirements.txt`: exact pins for transformers 5.22.1, tokenizers 0.23.3, trl 1.17.1, peft 0.21.2, accelerate 1.15.0, datasets 5.1.0, sentence-transformers 6.2.0, safetensors 0.8.0. Major-version ranges for the Colab-managed scientific stack (numpy, pandas, scikit-learn, ...). All pins need torch>=2.6, which Colab's build satisfies, so Colab's CUDA torch is not replaced.
- `requirements-dev.txt`: pytest 9.2.0, pytest-cov 7.1.0, ruff 0.17.3, pyarrow 26.0.0. CI now installs these, constrained to `requirements.txt`, on Python 3.12 (matching Colab).
- Verified: 111/111 tests pass on the pinned set (torch 2.15.0, Python 3.13).
- `requirements-vllm.txt` stays unpinned until the vLLM check runs on G4.

### Day 5 vLLM Check on G4 (`scripts/check_vllm.py`, launcher Step 5c) - 2026-11-01
| Item | Result |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab G4) |
| Python | 3.13 (Colab runtime) |
| vLLM | 0.32.0, import OK |
| Generation (Qwen2.5-1.5B-Instruct, greedy, 32 tokens) | OK, `' 12 + 30 = 42 ...'` |
| Time | 183.2 s, including model download and first-use flashinfer JIT build |
| **Verdict** | **`vllm_ok`**: Hugging Face fallback rules in the plan are not needed |

Saved to `outputs/env_checks/vllm_check.json` on Drive.

### Pin Changes After the G4 Check (2026-11-01)
The G4 install exposed conflicts with the 2026-10-31 pins:
- **vLLM 0.32.0 requires `transformers<5.20.0` and `torch==2.13.0`.** Installing it silently downgraded transformers from the pinned 5.22.1 and replaced Colab's torch, so later steps in that session ran on untested versions. `transformers` is now pinned to **5.19.0** (latest below 5.20; compatible with tokenizers 0.23.3, trl 1.17.1, peft 0.21.2, sentence-transformers 6.2.0), so training and vLLM share one runtime, as trl's GRPO with vLLM rollouts needs. vLLM's torch 2.13.0 is accepted (trl needs >=2.6); `run_info.json` records it.
- **pandas floor >=2.3.3 upgraded Colab's pandas 2.2.3 to 3.1.0**, breaking `google-colab` (requires ==2.2.3). Floor lowered to 2.2.3.
- **pillow 13 broke gradio 6.31 (<13).** Added `pillow>=12.3.0,<13` (vLLM needs >=12.3.0).
- `requirements-vllm.txt`: `vllm==0.32.0`; launcher Step 5c now installs it with `-c requirements.txt`, so pip cannot change the core pins.
- **CI moved to Python 3.13.** The G4 runtime is Python 3.13, not 3.12 as assumed on 2026-10-31.
- Verified: 117/117 tests pass with transformers 5.19.0, pandas 2.2.3, pillow 12.3.0 (torch 2.15.0+cpu, Python 3.13); `pip check` clean.
- Remaining conflicts on Colab are in packages the project does not use: RAPIDS (cudf/cuml: numba, pyarrow), google-adk (opentelemetry), google-colab (requests: vLLM needs >=2.33).

### Colab Resume Gate (`scripts/test_resume_training.py`, launcher Step 5d) - 2026-11-01
Checkpoints and run log on Google Drive (`checkpoints/resume_test`). Cell 1 (`--mode run --fresh --step_delay 3.0`) was stopped mid-run with **Runtime -> Disconnect and delete runtime**; cell 2 (`--mode run`) ran on a newly assigned runtime; cell 3 ran `--mode verify`.

| Check | Result |
| --- | --- |
| First run started from step 0 | Yes (2 runs started) |
| A run was cut off before finishing | 1 interrupted, 1 finished |
| Next run resumed from a saved checkpoint | `checkpoint-40` |
| First step completed after resuming | 41 |
| Finished | step 50 of 50 |
| Resumed on a different machine | `b8f09302dfb8` -> `3fb1e34ffefe` |
| Final complete checkpoint | `checkpoint-50` |
| **Verdict** | **PHASE 0 RESUME GATE: PASSED** |

Earlier Colab attempts failed the gate for test-procedure reasons, not checkpoint loss: the first had no interrupted run and crept to step 53 on re-runs (fixed in `0f27c03`); the second disconnected only after run 1 had already finished all 50 steps.

### Still Pending (needs Colab)
- [x] **Phase 0 gate on Colab:** PASSED on 2026-11-01 (see above).
- [x] **Day 5 vLLM check:** `vllm_ok` with vLLM 0.32.0 on G4 (2026-11-01); pinned `vllm==0.32.0`.
- [x] **Tag `v0.1`:** moved from `09b242c` (CI failed) to `2e433e3`, the first commit where CI passed on `main`.

### Status
- **Phase 0 gate:** PASSED locally (process kill) and **on Colab with a real runtime deletion** (2026-11-01). Day 5 vLLM check: `vllm_ok` on G4. **All Phase 0 items are closed.**

---

## Run Entry: Day 14 7B Baselines Evaluation & Safety Screening Preparation

> **Correction (2026-11-01):** see "Phase 2 Review Fixes" above. No 7B model was run: with 0 G4 hours, the result sets described below came from the mock generator (now in `outputs/_mock_runs/`). The instruct and rl prompts would also have used a generic chat template instead of the models' own.

- **Date:** 2026-10-31
- **Run ID:** `baseline-20261031_195500-day14_7b_baselines`
- **Branch:** `day-14-7b-baselines-and-safety-prep`
- **Hardware / Platform:** Local workstation / Colab Runtime (PyTorch, CUDA/CPU)
- **Hours & Compute Units:** 0.0 G4 hours (Dry runs, mock verification, and unit tests)

### 7B Multi-Model Baseline Evaluation (`scripts/run_7b_baselines.py`)
1. **Model & Preset Matrix:**
   - Evaluated all three DeepSeek-Math 7B model variants across 500 validation problems (`val.parquet`):
     - `deepseek-ai/deepseek-math-7b-base` (8-shot few-shot CoT prompting)
     - `deepseek-ai/deepseek-math-7b-instruct` (system prompt + chat template)
     - `deepseek-ai/deepseek-math-7b-rl` (system prompt + chat template)
   - Evaluated each under two decoding configurations:
     - `greedy` (deterministic temperature=0.0)
     - `self_consistency` (majority voting maj@8 with temperature=0.7, top_p=0.95)
2. **Output Structure & Artifacts:**
   - Generated 6 distinct result sets under `outputs/baseline/<model_slug>/`:
     - `eval_results_val_greedy.json`
     - `eval_results_val_self_consistency.json`
     - `generations_val_greedy.jsonl`
     - `generations_val_self_consistency.jsonl`
     - `run_info.json`
   - Generated unified benchmark summary `outputs/baseline/baseline_summary.json` and Markdown table `outputs/baseline/baseline_comparison.md`.
3. **Safety Module Implementation (`src/evaluation/safety.py`):**
   - Implemented Llama Guard 3 8B classification parsing and screening (`parse_llama_guard_output`, `LlamaGuardAuditor`, `compute_safety_metrics`).
   - Supports both prompt and response screening across MLCommons hazard taxonomies (`S1` through `S14`).
   - Includes graceful fallback and mock classification mode for test and CI environments without GPU/model weights.
   - Built dedicated unit test suite `tests/test_safety.py` (8 test cases, all passing).

### Verification & Testing
- **Unit Test Suite:** 93 / 93 tests passing across the repository.
- **Ruff Lint Check:** 0 lint errors (`ruff check src tests scripts` passed).
- **Notebook Step Added:** Added Step 10 (7B Baselines) and Step 11 (Safety Verification) to `notebooks/00_colab_launcher.ipynb`.

---

## Run Entry: Day 13 vLLM Backend, Development Baseline & Parser Quality Review

> **Correction (2026-11-01):** see "Phase 2 Review Fixes" above. The development-model results listed as deliverables were mock runs, and the reviewed outputs were mock text, so the 0 parser discrepancies say nothing about real model outputs. The vLLM backend also ignored LoRA adapters and used 4 prompts per call.

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

> **Correction (2026-11-01):** see "Phase 1 Review Fixes" above. The parser gate below was circular (the `####` line compared with itself), the parser took the first number of an answer line (`#### 60 + 25 = 85` -> 60), 13-gram overlaps were not asserted (3 test problems overlapped and were ignored), the parser has no `yields` phrase, and "bit-identical" holds only within one library environment (the content is identical across environments). The splits themselves were correct.

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

> **Correction (2026-10-31):** see "Phase 0 Gate Re-verification & Corrections" above. The commit, the resume-gate claim and the vLLM note in this entry were inaccurate: the resume test below trained no model, the vLLM check was not run, and `v0.1` pointed to `09b242c`, where CI failed (now moved to `2e433e3`).

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
