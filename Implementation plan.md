phases/07-transformers-deep-dive/02-self-attention-from-scratch/docs/en.md# Implementation plan

Oct 29, 2026 · @Abdul-Azeem Traoure Mohammed

This tab is the day-by-day plan for building the project from scratch: 63 working days over 13 weeks, from Nov 2, 2026 to Jan 29, 2027, for one person working 3–4 hours a day. Each phase gives its outputs, gate and pitfalls, then every working day's tasks with a "done when" check. Resources, metrics, testing, working practices, fallback rules and open decisions follow the phases.

## At a glance

The work runs back to back in nine phases, reaching v1.0 on Friday 22 January 2027 with the last week held as buffer. Phases 3, 5 and 6 take two weeks each; Phases 2 through 8 use G4 time.

&#91;embedded content: project schedule · 9 phases over 13 weeks\]

Each phase ends at its gate, described in its section below. A failed gate uses the fallback rules near the end of this tab rather than extending the schedule.

## Working assumptions

The schedule rests on six assumptions. If one changes, dates shift, but the phase order and gates stay the same.

- **Team:** one person doing all engineering, experiments and writing.
- **Time:** 3–4 hours per working day, Monday to Friday, about 200 hours over 63 working days. 25 December and 1 January are days off. Weekends are free, except that a long G4 run may keep going over a weekend.
- **Starting point:** an empty repository. Only `deepseek-math-7b-base` is fixed; no earlier code is reused.
- **Compute:** Colab with paid compute units. About 320 units for G4 work (see the compute budget in the overview tab), plus a small amount for cheaper development runtimes.
- **Start:** Monday 2 November 2026. The weeks of 21 and 28 December are planned lighter.
- **G4 days:** days that use the G4 show the expected G4 hours. On those days the run works in the background while you do the day's other tasks.

## Phase 0: Project setup

Week 1, Monday 2 to Friday 6 November, about 17 hours, no G4 time. The goal is an empty but working repository, with configs, Colab launcher, Drive persistence, run records, resume and CI in place before any math code exists.

**Outputs:** the `math-reasoning-llm` repository tagged `v0.1`; the Drive folder layout; a working launcher notebook; CI passing.

**Gate:** a forced disconnect resumes training from its checkpoint, and CI passes on `main`.

### Day by day

**Day 1 · Mon 2 Nov · 3 h · Accounts and access**

- [ ] Create the GitHub repository `math-reasoning-llm` with a README stub, licence, `.gitignore` and `.env.example`
- [ ] Create a Hugging Face access token and request access to Llama Guard 3 8B
- [ ] Read the licences on the three DeepSeek-Math 7B model pages
- [ ] Buy Colab compute units (or Colab Pro) and upgrade Drive storage to about 100 GB
- [ ] Create the Drive folder `math-reasoning-llm/` with `checkpoints/`, `outputs/`, `logs/` and `data/`

*Done when:* every account works and Llama Guard access is requested.

**Day 2 · Tue 3 Nov · 4 h · Skeleton and config system**

- [ ] Create the full folder tree from the overview tab, with empty modules
- [ ] Write `paths.yaml` (local and colab profiles), `base_config.yaml` and `pipeline.yaml`
- [ ] Write `src/config.py`: load and merge configs, and resolve each stage's folders for a given profile
- [ ] Write `test_config.py`

*Done when:* one command prints every stage's resolved paths under both profiles.

**Day 3 · Wed 4 Nov · 3 h · Run records and requirements**

- [ ] Write `src/run_info.py`: GPU name, CUDA version, library versions, git commit, a warning on uncommitted changes, and a config snapshot
- [ ] Add a seed helper that every script calls
- [ ] Write `requirements.txt` (on top of Colab's PyTorch), `requirements-vllm.txt` and `requirements-dev.txt`

*Done when:* a dummy script writes a complete `run_info.json` into its stage's output folder.

**Day 4 · Thu 5 Nov · 4 h · Colab launcher and resume**

- [ ] Build `00_colab_launcher.ipynb`: mount Drive, clone the repo, read the token from Colab Secrets, install requirements, print the GPU, run a script
- [ ] Write `src/training/checkpointing.py`: find the newest checkpoint in a stage folder
- [ ] On a cheap runtime, train a tiny test model for 50 steps, restart the runtime mid-run, and run again

*Done when:* the second run continues from the last checkpoint instead of step 0.

**Day 5 · Fri 6 Nov · 3 h · CI and vLLM check**

- [ ] Add `.github/workflows/ci.yml`: ruff and pytest on CPU for every push
- [ ] Try installing `requirements-vllm.txt` on a cheap runtime and note the result
- [ ] Create `EXPERIMENTS.md` with its first entry, merge to `main` and tag `v0.1`

*Done when:* CI passes on `main` and the gate is met.

**Pitfalls**

- Drive is slow with many small files. Store datasets as a few Parquet or JSONL files, not one file per example.
- Keep the Hugging Face model cache on Colab's local disk, not on Drive. Downloading a 7B model each session is faster than reading it from Drive.
- Never commit the token. The launcher reads it from Colab Secrets.

## Phase 1: Data and answer contract

Week 2, Monday 9 to Friday 13 November, about 18 hours, no G4 time. The goal is frozen GSM8K splits and an answer parser you can trust, because every later number depends on it.

**Outputs:** raw and processed GSM8K with stable problem IDs; fixed train, validation (analysis and gating halves) and test splits; the parser and normalizer; prompt templates; EDA findings; sample sets; tag `v0.2`.

**Gate:** the parser extracts the correct answer from all 8,792 gold solutions, rerunning data preparation reproduces identical splits, and all tests pass.

### Day by day

**Day 6 · Mon 9 Nov · 3 h · Download and store GSM8K**

- [ ] Write `src/data/loader.py`: load `openai/gsm8k` (main configuration) and give each problem a stable ID from a hash of its question
- [ ] Save train (7,473 problems) and test (1,319 problems) to `data/raw/gsm8k/` as Parquet
- [ ] Write `data.yaml` with the source, split sizes and seed
- [ ] Read 20 problems and their solutions by hand to learn the format

*Done when:* both raw files exist with IDs, and the counts match.

**Day 7 · Tue 10 Nov · 4 h · Cleaning and splits**

- [ ] In `preprocessing.py`, strip calculator annotations such as `<<48/2=24>>` from solutions and extract the gold answer after `####`
- [ ] Carve 500 validation problems out of train with a fixed seed, split them into an analysis half and a gating half of 250 each, and save the ID lists
- [ ] Write `scripts/prepare_data.py` to run the whole data preparation
- [ ] Write `test_splits.py`: no overlap between splits, correct counts, identical IDs on rerun

*Done when:* running `prepare_data.py` twice produces identical splits.

**Day 8 · Wed 11 Nov · 4 h · Answer parser**

- [ ] Write `answer_extraction.py`. Try, in order: the `####` line, a boxed answer, an "answer is" phrase, then the last number as a fallback. Return the value and the method that found it
- [ ] Write `normalization.py`: commas, currency signs, percent signs, trailing units, fractions, negatives, tolerance comparison
- [ ] Run the parser on all 8,792 gold solutions and fix every failure

*Done when:* the parser is correct on 8,792 of 8,792 gold solutions.

**Day 9 · Thu 12 Nov · 4 h · Tests, templates and contamination check**

- [ ] Write `test_answer_extraction.py` with at least 50 hand-written messy outputs, and `test_normalization.py`
- [ ] Write `formatting.py`: an 8-shot chain-of-thought prompt from 8 fixed, cleaned training examples; a zero-shot template; a chat-template wrapper for instruct models. Every template asks for the final line `#### <number>`
- [ ] Add the contamination check to `preprocessing.py` (exact match after normalizing text, plus long n-gram overlap) and write `test_no_contamination.py`

*Done when:* all tests pass in CI.

**Day 10 · Fri 13 Nov · 3 h · EDA and sample sets**

- [ ] Build `01_eda_gsm8k.ipynb` with the DeepSeek tokenizer: question and solution token lengths, number of solution steps, answer types (integer, decimal, negative)
- [ ] Choose `max_seq_length` and `max_new_tokens` from the EDA and record them in the configs
- [ ] Create `data/sample/`: 20 problems each from train, validation and test
- [ ] Log the phase in `EXPERIMENTS.md` and tag `v0.2`

*Done when:* the gate is met.

**Pitfalls**

- Some answers contain thousands separators, such as `1,000`. A naive parser reads `1`.
- Models sometimes write `####` more than once, or restate numbers from the question after the answer. Take the last `####` line, and make sure the fallback can't silently pick a number from the question.
- The base model ignores instructions without few-shot examples. Evaluating it zero-shot would understate the baseline and inflate your gains.

## Phase 2: Generation and baseline

Week 3, Monday 16 to Friday 20 November, about 18 hours, about 3 G4 hours. The goal is a generation and evaluation engine that every later phase reuses, plus validation baselines for the three DeepSeek-Math models.

**Outputs:** `generation.py`, `model.py`, the metrics and voting modules, `scripts/evaluate.py`, the safety module and prompt set, and `outputs/baseline/` with one folder per model; tag `v0.3`.

**Gate:** validation results exist for all three 7B models under greedy and self-consistency decoding, and the manual review finds no parsing errors.

### Day by day

**Day 11 · Mon 16 Nov · 4 h · Generation engine**

- [ ] Write `src/model.py`: load a model in bf16, with or without a LoRA adapter
- [ ] Write `generation.py` with the Hugging Face backend: batched, left-padded, chunked JSONL output that skips finished chunks on rerun
- [ ] Add stop rules: end of sequence, the line after `####`, and a new "Question:" for few-shot prompts
- [ ] Write `generation.yaml`: `greedy` (temperature 0), `sampled` (temperature 0.7, top-p 0.95) and `self_consistency` (8 samples at temperature 0.7), with `max_new_tokens` from the EDA

*Done when:* the development model answers the 20 sample problems on a cheap GPU, and a rerun skips the finished chunks.

**Day 12 · Tue 17 Nov · 4 h · Metrics, voting and the evaluate script**

- [ ] Write `math_metrics.py`: accuracy, maj@k, pass@k, format compliance, fallback rate, average tokens, truncation rate
- [ ] Write `self_consistency.py`: majority vote over parsed answers, ignoring unparseable ones, with a fixed tie-break rule
- [ ] Write `evaluator.py` and `scripts/evaluate.py`: stage, checkpoint, preset, split and an optional limit. It refuses the test split unless `--final` is given, and writes `eval_results.json` and `run_info.json`
- [ ] Write `test_math_metrics.py` on hand-made predictions with known results

*Done when:* the tests pass and `evaluate.py` runs end to end on the sample data.

**Day 13 · Wed 18 Nov · 3 h · vLLM backend and development run**

- [ ] Add the vLLM backend behind the same interface (skip it if the Day 5 check failed)
- [ ] Run the development model on all 500 validation problems with greedy and self-consistency decoding on a cheap GPU
- [ ] Read 20 outputs; add every parser miss to the tests and fix it

*Done when:* development-model results are in `outputs/baseline/`.

**Day 14 · Thu 19 Nov · 4 h · G4 ≈ 3 h · 7B baselines**

- [ ] Go through the G4 session checklist and confirm vLLM works on G4
- [ ] Run 7B base (8-shot prompt), instruct and rl on all 500 validation problems, with greedy and self-consistency decoding
- [ ] While the runs work, start `safety.py`

*Done when:* six result sets (3 models × 2 presets) are in `outputs/baseline/`.

**Day 15 · Fri 20 Nov · 3 h · Safety module and review**

- [ ] Finish `safety.py` around Llama Guard 3 8B
- [ ] Assemble `data/safety/`: about 50 harmful prompts from a public red-teaming set, plus about 50 benign math questions. Test the module on the development model
- [ ] Read 30 outputs per 7B model and check that each parsed answer matches what you'd read; add any misses to the tests
- [ ] Log the baselines in `EXPERIMENTS.md` and tag `v0.3`

*Done when:* the gate is met.

**Pitfalls**

- Decoder-only models must be left-padded for batched generation. Right padding silently corrupts outputs.
- Instruct and rl models need their chat template; the base model doesn't have one.
- A high truncation rate means the token limit is too low, and accuracy is understated.
- Llama Guard 3 is gated on Hugging Face. It was requested on Day 1; if it still isn't approved, the safety test waits until Phase 7.

## Phase 3: Reasoning data

Weeks 4–5, Monday 23 November to Friday 4 December, about 34 hours, about 5 G4 hours. The goal is a curated SFT set of correct, well-structured reasoning traces in the canonical answer format.

**Outputs:** `prompting.py`, `synthetic.py`, `judge.py`, `curation.py` and their scripts; `data/synthetic/generated/` (every trace with scores and per-problem pass rates) and `data/synthetic/curated/` (the SFT set); the inspection notebook; tag `v0.4`.

**Gate:** at least 4,000 curated traces covering at least 2,500 problems (adjust the target if the yield is very different), and the 100-trace review finds no systematic problems.

### Day by day

**Day 16 · Mon 23 Nov · 3 h · Reasoning prompts**

- [ ] Write `prompting.py`: chain-of-thought, solve-then-verify and alternative-method prompts, each asking for the final line `#### <number>`
- [ ] Try each prompt on 10 sample problems with the development model and adjust the wording

*Done when:* all three prompt types produce the canonical format on the sample problems.

**Day 17 · Tue 24 Nov · 4 h · Trace sampler**

- [ ] Write `synthetic.py`: several samples per problem, storing the trace, parsed answer, correct flag, prompt type and teacher model; chunked and resumable
- [ ] Choose a 3,000-problem training subset (excluding validation) with a fixed seed and record it in `data.yaml`
- [ ] Write `scripts/generate_reasoning_data.py` and run it on 100 problems with the development model on a cheap GPU

*Done when:* the 100-problem run produces complete records.

**Day 18 · Wed 25 Nov · 3 h · G4 ≈ 2.5 h · Chain-of-thought sampling**

- [ ] Generate 4 chain-of-thought samples for each of the 3,000 problems with 7B instruct, using the `sampled` preset
- [ ] While it runs, write the correctness filter in `curation.py` and the first cases of `test_curation.py`

*Done when:* 12,000 chain-of-thought traces are on Drive.

**Day 19 · Thu 26 Nov · 4 h · Yield analysis**

- [ ] Compute and save each problem's pass rate (correct samples out of 4); Phase 6 reuses it
- [ ] Measure the overall yield, the number of problems with no correct trace, and any format failures; fix problems in format handling
- [ ] Finish the correctness filter and its tests

*Done when:* pass rates are saved and the yield is known.

**Day 20 · Fri 27 Nov · 3 h · G4 ≈ 1.5 h · Verification and alternative-method traces**

- [ ] For every problem with at least one correct chain-of-thought trace, generate one solve-then-verify trace and one alternative-method trace
- [ ] Apply the correctness filter to both

*Done when:* both trace types are generated and filtered.

**Day 21 · Mon 30 Nov · 4 h · Judge**

- [ ] Write `judge.py`: a rubric for completeness, verification quality and novelty (alternative method compared with chain of thought), each scored 0–1, with robust score parsing
- [ ] Test it with the development model on a cheap GPU
- [ ] Score 20 traces yourself with the same rubric, for calibration

*Done when:* the judge runs on 20 traces and your own 20 scores are recorded.

**Day 22 · Tue 1 Dec · 3 h · G4 ≈ 1 h · Judge scoring**

- [ ] Score every correct trace with 7B instruct as the judge
- [ ] Compare the judge with your 20 scores and check how spread out the scores are

*Done when:* every correct trace has scores, and the agreement with your scores is recorded.

**Day 23 · Wed 2 Dec · 4 h · Curation**

- [ ] Finish `curation.py`: at most 2 traces per problem, preferring one of each type and the best judge score; remove near-duplicates, traces over the length limit and traces with repeated text
- [ ] Use the cleaned gold solution for problems with no correct trace
- [ ] Assert that no validation or test problem is present, then write `scripts/curate_data.py`

*Done when:* the curated JSONL exists and its counts are logged.

**Day 24 · Thu 3 Dec · 3 h · Inspection**

- [ ] Build `02_inspect_reasoning_traces.ipynb`: yield by problem, length distribution, judge-score histogram
- [ ] Read 50 kept and 50 rejected traces and list any systematic problems

*Done when:* the list of problems is written down.

**Day 25 · Fri 4 Dec · 3 h · Fix and freeze**

- [ ] Fix the systematic problems and rerun curation (CPU only)
- [ ] Run the contamination test; record the curated set's size and file hash in `EXPERIMENTS.md`
- [ ] Tag `v0.4`

*Done when:* the gate is met.

**Pitfalls**

- A correct final answer can hide wrong reasoning. The judge score and the manual read are there to catch it.
- Model judges tend to score everything high. If most traces score above 0.9, use the judge only to rank traces within a problem.
- Mixed answer formats in SFT data teach mixed formats. Every curated trace must end with the canonical line.

## Phase 4: SFT

Week 6, Monday 7 to Friday 11 December, about 17 hours, about 3 G4 hours. The goal is a first LoRA adapter that reasons step by step and always ends in the required format.

**Outputs:** `sft_trainer.py`, `callbacks.py`, `scripts/train_sft.py`, `lora_config.yaml` and `training_args.json`; the SFT adapter in `outputs/sft/adapter_model/` with validation results under both presets; training curves and sample generations; tag `v0.5`.

**Gate:** greedy validation accuracy beats the 8-shot 7B base model, format compliance is at least 98%, and the truncation rate is low.

### Day by day

**Day 26 · Mon 7 Dec · 4 h · SFT data and trainer**

- [ ] Turn the curated set into prompt–completion pairs with the zero-shot template, ending every completion with the end-of-sequence token
- [ ] Tokenize three examples and check by hand that the loss covers only the completion and that the end token is present and not masked
- [ ] Write `sft_trainer.py` on TRL's SFTTrainer
- [ ] Write `lora_config.yaml` (rank 16–32, alpha twice the rank, dropout 0.05, all linear layers) and `training_args.json` (bf16, gradient checkpointing, cosine schedule with a short warmup, 2–3 epochs, effective batch of about 32, `max_seq_length` from the EDA, TensorBoard, regular `save_steps`)

*Done when:* a training run starts on the sample data on CPU with the tiny test model.

**Day 27 · Tue 8 Dec · 4 h · Callback and sanity tests**

- [ ] Write `callbacks.py`: every few hundred steps, greedy generation on 100 validation problems, logging accuracy, format compliance and sample outputs to `logs/generations/`
- [ ] Write `scripts/train_sft.py`, resuming from the latest checkpoint by default
- [ ] Overfitting test: train the development model on 20 examples on a cheap GPU; loss should fall close to zero and the model should reproduce them
- [ ] Write a test that the training template matches the inference template exactly

*Done when:* the overfitting test passes.

**Day 28 · Wed 9 Dec · 3 h · Learning-rate sweep on the development model**

- [ ] Run three short trainings of the development model on a cheap GPU such as an L4, at learning rates 5e-5, 1e-4 and 2e-4
- [ ] Plot training loss, validation loss and validation accuracy; pick the learning rate
- [ ] Keep the best development adapter; Phase 6 uses it for its trial run

*Done when:* the learning rate is chosen and logged.

**Day 29 · Thu 10 Dec · 3 h · G4 ≈ 2 h · 7B SFT**

- [ ] Train the 7B model with the chosen settings, watching the curves and the callback's sample outputs
- [ ] Resume from the latest checkpoint if the session drops

*Done when:* training finishes and all checkpoints are on Drive.

**Day 30 · Fri 11 Dec · 3 h · G4 ≈ 1 h · Pick the best checkpoint**

- [ ] Evaluate every saved checkpoint on all 500 validation problems with greedy decoding
- [ ] Copy the best to `outputs/sft/adapter_model/` and evaluate it with self-consistency
- [ ] Log the results in `EXPERIMENTS.md` and tag `v0.5`

*Done when:* the gate is met.

**Pitfalls**

- If the pad token is the end-of-sequence token and padding is masked from the loss, the real end token can get masked too. The model then never learns to stop. The Day 26 hand check exists to catch this.
- Falling validation loss doesn't guarantee rising accuracy. Choose checkpoints by validation accuracy only.
- Too many epochs on a few thousand traces leads to memorized phrasing. Watch for validation accuracy peaking and then dropping.

## Phase 5: Error analysis and targeted SFT

Weeks 7–8, Monday 14 to Thursday 24 December, about 27 hours (the second week is lighter), about 3.5 G4 hours. The goal is to understand where the SFT model fails and to test whether targeted data fixes it.

**Outputs:** `error_analysis.py`, `selection.py` and `scripts/error_analysis.py`; `reports/error_analysis_sft.md` with a before-and-after comparison; `data/targeted/`; the round-2 adapter in `outputs/sft_targeted/`; tag `v0.6`.

**Gate:** round 2 clearly beats round 1 on the gating half. Otherwise Phase 6 starts from round 1, and the error report still goes in the README as a finding.

### Day by day

**Day 31 · Mon 14 Dec · 4 h · Error categories**

- [ ] Write `error_analysis.py`. Rules catch parse failures and truncation first; 7B instruct then classifies the remaining errors into a fixed list: arithmetic slip, misread question, wrong setup, incomplete reasoning, other
- [ ] Write tests for empty, truncated and unparseable outputs

*Done when:* error classification runs on sample errors with the development model.

**Day 32 · Tue 15 Dec · 3 h · G4 ≈ 0.5 h · Classify the SFT model's errors**

- [ ] Classify the SFT model's errors on the analysis half of validation only
- [ ] Label 30 of them yourself and compare; if they often disagree, tighten the category definitions and rerun

*Done when:* the automatic categories agree with most of your 30 labels.

**Day 33 · Wed 16 Dec · 3 h · Clustering and report**

- [ ] Embed the error questions with `all-MiniLM-L6-v2` and cluster them; name each cluster by topic, such as percentages, rates or time
- [ ] Write `reports/error_analysis_sft.md`: category counts, clusters and two examples of each

*Done when:* the report is written.

**Day 34 · Thu 17 Dec · 4 h · Targeted example selection**

- [ ] Write `selection.py`: embed the training pool (validation and test excluded) and, for each category, pick the 20–50 training problems most similar to that category's error questions. Attach each problem's curated trace, or its cleaned gold solution if it has none
- [ ] Write `test_selection.py`: no validation or test problem is ever returned, and per-category limits hold
- [ ] Wire both steps into `scripts/error_analysis.py`

*Done when:* the targeted set is in `data/targeted/`.

**Day 35 · Fri 18 Dec · 3 h · Round-2 dataset**

- [ ] Combine curated and targeted examples, with a config option to include targeted examples twice
- [ ] Run the contamination test and log the dataset counts

*Done when:* the round-2 dataset is frozen.

**Day 36 · Mon 21 Dec · 3 h · G4 ≈ 2 h · Round-2 training**

- [ ] Retrain from the base model on the combined data, with the same settings as round 1
- [ ] Resume from the latest checkpoint if the session drops

*Done when:* round-2 training is finished.

**Day 37 · Tue 22 Dec · 3 h · G4 ≈ 0.5 h · Compare the rounds**

- [ ] Evaluate the best round-2 checkpoint and the round-1 adapter on the gating half with greedy decoding
- [ ] Decide which adapter starts Phase 6, treating a 1–2 point difference as a tie, and copy the round-2 adapter to `outputs/sft_targeted/`

*Done when:* the decision is logged in `EXPERIMENTS.md`.

**Day 38 · Wed 23 Dec · 2 h · G4 ≈ 0.5 h · Category shift**

- [ ] Rerun error analysis on round 2's analysis-half errors
- [ ] Add the before-and-after category comparison to the report

*Done when:* the report shows how the categories moved.

**Day 39 · Thu 24 Dec · 2 h · Wrap-up**

- [ ] Finish the `EXPERIMENTS.md` entry and tag `v0.6`
- [ ] Read the GRPO items on the reading list before next week

*Done when:* the gate is met. Friday 25 December is a day off.

**Pitfalls**

- Analysis and gating use different validation halves on purpose. Data chosen to fix analysis-half errors would make round 2 look better than it is on that same half.
- A few hundred targeted examples can vanish among thousands of curated ones. If round 2 doesn't move, try including targeted examples twice before giving up.
- On 250 problems, a 1–2 point difference is only 3–5 problems. Treat it as a tie.

## Phase 6: GRPO

Weeks 9–10, Monday 28 December to Friday 8 January, about 27 hours, about 8 G4 hours. The goal is to raise accuracy with reinforcement learning from the best SFT adapter, without the model learning to game the reward.

**Outputs:** the `rewards/` package and `reward.yaml`; `grpo_trainer.py`, `grpo_args.json`, `scripts/build_grpo_set.py` and `scripts/train_grpo.py`; the GRPO problem set in `data/processed/grpo/`; the GRPO adapter in `outputs/grpo/adapter_model/`; reward and length curves; tag `v0.7`.

**Gate:** GRPO validation accuracy is at least as good as the starting SFT adapter's, format compliance stays at or above 98%, and response length hasn't grown without an accuracy gain.

### Day by day

**Day 40 · Mon 28 Dec · 3 h · Reward functions**

- [ ] Write `rewards/correctness.py`: full credit for a correct answer, partial credit for a close one, a penalty for an unparseable one
- [ ] Write `format.py` (canonical final line present), `quality.py` (visible steps, length within a sensible band) and `composite.py`
- [ ] Write `reward.yaml` with correctness carrying most of the weight; start at 1.0 for correctness, 0.2 for format and 0.1 for quality

*Done when:* the composite reward scores 10 sample outputs.

**Day 41 · Tue 29 Dec · 3 h · Reward tests**

- [ ] Write `test_rewards.py` as a table of response, gold answer and expected reward
- [ ] Include gaming attempts: an answer with no reasoning, long rambling, several `####` lines, the right number in the wrong place
- [ ] Score 50 real SFT outputs and check that the ranking matches your own judgment

*Done when:* the tests pass and the ranking looks right.

**Day 42 · Wed 30 Dec · 3 h · G4 ≈ 1 h · Difficulty filter**

- [ ] Write `scripts/build_grpo_set.py`: sample the starting SFT adapter 8 times on about 3,000 training problems, taken where possible from training problems outside the Phase 3 subset
- [ ] Keep problems solved between about 20% and 80% of the time, aiming for 1,000–1,500, and save them to `data/processed/grpo/`

*Done when:* the GRPO set is saved and its pass-rate distribution is logged.

**Day 43 · Thu 31 Dec · 3 h · GRPO trainer**

- [ ] Write `grpo_trainer.py` on TRL's GRPOTrainer: the base model plus the starting SFT adapter, kept trainable, with the composite reward
- [ ] Write `grpo_args.json`: 8 generations per problem, 400–512 new tokens, temperature 0.8, vLLM generation when available, a starting learning rate of 5e-6, small `save_steps`
- [ ] Log reward mean and spread, the share of groups where all 8 answers got the same reward, response length, format compliance and periodic validation accuracy
- [ ] Test that the loaded adapter's weights are trainable and non-zero

*Done when:* GRPO runs a few steps on CPU with the tiny test model and sample data. Friday 1 January is a day off.

**Day 44 · Mon 4 Jan · 3 h · Development GRPO run**

- [ ] Run about 100 GRPO steps with the development model and its Day 28 adapter on a cheaper GPU such as an L4
- [ ] Read 20 generations and check that rewards rise for the right reasons; adjust the learning rate if needed

*Done when:* the settings for the 7B run are chosen.

**Day 45 · Tue 5 Jan · 3 h · G4 ≈ 3 h · 7B GRPO, part 1**

- [ ] Confirm that step-0 validation accuracy matches the starting SFT adapter's result
- [ ] Train, watching reward, length and validation accuracy

*Done when:* about half the planned steps are done and checkpoints are on Drive.

**Day 46 · Wed 6 Jan · 3 h · G4 ≈ 3 h · 7B GRPO, part 2**

- [ ] Resume from the latest checkpoint and finish the run
- [ ] Stop early if a fallback rule triggers

*Done when:* training is finished.

**Day 47 · Thu 7 Jan · 3 h · G4 ≈ 1 h · Pick the best GRPO checkpoint**

- [ ] Evaluate the checkpoints on all 500 validation problems with greedy decoding
- [ ] Copy the best to `outputs/grpo/adapter_model/` and evaluate it with self-consistency

*Done when:* the best checkpoint is chosen.

**Day 48 · Fri 8 Jan · 3 h · Analysis and wrap-up**

- [ ] Plot reward and response length against validation accuracy
- [ ] Compare SFT and GRPO outputs on the same 20 problems
- [ ] Log the phase in `EXPERIMENTS.md` and tag `v0.7`

*Done when:* the gate is met.

**Pitfalls**

- GRPO learns from differences within a group. When all 8 answers are right, or all wrong, that problem teaches nothing. That's why the difficulty filter matters.
- Starting GRPO from the base model with a fresh adapter, instead of the SFT adapter, is an easy mistake. The step-0 check on Day 45 catches it.
- Rising response length with flat accuracy is a warning sign of reward gaming. Check the length curve against validation accuracy, not just the reward.

## Phase 7: Final evaluation

Week 11, Monday 11 to Friday 15 January, about 16 hours, about 5 G4 hours. The goal is the results table, computed once on the test set, with honest error bars.

**Outputs:** `stats.py` and `scripts/compare_stages.py`; test results for six models under two presets; transfer-benchmark and safety results; the results table and plots in `outputs/reports/`; the results notebook; tag `v0.8`.

**Gate:** every number in the table traces to an `eval_results.json` with a matching `run_info.json`, and every model's test run happened exactly once.

### Day by day

**Day 49 · Mon 11 Jan · 4 h · Pre-commitment and comparison tooling**

- [ ] Before touching test, write in `EXPERIMENTS.md` which checkpoint is final for each stage, and commit it
- [ ] Write `stats.py`: 95% bootstrap confidence intervals and paired bootstrap comparisons between stages, plus `test_stats.py`
- [ ] Write `scripts/compare_stages.py`: the results table and three plots (accuracy by stage, greedy against self-consistency, response length against accuracy). Try it on the validation results

*Done when:* `compare_stages.py` produces a complete table from the validation results.

**Day 50 · Tue 12 Jan · 3 h · G4 ≈ 3 h · Test runs, greedy**

- [ ] Run `evaluate.py --final` on test with greedy decoding for base (8-shot), instruct, rl, SFT, targeted SFT and GRPO
- [ ] Start the self-consistency test runs

*Done when:* six greedy test results exist.

**Day 51 · Wed 13 Jan · 3 h · G4 ≈ 1.5 h · Self-consistency and transfer**

- [ ] Finish self-consistency test runs for all six models
- [ ] Run the transfer benchmark (default: SVAMP) on the base and final models

*Done when:* all twelve test result sets and the transfer results exist.

**Day 52 · Thu 14 Jan · 3 h · G4 ≈ 0.5 h · Safety and the results table**

- [ ] Run `safety.py` with the `data/safety/` prompts on the base and final models
- [ ] Run `compare_stages.py` on the test results to produce the table with confidence intervals and the three plots in `outputs/reports/`

*Done when:* the table and plots are in `outputs/reports/`.

**Day 53 · Fri 15 Jan · 3 h · Results notebook**

- [ ] Build `03_results.ipynb`: how error categories changed across stages, and one example problem that each stage fixed
- [ ] Draft the findings for the README
- [ ] Log the phase in `EXPERIMENTS.md` and tag `v0.8`

*Done when:* the gate is met.

**Pitfalls**

- Don't retune anything after seeing test results. If you do, report it and treat those numbers as validation-quality.
- With 1,319 problems, confidence intervals are roughly ±2–2.5 points. Compare stages with the paired test, not by eye.
- vLLM sampling isn't perfectly reproducible. Record seeds and rely on the confidence intervals.

## Phase 8: Serve, monitor, publish

Week 12, Monday 18 to Friday 22 January, about 18 hours, about 1 G4 hour. The goal is a working demo, a monitoring report and a repository a stranger can follow.

**Outputs:** the merged model in `outputs/final/merged_model/`; `app/server.py` and `app/request_logging.py`; `scripts/infer.py`, `send_traffic.py` and `monitor.py`; the `monitoring/` package and `monitoring.yaml`; `outputs/reports/monitoring_report.md`; the README and `docs/model_card.md`; release `v1.0`.

**Gate:** someone else can go from a clean Colab session to the running demo using only the README.

### Day by day

**Day 54 · Mon 18 Jan · 3 h · G4 ≈ 0.5 h · Merge the final adapter**

- [ ] Write `merge_adapter.py` and merge the final adapter into the bf16 base model on a High-RAM runtime
- [ ] Check that the merged model gives the same answers as base plus adapter on 20 problems
- [ ] Optionally push the adapter to the Hugging Face Hub, after checking the base model's licence

*Done when:* the merged model is in `outputs/final/merged_model/` and passes the check.

**Day 55 · Tue 19 Jan · 4 h · G4 ≈ 0.5 h · Demo app**

- [ ] Write `app/server.py` in Gradio: a question box, the reasoning, the final answer and a self-consistency toggle, reachable through Gradio's share link from Colab
- [ ] Write `app/request_logging.py`: one JSONL line per request in `logs/production/`, with timestamp, question, response, parsed answer, tokens, latency and any error
- [ ] Write `scripts/infer.py` for quick generation from any checkpoint

*Done when:* the app answers 5 questions through the share link and logs all of them.

**Day 56 · Wed 20 Jan · 3 h · Test traffic**

- [ ] Write `scripts/send_traffic.py` and send about 300 questions: mostly validation problems, plus off-topic, very long and adversarial ones
- [ ] Check that every request was logged

*Done when:* about 300 log lines exist.

**Day 57 · Thu 21 Jan · 4 h · Monitoring**

- [ ] Write `monitoring/metrics.py` (median and 95th-percentile latency, tokens per request, error rate, format compliance), `alerts.py` with thresholds from `monitoring.yaml`, and `diagnosis.py` (extract failures and sort them by rule: off-topic, too long, unparseable, error)
- [ ] Write `scripts/monitor.py` and generate `outputs/reports/monitoring_report.md`

*Done when:* the monitoring report is generated.

**Day 58 · Fri 22 Jan · 4 h · Release v1.0**

- [ ] Write the README (overview, pipeline diagram, Colab quickstart, results table, findings, limitations) and `docs/model_card.md`
- [ ] Follow the README in a fresh Colab session, from clone to running demo
- [ ] Run CI one last time and tag `v1.0`

*Done when:* the gate is met.

**Pitfalls**

- Merging needs the base model in bf16, not 4-bit. Run it on a High-RAM runtime.
- The demo should show the full reasoning, not just the number. The reasoning is what shows that the model works rather than guesses.

## Buffer week

Week 13, Monday 25 to Friday 29 January, about 14 hours. If a phase slipped, these days go to it first and the write-up moves later.

**Day 59 · Mon 25 Jan · 3 h · Catch-up**

- [ ] Finish anything left from earlier phases; if nothing is left, tidy code and docs

**Day 60 · Tue 26 Jan · 3 h · Write-up draft**

- [ ] Draft a public write-up: the problem, the pipeline, the results table and the lessons learned

**Day 61 · Wed 27 Jan · 3 h · Write-up figures**

- [ ] Finish the write-up using the plots in `outputs/reports/`

**Day 62 · Thu 28 Jan · 3 h · Retrospective**

- [ ] Add a retrospective to `EXPERIMENTS.md`: what worked, what didn't, and what to try next

**Day 63 · Fri 29 Jan · 2 h · Publish**

- [ ] Publish the write-up with links to the repository and the model

*Done when:* the write-up is published and the project is closed.

## Resources

The project needs four accounts, six models, one core dataset, about 320 G4 compute units, roughly 100 GB of Drive storage and about 200 working hours. Set up every account and model access in week 1, because gated approvals can take days.

**Accounts and services**

| Item | Used for | Needed by | Notes |
| --- | --- | --- | --- |
| Google Colab (Pro or pay-as-you-go units) | All GPU work | Phase 0 | About 320 G4 units, plus a few for cheaper development runtimes |
| Google Drive storage | Checkpoints, outputs, data, logs | Phase 0 | The free 15 GB is too small; plan for about 100 GB |
| Hugging Face account and token | Model downloads, gated models, optional publishing | Phase 0 | Request Llama Guard 3 access in week 1; read each model's license |
| GitHub | Code, issues, releases | Phase 0 | One branch per phase, tag at each gate |
| TensorBoard (optional: Weights & Biases free tier) | Training curves | Phase 4 | TensorBoard logs live in `logs/` on Drive |

**Models**

| Model | Role | Approx. size in bf16 | Phases |
| --- | --- | --- | --- |
| `deepseek-math-7b-base` | Main training base | \~14 GB | 2–8 |
| `deepseek-math-7b-instruct` | Reference, teacher for traces, judge, error classifier | \~14 GB | 2, 3, 5, 7 |
| `deepseek-math-7b-rl` | Reference only | \~14 GB | 2, 7 |
| A \~1.5B instruct model (for example Qwen2.5-1.5B-Instruct) | Debugging every stage cheaply | \~3 GB | 0–6 |
| `Llama-Guard-3-8B` (gated) | Safety evaluation | \~16 GB | 2, 7 |
| `all-MiniLM-L6-v2` | Embeddings for clustering and example selection | <100 MB | 5 |

**Data**

- **GSM8K** (`openai/gsm8k`, main configuration): the core dataset for training, validation and test.
- **Optional transfer benchmark:** one of SVAMP or GSM-Hard for Phase 7.
- **Safety prompts:** about 50 harmful prompts from a public red-teaming set plus about 50 benign math questions, assembled on Day 15 and stored in data/safety/.

**Storage on Drive (rough)**

The merged 7B model takes about 14 GB. Each LoRA checkpoint with optimizer state is roughly 0.5 GB; keeping the last 3 per stage comes to about 10 GB. Generated traces and evaluation outputs stay under a few GB. That totals around 30 GB, so 100 GB leaves room for reruns. Model weights downloaded from Hugging Face stay on Colab's local disk, not on Drive.

**Libraries**

transformers, trl, peft, accelerate, datasets, sentence-transformers, scikit-learn, pandas, numpy, matplotlib, tensorboard, gradio, pyyaml and pytest, installed on top of Colab's PyTorch. vLLM is optional and lives in `requirements-vllm.txt`.

**Time**

The phases add up to about 190 hours over 12 weeks, plus about 14 hours in the buffer week. Reasoning data (34 hours), targeted SFT (27 hours) and GRPO (27 hours) are the largest phases.

**Skills worth refreshing first**

How GRPO computes advantages within a group; LoRA and adapter loading in PEFT; batched generation and padding; bootstrap confidence intervals. The reading list at the end covers each.

## Metrics to track

Validation accuracy with greedy decoding decides every gate. The other metrics explain why it moved, or warn before something breaks.

| Metric | What it tells you | Phases |
| --- | --- | --- |
| Validation accuracy, greedy | The decision metric for every gate | 2–7 |
| maj@8 accuracy | How much self-consistency adds on top of training | 2–7 |
| pass@8 | Share of problems solvable by sampling at all; the headroom GRPO can use | 2, 4, 6 |
| Format compliance | Whether the answer contract holds | 2–8 |
| Parser fallback rate | Outputs drifting away from the `####` format | 2–8 |
| Truncation rate | Token limit too low, or the model rambling | 2–8 |
| Average response tokens | Verbosity; an early sign of reward gaming | 4–8 |
| Trace yield (correct teacher samples) | How efficiently Phase 3 turns compute into data | 3 |
| Judge-score spread | Whether the judge separates good traces from weak ones | 3 |
| Training and validation loss | Fitting and overfitting during SFT | 4–5 |
| Reward mean and spread; share of zero-spread groups | Whether GRPO has a learning signal | 6 |
| Error category counts | Where the model fails, and how that changes by stage | 5, 7 |
| Unsafe-response rate and benign refusal rate | Safety before and after training | 2, 7 |
| Latency (median and 95th percentile), tokens per request, error rate | Production health of the demo | 8 |
| G4 hours and units spent | Budget versus plan | All |

Record compute spend per phase in `EXPERIMENTS.md` alongside results, so budget problems show up early.

## Testing and quality strategy

Quality comes from six layers, from fast automatic checks to careful human reading. The first three run automatically on every push through a free GitHub Actions workflow on CPU, so no G4 time is spent catching basic bugs.

| Layer | What it checks | When it runs |
| --- | --- | --- |
| Unit tests | Parser, normalization, rewards, metrics, majority vote, curation filters, example selection | Every push (CI) |
| Data tests | No contamination, splits unchanged and non-overlapping, every curated trace ends in the canonical format | Every push (CI) and after each data-producing script |
| Smoke tests | The whole pipeline runs end to end on sample data with a tiny model | Every push (CI) |
| Training sanity checks | Overfit 20 examples; step-0 accuracy equals the previous stage; settings tried on the 1.5B model first | Before each 7B training run |
| Human review | 30 outputs in Phase 2, 100 traces in Phase 3, 30 errors in Phase 5, 20 GRPO generations in Phase 6 | At each gate |
| Statistical checks | Bootstrap confidence intervals and paired tests between stages | Phase 7 |

**Rule of thumb:** any bug found by hand becomes a test case before it's fixed. The parser and reward tests should keep growing throughout the project.

## Working practices

Three habits do most of the work: never run uncommitted code on G4, log every run with its decision, and close each week with a budget check.

**Git**

- `main` always passes CI. Each phase gets a branch, such as `phase-3-reasoning-data`, merged through a self-reviewed pull request.
- Tag the repo at each gate: `v0.1` after Phase 0 through `v1.0` after Phase 8.
- `run_info.py` warns when the working tree has uncommitted changes, so every result maps to an exact commit.

**Experiment log**

Keep `EXPERIMENTS.md` at the repo root, one entry per run, newest first. Each entry records: date, run ID (`<stage>-<date>-<short description>`), commit, config changes, GPU, hours and units, the key metrics, the decision taken, and what comes next. This log becomes the source for the README findings and the write-up.

**Weekly cadence**

- **Monday:** plan the week's tasks and decide which G4 runs are ready. Group G4 work into one or two sessions per week.
- **Midweek:** develop and test on CPU or a cheap GPU with the 1.5B model.
- **Friday:** update `EXPERIMENTS.md`, compare units spent against the budget, and update the README's progress section.

**G4 session checklist**

- [ ] The code for this run already worked on the 1.5B model and sample data
- [ ] Configs are committed, and the tree is clean
- [ ] Drive is mounted, and the resume path points at the right stage folder
- [ ] The GPU printed by the launcher is noted (G4 or A100)
- [ ] TensorBoard or logged generations are being watched during the run
- [ ] Results are on Drive and logged in `EXPERIMENTS.md`
- [ ] The runtime is disconnected as soon as the run ends, since units are spent while it stays connected

**Feedback**

Share progress after Phase 4 with a peer or an online ML community. A second reader catches evaluation mistakes far earlier than a final review does.

## Fallback rules

These rules are decided now, so a bad result triggers a planned response instead of an unplanned detour. Check this table whenever a gate fails.

| If this happens | Do this |
| --- | --- |
| vLLM won't install on G4 | Use Hugging Face generation. Halve the Phase 3 subset, use 4–6 GRPO generations on about 600 problems, and use k=4 for self-consistency on the reference models |
| Curated set is under 4,000 traces | Sample 8 solutions per problem instead of 4, widen the subset, and fill gaps with cleaned gold solutions |
| SFT doesn't beat the base model | Check that training and inference templates match, that the end token isn't masked, and that the base model got a fair few-shot prompt. Then lower the learning rate |
| SFT format compliance is under 98% | Check that every curated trace ends with `####`, and check end-token handling |
| Targeted SFT doesn't beat round 1 | Start GRPO from round 1, and report the error analysis as a finding |
| GRPO reward is flat after \~100 steps, or many groups have zero spread | Redo the difficulty filter with the current model; try a higher temperature or more generations |
| GRPO response length grows with flat accuracy | Lower the quality-bonus weight, cap tokens tighter, and restart from the last good checkpoint |
| GRPO validation accuracy drops below SFT | Stop, keep the best earlier checkpoint, and retry with a lower learning rate |
| Fewer than 120 units left at the start of Phase 6 | Cut GRPO to about 500 problems and one pass; evaluate reference models with greedy decoding only |
| G4 unavailable for days | Run on A100 80 GB; `run_info.json` records the change |
| More than a week behind after Phase 5 | Drop the transfer benchmark and monitoring polish; protect GRPO and the final evaluation |
| Llama Guard access still pending in Phase 7 | Report safety as not evaluated, rather than substituting an unverified check |

## Decisions to make

Eleven choices are still open. Each has a default that the plan already assumes, so work can start now; record the final choice in `EXPERIMENTS.md` and set its status here.

| Decision | Decide by | Default | Alternatives | Status |
| --- | --- | --- | --- | --- |
| Development model | Phase 0 | Qwen2.5-1.5B-Instruct | Another \~1.5B instruct model | Open |
| Final answer format | Phase 1 | Last line `#### <number>` | Boxed answer | Open |
| Few-shot examples for the base model | Phase 1 | 8 chain-of-thought examples | 4 examples | Open |
| Self-consistency sample count (k) | Phase 2 | 8 | 4 or 16 | Open |
| Source of reasoning traces | Phase 3 | Rejection sampling from 7B instruct | Sampling from 7B base | Open |
| Gold solutions for unsolved problems | Phase 3 | Include them | Leave those problems out | Open |
| Round-2 training start | Phase 5 | Retrain from base on combined data | Continue round 1 at a lower learning rate | Open |
| GRPO generations per problem | Phase 6 | 8 | 12, or 4–6 as a fallback | Open |
| Transfer benchmark | Phase 7 | SVAMP | GSM-Hard, or none | Open |
| Demo framework | Phase 8 | Gradio | FastAPI with a simple page | Open |
| What to publish on the Hub | Phase 8 | Adapter and model card | Merged model too, or nothing; check the base model's license first | Open |

## Reading list

Read each item just before the phase that uses it, not all at the start. Paper titles are listed from memory; search for them by name.

| Read before | Item | Why |
| --- | --- | --- |
| Phase 1 | Cobbe et al., 2021, "Training Verifiers to Solve Math Word Problems" | Introduces GSM8K and its answer format |
| Phase 2 | Wei et al., 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" | Few-shot chain-of-thought for the base model |
| Phase 2 | Wang et al., 2022, "Self-Consistency Improves Chain of Thought Reasoning in Language Models" | Majority voting at inference time |
| Phase 3 | Zelikman et al., 2022, "STaR: Bootstrapping Reasoning With Reasoning" | Training on the model's own correct reasoning |
| Phase 3 | Yuan et al., 2023, "Scaling Relationship on Learning Mathematical Reasoning with Large Language Models" | Rejection-sampling fine-tuning, the method behind Phase 3 |
| Phase 3 | Bai et al., 2022, "Constitutional AI: Harmlessness from AI Feedback" | Background for rubric-based judging of reasoning traces |
| Phase 4 | Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models" | The adapter method used in every training stage |
| Phase 4 | Hugging Face TRL documentation: SFTTrainer | Prompt–completion data and completion-only loss |
| Phase 6 | Shao et al., 2024, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" | Introduces GRPO and the 7B models this project uses |
| Phase 6 | Hugging Face TRL documentation: GRPOTrainer | Settings, reward functions and vLLM generation |
| Phase 7 | Any introduction to bootstrap confidence intervals | Error bars and paired comparisons for the results table |
