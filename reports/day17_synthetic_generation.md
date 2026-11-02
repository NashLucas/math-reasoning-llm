# Day 17 Reasoning Data Generation Report

- **Date:** 2026-11-02 16:07:53 UTC
- **Teacher Model:** `Qwen/Qwen2.5-1.5B-Instruct` (MOCK / Simulation)
- **Prompt Strategies:** `chain_of_thought`
- **Samples per Problem:** 4
- **Traces Output File:** [`traces_Qwen2.5-1.5B-Instruct_chain_of_thought.jsonl`](C:/Users/DELL/Desktop/math-reasoning-llm/data/synthetic/generated/traces_Qwen2.5-1.5B-Instruct_chain_of_thought.jsonl)

> [!WARNING]
> **Mock run.** The mock generator writes `#### <gold answer>` into every completion, so the
> 100% accuracy and format compliance numbers below verify the pipeline plumbing only.
> Run with `--use_real_model` on a Colab GPU to measure real yield.

## Yield & Quality Metrics

| Metric | Value |
|---|---|
| Total Problems Sampled | 100 |
| Total Traces Generated | 400 |
| Format Compliance Rate | 100.0% (1.0000) |
| Correct Traces Count | 400 |
| Trace Accuracy | 100.0% (1.0000) |
| Problems with $\ge 1$ Correct Trace | 100 / 100 (100.0%) |

## Pass-Rate Distribution (Correct Samples per Problem)

| Bucket | Problem Count | Share (%) |
|---|---|---|
| 0% | 0 | 0.0% |
| (0, 25%] | 0 | 0.0% |
| (25, 50%] | 0 | 0.0% |
| (50, 75%] | 0 | 0.0% |
| (75, 100%) | 0 | 0.0% |
| 100% | 100 | 100.0% |

## Phase 3 Day 17 Status

**Status:** MOCK (Plumbing Verified; Real Model Run Pending)
- Records contain prompt, completion, parsed answer, extraction method, correctness flag, token counts, and timestamp.
- Next: Run `--use_real_model` on 100 problems with `Qwen/Qwen2.5-1.5B-Instruct` on a cheap GPU.
