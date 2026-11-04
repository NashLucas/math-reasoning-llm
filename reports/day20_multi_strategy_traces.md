# Day20 Multi Strategy Traces

- **Date:** 2026-11-02 23:31:02 UTC
- **Teacher Model:** `Qwen/Qwen2.5-1.5B-Instruct` (MOCK / Simulation)
- **Prompt Strategies:** `solve_then_verify`, `alternative_method`
- **Samples per Problem:** 1
- **Traces Output File:** [`traces_Qwen2.5-1.5B-Instruct_solve_then_verify_alternative_method.jsonl`](C:/Users/DELL/Desktop/math-reasoning-llm/data/synthetic/generated/traces_Qwen2.5-1.5B-Instruct_solve_then_verify_alternative_method.jsonl)

> [!WARNING]
> **Mock run.** The mock generator writes `#### <gold answer>` into every completion, so the
> 100% accuracy and format compliance numbers below verify the pipeline plumbing only.
> Run with `--use_real_model` on a Colab GPU to measure real yield.

## Overall Yield & Quality Metrics

| Metric | Value |
|---|---|
| Total Problems Sampled | 100 |
| Total Traces Generated | 200 |
| Format Compliance Rate | 100.0% (1.0000) |
| Correct Traces Count | 200 |
| Trace Accuracy | 100.0% (1.0000) |
| Problems with $\ge 1$ Correct Trace | 100 / 100 (100.0%) |

## Per-Strategy Breakdown

| Strategy | Traces | Format Compliance | Correct Count | Accuracy (%) | Kept by Filter |
|---|---|---|---|---|---|
| `solve_then_verify` | 100 | 100.0% | 100 | 100.0% | 100 |
| `alternative_method` | 100 | 100.0% | 100 | 100.0% | 100 |

## Curation Filter Results

| Curation Metric | Value |
|---|---|
| Evaluated Traces | 200 |
| Kept (Correct & Clean) | 200 (100.0%) |
| Rejected Count | 0 |

### Rejection Breakdown

No traces rejected (100% retention).

## Pass Rate Distribution

| Bucket | Problem Count | Share (%) |
|---|---|---|
| 0% | 0 | 0.0% |
| (0, 25%] | 0 | 0.0% |
| (25, 50%] | 0 | 0.0% |
| (50, 75%] | 0 | 0.0% |
| (75, 100%) | 0 | 0.0% |
| 100% | 100 | 100.0% |

## Status & Next Steps

**Status:** MOCK (Plumbing Verified; Real Model Run Pending)
- Records contain prompt, completion, parsed answer, extraction method, correctness flag, token counts, and timestamp.
- Mock run verified: run `--use_real_model` on Colab G4 for real generation.
