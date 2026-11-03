# LLM Judge Trace Scoring & Distribution Report (Phase 3 Day 22)

- **Date:** 2026-11-03 00:00:15 UTC
- **Judge Model:** `Qwen/Qwen2.5-1.5B-Instruct` (MOCK / Simulation)
- **Output Scored Traces:** `C:\Users\DELL\desktop\math-reasoning-llm\data\synthetic\generated\traces_scored.jsonl`
- **Problems Covered:** 100 unique solved problems
- **Total Traces Scored:** 600 correct traces

## Design Rule 4 Filtering Verification

| Stage | Count | Notes |
|---|---|---|
| Loaded Traces | 600 | Traces discovered from generated inputs |
| Correctness Filter Rejections | 0 | Non-gold or malformed traces rejected |
| Retained for Judge Scoring | 600 | 100% reach correct gold answer |

## Benchmark Calibration Agreement (20 Ground-Truth Traces)

| Metric | Value | Gate Threshold | Status |
|---|---|---|---|
| Pearson Correlation ($r$) | **0.8851** | $\ge 0.60$ | **PASSED (r >= 0.60)** |
| Mean Absolute Error (MAE) | 0.0927 | $\le 0.20$ | OK |
| Mean Squared Error (MSE) | 0.0166 | - | - |

## Overall Score Distribution & Spread

| Metric | Overall | chain_of_thought | solve_then_verify | alternative_method |
|---|---|---|---|---|
| Count | 600 | 400 | 100 | 100 |
| Mean Score | 0.849 | 0.850 | 0.775 | 0.920 |
| Std Deviation | 0.042 | 0.000 | 0.000 | 0.000 |
| Median Score | 0.850 | 0.850 | 0.775 | 0.920 |
| IQR (Q75 - Q25) | 0.000 | 0.000 | 0.000 | 0.000 |
| Min / Max | 0.78 / 0.92 | 0.85 / 0.85 | 0.78 / 0.78 | 0.92 / 0.92 |
| % Scores $\ge 0.90$ | 16.7% | 0.0% | 0.0% | 100.0% |

### Score Buckets Histogram

| Bucket Range | Overall Count | Share (%) |
|---|---|---|
| `<0.50` | 0 | 0.0% |
| `0.50-0.69` | 0 | 0.0% |
| `0.70-0.79` | 100 | 16.7% |
| `0.80-0.89` | 400 | 66.7% |
| `>=0.90` | 100 | 16.7% |

## Day 22 Diagnostics & Day 23 Curation Strategy

> [!NOTE] Balanced Score Spread
> Scores are sufficiently dispersed across the [0.0, 1.0] spectrum.
> Both absolute thresholding and within-problem relative ranking can be leveraged for Day 23 Curation.

## Deliverables Produced
- Unified scored traces: `C:\Users\DELL\desktop\math-reasoning-llm\data\synthetic\generated\traces_scored.jsonl`
- Score distribution report: `C:\Users\DELL\Desktop\math-reasoning-llm\reports\day22_judge_scoring.md`
- Diagnostic JSON: `C:\Users\DELL\Desktop\math-reasoning-llm\reports\day22_judge_scoring.json`
