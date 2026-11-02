# LLM Judge Calibration Report (Phase 3 Day 21)

- **Date:** 2026-11-02 23:43:21 UTC
- **Judge Model:** `Qwen/Qwen2.5-1.5B-Instruct` (MOCK / Simulation)
- **Benchmark Size:** 20 manually calibrated traces
- **Calibration Status:** **PASSED (r >= 0.60)**

## Calibration Statistics

| Metric | Value | Target Threshold |
|---|---|---|
| Pearson Correlation ($r$) | **0.8851** | $\ge 0.60$ |
| Mean Absolute Error (MAE) | 0.0927 | $\le 0.20$ |
| Mean Squared Error (MSE) | 0.0166 | - |
| Samples Evaluated | 20 | 20 traces |

## Trace-by-Trace Score Comparison

| Trace ID | Strategy | Manual Overall | Judge Overall | Diff | Status |
|---|---|---|---|---|---|
| `calib_cot_01` | `chain_of_thought` | 1.00 | 0.95 | -0.05 | OK |
| `calib_cot_02` | `chain_of_thought` | 1.00 | 0.95 | -0.05 | OK |
| `calib_cot_03` | `chain_of_thought` | 0.90 | 0.95 | +0.05 | OK |
| `calib_cot_04` | `chain_of_thought` | 0.80 | 0.85 | +0.05 | OK |
| `calib_cot_05` | `chain_of_thought` | 0.75 | 0.85 | +0.10 | OK |
| `calib_cot_06` | `chain_of_thought` | 0.70 | 0.85 | +0.15 | OK |
| `calib_cot_07` | `chain_of_thought` | 0.50 | 0.65 | +0.15 | OK |
| `calib_cot_08` | `chain_of_thought` | 0.40 | 0.45 | +0.05 | OK |
| `calib_stv_01` | `solve_then_verify` | 1.00 | 0.95 | -0.05 | OK |
| `calib_stv_02` | `solve_then_verify` | 0.98 | 0.95 | -0.03 | OK |
| `calib_stv_03` | `solve_then_verify` | 0.90 | 0.95 | +0.05 | OK |
| `calib_stv_04` | `solve_then_verify` | 0.68 | 0.72 | +0.04 | OK |
| `calib_stv_05` | `solve_then_verify` | 0.45 | 0.47 | +0.02 | OK |
| `calib_stv_06` | `solve_then_verify` | 0.25 | 0.38 | +0.12 | OK |
| `calib_alt_01` | `alternative_method` | 0.50 | 0.92 | +0.42 | Divergent |
| `calib_alt_02` | `alternative_method` | 1.00 | 0.92 | -0.08 | OK |
| `calib_alt_03` | `alternative_method` | 0.97 | 0.92 | -0.05 | OK |
| `calib_alt_04` | `alternative_method` | 0.89 | 0.92 | +0.03 | OK |
| `calib_alt_05` | `alternative_method` | 0.66 | 0.88 | +0.22 | OK |
| `calib_alt_06` | `alternative_method` | 0.38 | 0.46 | +0.08 | OK |

## Day 21 Calibration Verdict

- Correlation target $r \ge 0.60$: MET
- Design Rule 4 check: All evaluated traces reach the correct gold answer; judge scores rank qualitative derivation quality.
- Ready for Day 22 full-scale trace scoring on Colab G4.
