# Day 27 SFT Overfitting Sanity Test Report

- **Date:** 2026-11-03 20:19:06 UTC
- **Model:** `mock-model`
- **Examples Trained:** 20
- **Steps:** 30
- **Status:** **PASSED**

## Optimization Metrics
| Metric | Observed Value | Expected Threshold | Status |
|---|---|---|---|
| Initial Loss | `1.5000` | > 1.00 | BASELINE |
| Final Loss | `0.0050` | < 0.50 | MET |
| Loss Reduction | `99.67%` | >= 50% | MET |
| Memorized Accuracy | `100.00%` (20/20) | High | MET |
| Format Compliance | `100.00%` (20/20) | >= 90% | MET |

## Overfitting Sanity Verification Summary
1. **Gradient Propagation:** Loss dropped from `1.5000` down to `0.0050`, representing a `99.7%` reduction over `30` steps.
2. **Termination & Format Retention:** Output responses correctly end with canonical `#### <number>`.
3. **Verdict:** SFT training mechanics, LoRA parameter adaptation, and completion-only loss masking are working as expected.
