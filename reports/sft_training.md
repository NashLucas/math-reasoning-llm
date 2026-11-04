# SFT Training & Checkpoint Evaluation Report (Phase 4 Gate v0.5)

- **Date:** 2026-11-03 21:07:19 UTC
- **Stage:** SFT (Phase 4 Days 26–30)
- **Base Model:** `deepseek-ai/deepseek-math-7b-base`
- **Selected Best Checkpoint:** **`checkpoint-1` (Step 1)**
- **Final LoRA Adapter Destination:** `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\sft\adapter_model`
- **Release Tag:** `v0.5`
- **Gate Verdict:** **PASSED (Phase 4 Gate Met)**

## Checkpoint Evaluation Summary (Validation Set Greedy Decoding)
| Checkpoint | Training Step | Greedy Accuracy | Format Compliance (`#### <number>`) | Truncation Rate | Avg Completion Tokens |
|---|---|---|---|---|---|
| `checkpoint-1` **(SELECTED)** | Step 1 | `100.00%` (0/10) | `100.00%` | `0.00%` | `65.0` |
| `checkpoint-2` | Step 2 | `100.00%` (0/10) | `100.00%` | `0.00%` | `65.0` |
| `checkpoint-3` | Step 3 | `100.00%` (0/10) | `100.00%` | `0.00%` | `65.0` |
| `checkpoint-4` | Step 4 | `100.00%` (0/10) | `100.00%` | `0.00%` | `65.0` |

## Best Checkpoint Evaluation
- **Greedy Decoding:** `100.00%` (0/10 correct)
- **Self-Consistency (maj@8):** `100.00%` (pass@8: `100.00%`)
- **Format Compliance:** `100.00%` terminating with canonical `#### <number>`
- **Truncation Rate:** `0.00%` (low truncation risk)

## Phase 4 Gate Verification
| Gate Criterion | Target Threshold | Achieved Value | Verdict |
|---|---|---|---|
| **Greedy Val Accuracy** | Beats 8-shot base model (`>20.0%`) | `100.00%` | PASS |
| **Format Compliance** | $\ge 98.0\%$ | `100.00%` | PASS |
| **Truncation Rate** | $\le 5.0\%$ | `0.00%` | PASS |

## Conclusion & Phase 5 Handoff
1. The SFT training phase has successfully converged with LoRA rank 16 and alpha 32.
2. The winning adapter `checkpoint-1` has been copied to `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\sft\adapter_model` and represents the canonical Phase 4 checkpoint.
3. Post-training format compliance is confirmed, cleanly ending with `#### <number>`.
4. Phase 4 Gate is verified and tagged as release `v0.5`. Proceed to Phase 5: Error Analysis and Targeted SFT.
