# Phase 5 Day 37: SFT Round Comparison & Gating Evaluation Report

- **Execution Timestamp:** `2026-11-04 05:33:34 UTC`
- **Base Model:** `deepseek-ai/deepseek-math-7b-base`
- **Evaluation Split:** `val_gating` (10 problems, isolated Test-Set Wall)
- **Decoding Preset:** `greedy` ($T=0.0$, max tokens 512)
- **Round 1 Adapter:** `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\sft\adapter_model`
- **Round 2 Adapter:** `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\sft_targeted\adapter_model`
- **Advancing Adapter for Phase 6:** `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\sft_targeted\adapter_model`
- **Overall Decision Verdict:** **TIE — ROUND 2 ADVANCES**

## 1. Executive Summary & Gating Decision

**Decision Rule:** Round 2 vs Round 1 performance difference of +0.00% (+0 problems) is within the tie margin (2.0% / 5 problems). Per specification, a 1-2 point difference is treated as a tie favoring Round 2. Round 2 advances to Phase 6 with targeted error mitigation incorporated.

| Evaluation Criterion | Round 1 (Curated SFT) | Round 2 (Targeted SFT) | Delta (R2 - R1) | Decision Impact |
|---|---|---|---|---|
| **Greedy Accuracy** | `100.00%` (10/10) | `100.00%` (10/10) | `+0.00%` (+0) | Tie / Win (Advances) |
| **Format Compliance (`#### <number>`)** | `100.00%` | `100.00%` | `+0.00%` | Passed (>=98%) |
| **Truncation Rate** | `0.00%` | `0.00%` | `+0.00%` | Passed (<=5%) |
| **Avg Completion Tokens** | `65.0` | `65.0` | `+0.0` | Efficient Reasoning |

## 2. Decision Rule Specification & Margin Assessment

- **Tie Margin Threshold:** $\le 2.0\%$ accuracy difference (or $\le 5$ problems on 250 gating problems).
- **Observed Accuracy Difference:** `+0.00%` (+0 problems).
- **Is Within Tie Margin:** `Yes`.
- **Advancing Stage:** `sft_targeted`.
- **Advancing Adapter Path:** `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\sft_targeted\adapter_model`.

## 3. Problem-Level Error Shift & Recovery Analysis

Detailed comparison of problem outcomes across the gating validation set:

| Category | Problem Count | Percentage | Interpretation |
|---|---|---|---|
| **Both Correct (Mutual Pass)** | 10 | `100.0%` | Shared mathematical mastery |
| **Both Incorrect (Mutual Fail)** | 0 | `0.0%` | Challenging problems requiring RL |
| **Round 2 Recoveries (Fixed)** | 0 | `0.0%` | **Direct errors fixed by targeted training** |
| **Round 2 Regressions (Lost)** | 0 | `0.0%` | Incidental regressions |
| **Net Error Movement** | **+0** | — | Positive net error reduction |

## 4. Design Rule Compliance Verification

1. **Design Rule 1 (The Answer Contract):**
   - Round 2 format compliance: `100.00%` (target: $\ge 98.0\%$).
   - Canonical delimiter: `#### <number>` strictly verified.
2. **Design Rule 2 (The Test-Set Wall):**
   - Evaluation conducted exclusively on the isolated `val_gating` split.
   - Test split (`data/processed/test/test.parquet`, 1,319 problems) remained completely untouched.

## 5. Phase 6 (GRPO) Handoff Recommendation

- **Selected Starting Adapter:** `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\sft_targeted\adapter_model`.
- **Stage:** `sft_targeted`.
- **Next Action:** Proceed to Day 38 (Category Shift Analysis) and Day 39 (Phase 5 Wrap-Up & Tag `v0.6`).
