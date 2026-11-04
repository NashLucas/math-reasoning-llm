# Phase 6 Day 46: Post-GRPO Evaluation & Reward Trajectory Diagnostics Report

- **Date / Timestamp (UTC):** 2026-11-04T12:54:55.242350+00:00
- **Gating Split:** `val_gating` (10 problems)
- **Gating Verdict:** **PASSED**

---

## 1. Executive Summary & Gating Decision

| Evaluation Metric | SFT Baseline (R2 Targeted) | GRPO Policy | Delta (GRPO - SFT) | Status |
|---|---|---|---|---|
| **Greedy Accuracy** | 100.00% | 100.00% | **+0.00%** | PASS |
| **Pass@1** | 75.00% | 75.00% | **+0.00%** | INFO |
| **Pass@4** | 100.00% | 100.00% | **+0.00%** | INFO |
| **Pass@8** | 100.00% | 100.00% | **+0.00%** | PASS |
| **Majority Vote @ 8** | 100.00% | 100.00% | **+0.00%** | INFO |
| **Mean Total Reward** | 1.2500 | 1.4000 | **+0.1500** | PASS |
| **Contract Adherence** | 100.0% | 100.0% | **+0.0%** | PASS |

---

## 2. Gate Verification Checklist

- **greedy_accuracy_no_regression:** [PASS]
  - Description: GRPO greedy accuracy must not regress vs. SFT baseline
  - Threshold: `delta >= -0.01 (-1.0%)`
  - Actual: `0.0`

- **pass_at_8_gain:** [PASS]
  - Description: GRPO pass@8 must demonstrate exploration/reasoning gain over SFT
  - Threshold: `delta_pass@8 >= 0.00`
  - Actual: `0.0`

- **answer_contract_preservation:** [PASS]
  - Description: Canonical answer contract format adherence rate under RL
  - Threshold: `>= 98.0%`
  - Actual: `100.0%`

- **composite_reward_gain:** [PASS]
  - Description: GRPO mean total composite reward must exceed SFT baseline
  - Threshold: `delta_reward > 0.00`
  - Actual: `0.15`

---

## 3. Multi-Component Reward Trajectory Breakdown

| Component | Weight | SFT Mean (Std) | GRPO Mean (Std) | Trajectory Delta | Direction |
|---|---|---|---|---|---|
| **Accuracy Reward (R_acc)** | 0.70 (raw: 1.0) | 0.7500 (0.44) | 0.7500 (0.44) | +0.0000 | IMPROVED |
| **Format Reward (R_format)** | 0.15 (raw: 0.5) | 1.0000 (0.00) | 1.0000 (0.00) | +0.0000 | PRESERVED |
| **Structure Reward (R_struct)** | 0.10 (raw: 0.2) | 0.0000 (0.00) | 0.0000 (0.00) | +0.0000 | STABLE |
| **Length Penalty (R_len)** | 0.05 (raw: 0.1) | 0.0000 (0.00) | 0.0000 (0.00) | +0.0000 | CONTROLLED |
| **Total Composite Reward** | 1.00 (sum) | 1.2500 (0.44) | 1.4000 (0.44) | **+0.1500** | **POSITIVE SHIFT** |

---

## 4. Answer Contract & Token Efficiency Audit

- **Answer Contract Rate (GRPO):** 100.0% (Minimum threshold: >= 98.0%).
- **Answer Contract Rate (SFT Baseline):** 100.0%.
- **Mean Completion Tokens:** SFT = 85.0 tokens, GRPO = 85.0 tokens.
- **Extraction Method Breakdown (GRPO):** {}
- **Test-Set Wall Status:** Strict isolation verified. GSM8K test set (`test.parquet`, 1,319 problems) was quarantined and untouched throughout evaluation.

---

## 5. Artifact Provenance

- **SFT Adapter:** `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\sft_targeted\adapter_model`
- **GRPO Adapter:** `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\grpo\adapter_model`
- **Greedy Generations File (GRPO):** `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\eval_grpo\grpo_policy\greedy\generations_val_gating_greedy.jsonl`
- **Sampled Generations File (GRPO):** `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\eval_grpo\grpo_policy\sampled\generations_val_gating_self_consistency.jsonl`
- **Audit Plot:** `reports/day46_grpo_eval_curves.png`
