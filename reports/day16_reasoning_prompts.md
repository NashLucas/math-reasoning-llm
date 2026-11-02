# Day 16 Reasoning Prompts Verification Report

- **Date:** 2026-11-02 16:07:52 UTC
- **Model Evaluated:** `Qwen/Qwen2.5-1.5B-Instruct` (MOCK / simulation)
- **Decoding:** greedy, engine default stop rules
- **Phase:** Phase 3 (Reasoning Data)
- **Milestone:** Day 16 — Reasoning Prompts Verification

> [!WARNING]
> **Mock run.** The mock generator writes `#### <gold answer>` itself, so the numbers below are
> 100% by construction and say nothing about the prompts. Rerun with `--use_real_model`.

## Summary Results Table

| Prompt Strategy | Type Key | Problems | Format Compliance | Accuracy | Truncated | Text after `####` |
|---|---|---|---|---|---|---|
| Chain-of-Thought | `chain_of_thought` | 10 | 100.0% | 100.0% | 0 | 0 |
| Solve-then-Verify | `solve_then_verify` | 10 | 100.0% | 100.0% | 0 | 0 |
| Alternative-Method | `alternative_method` | 10 | 100.0% | 100.0% | 0 | 0 |

Format compliance means the parser found the answer on a `####` line. Accuracy is only a sanity
check here; the Day 16 check is about the format. "Text after `####`" counts completions where
the model kept writing after the answer line (cut off by the engine, but a sign the prompt is not
followed literally).

## Non-compliant and Notable Completions

None: every completion ended on a `####` line with nothing after it.

## One Completion per Strategy

### Strategy: Chain-of-Thought
- **Problem ID:** `gsm8k_4b291b0c6a0ca2da`
- **Question:** I went to the music shop and there were CDs of The Life Journey for $100, A Day a Life for $50, and When You Rescind for $85 on display. If I bought 3 of each CD to share with my friends, what's the total amount of money I spent in the shop?
- **Gold Answer:** `705`
- **Extracted Answer:** `705` (Method: `hash`)
- **Compliant:** `True` | **Match:** `True`

```text
Let's solve this step by step.
Given the problem: I went to the music shop and there were CDs of The Life Journey for $100, A Day ...
Step 1: Identify given quantities and relationships.
Step 2: Compute the intermediate operations.
Step 3: Calculating final total yields 705.
#### 705
```

### Strategy: Solve-then-Verify
- **Problem ID:** `gsm8k_4b291b0c6a0ca2da`
- **Question:** I went to the music shop and there were CDs of The Life Journey for $100, A Day a Life for $50, and When You Rescind for $85 on display. If I bought 3 of each CD to share with my friends, what's the total amount of money I spent in the shop?
- **Gold Answer:** `705`
- **Extracted Answer:** `705` (Method: `hash`)
- **Compliant:** `True` | **Match:** `True`

```text
Stage 1 - Initial Solution:
Working through the problem step by step gives an initial result of 705.

Stage 2 - Verification:
Let us check this by reversing the arithmetic and verifying constraints.
All conditions check out and confirm the initial calculation.
#### 705
```

### Strategy: Alternative-Method
- **Problem ID:** `gsm8k_4b291b0c6a0ca2da`
- **Question:** I went to the music shop and there were CDs of The Life Journey for $100, A Day a Life for $50, and When You Rescind for $85 on display. If I bought 3 of each CD to share with my friends, what's the total amount of money I spent in the shop?
- **Gold Answer:** `705`
- **Extracted Answer:** `705` (Method: `hash`)
- **Compliant:** `True` | **Match:** `True`

```text
Alternative Strategy (Algebraic Formulation / Working Backwards):
Instead of sequential arithmetic, let x represent the unknown quantity.
Formulating the equation according to the problem constraints:
Solving for x directly gives x = 705.
#### 705
```

## Day 16 Check

**Verdict:** NOT VERIFIED (mock run)
- Day 16's done-when needs real development-model outputs.
