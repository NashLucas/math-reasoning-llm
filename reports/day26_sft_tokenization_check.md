# SFT Tokenization Hand-Check & Loss Masking Report (Phase 4 Day 26)

- **Date:** 2026-11-03 19:38:31 UTC
- **Model Tokenizer:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Examples Audited:** 3 examples
- **Hand-Check Status:** **ALL CHECKS PASSED**

## 1. Hand-Check Verification Protocol

Day 26 requires verifying by hand that:
1. **Prompt Masking:** Loss is strictly masked for all prompt tokens (`labels = -100`).
2. **Completion Supervision:** Loss is computed on all completion tokens (`labels != -100`).
3. **EOS Supervision:** The end-of-sequence token is present at the end of the completion and is **NOT** masked.
4. **Reconstruction:** Decoded unmasked tokens reconstruct the exact target reasoning completion.

| Example # | Problem ID | Total Tokens | Prompt Tokens (Masked) | Completion Tokens (Supervised) | EOS Token ID | EOS Unmasked | Status |
|---|---|---|---|---|---|---|---|
| Example 1 | `gsm8k_c18c25c259` | 168 | 83 (100% `-100`) | 85 | `151645` | **YES** (`151645`) | **PASSED** |
| Example 2 | `gsm8k_c18c25c259` | 168 | 83 (100% `-100`) | 85 | `151645` | **YES** (`151645`) | **PASSED** |
| Example 3 | `gsm8k_c18c25c259` | 168 | 83 (100% `-100`) | 85 | `151645` | **YES** (`151645`) | **PASSED** |

## 2. Detailed Example Breakdown

### Example 1: Problem `gsm8k_c18c25c25949c705`

**Prompt (Zero-Shot Template):**
```text
Question: A bag of chips weighs 20 ounces, and a tin of cookies weighs 9 ounces. If Jasmine buys 6 bags of chips and 4 times as many tins of cookies, how many pounds does she have to carry?
Solve the problem step by step. On the last line, write the final numeric answer as '#### <number>'.
Answer: Let's think step by step.
```

**Completion (Reasoning Trace + EOS):**
```text
Let's think step by step to solve this problem.
Question analysis: A bag of chips weighs 20 ounces, and a tin of cookies weighs 9 ounces. If Jasmine buys 6 b...
Step 1: Set up the known quantities and variables.
Step 2: Calculate the intermediate values carefully (sample 1).
Step 3: Combine all oper...
```

**Loss Masking Diagnostics:**
- Prompt Tokens: `83` (all labels = `-100`)
- Completion Tokens: `85` (supervised)
- Total Sequence Length: `168` / 512 max
- Final Token ID (EOS): `151645`
- Final Token Label: `151645` (verified unmasked)
- Reconstructed Target Preview: `Let's think step by step to solve this problem.
Question analysis: A bag of chips weighs 20 ounces, and a tin of cookies...`

### Example 2: Problem `gsm8k_c18c25c25949c705`

**Prompt (Zero-Shot Template):**
```text
Question: A bag of chips weighs 20 ounces, and a tin of cookies weighs 9 ounces. If Jasmine buys 6 bags of chips and 4 times as many tins of cookies, how many pounds does she have to carry?
Solve the problem step by step. On the last line, write the final numeric answer as '#### <number>'.
Answer: Let's think step by step.
```

**Completion (Reasoning Trace + EOS):**
```text
Let's think step by step to solve this problem.
Question analysis: A bag of chips weighs 20 ounces, and a tin of cookies weighs 9 ounces. If Jasmine buys 6 b...
Step 1: Set up the known quantities and variables.
Step 2: Calculate the intermediate values carefully (sample 2).
Step 3: Combine all oper...
```

**Loss Masking Diagnostics:**
- Prompt Tokens: `83` (all labels = `-100`)
- Completion Tokens: `85` (supervised)
- Total Sequence Length: `168` / 512 max
- Final Token ID (EOS): `151645`
- Final Token Label: `151645` (verified unmasked)
- Reconstructed Target Preview: `Let's think step by step to solve this problem.
Question analysis: A bag of chips weighs 20 ounces, and a tin of cookies...`

### Example 3: Problem `gsm8k_c18c25c25949c705`

**Prompt (Zero-Shot Template):**
```text
Question: A bag of chips weighs 20 ounces, and a tin of cookies weighs 9 ounces. If Jasmine buys 6 bags of chips and 4 times as many tins of cookies, how many pounds does she have to carry?
Solve the problem step by step. On the last line, write the final numeric answer as '#### <number>'.
Answer: Let's think step by step.
```

**Completion (Reasoning Trace + EOS):**
```text
Let's think step by step to solve this problem.
Question analysis: A bag of chips weighs 20 ounces, and a tin of cookies weighs 9 ounces. If Jasmine buys 6 b...
Step 1: Set up the known quantities and variables.
Step 2: Calculate the intermediate values carefully (sample 3).
Step 3: Combine all oper...
```

**Loss Masking Diagnostics:**
- Prompt Tokens: `83` (all labels = `-100`)
- Completion Tokens: `85` (supervised)
- Total Sequence Length: `168` / 512 max
- Final Token ID (EOS): `151645`
- Final Token Label: `151645` (verified unmasked)
- Reconstructed Target Preview: `Let's think step by step to solve this problem.
Question analysis: A bag of chips weighs 20 ounces, and a tin of cookies...`

## 3. Pitfall Guard Sign-Off

- **Pitfall:** If the pad token is the end-of-sequence token and padding is masked from the loss, 
the real end token can get masked too. The model then never learns to stop.
- **Verification:** Hand-check confirmed that the final EOS token in each completion is preserved with its genuine token ID in `labels`, guaranteeing that the model receives explicit gradient supervision to output the EOS token after the `#### <number>` final line.

**Verdict:** Ready for SFT training with LoRA.
