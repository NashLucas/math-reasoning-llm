# Day 13 Model Output & Parser Audit Report

- **Source File:** `outputs\baseline\generations_val_greedy.jsonl`
- **Total Records in File:** 20
- **Audited Samples:** 20

## Summary Findings

- **Correct Answers:** 19/20 (95.0%)
- **Format Compliance (####):** 17/20 (85.0%)
- **Fallback Rate:** 3/20 (15.0%)
- **Unparseable Rate:** 0/20 (0.0%)
- **Parser Discrepancies:** 0

## Detailed Output Audit Log

| # | Problem ID | Gold | Extracted | Method | Match | Snippet |
|---|---|---|---|---|---|---|
| 1 | `gsm8k_2e0ade69589fa376` | 18 | `18` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 18. #### 18 |
| 2 | `gsm8k_148db5b4cd90eeb8` | 35 | `35` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 35. #### 35 |
| 3 | `gsm8k_bc7fb141d5a98912` | 630 | `630` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 630. #### 630 |
| 4 | `gsm8k_942394c590922956` | 42 | `42` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 42. #### 42 |
| 5 | `gsm8k_f3f6eed664e61a36` | 60 | `60` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 60. #### 60 |
| 6 | `gsm8k_ea7e4a321247fafc` | 9 | `9` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 9. #### 9 |
| 7 | `gsm8k_96817dd03351f2e4` | 32000 | `32000` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 32000. #### 32000 |
| 8 | `gsm8k_3a16324c1f7df387` | 6 | `6` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 6. #### 6 |
| 9 | `gsm8k_cc82c81c82bf907e` | 121 | `121` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 121. #### 121 |
| 10 | `gsm8k_da9a9eb03cb6bd99` | 15 | `15` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 15. #### 15 |
| 11 | `gsm8k_76515568e3cd0a9c` | 130 | `130` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 130. #### 130 |
| 12 | `gsm8k_3abe12b674001142` | 3000 | `3000` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 3000. #### 3000 |
| 13 | `gsm8k_893e3a5ca02afb55` | 62 | `62` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 62. #### 62 |
| 14 | `gsm8k_b340b666748e7eea` | 108 | `108` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 108. #### 108 |
| 15 | `gsm8k_24a2ac62f9a7533c` | 21 | `21` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 21. #### 21 |
| 16 | `gsm8k_c863f793ce603590` | 97 | `97` | `hash` | ✓ | Step 1: Calculate. Therefore the answer is 97. #### 97 |
| 17 | `gsm8k_a21beba226c1f77d` | 36 | `36` | `boxed` | ✓ | Step 1: Calculate. Final answer: \boxed{36} |
| 18 | `gsm8k_2648fe1cc0ebcd47` | 72 | `72` | `boxed` | ✓ | Step 1: Calculate. Final answer: \boxed{72} |
| 19 | `gsm8k_76dc36f750b913a0` | 40 | `40` | `boxed` | ✓ | Step 1: Calculate. Final answer: \boxed{40} |
| 20 | `gsm8k_b95879a14667dc7b` | 21 | `9999` | `hash` | ✗ | Step 1: Mistake. #### 9999 |

## Discrepancies Requiring Investigation

Zero parser discrepancies found! Every extracted value matches the ground truth and format contract as expected.
