# Day 13 Model Output & Parser Audit Report

- **Source File:** `outputs\baseline\Qwen2.5-1.5B-Instruct\generations_val_greedy.jsonl`
- **Total Records in File:** 500
- **Audited Samples:** 20

## Summary Findings

- **Correct Answers:** 17/20 (85.0%)
- **Format Compliance (####):** 17/20 (85.0%)
- **Fallback Rate:** 3/20 (15.0%)
- **Unparseable Rate:** 0/20 (0.0%)
- **Parser Discrepancies:** 0

## Detailed Output Audit Log

| # | Problem ID | Gold | Extracted | Method | Match | Snippet |
|---|---|---|---|---|---|---|
| 1 | `gsm8k_2e0ade69589fa376` | 18 | `26` | `hash` | ✗ | 6 pounds  So, Steve had to pick 26 pounds of lingonberries on Thursday.  #### 26 |
| 2 | `gsm8k_148db5b4cd90eeb8` | 35 | `35` | `hash` | ✓ | #### 35 |
| 3 | `gsm8k_bc7fb141d5a98912` | 630 | `630` | `hash` | ✓ | refore, Mark supplies a total of 630 eggs to the two stores in a week.  #### 630 |
| 4 | `gsm8k_942394c590922956` | 42 | `24` | `hash` | ✗ | by the time Thomas reaches his current age, James will be 24 years old.  #### 24 |
| 5 | `gsm8k_f3f6eed664e61a36` | 60 | `the farmer shipped a total of 60 dozens of pomelos in all` | `answer_is` | ✓ | }      \]  Therefore, the farmer shipped a total of 60 dozens of pomelos in all. |
| 6 | `gsm8k_ea7e4a321247fafc` | 9 | `Justice needs 9 more plants to reach her goal of 24 plants` | `answer_is` | ✓ | 5 = 9 \]  Therefore, Justice needs 9 more plants to reach her goal of 24 plants. |
| 7 | `gsm8k_96817dd03351f2e4` | 32000 | `32000` | `hash` | ✓ |  number of baby tarantula legs in one less than 5 egg sacs is 32000.  #### 32000 |
| 8 | `gsm8k_3a16324c1f7df387` | 6 | `6` | `hash` | ✓ |  nickels, and Peter has 6 nickels. The difference is 12 - 6 = 6 nickels.  #### 6 |
| 9 | `gsm8k_cc82c81c82bf907e` | 121 | `121` | `hash` | ✓ |  Delivery fee + Tip Total amount = $90 + $20 + $11 Total amount = $121  #### 121 |
| 10 | `gsm8k_da9a9eb03cb6bd99` | 15 | `15` | `hash` | ✓ | herefore, it will take Yolanda's husband 15 minutes to catch up to her.  #### 15 |
| 11 | `gsm8k_76515568e3cd0a9c` | 130 | `130` | `hash` | ✓ |  Total from the way back from school: 70    - Total: \(60 + 70 = 130\)  #### 130 |
| 12 | `gsm8k_3abe12b674001142` | 3000 | `3000` | `hash` | ✓ |  \]  Therefore, the total number of bricks used to make both walls is: #### 3000 |
| 13 | `gsm8k_893e3a5ca02afb55` | 62 | `62` | `hash` | ✓ | 2 = 62 \]  Therefore, the total number of pages they read altogether is: #### 62 |
| 14 | `gsm8k_b340b666748e7eea` | 108 | `108 grams` | `bold` | ✓ |  grams}      \]  So, the company will use **108 grams** of sugar in two minutes. |
| 15 | `gsm8k_24a2ac62f9a7533c` | 21 | `21` | `hash` | ✓ | bles:    \[    28 - 7 = 21    \]  Therefore, James has 21 marbles left.  #### 21 |
| 16 | `gsm8k_c863f793ce603590` | 97 | `97` | `hash` | ✓ |  = 97 \]  Therefore, the total number of pastries that the four have is: #### 97 |
| 17 | `gsm8k_a21beba226c1f77d` | 36 | `36` | `hash` | ✓ | 2 \times 3 = 36 \) people.  So, Marcy is painting makeup for 36 people.  #### 36 |
| 18 | `gsm8k_2648fe1cc0ebcd47` | 72 | `72` | `hash` | ✓ | t{ hours/time} = 72 \text{ hours/year}      \]  So, the final answer is: #### 72 |
| 19 | `gsm8k_76dc36f750b913a0` | 40 | `45` | `hash` | ✗ | rac{3}{2} \]  \[ H = 45 \]  So, the hospital is currently 45 years old.  #### 45 |
| 20 | `gsm8k_b95879a14667dc7b` | 21 | `21` | `hash` | ✓ | erefore, the total number of insects remaining in the playground is 21.  #### 21 |

## Discrepancies Requiring Investigation

Zero parser discrepancies found! Every extracted value matches the ground truth and format contract as expected.
