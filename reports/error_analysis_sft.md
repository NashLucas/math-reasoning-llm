# Error Analysis Report — SFT Round 1

## 1. Executive Summary
- **Total Problems Evaluated:** 11
- **Correct Solutions:** 2 (18.18%)
- **Total Errors Analyzed:** 9 (81.82%)
- **Rule-Based Failures:** 3 (33.33%)
- **Semantic Reasoning Errors:** 6 (66.67%)

## 2. Error Category Breakdown

| Category | Type | Count | % of Errors | Description |
| :--- | :--- | :---: | :---: | :--- |
| `parse_error` | Rule-Based | 2 | 22.2% | Output empty, unparseable, or missing numeric final answer |
| `truncation` | Rule-Based | 1 | 11.1% | Output exceeded maximum length without terminating |
| `arithmetic_slip` | Semantic | 2 | 22.2% | Correct mathematical setup, but arithmetic calculation error |
| `misread_question` | Semantic | 1 | 11.1% | Misread numbers, entities, or solved for wrong quantity |
| `wrong_setup` | Semantic | 1 | 11.1% | Incorrect formula, equation modeling, or conceptual approach |
| `incomplete_reasoning` | Semantic | 1 | 11.1% | Derivation omitted steps or guessed answer prematurely |
| `other` | Semantic | 1 | 11.1% | Degenerate repetition, language mixing, or incoherent failure |

## 3. Representative Examples per Category

### Category: `parse_error`
**Example 1 (ID: err_01_parse_empty)**
- **Question:** What is 10 plus 15?
- **Gold Answer:** `25`
- **Extracted Answer:** `None`
- **Rationale:** Empty or whitespace-only model generation
```text

```

**Example 2 (ID: err_02_parse_no_answer)**
- **Question:** How many books are on the shelf if there are 3 shelves with 12 books each?
- **Gold Answer:** `36`
- **Extracted Answer:** `None`
- **Rationale:** No numeric answer could be extracted from generation
```text
I think there are multiple shelves and they have several books on them. Let me consider how to read.
```

### Category: `truncation`
**Example 1 (ID: err_03_truncation_token_limit)**
- **Question:** A bakery makes 40 loaves of bread in the morning and 60 in the afternoon. How many loaves did they make in total?
- **Gold Answer:** `100`
- **Extracted Answer:** `40`
- **Rationale:** Explicit finish_reason='length' reported by engine
```text
In the morning, the bakery makes 40 loaves. In the afternoon, they make 60 loaves. To find the total, we add 40 and
```

### Category: `arithmetic_slip`
**Example 1 (ID: err_05_arithmetic_slip_addition)**
- **Question:** Janet has 15 apples and buys 12 more. How many does she have now?
- **Gold Answer:** `27`
- **Extracted Answer:** `28`
- **Rationale:** Equations present (1 '='); predicted 28.0 close to gold 27.0
```text
Janet starts with 15 apples. She buys 12 more.
15 + 12 = 28.
Therefore she has 28 apples.
#### 28
```

**Example 2 (ID: err_06_arithmetic_slip_multiplication)**
- **Question:** A box contains 7 bags of marbles with 8 marbles per bag. How many marbles in total?
- **Gold Answer:** `56`
- **Extracted Answer:** `54`
- **Rationale:** Equations present (1 '='); predicted 54.0 close to gold 56.0
```text
There are 7 bags with 8 marbles each.
7 * 8 = 54.
#### 54
```

### Category: `misread_question`
**Example 1 (ID: err_07_misread_question)**
- **Question:** Sam bought 4 notebooks for $5 each and 2 pens for $3 each. How much did he spend on notebooks?
- **Gold Answer:** `20`
- **Extracted Answer:** `26`
- **Rationale:** Question constraint or quantity misread
```text
Sam bought 4 notebooks for $5 each, which is 4 * 5 = 20 dollars.
He also bought 2 pens for $3 each, which is 2 * 3 = 6 dollars.
Adding them together gives 20 + 6 = 26 dollars in total instead of notebooks.
#### 26
```

### Category: `wrong_setup`
**Example 1 (ID: err_08_wrong_setup)**
- **Question:** A car travels 180 miles in 3 hours. What is its speed in miles per hour?
- **Gold Answer:** `60`
- **Extracted Answer:** `540`
- **Rationale:** Mathematical formulation or operation setup diverged from problem requirements
```text
To find the speed, we multiply the distance by the time:
180 * 3 = 540 mph.
#### 540
```

### Category: `incomplete_reasoning`
**Example 1 (ID: err_09_incomplete_reasoning)**
- **Question:** A train has 5 carriages with 40 seats each. At the first stop, 50 passengers board. At the second stop, 30 passengers leave. How many empty seats remain?
- **Gold Answer:** `180`
- **Extracted Answer:** `150`
- **Rationale:** Answer provided with minimal or no preceding derivation steps
```text
There are seats on the train.
#### 150
```

### Category: `other`
**Example 1 (ID: err_10_other_degenerate)**
- **Question:** Solve 20 divided by 4.
- **Gold Answer:** `5`
- **Extracted Answer:** `999`
- **Rationale:** Degenerate repetition detected in generated reasoning
```text
repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat
#### 999
```
