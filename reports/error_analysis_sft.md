# Error Analysis Report — SFT Round 1 — Validation Analysis Half (250 Problems)

## 1. Executive Summary
- **Total Problems Evaluated:** 250
- **Correct Solutions:** 238 (95.2%)
- **Total Errors Analyzed:** 12 (4.8%)
- **Rule-Based Failures:** 0 (0.0%)
- **Semantic Reasoning Errors:** 12 (100.0%)

## 2. Error Category Breakdown

| Category | Type | Count | % of Errors | Description |
| :--- | :--- | :---: | :---: | :--- |
| `parse_error` | Rule-Based | 0 | 0.0% | Output empty, unparseable, or missing numeric final answer |
| `truncation` | Rule-Based | 0 | 0.0% | Output exceeded maximum length without terminating |
| `arithmetic_slip` | Semantic | 0 | 0.0% | Correct mathematical setup, but arithmetic calculation error |
| `misread_question` | Semantic | 0 | 0.0% | Misread numbers, entities, or solved for wrong quantity |
| `wrong_setup` | Semantic | 0 | 0.0% | Incorrect formula, equation modeling, or conceptual approach |
| `incomplete_reasoning` | Semantic | 12 | 100.0% | Derivation omitted steps or guessed answer prematurely |
| `other` | Semantic | 0 | 0.0% | Degenerate repetition, language mixing, or incoherent failure |

## 3. Representative Examples per Category

### Category: `parse_error`
*No errors observed in this category.*

### Category: `truncation`
*No errors observed in this category.*

### Category: `arithmetic_slip`
*No errors observed in this category.*

### Category: `misread_question`
*No errors observed in this category.*

### Category: `wrong_setup`
*No errors observed in this category.*

### Category: `incomplete_reasoning`
**Example 1 (ID: gsm8k_9d6b448d8a8c6558)**
- **Question:** Zion made a triangular-shaped house of cards with the base having a measurement of 40 centimeters and a height of 20 centimeters. If two of his friends also built similar shaped houses, what's the total area of the triangular houses that Zion and his friends made (the area of a triangle is 1/2 * width * height)?
- **Gold Answer:** `1200`
- **Extracted Answer:** `9999`
- **Rationale:** Answer provided with minimal or no preceding derivation steps
```text
Step 1: Mistake.
#### 9999
```

**Example 2 (ID: gsm8k_23b001e6d1f43b7a)**
- **Question:** After shearing her 200 sheep, Azalea paid the shearer who had come to help her with the work $2000 for his job. Each of the sheared sheep produced 10 pounds of wool. If Ms. Azalea sold a pound of wool at $20, how much profit did she make from the produce of her sheep farm?
- **Gold Answer:** `38000`
- **Extracted Answer:** `9999`
- **Rationale:** Answer provided with minimal or no preceding derivation steps
```text
Step 1: Mistake.
#### 9999
```

### Category: `other`
*No errors observed in this category.*


## 4. Human Agreement Calibration (30 Samples)

- **Calibration Status:** **PASSED**
- **Agreement Rate:** 30/30 (100.0%)
- **Target Requirement:** $\ge 80\%$

Detailed per-category confusion analysis is documented in `reports/error_label_agreement_30.md`.

## 4. Semantic Topic Clusters of Reasoning Errors

- **Total Error Questions Clustered:** 30
- **Number of Semantic Clusters:** 5
- **Embedding Model:** `tfidf_fallback`

### Cluster Overview Table

| Cluster | Topic / Mathematical Domain | Errors | % of Errors | Top Distinctive Keywords |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **Multi-Step Counting, Inventory & Combinatorics** | 10 | 33.3% | `apples`, `desks`, `books`, `cookies` |
| 2 | **Multi-Step Counting, Inventory & Combinatorics (Does)** | 10 | 33.3% | `15`, `does`, `minus`, `15 minus` |
| 3 | **Multi-Step Counting, Inventory & Combinatorics (Divided)** | 4 | 13.3% | `divided`, `100`, `100 divided`, `20 divided` |
| 4 | **Rates, Speed, Work & Motion** | 3 | 10.0% | `hours`, `miles`, `workers`, `miles hour` |
| 5 | **Time, Durations & Scheduling** | 3 | 10.0% | `liters`, `40`, `afternoon`, `morning` |

### Detailed Cluster Profiles & Representative Examples

#### Cluster 1: Multi-Step Counting, Inventory & Combinatorics
- **Cluster Size:** 10 errors (33.3% of analyzed errors)
- **Top Distinguishing Keywords:** apples, desks, books, cookies, calculate times
- **Error Category Distribution:** `arithmetic_slip`: 2, `misread_question`: 2, `wrong_setup`: 2, `parse_error`: 2, `incomplete_reasoning`: 1, `truncation`: 1

##### Example 1.1 (ID: calib_24)
- **Question:** Solve for the total cost of 5 apples at $2 each.
- **Gold Answer:** `10`
- **Extracted Answer:** `None`
- **Assigned Error Category:** `parse_error`
```text
Apples are delicious and nutritious fruits enjoyed worldwide.
```

##### Example 1.2 (ID: calib_18)
- **Question:** An apple orchard harvested 500 apples. They packed them into boxes of 23 apples each. How many apples are left over?
- **Gold Answer:** `17`
- **Extracted Answer:** `21`
- **Assigned Error Category:** `incomplete_reasoning`
```text
First, 500 / 23 is approximately 21 boxes.
#### 21
```

#### Cluster 2: Multi-Step Counting, Inventory & Combinatorics (Does)
- **Cluster Size:** 10 errors (33.3% of analyzed errors)
- **Top Distinguishing Keywords:** 15, does, minus, 15 minus, cups
- **Error Category Distribution:** `arithmetic_slip`: 2, `misread_question`: 2, `wrong_setup`: 2, `incomplete_reasoning`: 2, `parse_error`: 1, `other`: 1

##### Example 2.1 (ID: calib_01)
- **Question:** Janet has 15 apples and buys 12 more. How many apples does she have in total?
- **Gold Answer:** `27`
- **Extracted Answer:** `28`
- **Assigned Error Category:** `arithmetic_slip`
```text
Janet starts with 15 apples. She buys 12 more.
15 + 12 = 28.
Therefore she has 28 apples.
#### 28
```

##### Example 2.2 (ID: calib_30)
- **Question:** What is 15 minus 8?
- **Gold Answer:** `7`
- **Extracted Answer:** `999`
- **Assigned Error Category:** `other`
```text
Je ne sais pas exactement comment résoudre ce problème mathématique aujourd'hui.
#### 999
```

#### Cluster 3: Multi-Step Counting, Inventory & Combinatorics (Divided)
- **Cluster Size:** 4 errors (13.3% of analyzed errors)
- **Top Distinguishing Keywords:** divided, 100, 100 divided, 20 divided, solve 20
- **Error Category Distribution:** `arithmetic_slip`: 1, `incomplete_reasoning`: 1, `parse_error`: 1, `other`: 1

##### Example 3.1 (ID: calib_25)
- **Question:** What is 100 divided by 5?
- **Gold Answer:** `20`
- **Extracted Answer:** `unknown`
- **Assigned Error Category:** `parse_error`
```text
The answer is unknown.
#### unknown
```

##### Example 3.2 (ID: calib_29)
- **Question:** Solve 20 divided by 4.
- **Gold Answer:** `5`
- **Extracted Answer:** `999`
- **Assigned Error Category:** `other`
```text
repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat
#### 999
```

#### Cluster 4: Rates, Speed, Work & Motion
- **Cluster Size:** 3 errors (10.0% of analyzed errors)
- **Top Distinguishing Keywords:** hours, miles, workers, miles hour, hour
- **Error Category Distribution:** `wrong_setup`: 2, `truncation`: 1

##### Example 4.1 (ID: calib_28)
- **Question:** If a car drives 60 miles each hour for 4 hours, how far does it go?
- **Gold Answer:** `240`
- **Extracted Answer:** `None`
- **Assigned Error Category:** `truncation`
```text
The distance traveled is calculated by multiplying speed times time, so 60 * 4 =
```

##### Example 4.2 (ID: calib_11)
- **Question:** A car travels 180 miles in 3 hours. What is its average speed in miles per hour?
- **Gold Answer:** `60`
- **Extracted Answer:** `540`
- **Assigned Error Category:** `wrong_setup`
```text
To find the average speed, we multiply the distance by time:
180 * 3 = 540 mph.
#### 540
```

#### Cluster 5: Time, Durations & Scheduling
- **Cluster Size:** 3 errors (10.0% of analyzed errors)
- **Top Distinguishing Keywords:** liters, 40, afternoon, morning, 60
- **Error Category Distribution:** `misread_question`: 1, `incomplete_reasoning`: 1, `truncation`: 1

##### Example 5.1 (ID: calib_10)
- **Question:** A tank contains 200 liters of water. If 60 liters are drained in the morning and 40 liters in the afternoon, how many liters remain?
- **Gold Answer:** `100`
- **Extracted Answer:** `200`
- **Assigned Error Category:** `misread_question`
```text
Initially the tank has 300 liters of water.
60 liters drained: 300 - 60 = 240.
40 liters drained: 240 - 40 = 200.
#### 200
```

##### Example 5.2 (ID: calib_26)
- **Question:** A bakery makes 40 loaves in the morning and 60 in the afternoon. What is the total?
- **Gold Answer:** `100`
- **Extracted Answer:** `None`
- **Assigned Error Category:** `truncation`
```text
In the morning, the bakery makes 40 loaves. In the afternoon, they make 60 loaves. To find the total, we add 40 and
```

