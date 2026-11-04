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
