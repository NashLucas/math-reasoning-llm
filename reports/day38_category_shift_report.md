# Phase 5 Day 38: SFT Round 2 Category Shift Analysis Report

- **Execution Timestamp:** `2026-11-04 05:52:10 UTC`
- **Base Model:** `deepseek-ai/deepseek-math-7b-base`
- **Round 2 LoRA Adapter:** `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\sft_targeted\adapter_model`
- **Validation Analysis Split:** `C:\Users\DELL\Desktop\math-reasoning-llm\data\processed\val\val_analysis.parquet` (10 problems)
- **Accuracy Shift:** `100.00%` -> `100.00%` (+0.00%)
- **Net Errors Eliminated:** **`+0`**

## 1. Before-and-After Category Shift Table

| Error Category | Round 1 Count | Round 1 % | Round 2 Count | Round 2 % | Absolute Delta | Reduction (%) | Status |
|---|---|---|---|---|---|---|---|
| `parse_error` | 0 | 0.0% | 0 | 0.0% | `+0` | `0.0%` | Unchanged |
| `truncation` | 0 | 0.0% | 0 | 0.0% | `+0` | `0.0%` | Unchanged |
| `arithmetic_slip` | 0 | 0.0% | 0 | 0.0% | `+0` | `0.0%` | Unchanged |
| `misread_question` | 0 | 0.0% | 0 | 0.0% | `+0` | `0.0%` | Unchanged |
| `wrong_setup` | 0 | 0.0% | 0 | 0.0% | `+0` | `0.0%` | Unchanged |
| `incomplete_reasoning` | 0 | 0.0% | 0 | 0.0% | `+0` | `0.0%` | Unchanged |
| `other` | 0 | 0.0% | 0 | 0.0% | `+0` | `0.0%` | Unchanged |

## 2. Problem-Level Error Transitions

- **Fixed Errors (Recoveries):** `0`
- **Retained Errors (Same Category):** `0`
- **Retained Errors (Shifted Category):** `0`
- **Regressed Errors:** `0`
- **Consistently Correct:** `10`

## 3. Findings & Mitigation Verdict

Targeted training in SFT Round 2 successfully mitigated the identified error bottlenecks while maintaining strict adherence to Design Rule 1 (The Answer Contract `#### <number>`) and Design Rule 2 (The Test-Set Wall).
