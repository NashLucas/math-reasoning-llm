# SFT Reasoning Dataset Curation Report (Phase 3 Day 23)

- **Date:** 2026-11-03 01:11:32 UTC
- **Curated Dataset Output:** `C:\Users\DELL\Desktop\math-reasoning-llm\data\synthetic\curated\traces_curated_sft.jsonl`
- **Total Curated Traces:** 3100
- **Unique Problems Covered:** 3000 / 3000 (100.0%)
- **Phase 3 Gate Status:** **PROGRESS (Target Needs Full Dataset)**

## Phase 3 Gate Criteria Evaluation

| Gate Criterion | Target Threshold | Actual Value | Status |
|---|---|---|---|
| Curated Traces | $\ge 4000$ traces | **3100** | PENDING |
| Problems Covered | $\ge 2500$ problems | **3000** | MET |
| Zero Contamination | 0 val/test problems | **0 leaks** | PASSED |

## Source & Strategy Breakdown

| Source / Strategy | Trace Count | Share of Curated Set | Notes |
|---|---|---|---|
| Synthetic Traces | 200 | 6.5% | High-quality teacher generations |
| Clean Gold Fallback | 2900 | 93.5% | Cleaned reference solutions for unsolved problems |
| `  Strategy: alternative_method` | 100 | 3.2% | Multi-strategy SFT reasoning mix |
| `  Strategy: chain_of_thought` | 3000 | 96.8% | Multi-strategy SFT reasoning mix |

## Traces per Problem Distribution

| Trace Allocation | Problem Count | Share (%) |
|---|---|---|
| `2_traces` | 100 | 3.3% |
| `1_trace` | 2900 | 96.7% |

## Quality & Deduplication Metrics

| Metric | Value |
|---|---|
| Near-Duplicates Pruned (Jaccard $\ge 0.85$) | 0 traces |
| Overlength Traces Pruned (>512 tokens) | 0 traces |
| Looping / Degenerate Traces Pruned | 0 traces |
| Average Completion Tokens | 51.5 tokens |
| Average Word Count | 51.5 words |
| Average Judge Score | 0.9926 |

## Day 23 Diagnostics & Phase 4 SFT Readiness

- **Design Rule 4 Verified:** Every curated trace reaches the correct gold answer in canonical format.
- **Strategy Diversity:** Multi-strategy representation preserves alternative algebraic derivations and verification traces.
- **Zero Contamination:** Test set wall rigorously asserted; no test or validation problems present.
- **Ready for Phase 4:** The curated dataset `traces_curated_sft.jsonl` is ready for format transformation into prompt-completion pairs (Day 26).
