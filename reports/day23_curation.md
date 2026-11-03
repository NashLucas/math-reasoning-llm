# SFT Reasoning Dataset Curation Report (Phase 3 Day 23)

- **Date:** 2026-11-03 00:16:10 UTC
- **Curated Dataset Output:** `C:\Users\DELL\Desktop\math-reasoning-llm\data\synthetic\curated\traces_curated_sft.jsonl`
- **Total Curated Traces:** 200
- **Unique Problems Covered:** 100 / 100 (100.0%)
- **Phase 3 Gate Status:** **PASSED (Phase 3 Gate Met)** *(Dev sample subset evaluated)*

## Phase 3 Gate Criteria Evaluation

| Gate Criterion | Target Threshold | Actual Value | Status |
|---|---|---|---|
| Curated Traces | $\ge 133$ traces | **200** | MET |
| Problems Covered | $\ge 83$ problems | **100** | MET |
| Zero Contamination | 0 val/test problems | **0 leaks** | PASSED |

## Source & Strategy Breakdown

| Source / Strategy | Trace Count | Share of Curated Set | Notes |
|---|---|---|---|
| Synthetic Traces | 200 | 100.0% | High-quality teacher generations |
| Clean Gold Fallback | 0 | 0.0% | Cleaned reference solutions for unsolved problems |
| `  Strategy: alternative_method` | 100 | 50.0% | Multi-strategy SFT reasoning mix |
| `  Strategy: chain_of_thought` | 100 | 50.0% | Multi-strategy SFT reasoning mix |

## Traces per Problem Distribution

| Trace Allocation | Problem Count | Share (%) |
|---|---|---|
| `2_traces` | 100 | 100.0% |

## Quality & Deduplication Metrics

| Metric | Value |
|---|---|
| Near-Duplicates Pruned (Jaccard $\ge 0.85$) | 0 traces |
| Overlength Traces Pruned (>512 tokens) | 0 traces |
| Looping / Degenerate Traces Pruned | 0 traces |
| Average Completion Tokens | 46.4 tokens |
| Average Word Count | 46.4 words |
| Average Judge Score | 0.8850 |

## Day 23 Diagnostics & Phase 4 SFT Readiness

- **Design Rule 4 Verified:** Every curated trace reaches the correct gold answer in canonical format.
- **Strategy Diversity:** Multi-strategy representation preserves alternative algebraic derivations and verification traces.
- **Zero Contamination:** Test set wall rigorously asserted; no test or validation problems present.
- **Ready for Phase 4:** The curated dataset `traces_curated_sft.jsonl` is ready for format transformation into prompt-completion pairs (Day 26).
