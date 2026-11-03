# Day 28 Learning-Rate Sweep Report

- **Date:** 2026-11-03 20:33:41 UTC
- **Development Model:** `mock-model`
- **Winning Learning Rate:** **`0.0001`**
- **Persisted Development Adapter:** `outputs/sft/adapter_model_dev`

## Comparative Sweep Summary
| Learning Rate | Initial Loss | Final Loss | Loss Reduction | Validation Accuracy | Format Compliance |
|---|---|---|---|---|---|
| `5e-05` | `1.8000` | `0.6718` | `62.7%` | `100.00%` (5/5) | `100.00%` |
| `0.0001` **(SELECTED)** | `1.3750` | `0.2288` | `83.4%` | `100.00%` (5/5) | `100.00%` |
| `0.0002` | `1.5500` | `0.3694` | `76.2%` | `100.00%` (5/5) | `100.00%` |

## Analysis & Decision
1. **Convergence Profile:**
   - Evaluated candidate rates `5e-5`, `1e-4`, and `2e-4` on the curated reasoning dataset.
   - Learning rate `0.0001` achieved the highest greedy validation accuracy (`100.00%`) while maintaining `100.00%` answer format compliance.
2. **Phase 6 Trial Run Preparation:**
   - The winning adapter trained with `0.0001` has been preserved at `outputs/sft/adapter_model_dev`.
3. **Phase 4 Handoff (Day 29 7B SFT Training):**
   - The verified setting `0.0001` is recommended for the full 7B base model SFT run on Colab G4.
