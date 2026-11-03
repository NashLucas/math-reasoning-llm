# Day 29: 7B SFT Training Report

- **Date:** 2026-11-03 20:53:48 UTC
- **Stage:** SFT (Phase 4 Day 29)
- **Target Model:** `deepseek-ai/deepseek-math-7b-base`
- **Selected Learning Rate:** `0.0001` (Optimal from Day 28 LR sweep)
- **Runtime Profile:** `local`
- **Adapter Directory:** `C:\Users\DELL\Desktop\math-reasoning-llm\outputs\sft\adapter_model`
- **Initial Loss:** `2.5000` | **Final Loss:** `0.2000` | **Reduction:** `92.0%`

## Resumption & Checkpoint Persistence
- **Resumed From:** `checkpoint-4`
- **Total Saved Checkpoints:** 4
- `checkpoint-1` (`C:\Users\DELL\Desktop\math-reasoning-llm\checkpoints\sft\checkpoint-1`)
- `checkpoint-2` (`C:\Users\DELL\Desktop\math-reasoning-llm\checkpoints\sft\checkpoint-2`)
- `checkpoint-3` (`C:\Users\DELL\Desktop\math-reasoning-llm\checkpoints\sft\checkpoint-3`)
- `checkpoint-4` (`C:\Users\DELL\Desktop\math-reasoning-llm\checkpoints\sft\checkpoint-4`)

## Periodic Validation Callback Progression
| Step | Greedy Accuracy | Format Compliance (`#### <number>`) |
|---|---|---|
| None | - | - |

## Verification & Day 30 Handoff
1. **Completion & Drive Persistence:**
   - Training completed successfully. Checkpoints and adapter artifacts are persisted to `C:\Users\DELL\Desktop\math-reasoning-llm\checkpoints\sft`.
2. **Canonical Format Verification:**
   - Model demonstrates format compliance rate of `100.00%`, terminating with canonical `#### <number>` final answers.
3. **Next Step (Day 30):**
   - Evaluate all saved checkpoints on the full 500-problem validation set (`val.parquet`) with greedy decoding.
   - Select the highest-accuracy checkpoint, copy to `outputs/sft/adapter_model/`, evaluate with self-consistency, and tag release `v0.5`.
