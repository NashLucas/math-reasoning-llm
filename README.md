# math-reasoning-llm

Post-training `deepseek-math-7b-base` on mathematical reasoning (GSM8K) through four progressive stages: Baseline, SFT on curated reasoning traces, Error Analysis & Targeted SFT, and Group Relative Policy Optimization (GRPO).

## Overview

This project implements an end-to-end post-training pipeline designed for Google Colab G4 runtimes with strict evaluation walls, answer extraction contracts, and reproducible artifact logging.

### Model Lineup
- **Base Model (Trained):** `deepseek-ai/deepseek-math-7b-base`
- **Development Model (Fast validation/testing):** `Qwen/Qwen2.5-1.5B-Instruct`
- **Reference Models:** `deepseek-ai/deepseek-math-7b-instruct` and `deepseek-ai/deepseek-math-7b-rl`
- **Safety Evaluation:** `meta-llama/Llama-Guard-3-8B`

---

## Post-Training Phase Roadmap & Gate Status

| Phase | Description | Release Tag | Gate Status | Advancing Artifact |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 0** | Environment, Foundation & Tooling | `v0.1` | **MET** | Central config, run provenance, CI pipeline |
| **Phase 1** | Data Contract, Answer Extraction & EDA | `v0.2` | **MET** | 100% extraction contract, split manifests |
| **Phase 2** | Generation Engine, Baselines & Safety | `v0.3` | **MET** | vLLM/HF engine, 7B baseline benchmarks |
| **Phase 3** | Reasoning Data Synthesis & Curation | `v0.4` | **MET** | `data/synthetic/curated/traces_curated_sft.jsonl` |
| **Phase 4** | Supervised Fine-Tuning (SFT Round 1) | `v0.5` | **MET** | `outputs/sft/adapter_model/` |
| **Phase 5** | Error Analysis & Targeted SFT (Round 2) | `v0.6` | **MET** | `outputs/sft_targeted/adapter_model/` |
| **Phase 6** | Group Relative Policy Optimization (GRPO) | `v0.7` | *In Progress* | RL Policy Adapter (`outputs/grpo/`) |
| **Phase 7** | Final Evaluation, Benchmarks & Demo | `v1.0` | *Planned* | Final Test Evaluation & Gradio App |

---

## Results Benchmark

*Per Design Rule 2 (The Test-Set Wall), the GSM8K test set (1,319 problems) remains strictly quarantined until Phase 7 final evaluation. Validation numbers represent greedy decoding on the validation sub-partitions (`val_gating` & `val_analysis`). `maj@8` denotes majority vote self-consistency over 8 sampled paths.*

| Model / Stage | Training Data | Validation Greedy (%) | maj@8 Accuracy (%) | Format Compliance (%) | Advancing Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `deepseek-math-7b-base` | None (8-shot CoT) | Baseline | — | — | Base Reference |
| `deepseek-math-7b-instruct` | DeepSeek (reference) | Benchmark | — | — | Distillation Reference |
| `deepseek-math-7b-rl` | DeepSeek (reference) | Benchmark | — | — | RL Reference |
| **Ours: SFT Round 1** | Curated reasoning traces | Evaluated | Evaluated | $\ge 98.0\%$ | Checkpoint Selected (`v0.5`) |
| **Ours: Targeted SFT (Round 2)** | + Error-targeted examples | Parity / Win | — | $\ge 98.0\%$ | **Advancing to Phase 6 (`v0.6`)** |
| **Ours: GRPO** | RL from best SFT adapter | *Phase 6* | *Phase 6* | *Phase 6* | *Phase 6* |

---

## Key Design Rules

1. **The Answer Contract (Design Rule 1):** All reasoning completions must terminate with a canonical final answer line: `\n#### <number>`.
2. **The Test-Set Wall (Design Rule 2):** Zero test contamination. The GSM8K test set (`data/processed/test/test.parquet`, 1,319 problems) is never evaluated, parsed, or prompted during intermediate training, selection, or gating rounds.
3. **Validation Accuracy Selection (Design Rule 3):** Checkpoints and round advancement decisions are governed strictly by held-out validation accuracy under the Gating Decision Rule rather than training loss.

---

## Repository Structure

```text
math-reasoning-llm/
├── configs/              # Central configuration profiles (local & colab)
├── data/                 # Raw, processed, synthetic, targeted, and safety datasets
├── src/                  # Core modules (data, reasoning, training, rewards, evaluation)
├── scripts/              # Executable stage workflows & gate verifiers
├── app/                  # Gradio demo and request logging
├── notebooks/            # Colab operational launcher and EDA notebooks
├── tests/                # Automated pytest & unittest suite
├── reports/              # Phase gate reports, error taxonomies, and audit logs
└── outputs/              # Staged adapters (outputs/sft_targeted/adapter_model)
```

---

## License

This project is licensed under the [MIT License](LICENSE).
