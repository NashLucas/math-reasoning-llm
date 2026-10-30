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

## Results Benchmark

*All numbers evaluated on the GSM8K test set (1,319 problems). `maj@8` denotes majority vote self-consistency over 8 samples.*

| Model | Training | Greedy Accuracy (%) | maj@8 Accuracy (%) | Format Compliance (%) | Avg Response Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `deepseek-math-7b-base` | None (8-shot CoT) | — | — | — | — |
| `deepseek-math-7b-instruct` | DeepSeek (reference) | — | — | — | — |
| `deepseek-math-7b-rl` | DeepSeek (reference) | — | — | — | — |
| **Ours: SFT** | Curated reasoning traces | — | — | — | — |
| **Ours: Targeted SFT** | + Error-targeted examples | — | — | — | — |
| **Ours: GRPO** | RL from best SFT adapter | — | — | — | — |

---

## Repository Structure

```text
math-reasoning-llm/
├── configs/              # Central configuration profiles (local & colab)
├── data/                 # Raw, processed, synthetic, and safety datasets
├── src/                  # Core modules (data, reasoning, training, rewards, evaluation)
├── scripts/              # Executable stage workflows
├── app/                  # Gradio demo and request logging
├── notebooks/            # Colab launcher and EDA notebooks
├── tests/                # Automated pytest suite (smoke tests & unit tests)
└── outputs/              # Checkpoints, evaluation results, and reports
```

---

## License

This project is licensed under the [MIT License](LICENSE).
