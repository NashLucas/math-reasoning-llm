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
| **Phase 6** | Group Relative Policy Optimization (GRPO) | `v0.7` | **MET** | `outputs/grpo/adapter_model/` |
| **Phase 7** | Final Evaluation, Benchmarks & Serving | `v1.0` | **MET** | Production Engine & Gradio App (`v1.0`) |

---

## Final Results Benchmark (GSM8K Quarantined Test Set, $N=1,319$)

*Per Design Rule 2 (The Test-Set Wall), the GSM8K test set (1,319 problems) was strictly quarantined until Phase 7 unfreezing with cryptographic SHA-256 verification. Numbers report performance under greedy decoding (with 95% bootstrap confidence intervals), pass@8, majority vote self-consistency (maj@8), completion efficiency, and Answer Contract compliance.*

| Model / Pipeline Stage | Role | Greedy Acc (%) [95% CI] | pass@8 (%) | maj@8 (%) | Tokens / Correct | Contract Adherence (%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **DeepSeekMath-7B-Base** | Base Foundation (8-shot) | **60.1%** [57.4, 62.6] | 78.2% | 68.4% | 240.5 tok | 96.1% | Base Reference |
| **DeepSeekMath-7B-Instruct** | Distillation Reference | **68.6%** [65.9, 71.2] | 83.9% | 76.8% | 209.8 tok | 98.0% | Reference Model |
| **DeepSeekMath-7B-RL** | RL Reference | **71.6%** [68.8, 73.9] | 85.1% | 79.4% | 202.4 tok | 98.7% | Reference Model |
| **Ours: SFT Round 1** | Phase 4 Post-Trained | **69.6%** [66.9, 72.0] | 85.1% | 77.0% | 207.2 tok | 99.5% | Released `v0.5` |
| **Ours: Targeted SFT (Round 2)** | Phase 5 Error-Targeted | **73.7%** [71.1, 76.0] | 87.6% | 81.5% | 195.7 tok | 99.6% | Released `v0.6` |
| **Ours: Post-GRPO Policy** | **Phase 6 RL Policy** | **76.9%** [74.6, 79.2] | **90.3%** | **86.0%** | **188.6 tok** | **100.0%** | **Production Release `v1.0`** |

---

## Production Serving & Interactive Demo

The production serving engine (`src/serving/engine.py`) and Gradio web interface (`app/app.py`) deliver real-time inference with strict Answer Contract adherence and operational telemetry.

### Launching the Interactive Web UI
```bash
# Launch interactive Gradio interface locally
python scripts/serve_model.py --port 7860

# Run in headless programmatic mode (smoke test)
python scripts/serve_model.py --use_mock --smoke_test

# Generate a public share link (Colab / cloud)
python scripts/serve_model.py --port 7860 --share
```

### Operational Serving Verification
```bash
# Verify greedy, sampled, maj@8 consensus, concurrency, and traffic logging
python scripts/verify_serving.py --output_dir reports
```

### Phase 7 Gate Verification & Release v1.0 Certification
```bash
# Formally audit all 7 Phase 7 gate criteria
python scripts/verify_phase7_gate.py
```

---

## Key Design Rules

1. **The Answer Contract (Design Rule 1):** All reasoning completions must terminate with a canonical final answer line: `\n#### <number>`. Post-GRPO Policy achieves 100.0% adherence on the benchmark test set.
2. **The Test-Set Wall (Design Rule 2):** Zero test contamination. The GSM8K test set (`data/processed/test/test.parquet`, 1,319 problems) remained isolated until Phase 7 final unfreezing with verified cryptographic SHA-256 integrity.
3. **Validation Accuracy Selection (Design Rule 3):** Checkpoints and round advancement decisions are governed strictly by held-out validation accuracy under the Gating Decision Rule rather than training loss.

---

## Repository Structure

```text
math-reasoning-llm/
├── configs/              # Central configuration profiles (local & colab)
├── data/                 # Raw, processed, synthetic, targeted, and split manifests
├── src/                  # Core modules
│   ├── data/             # Dataset loaders, preprocessors, and contracts
│   ├── models/           # Model loading, LoRA configuration, and backends
│   ├── reasoning/        # Prompt formats, trace synthesis, and self-consistency
│   ├── training/         # SFT training, callbacks, and sweeps
│   ├── grpo/             # GRPO rewards, advantages, rollouts, and trainer
│   ├── serving/          # Production serving engine & traffic logger
│   └── evaluation/       # Benchmark evaluators, diagnostics, and gate verifiers
├── app/                  # Interactive Gradio web application
├── scripts/              # Executable stage workflows & gate verifiers
├── notebooks/            # Colab operational launcher (Steps 1–48)
├── tests/                # Automated pytest & unittest suite
├── reports/              # Phase gate reports, benchmarks, and audit logs
└── outputs/              # Advancing LoRA adapters (outputs/grpo/adapter_model)
```

---

## License

This project is licensed under the [MIT License](LICENSE).
