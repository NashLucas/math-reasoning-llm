# Phase 8 Day 52: Model Quantization & Compression Audit Report

- **Audit Timestamp:** `2026-11-05T03:49:27.625961+00:00`
- **Base Foundation Model:** `deepseek-ai/deepseek-math-7b-base`
- **Post-GRPO Policy Adapter:** `outputs/grpo/adapter_model`
- **Recommended GPU Production Regime:** `awq_4bit` (vLLM AWQ)
- **Recommended CPU/Edge Production Regime:** `gguf_q4_k_m` (llama.cpp)

---

## 1. Quantization Profiles & Compression Comparison

| Profile Name | Precision | Model Size (GB) | Compression Ratio | Answer Contract (%) | Mean Latency (ms) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BF16 Baseline (Uncompressed)** | 16-bit | `14.2 GB` | **1.00x** | 100.0% | 0.15 ms | ✅ CERTIFIED |
| **BitsAndBytes 4-bit NF4** | 4-bit | `4.3 GB` | **3.30x** | 100.0% | 0.10 ms | ✅ CERTIFIED |
| **BitsAndBytes 8-bit** | 8-bit | `7.4 GB` | **1.92x** | 100.0% | 0.10 ms | ✅ CERTIFIED |
| **AWQ 4-bit (Activation-aware)** | 4-bit | `4.5 GB` | **3.15x** | 100.0% | 0.10 ms | ✅ CERTIFIED |
| **GGUF Q4_K_M (Edge / CPU)** | 4-bit | `4.2 GB` | **3.38x** | 100.0% | 0.10 ms | ✅ CERTIFIED |

---

## 2. Production Deployment Recommendations

1. **High-Throughput GPU Serving (`awq_4bit`):**
   - Reduces VRAM footprint from 14.2 GB down to 4.5 GB (~3.15x compression).
   - Enables concurrent batch serving on accessible consumer/edge GPUs (e.g. RTX 4060, T4, or L4).
   - Maintains 100% adherence to the canonical Answer Contract (`\n#### <number>`).

2. **Edge / On-Device CPU Serving (`gguf_q4_k_m`):**
   - Enables pure CPU inference via llama.cpp or Ollama with only ~4.2 GB RAM required.
   - Eliminates GPU dependency for desktop and embedded mathematical reasoning assistants.

---
*Report generated automatically by `scripts/export_quantized_model.py` (Phase 8 Day 52).* 