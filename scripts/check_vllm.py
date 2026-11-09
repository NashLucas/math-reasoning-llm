"""Day 5 check: does vLLM install and generate on this runtime?

Records the result to <profile outputs_dir>/env_checks/vllm_check.json so it can be logged in
EXPERIMENTS.md. The plan's fallback rules depend on it: if vLLM fails on G4, use Hugging Face
generation, halve the Phase 3 subset, use 4-6 GRPO generations on ~600 problems, and k=4 for
self-consistency on the reference models.

Run after `pip install -r requirements-vllm.txt` on the target runtime.

Usage:
  python scripts/check_vllm.py                 # import + tiny generation with the dev model
  python scripts/check_vllm.py --import_only   # import check only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager  # noqa: E402
from src.run_info import get_gpu_info, get_library_versions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that vLLM imports and generates on this runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=None, help="Model for the generation test (defaults to dev_model).")
    parser.add_argument("--import_only", action="store_true", help="Skip the generation test.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ConfigManager()
    model_name = args.model or cfg.base_config.get("models", {}).get("dev_model", "Qwen/Qwen2.5-1.5B-Instruct")

    result: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": get_gpu_info(),
        "library_versions": get_library_versions(),
        "import_ok": False,
        "generation_ok": None,
        "error": None,
    }

    try:
        import torch
        try:
            _ = torch.ops.aten.cholesky
        except AttributeError:
            try:
                torch.ops.aten.__dict__["cholesky"] = getattr(
                    torch.ops.aten, "linalg_cholesky", lambda *a, **kw: None
                )
            except Exception:
                pass

        import vllm

        result["import_ok"] = True
        result["vllm_version"] = vllm.__version__
        print(f"vLLM {vllm.__version__} imported.")

        if not args.import_only:
            from vllm import LLM, SamplingParams

            start = time.time()
            llm = LLM(model=model_name, dtype="auto", gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=1024)
            out = llm.generate(["Question: What is 12 + 30?\nAnswer:"], SamplingParams(temperature=0.0, max_tokens=32))
            result["generation_ok"] = True
            result["generation_model"] = model_name
            result["generation_seconds"] = round(time.time() - start, 1)
            result["sample_output"] = out[0].outputs[0].text
            print(f"Generation OK in {result['generation_seconds']}s: {result['sample_output']!r}")
    except Exception as e:
        if result["import_ok"]:
            result["generation_ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        print(f"vLLM check FAILED: {result['error']}")

    passed = result["import_ok"] and result["generation_ok"] is not False
    result["verdict"] = "vllm_ok" if passed else "use_hf_fallback"

    out_dir = cfg.outputs_path("env_checks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "vllm_check.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Verdict: {result['verdict']} | GPU: {result['hardware'].get('gpu_names')} | saved to {out_file}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
