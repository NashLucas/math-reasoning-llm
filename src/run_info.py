"""Run tracking, environment provenance, and reproducible seeding.

Captures system information (GPU, CUDA, platform), installed library versions,
git commit hash and dirty status, and a full snapshot of stage configuration.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info(f"Reproducibility seed set to {seed} (deterministic={deterministic})")
    return seed


def get_git_info() -> Dict[str, Any]:
    """Retrieve current git commit hash, branch, and uncommitted status."""
    git_info: Dict[str, Any] = {
        "commit_hash": "unknown",
        "commit_short": "unknown",
        "branch": "unknown",
        "is_dirty": False,
        "warning": None,
    }

    try:
        # Commit hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_info["commit_hash"] = commit
        git_info["commit_short"] = commit[:7]

        # Branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_info["branch"] = branch

        # Status check for uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        if status:
            git_info["is_dirty"] = True
            msg = (
                "WARNING: Working directory has uncommitted changes. "
                "The recorded commit hash does NOT represent the exact state of this run!"
            )
            git_info["warning"] = msg
            print(f"\n[bold yellow]! {msg}\n")
    except Exception as e:
        git_info["warning"] = f"Failed to retrieve git info: {e}"

    return git_info


def get_gpu_info() -> Dict[str, Any]:
    """Detect available GPU devices and CUDA runtime details."""
    gpu_info: Dict[str, Any] = {
        "device": "cpu",
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "cuda_version": None,
    }

    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["device"] = "cuda"
            gpu_info["cuda_available"] = True
            gpu_info["gpu_count"] = torch.cuda.device_count()
            gpu_info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(gpu_info["gpu_count"])
            ]
            gpu_info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    return gpu_info


def get_library_versions() -> Dict[str, str]:
    """Inspect versions of all critical ML, data, and training libraries."""
    tracked_packages = [
        "torch",
        "transformers",
        "trl",
        "peft",
        "accelerate",
        "datasets",
        "vllm",
        "sentence_transformers",
        "sklearn",
        "pandas",
        "numpy",
        "yaml",
        "gradio",
        "pytest",
    ]

    versions: Dict[str, str] = {}
    for pkg_name in tracked_packages:
        try:
            mod = __import__(pkg_name)
            versions[pkg_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg_name] = "not_installed"
    return versions


def collect_run_info(
    stage: str,
    config: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble a full provenance record for the run."""
    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    desc_slug = (description or "run").replace(" ", "_").lower()
    run_id = f"{stage}-{timestamp_str}-{desc_slug}"

    return {
        "run_id": run_id,
        "stage": stage,
        "timestamp_utc": now.isoformat(),
        "description": description or "",
        "git": get_git_info(),
        "hardware": get_gpu_info(),
        "system": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "library_versions": get_library_versions(),
        "config_snapshot": config or {},
    }


def save_run_info(
    stage: str,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
) -> Path:
    """Collect and persist run_info.json in the stage's output folder."""
    if output_dir is None:
        try:
            from src.config import ConfigManager
            mgr = ConfigManager()
            stage_paths = mgr.get_stage_paths(stage)
            target_dir = stage_paths.get("outputs", Path(f"outputs/{stage}"))
        except Exception:
            target_dir = Path(f"outputs/{stage}")
    else:
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "run_info.json"

    run_info = collect_run_info(stage=stage, config=config, description=description)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print(f"Run provenance record saved to: {out_file.resolve()}")
    return out_file
