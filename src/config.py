"""Configuration loading, merging, and path resolution system.

Supports multiple runtime profiles (e.g. 'local' for development and 'colab' for
Google Drive persistence) and stage-based folder resolution across the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# Locate project root (the directory containing configs/ and src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def get_default_profile() -> str:
    """Detect default profile based on environment."""
    if "APP_PROFILE" in os.environ:
        return os.environ["APP_PROFILE"].lower()
    # Detect Google Colab environment
    if os.path.exists("/content") or "COLAB_GPU" in os.environ:
        return "colab"
    return "local"


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    path = Path(file_path)
    if not path.is_absolute():
        path = CONFIGS_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge multiple configuration dictionaries."""
    merged: Dict[str, Any] = {}
    for cfg in configs:
        for k, v in cfg.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = merge_configs(merged[k], v)
            else:
                merged[k] = v
    return merged


class ConfigManager:
    """Manages project configurations and resolves profile-dependent paths."""

    def __init__(self, configs_dir: Optional[Union[str, Path]] = None):
        self.configs_dir = Path(configs_dir) if configs_dir else CONFIGS_DIR
        self._paths_cfg = self._load("paths.yaml")
        self._base_cfg = self._load("base_config.yaml")
        self._pipeline_cfg = self._load("pipeline.yaml")

    def _load(self, filename: str) -> Dict[str, Any]:
        target = self.configs_dir / filename
        if not target.exists():
            return {}
        with open(target, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def base_config(self) -> Dict[str, Any]:
        return self._base_cfg

    @property
    def pipeline_config(self) -> Dict[str, Any]:
        return self._pipeline_cfg

    @property
    def paths_config(self) -> Dict[str, Any]:
        return self._paths_cfg

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load and return any config YAML file from the configs directory."""
        return self._load(filename)

    def get_profile(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """Return raw profile variables for a given profile."""
        name = profile_name or get_default_profile()
        profiles = self._paths_cfg.get("profiles", {})
        if name not in profiles:
            raise KeyError(
                f"Unknown profile '{name}'. Available: {list(profiles.keys())}"
            )
        return profiles[name]

    def resolve_path(self, raw_path: str, profile_name: Optional[str] = None) -> Path:
        """Resolve a path template with profile variables."""
        profile = self.get_profile(profile_name)
        formatted = raw_path.format(**profile)
        p = Path(formatted)
        if not p.is_absolute():
            # If relative, resolve against project root
            return (PROJECT_ROOT / p).resolve()
        return p

    def get_stage_paths(
        self, stage_name: str, profile_name: Optional[str] = None
    ) -> Dict[str, Path]:
        """Resolve all configured output/checkpoint/log paths for a specific stage."""
        profile = self.get_profile(profile_name)
        stage_defs = self._paths_cfg.get("stages", {})

        if stage_name not in stage_defs:
            # Fallback to standard convention if stage is not explicitly listed in stages dict
            checkpoints_dir = profile.get("checkpoints_dir", "checkpoints")
            outputs_dir = profile.get("outputs_dir", "outputs")
            logs_dir = profile.get("logs_dir", "logs")
            stage_def = {
                "checkpoints": f"{checkpoints_dir}/{stage_name}",
                "outputs": f"{outputs_dir}/{stage_name}",
                "logs": f"{logs_dir}/{stage_name}",
            }
        else:
            stage_def = stage_defs[stage_name]

        resolved: Dict[str, Path] = {}
        for key, template_str in stage_def.items():
            if isinstance(template_str, str):
                formatted = template_str.format(**profile)
                p = Path(formatted)
                if not p.is_absolute():
                    resolved[key] = (PROJECT_ROOT / p).resolve()
                else:
                    resolved[key] = p

        return resolved

    def get_all_stages(self) -> list[str]:
        """Return ordered list of pipeline stage names."""
        stages = self._pipeline_cfg.get("stages", [])
        return [s["name"] for s in stages]

    def get_all_stage_paths(
        self, profile_name: Optional[str] = None
    ) -> Dict[str, Dict[str, Path]]:
        """Return resolved paths for every stage in the pipeline."""
        stages = self.get_all_stages()
        return {s: self.get_stage_paths(s, profile_name=profile_name) for s in stages}


def print_resolved_paths(profile_name: Optional[str] = None) -> None:
    """Pretty print resolved stage paths for a given profile."""
    mgr = ConfigManager()
    target_profiles = [profile_name] if profile_name else ["local", "colab"]

    for prof in target_profiles:
        profile_info = mgr.get_profile(prof)
        print(f"\n{'='*70}")
        print(f"PROFILE: {prof.upper()} ({profile_info.get('description', '')})")
        print(f"Root: {profile_info.get('root')}")
        print(f"Data Dir: {profile_info.get('data_dir')}")
        print(f"Checkpoints Dir: {profile_info.get('checkpoints_dir')}")
        print(f"Outputs Dir: {profile_info.get('outputs_dir')}")
        print(f"Logs Dir: {profile_info.get('logs_dir')}")
        print(f"HF Cache: {profile_info.get('hf_cache_dir')}")
        print(f"{'-'*70}")
        print(f"{'STAGE':<16} | {'KEY':<14} | {'RESOLVED PATH'}")
        print(f"{'-'*70}")

        all_paths = mgr.get_all_stage_paths(profile_name=prof)
        for stage, paths in all_paths.items():
            first = True
            for k, p in paths.items():
                stage_col = stage if first else ""
                print(f"{stage_col:<16} | {k:<14} | {str(p)}")
                first = False
        print(f"{'='*70}\n")


if __name__ == "__main__":
    print("Printing resolved stage paths under both 'local' and 'colab' profiles:")
    print_resolved_paths()
