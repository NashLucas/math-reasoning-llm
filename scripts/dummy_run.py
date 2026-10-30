"""Dummy verification script for Day 3.

Validates that set_seed works and save_run_info generates a complete
run_info.json record in the target stage's outputs directory.
"""

import json
from pathlib import Path
import sys

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigManager
from src.run_info import set_seed, save_run_info


def main() -> None:
    print("Running Day 3 dummy verification run...")

    # 1. Set seed
    seed = set_seed(42)
    assert seed == 42, "Failed to set seed"

    # 2. Load stage config snapshot
    mgr = ConfigManager()
    stage = "baseline"
    config_snapshot = {
        "base": mgr.base_config,
        "paths": mgr.get_profile(),
        "stage_paths": {k: str(v) for k, v in mgr.get_stage_paths(stage).items()},
    }

    # 3. Save run info into stage output folder
    out_file = save_run_info(
        stage=stage,
        config=config_snapshot,
        description="Day 3 verification dummy run",
    )

    # 4. Validate output contents
    assert out_file.exists(), f"File {out_file} was not created!"

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = [
        "run_id",
        "stage",
        "timestamp_utc",
        "git",
        "hardware",
        "system",
        "library_versions",
        "config_snapshot",
    ]
    for k in required_keys:
        assert k in data, f"Missing required key in run_info.json: {k}"

    assert data["stage"] == stage
    assert "commit_hash" in data["git"]
    assert "device" in data["hardware"]

    print("\nVerification SUCCESSFUL!")
    print(f"Generated Run ID: {data['run_id']}")
    print(f"Git Commit: {data['git'].get('commit_hash')[:7]}")
    print(f"Device: {data['hardware'].get('device')}")
    print(f"File location: {out_file.resolve()}\n")


if __name__ == "__main__":
    main()
