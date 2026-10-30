"""Checkpoint detection, sorting, and resume utilities.

Enables automatic training resumption from Google Drive following Colab session
disconnects or preemptions, and manages storage quotas by pruning old checkpoints.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def parse_checkpoint_step(path: Union[str, Path]) -> int:
    """Extract the integer step count from a checkpoint directory name."""
    name = Path(path).name
    match = CHECKPOINT_PATTERN.match(name)
    if match:
        return int(match.group(1))
    return -1


def get_checkpoints(checkpoint_dir: Union[str, Path]) -> List[Path]:
    """Find all valid checkpoint directories sorted chronologically / by step.

    Sorted in ascending order (latest checkpoint is the last element).
    """
    path = Path(checkpoint_dir)
    if not path.exists() or not path.is_dir():
        return []

    checkpoints: List[Path] = []
    for item in path.iterdir():
        if not item.is_dir():
            continue
        # Standard Hugging Face / TRL pattern (checkpoint-<step>)
        if CHECKPOINT_PATTERN.match(item.name):
            checkpoints.append(item)
        # Or directory containing checkpoint artifacts
        elif (item / "trainer_state.json").exists() or (item / "adapter_config.json").exists():
            checkpoints.append(item)

    # Sort primarily by parsed step if available, else by directory modification time
    def sort_key(p: Path) -> Tuple[int, float]:
        step = parse_checkpoint_step(p)
        mtime = p.stat().st_mtime
        return (step if step >= 0 else 0, mtime)

    checkpoints.sort(key=sort_key)
    return checkpoints


def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Return the most recent checkpoint in the given directory, or None."""
    checkpoints = get_checkpoints(checkpoint_dir)
    if not checkpoints:
        return None
    return checkpoints[-1]


def find_stage_checkpoint(
    stage: str, profile_name: Optional[str] = None
) -> Optional[Path]:
    """Locate the latest checkpoint for a specific pipeline stage."""
    try:
        from src.config import ConfigManager
        mgr = ConfigManager()
        stage_paths = mgr.get_stage_paths(stage, profile_name=profile_name)
        ckpt_dir = stage_paths.get("checkpoints")
        if ckpt_dir and ckpt_dir.exists():
            return get_latest_checkpoint(ckpt_dir)
    except Exception as e:
        logger.warning(f"Error resolving stage checkpoint for '{stage}': {e}")
    return None


def should_resume(
    checkpoint_dir: Union[str, Path], overwrite: bool = False
) -> Tuple[bool, Optional[Path]]:
    """Determine whether a training run should resume from a previous checkpoint.

    Args:
        checkpoint_dir: Directory where stage checkpoints are saved.
        overwrite: If True, explicitly discard previous checkpoints and restart from step 0.

    Returns:
        (resume_bool, checkpoint_path_or_none)
    """
    if overwrite:
        logger.info(f"Overwrite flag is True: starting fresh in {checkpoint_dir}")
        return False, None

    latest = get_latest_checkpoint(checkpoint_dir)
    if latest is not None:
        logger.info(f"Found existing checkpoint to resume from: {latest}")
        return True, latest

    logger.info(f"No existing checkpoints found in {checkpoint_dir}. Starting new run.")
    return False, None


def cleanup_old_checkpoints(
    checkpoint_dir: Union[str, Path], keep_last_n: int = 3
) -> List[Path]:
    """Prune older checkpoints to conserve Google Drive storage.

    Keeps the most recent `keep_last_n` checkpoints.
    Returns the list of deleted checkpoint paths.
    """
    checkpoints = get_checkpoints(checkpoint_dir)
    if len(checkpoints) <= keep_last_n:
        return []

    to_delete = checkpoints[:-keep_last_n]
    deleted: List[Path] = []
    for ckpt in to_delete:
        try:
            shutil.rmtree(ckpt)
            deleted.append(ckpt)
            logger.info(f"Pruned older checkpoint to save space: {ckpt}")
        except Exception as e:
            logger.error(f"Failed to delete old checkpoint {ckpt}: {e}")

    return deleted
