"""Checkpoint detection, sorting, and resume utilities.

Enables automatic training resumption from Google Drive following Colab session
disconnects or preemptions, and manages storage quotas by pruning old checkpoints.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")

# Hugging Face Trainer writes trainer_state.json after the weights, optimizer and RNG state,
# so its presence marks a checkpoint whose save finished.
COMPLETION_MARKER = "trainer_state.json"


def parse_checkpoint_step(path: Union[str, Path]) -> int:
    """Extract the integer step count from a checkpoint directory name."""
    name = Path(path).name
    match = CHECKPOINT_PATTERN.match(name)
    if match:
        return int(match.group(1))
    return -1


def is_complete_checkpoint(path: Union[str, Path]) -> bool:
    """True if the checkpoint save finished (a disconnect mid-save leaves no completion marker)."""
    return (Path(path) / COMPLETION_MARKER).is_file()


def get_checkpoints(checkpoint_dir: Union[str, Path], complete_only: bool = True) -> List[Path]:
    """Find `checkpoint-<step>` directories sorted by step (latest checkpoint is the last element).

    Only directories named `checkpoint-<step>` count. Other folders (copied adapters, best
    models, logs) are never treated as checkpoints, so they are never resumed from or pruned.

    Args:
        checkpoint_dir: Stage checkpoint directory.
        complete_only: If True, skip checkpoints whose save did not finish.
    """
    path = Path(checkpoint_dir)
    if not path.exists() or not path.is_dir():
        return []

    checkpoints: List[Path] = []
    for item in path.iterdir():
        if not item.is_dir() or not CHECKPOINT_PATTERN.match(item.name):
            continue
        if complete_only and not is_complete_checkpoint(item):
            continue
        checkpoints.append(item)

    checkpoints.sort(key=parse_checkpoint_step)
    return checkpoints


def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Return the most recent complete checkpoint in the given directory, or None.

    Incomplete checkpoints newer than it (e.g. from a disconnect during save) are skipped
    with a warning.
    """
    complete = get_checkpoints(checkpoint_dir, complete_only=True)
    latest = complete[-1] if complete else None

    latest_step = parse_checkpoint_step(latest) if latest else -1
    for ckpt in get_checkpoints(checkpoint_dir, complete_only=False):
        if parse_checkpoint_step(ckpt) > latest_step and not is_complete_checkpoint(ckpt):
            logger.warning(f"Skipping incomplete checkpoint (no {COMPLETION_MARKER}): {ckpt}")

    return latest


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

    Keeps the most recent `keep_last_n` complete checkpoints and deletes `checkpoint-<step>`
    directories (complete or not) older than the oldest one kept. Incomplete checkpoints newer
    than that may still be saving and are left alone; non-checkpoint folders are never touched.
    Returns the list of deleted checkpoint paths.
    """
    if keep_last_n < 1:
        raise ValueError("keep_last_n must be at least 1 so a resumable checkpoint always remains")

    complete = get_checkpoints(checkpoint_dir, complete_only=True)
    if len(complete) <= keep_last_n:
        return []

    oldest_kept_step = parse_checkpoint_step(complete[-keep_last_n])
    to_delete = [
        ckpt
        for ckpt in get_checkpoints(checkpoint_dir, complete_only=False)
        if parse_checkpoint_step(ckpt) < oldest_kept_step
    ]
    deleted: List[Path] = []
    for ckpt in to_delete:
        try:
            shutil.rmtree(ckpt)
            deleted.append(ckpt)
            logger.info(f"Pruned older checkpoint to save space: {ckpt}")
        except Exception as e:
            logger.error(f"Failed to delete old checkpoint {ckpt}: {e}")

    return deleted
