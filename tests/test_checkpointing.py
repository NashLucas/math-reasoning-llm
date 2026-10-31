"""Unit tests for src/training/checkpointing.py."""

import json
import tempfile
import unittest
from pathlib import Path

from src.training.checkpointing import (
    cleanup_old_checkpoints,
    get_checkpoints,
    get_latest_checkpoint,
    is_complete_checkpoint,
    parse_checkpoint_step,
    should_resume,
)


def make_checkpoint(base: Path, step: int, complete: bool = True) -> Path:
    """Create checkpoint-<step>; a complete one has the trainer_state.json completion marker."""
    ckpt = base / f"checkpoint-{step}"
    ckpt.mkdir()
    if complete:
        (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return ckpt


class TestCheckpointing(unittest.TestCase):
    def test_parse_checkpoint_step(self):
        self.assertEqual(parse_checkpoint_step("checkpoint-50"), 50)
        self.assertEqual(parse_checkpoint_step("checkpoint-1250"), 1250)
        self.assertEqual(parse_checkpoint_step("checkpoint-0"), 0)
        self.assertEqual(parse_checkpoint_step("model_weights"), -1)
        self.assertEqual(parse_checkpoint_step("checkpoint_invalid"), -1)

    def test_get_checkpoints_sorting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            for step in [100, 20, 500, 5]:
                make_checkpoint(base, step)

            checkpoints = get_checkpoints(base)
            steps = [parse_checkpoint_step(c) for c in checkpoints]
            self.assertEqual(steps, [5, 20, 100, 500])

    def test_get_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            self.assertIsNone(get_latest_checkpoint(base))

            for step in [50, 150, 75]:
                make_checkpoint(base, step)

            latest = get_latest_checkpoint(base)
            self.assertIsNotNone(latest)
            self.assertEqual(latest.name, "checkpoint-150")

    def test_latest_skips_incomplete_checkpoint(self):
        """A disconnect during save leaves checkpoint-40 without its marker; resume from 30."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            for step in [10, 20, 30]:
                make_checkpoint(base, step)
            partial = make_checkpoint(base, 40, complete=False)

            self.assertFalse(is_complete_checkpoint(partial))
            self.assertEqual(get_latest_checkpoint(base).name, "checkpoint-30")
            self.assertEqual(len(get_checkpoints(base, complete_only=False)), 4)

    def test_non_checkpoint_folders_are_ignored(self):
        """Copied adapters or best-model folders are never treated as checkpoints."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            best = base / "best_adapter"
            best.mkdir()
            (best / "adapter_config.json").write_text("{}")
            (best / "trainer_state.json").write_text("{}")
            make_checkpoint(base, 10)

            self.assertEqual([c.name for c in get_checkpoints(base)], ["checkpoint-10"])
            self.assertEqual(get_latest_checkpoint(base).name, "checkpoint-10")

    def test_should_resume(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            resume, ckpt = should_resume(base, overwrite=False)
            self.assertFalse(resume)
            self.assertIsNone(ckpt)

            # An incomplete checkpoint alone is not resumable
            make_checkpoint(base, 10, complete=False)
            resume, ckpt = should_resume(base, overwrite=False)
            self.assertFalse(resume)
            self.assertIsNone(ckpt)

            make_checkpoint(base, 30)
            resume, ckpt = should_resume(base, overwrite=False)
            self.assertTrue(resume)
            self.assertEqual(ckpt.name, "checkpoint-30")

            # With overwrite=True, should ignore existing checkpoint
            resume, ckpt = should_resume(base, overwrite=True)
            self.assertFalse(resume)
            self.assertIsNone(ckpt)

    def test_cleanup_old_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            for step in [10, 20, 30, 40, 50]:
                make_checkpoint(base, step)

            # Keep only last 2 checkpoints (40 and 50)
            deleted = cleanup_old_checkpoints(base, keep_last_n=2)
            self.assertEqual(len(deleted), 3)

            remaining = [c.name for c in get_checkpoints(base)]
            self.assertEqual(remaining, ["checkpoint-40", "checkpoint-50"])

    def test_cleanup_never_deletes_non_checkpoint_folders(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            best = base / "best_adapter"
            best.mkdir()
            (best / "adapter_config.json").write_text("{}")
            for step in [10, 20, 30, 40]:
                make_checkpoint(base, step)

            deleted = cleanup_old_checkpoints(base, keep_last_n=3)
            self.assertEqual([d.name for d in deleted], ["checkpoint-10"])
            self.assertTrue(best.exists())

    def test_cleanup_keeps_complete_checkpoints_and_in_progress_save(self):
        """Incomplete checkpoints don't count toward keep_last_n; a newer one may still be saving."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            make_checkpoint(base, 10)
            make_checkpoint(base, 15, complete=False)  # stale partial save
            make_checkpoint(base, 20)
            make_checkpoint(base, 30)
            make_checkpoint(base, 40, complete=False)  # save in progress

            deleted = cleanup_old_checkpoints(base, keep_last_n=2)
            self.assertEqual(sorted(d.name for d in deleted), ["checkpoint-10", "checkpoint-15"])
            remaining = [c.name for c in get_checkpoints(base, complete_only=False)]
            self.assertEqual(remaining, ["checkpoint-20", "checkpoint-30", "checkpoint-40"])

    def test_cleanup_rejects_keep_zero(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                cleanup_old_checkpoints(tmp_dir, keep_last_n=0)


if __name__ == "__main__":
    unittest.main()
