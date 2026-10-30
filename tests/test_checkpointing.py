"""Unit tests for src/training/checkpointing.py."""

from pathlib import Path
import tempfile
import unittest

from src.training.checkpointing import (
    parse_checkpoint_step,
    get_checkpoints,
    get_latest_checkpoint,
    should_resume,
    cleanup_old_checkpoints,
)


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
            (base / "checkpoint-100").mkdir()
            (base / "checkpoint-20").mkdir()
            (base / "checkpoint-500").mkdir()
            (base / "checkpoint-5").mkdir()

            checkpoints = get_checkpoints(base)
            steps = [parse_checkpoint_step(c) for c in checkpoints]
            self.assertEqual(steps, [5, 20, 100, 500])

    def test_get_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            self.assertIsNone(get_latest_checkpoint(base))

            (base / "checkpoint-50").mkdir()
            (base / "checkpoint-150").mkdir()
            (base / "checkpoint-75").mkdir()

            latest = get_latest_checkpoint(base)
            self.assertIsNotNone(latest)
            self.assertEqual(latest.name, "checkpoint-150")

    def test_should_resume(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            resume, ckpt = should_resume(base, overwrite=False)
            self.assertFalse(resume)
            self.assertIsNone(ckpt)

            (base / "checkpoint-30").mkdir()
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
                (base / f"checkpoint-{step}").mkdir()

            # Keep only last 2 checkpoints (40 and 50)
            deleted = cleanup_old_checkpoints(base, keep_last_n=2)
            self.assertEqual(len(deleted), 3)

            remaining = [c.name for c in get_checkpoints(base)]
            self.assertEqual(remaining, ["checkpoint-40", "checkpoint-50"])


if __name__ == "__main__":
    unittest.main()
