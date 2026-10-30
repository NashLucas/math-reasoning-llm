"""Unit tests for run_info.py and set_seed."""

import json
from pathlib import Path
import tempfile
import unittest

from src.run_info import (
    set_seed,
    get_git_info,
    get_gpu_info,
    get_library_versions,
    collect_run_info,
    save_run_info,
)


class TestRunInfo(unittest.TestCase):
    def test_set_seed(self):
        s1 = set_seed(1234)
        self.assertEqual(s1, 1234)

    def test_get_git_info(self):
        git_info = get_git_info()
        self.assertIn("commit_hash", git_info)
        self.assertIn("is_dirty", git_info)
        self.assertIn("branch", git_info)

    def test_get_gpu_info(self):
        gpu_info = get_gpu_info()
        self.assertIn("device", gpu_info)
        self.assertIn("cuda_available", gpu_info)

    def test_get_library_versions(self):
        versions = get_library_versions()
        self.assertIn("yaml", versions)
        self.assertIn("torch", versions)
        self.assertIn("transformers", versions)

    def test_collect_run_info(self):
        info = collect_run_info(stage="test_stage", description="unit_test")
        self.assertEqual(info["stage"], "test_stage")
        self.assertIn("run_id", info)
        self.assertIn("timestamp_utc", info)
        self.assertIn("git", info)
        self.assertIn("hardware", info)
        self.assertIn("library_versions", info)

    def test_save_run_info_to_custom_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = save_run_info(
                stage="unit_test_stage",
                output_dir=tmp_dir,
                config={"lr": 1e-4, "batch_size": 32},
                description="test run",
            )
            self.assertTrue(out_path.exists())
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["stage"], "unit_test_stage")
            self.assertEqual(data["config_snapshot"]["lr"], 1e-4)


if __name__ == "__main__":
    unittest.main()
