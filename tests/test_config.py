"""Unit tests for the configuration system and profile path resolver.

Compatible with both pytest and python's standard unittest.
"""

import unittest

from src.config import (
    CONFIGS_DIR,
    PROJECT_ROOT,
    ConfigManager,
    merge_configs,
)


class TestConfig(unittest.TestCase):
    def test_project_root_and_configs_dir(self):
        self.assertTrue(PROJECT_ROOT.exists())
        self.assertTrue(CONFIGS_DIR.exists())
        self.assertTrue((CONFIGS_DIR / "paths.yaml").exists())
        self.assertTrue((CONFIGS_DIR / "base_config.yaml").exists())
        self.assertTrue((CONFIGS_DIR / "pipeline.yaml").exists())

    def test_merge_configs(self):
        c1 = {"a": 1, "b": {"x": 10, "y": 20}}
        c2 = {"b": {"y": 99, "z": 30}, "c": 3}
        merged = merge_configs(c1, c2)
        self.assertEqual(merged["a"], 1)
        self.assertEqual(merged["c"], 3)
        self.assertEqual(merged["b"], {"x": 10, "y": 99, "z": 30})

    def test_base_config_contents(self):
        cfg = ConfigManager()
        base = cfg.base_config
        self.assertEqual(base["project_name"], "math-reasoning-llm")
        self.assertIn("models", base)
        self.assertEqual(base["models"]["train_base"], "deepseek-ai/deepseek-math-7b-base")
        self.assertIn("dev_model", base["models"])
        self.assertIn("answer_contract", base)
        self.assertEqual(base["answer_contract"]["prefix"], "####")

    def test_pipeline_stages_order(self):
        cfg = ConfigManager()
        stages = cfg.get_all_stages()
        expected_stages = [
            "baseline",
            "reasoning_data",
            "sft",
            "error_analysis",
            "sft_targeted",
            "grpo",
            "final_eval",
            "serve",
        ]
        self.assertEqual(stages, expected_stages)

    def test_profiles_exist(self):
        cfg = ConfigManager()
        local_prof = cfg.get_profile("local")
        colab_prof = cfg.get_profile("colab")

        self.assertEqual(local_prof["root"], ".")
        self.assertEqual(colab_prof["root"], "/content/drive/MyDrive/math-reasoning-llm")
        # Verify HF cache is kept off Drive on Colab
        self.assertNotIn("drive", colab_prof["hf_cache_dir"].lower())

    def test_stage_paths_resolution_local(self):
        cfg = ConfigManager()
        sft_paths = cfg.get_stage_paths("sft", profile_name="local")

        self.assertIn("checkpoints", sft_paths)
        self.assertIn("outputs", sft_paths)
        self.assertIn("logs", sft_paths)

        # Verify that paths resolve within the project directory for local profile
        self.assertTrue(
            PROJECT_ROOT in sft_paths["checkpoints"].parents
            or sft_paths["checkpoints"] == (PROJECT_ROOT / "checkpoints/sft").resolve()
        )
        self.assertTrue(
            PROJECT_ROOT in sft_paths["outputs"].parents
            or sft_paths["outputs"] == (PROJECT_ROOT / "outputs/sft").resolve()
        )

    def test_stage_paths_resolution_colab(self):
        cfg = ConfigManager()
        sft_paths = cfg.get_stage_paths("sft", profile_name="colab")

        self.assertIn("checkpoints", sft_paths)
        self.assertIn("outputs", sft_paths)
        self.assertIn("logs", sft_paths)

        str_path = str(sft_paths["checkpoints"]).replace("\\", "/")
        self.assertIn("drive/MyDrive/math-reasoning-llm", str_path)

    def test_all_stages_have_resolved_paths(self):
        cfg = ConfigManager()
        for profile in ["local", "colab"]:
            all_stage_paths = cfg.get_all_stage_paths(profile_name=profile)
            self.assertEqual(len(all_stage_paths), 8)
            for stage_name, paths in all_stage_paths.items():
                self.assertGreaterEqual(
                    len(paths), 2, f"Stage {stage_name} has insufficient paths in {profile}"
                )
                for p in paths.values():
                    self.assertIsNotNone(p)
                    self.assertGreater(len(str(p)), 0)


if __name__ == "__main__":
    unittest.main()
