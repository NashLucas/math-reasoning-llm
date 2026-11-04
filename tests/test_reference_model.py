"""Unit tests for Reference Model Management & KL Divergence Tracking (Phase 6 Day 42)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rewards.reference_model import (
    KLController,
    KLEstimatorType,
    KLSummary,
    ReferenceModelManager,
    compute_per_token_kl,
    compute_trajectory_kl,
    load_kl_config,
)


class TestReferenceModelAndKL(unittest.TestCase):
    """Test suite for KL estimators, completion masking, controller, and reference manager."""

    def test_k1_estimator(self):
        """Test standard log-ratio estimator k1: log(p) - log(q)."""
        p_lp = np.array([-1.0, -2.0, -0.5])
        r_lp = np.array([-1.5, -1.0, -0.5])

        k1 = compute_per_token_kl(p_lp, r_lp, estimator=KLEstimatorType.K1)
        expected = np.array([0.5, -1.0, 0.0])
        np.testing.assert_allclose(k1, expected, rtol=1e-5)

    def test_k2_estimator(self):
        """Test squared Taylor approximation estimator k2: 0.5 * (log(p) - log(q))^2."""
        p_lp = np.array([-1.0, -2.0, -0.5])
        r_lp = np.array([-1.5, -1.0, -0.5])

        k2 = compute_per_token_kl(p_lp, r_lp, estimator=KLEstimatorType.K2)
        expected = np.array([0.125, 0.5, 0.0])
        np.testing.assert_allclose(k2, expected, rtol=1e-5)

    def test_k3_estimator_non_negativity(self):
        """Test Schulman k3 estimator non-negativity across positive, negative, and zero deltas."""
        # deltas: -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0
        p_lp = np.array([-5.0, -2.0, -1.1, -1.0, -0.9, 0.0, 4.0])
        r_lp = np.array([0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

        k3 = compute_per_token_kl(p_lp, r_lp, estimator=KLEstimatorType.K3)

        # k3 must be strictly non-negative
        self.assertTrue(np.all(k3 >= 0.0))
        # At delta = 0 (index 3), k3 must be 0.0
        self.assertAlmostEqual(float(k3[3]), 0.0, places=6)

    def test_k3_identical_distribution(self):
        """Test that identical policy and reference distributions produce exactly 0 KL."""
        p_lp = np.array([-0.3, -1.5, -4.2])
        r_lp = np.array([-0.3, -1.5, -4.2])

        k3 = compute_per_token_kl(p_lp, r_lp, estimator=KLEstimatorType.K3)
        np.testing.assert_allclose(k3, np.zeros_like(p_lp), atol=1e-7)

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched logprob array shapes raise ValueError."""
        p_lp = np.array([1.0, 2.0])
        r_lp = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            compute_per_token_kl(p_lp, r_lp, estimator=KLEstimatorType.K3)

    def test_numerical_stability_extreme_deltas(self):
        """Test extreme delta values to verify clipping prevents NaN and Inf."""
        p_lp = np.array([100.0, -100.0])
        r_lp = np.array([0.0, 0.0])

        k3 = compute_per_token_kl(p_lp, r_lp, estimator=KLEstimatorType.K3)
        self.assertTrue(np.all(np.isfinite(k3)))
        self.assertTrue(np.all(k3 >= 0.0))

    def test_trajectory_kl_masking(self):
        """Test completion masking ignores prompt tokens and padding."""
        # 2 sequences, 4 tokens each: prompt is first 2 tokens (mask 0), completion is last 2 tokens (mask 1)
        per_token_kl = np.array([
            [10.0, 10.0, 0.2, 0.4],  # prompt has massive KL 10.0
            [10.0, 10.0, 0.3, 0.0],  # seq 2 has padding at token 4 (mask 0)
        ])
        mask = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 0],
        ])

        summary = compute_trajectory_kl(per_token_kl, completion_mask=mask, max_kl_threshold=0.5)

        self.assertEqual(summary.total_valid_tokens, 3)
        # Expected tokens: 0.2, 0.4, 0.3 -> mean = 0.3
        self.assertAlmostEqual(summary.mean_token_kl, 0.3, places=5)
        self.assertAlmostEqual(summary.max_token_kl, 0.4, places=5)
        self.assertAlmostEqual(summary.min_token_kl, 0.2, places=5)
        self.assertTrue(summary.is_safe)
        self.assertIsNone(summary.warning)

    def test_trajectory_empty_mask(self):
        """Test zero valid completion tokens returns safe zero-summary."""
        per_token_kl = np.array([[1.0, 2.0]])
        mask = np.array([[0, 0]])

        summary = compute_trajectory_kl(per_token_kl, completion_mask=mask)
        self.assertEqual(summary.total_valid_tokens, 0)
        self.assertEqual(summary.mean_token_kl, 0.0)
        self.assertTrue(summary.is_safe)
        self.assertIn("No valid completion tokens", summary.warning)

    def test_trajectory_drift_warning(self):
        """Test policy drift exceeding max_kl_threshold triggers warning."""
        per_token_kl = np.array([[0.8, 0.9]])
        mask = np.array([[1, 1]])

        summary = compute_trajectory_kl(per_token_kl, completion_mask=mask, max_kl_threshold=0.5)
        self.assertFalse(summary.is_safe)
        self.assertIsNotNone(summary.warning)
        self.assertIn("exceeds safe ceiling", summary.warning)

    def test_kl_controller_fixed_and_adaptive(self):
        """Test fixed beta invariance and adaptive beta dynamic response."""
        # Fixed controller: beta never changes
        c_fixed = KLController(init_beta=0.04, adaptive=False)
        dummy_summary = KLSummary(
            estimator="k3",
            mean_token_kl=0.25,
            max_token_kl=0.5,
            min_token_kl=0.1,
            std_token_kl=0.1,
            sequence_mean_kl=0.25,
            sequence_sum_kl=5.0,
            total_valid_tokens=20,
            num_sequences=1,
            is_safe=True,
        )
        beta_f = c_fixed.step(dummy_summary)
        self.assertEqual(beta_f, 0.04)

        # Adaptive controller: high KL increases beta
        c_adapt = KLController(init_beta=0.04, target_kl=0.10, adaptive=True, proportional_gain=0.2)
        high_kl_summary = KLSummary(
            estimator="k3",
            mean_token_kl=0.30,  # 3x target KL
            max_token_kl=0.6,
            min_token_kl=0.1,
            std_token_kl=0.1,
            sequence_mean_kl=0.30,
            sequence_sum_kl=6.0,
            total_valid_tokens=20,
            num_sequences=1,
            is_safe=True,
        )
        beta_a = c_adapt.step(high_kl_summary)
        self.assertGreater(beta_a, 0.04)

        summary_dict = c_adapt.get_summary()
        self.assertEqual(summary_dict["step_count"], 1)
        self.assertAlmostEqual(summary_dict["mean_kl"], 0.30, places=4)

    def test_mock_reference_model_manager(self):
        """Test ReferenceModelManager in mock mode."""
        mgr = ReferenceModelManager(mock_mode=True)
        input_ids = [[1, 2, 3, 4]]
        completion_mask = [[0, 0, 1, 1]]

        logprobs = mgr.get_ref_logprobs(input_ids, completion_mask=completion_mask)
        self.assertEqual(logprobs.shape, (1, 4))
        # First 2 tokens masked -> 0.0
        self.assertEqual(float(logprobs[0, 0]), 0.0)
        self.assertEqual(float(logprobs[0, 1]), 0.0)
        # Last 2 tokens unmasked -> negative logprob
        self.assertLess(float(logprobs[0, 2]), 0.0)
        self.assertLess(float(logprobs[0, 3]), 0.0)

    def test_load_kl_config(self):
        """Test configuration loader loads valid parameters."""
        cfg = load_kl_config()
        self.assertIn("kl_divergence", cfg)
        self.assertIn("reference_model", cfg)
        self.assertEqual(cfg["kl_divergence"]["estimator"], "k3")
        self.assertEqual(cfg["kl_divergence"]["beta"], 0.04)

    def test_cli_execution(self):
        """Test standalone CLI execution via subprocess."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                "scripts/verify_reference_model.py",
                "--output-dir",
                str(Path(tmpdir)),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"CLI failed:\nStdout: {proc.stdout}\nStderr: {proc.stderr}")

            md_path = Path(tmpdir) / "day42_reference_model_report.md"
            json_path = Path(tmpdir) / "day42_reference_model_report.json"
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertIn("benchmark", data)
                self.assertIn("masking_aggregation", data)
                self.assertIn("controller", data)


if __name__ == "__main__":
    unittest.main()
