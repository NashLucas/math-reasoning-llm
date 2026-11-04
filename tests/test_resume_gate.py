"""Unit tests for the Phase 0 resume gate verdict in scripts/test_resume_training.py."""

import unittest

from scripts.test_resume_training import MAX_STEPS, evaluate_run_log


def start(run_id, host, resumed_from, t):
    return {"event": "start", "run_id": run_id, "host": host, "resumed_from_step": resumed_from, "time": t}


def finish(run_id, host, resumed_from, first, final, t):
    return {
        "event": "finish", "run_id": run_id, "host": host, "resumed_from_step": resumed_from,
        "first_global_step": first, "final_global_step": final, "time": t,
    }


class TestResumeGateVerdict(unittest.TestCase):
    def test_real_disconnect_passes(self):
        """Run 1 cut off on one machine; run 2 on a new machine resumes from checkpoint-20."""
        records = [
            start("a", "vm-1", 0, 1.0),
            start("b", "vm-2", 20, 2.0),
            finish("b", "vm-2", 20, 21, MAX_STEPS, 3.0),
        ]
        self.assertTrue(evaluate_run_log(records, MAX_STEPS)["passed"])

    def test_no_interruption_fails(self):
        """The Colab result: every run finished, and re-runs crept past the last step."""
        records = [
            start("a", "vm-1", 0, 1.0), finish("a", "vm-1", 0, 1, 50, 2.0),
            start("b", "vm-2", 50, 3.0), finish("b", "vm-2", 50, 51, 51, 4.0),
            start("c", "vm-2", 51, 5.0), finish("c", "vm-2", 51, 52, 52, 6.0),
            start("d", "vm-2", 52, 7.0), finish("d", "vm-2", 52, 53, 53, 8.0),
        ]
        result = evaluate_run_log(records, 53)
        self.assertFalse(result["passed"])
        failed = [desc for desc, ok, _ in result["checks"] if not ok]
        self.assertIn("A run was cut off before finishing", failed)
        self.assertIn("Final checkpoint is the last step", failed)

    def test_restart_from_step_zero_fails(self):
        """If the next machine found no checkpoint (e.g. Drive had not synced), it restarts at 0."""
        records = [
            start("a", "vm-1", 0, 1.0),
            start("b", "vm-2", 0, 2.0),
            finish("b", "vm-2", 0, 1, MAX_STEPS, 3.0),
        ]
        result = evaluate_run_log(records, MAX_STEPS)
        self.assertFalse(result["passed"])
        failed = [desc for desc, ok, _ in result["checks"] if not ok]
        self.assertIn("Next run resumed from a saved checkpoint (not step 0)", failed)

    def test_same_machine_requires_flag(self):
        """A process restart on the same machine passes auto mode but not the Colab gate."""
        records = [
            start("a", "vm-1", 0, 1.0),
            start("b", "vm-1", 20, 2.0),
            finish("b", "vm-1", 20, 21, MAX_STEPS, 3.0),
        ]
        self.assertFalse(evaluate_run_log(records, MAX_STEPS, require_new_machine=True)["passed"])
        self.assertTrue(evaluate_run_log(records, MAX_STEPS, require_new_machine=False)["passed"])

    def test_old_log_format_fails(self):
        records = [{"event": "start", "resumed_from_step": 0, "time": 1.0}]
        result = evaluate_run_log(records, MAX_STEPS)
        self.assertFalse(result["passed"])
        self.assertIn("--fresh", result["checks"][0][2])

    def test_empty_log_fails(self):
        self.assertFalse(evaluate_run_log([], -1)["passed"])


if __name__ == "__main__":
    unittest.main()
