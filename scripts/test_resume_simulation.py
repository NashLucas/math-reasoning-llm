"""Day 4 Checkpoint Resume Simulation Script.

Simulates:
1. First training run: trains for 30 steps, saving checkpoints every 10 steps.
2. Simulated runtime crash / disconnect.
3. Second training run: discovers latest checkpoint (checkpoint-30) and resumes
   from step 30 up to target step 50.

Validates the Day 4 condition:
'Done when: the second run continues from the last checkpoint instead of step 0.'
"""

import json
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.checkpointing import (  # noqa: E402
    get_latest_checkpoint,
    parse_checkpoint_step,
    should_resume,
)


def mock_training_run(
    checkpoint_dir: Path,
    target_steps: int,
    save_every: int = 10,
    overwrite: bool = False,
) -> int:
    """Mock training loop demonstrating checkpointing and automatic resumption."""
    resume, latest_ckpt = should_resume(checkpoint_dir, overwrite=overwrite)

    if resume and latest_ckpt:
        start_step = parse_checkpoint_step(latest_ckpt)
        print(f"--> Resuming training from {latest_ckpt.name} at step {start_step}")
    else:
        start_step = 0
        print("--> Starting fresh training run from step 0")

    current_step = start_step
    for step in range(start_step + 1, target_steps + 1):
        current_step = step
        # Save checkpoint periodically
        if current_step % save_every == 0:
            ckpt_path = checkpoint_dir / f"checkpoint-{current_step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            # Write state metadata
            with open(ckpt_path / "trainer_state.json", "w", encoding="utf-8") as f:
                json.dump({"global_step": current_step, "loss": 1.0 / current_step}, f)
            print(f"    Saved checkpoint at step {current_step}: {ckpt_path.name}")

    return start_step


def main() -> None:
    print("=" * 65)
    print("DAY 4 VERIFICATION: CHECKPOINT RESUME SIMULATION")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir) / "checkpoints" / "sft"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        print("\n--- PHASE 1: RUN 1 (Interrupted at Step 30) ---")
        start_1 = mock_training_run(ckpt_dir, target_steps=30, save_every=10)
        assert start_1 == 0, "Run 1 did not start at step 0"

        latest_after_run1 = get_latest_checkpoint(ckpt_dir)
        print(f"\n[Simulated Disconnect] Checkpoint on storage: {latest_after_run1.name}")
        assert latest_after_run1.name == "checkpoint-30", "Run 1 did not reach checkpoint-30"

        print("\n--- PHASE 2: RUN 2 (New Session / Resumed Run to Step 50) ---")
        start_2 = mock_training_run(ckpt_dir, target_steps=50, save_every=10)

        # Verification check
        assert start_2 == 30, f"FAILED: Run 2 started at step {start_2} instead of step 30!"
        latest_after_run2 = get_latest_checkpoint(ckpt_dir)
        assert latest_after_run2.name == "checkpoint-50", "Run 2 did not complete to checkpoint-50"

        print("\n" + "=" * 65)
        print("VERIFICATION PASSED!")
        print(f"Run 1 started at: step {start_1}")
        print(f"Run 2 resumed at: step {start_2} (continued from last checkpoint)")
        print(f"Final checkpoint: {latest_after_run2.name}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
