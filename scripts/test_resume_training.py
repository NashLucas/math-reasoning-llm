"""Phase 0 gate: a forced disconnect resumes real training from its last checkpoint.

Trains a tiny GPT-2 (built from a config, no download) with the Hugging Face Trainer for
50 steps, saving every 10, and resumes through src/training/checkpointing.py. Unlike
scripts/test_resume_simulation.py, this runs a real optimizer, scheduler and RNG state.

Modes:
  --mode auto   Run 1 in a subprocess, hard-kill it once a complete checkpoint exists
                (the same abrupt stop as a runtime disconnect), then run 2 in a fresh
                process and verify it continues from that checkpoint, not step 0.
  --mode run    One training run that resumes if a complete checkpoint exists. Use it for
                the manual Colab test: start it, restart the runtime mid-run
                (Runtime -> Restart session), then run it again.
  --mode verify Check the finished run: it reached the final step and resumed exactly once.

Checkpoints go to <profile checkpoints_dir>/resume_test (Google Drive under colab).

Usage:
  python scripts/test_resume_training.py --mode auto
  python scripts/test_resume_training.py --mode run      # Colab: restart mid-run, run again
  python scripts/test_resume_training.py --mode verify
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager  # noqa: E402
from src.training.checkpointing import (  # noqa: E402
    get_latest_checkpoint,
    parse_checkpoint_step,
    should_resume,
)

MAX_STEPS = 50
SAVE_STEPS = 10
RUN_LOG = "resume_runs.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0 gate: verify training resumes from its checkpoint after a forced stop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["auto", "run", "verify"], default="auto")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(ConfigManager().get_profile_dir("checkpoints_dir") / "resume_test"),
        help="Where checkpoints are written (profile checkpoints dir; Google Drive under colab).",
    )
    parser.add_argument("--step_delay", type=float, default=0.3, help="Seconds slept per step, so a run can be interrupted.")
    parser.add_argument("--fresh", action="store_true", help="Delete existing checkpoints before starting.")
    return parser.parse_args()


def append_run_record(checkpoint_dir: Path, record: Dict[str, Any]) -> None:
    with open(checkpoint_dir / RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_run_records(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    path = checkpoint_dir / RUN_LOG
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_once(checkpoint_dir: Path, step_delay: float) -> int:
    """One training run of a tiny GPT-2; resumes from the latest complete checkpoint if present."""
    import torch
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    from src.run_info import set_seed

    set_seed(42)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config)

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 512

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            g = torch.Generator().manual_seed(idx)
            ids = torch.randint(0, 64, (16,), generator=g)
            return {"input_ids": ids, "labels": ids.clone()}

    resume, latest = should_resume(checkpoint_dir)
    resume_path: Optional[str] = str(latest) if resume and latest else None
    first_step: Dict[str, int] = {}

    class StepRecorder(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            if "global_step" not in first_step:
                first_step["global_step"] = state.global_step
                print(f"--> First step of this run finished at global_step={state.global_step}", flush=True)
            time.sleep(step_delay)
            return control

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=None,
        per_device_train_batch_size=8,
        learning_rate=1e-3,
        logging_steps=10,
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
        seed=42,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=TinyDataset(), callbacks=[StepRecorder()])

    resumed_from_step = parse_checkpoint_step(resume_path) if resume_path else 0
    print(f"--> {'Resuming from ' + Path(resume_path).name if resume_path else 'Starting fresh from step 0'}", flush=True)
    append_run_record(checkpoint_dir, {"event": "start", "resumed_from_step": resumed_from_step, "time": time.time()})

    trainer.train(resume_from_checkpoint=resume_path)

    append_run_record(checkpoint_dir, {
        "event": "finish",
        "resumed_from_step": resumed_from_step,
        "first_global_step": first_step.get("global_step"),
        "final_global_step": trainer.state.global_step,
        "time": time.time(),
    })
    print(f"--> Run finished at global_step={trainer.state.global_step}", flush=True)
    return trainer.state.global_step


def verify(checkpoint_dir: Path) -> bool:
    """The finished run must have resumed from a checkpoint > 0 and continued from it to MAX_STEPS."""
    records = load_run_records(checkpoint_dir)
    starts = [r for r in records if r["event"] == "start"]
    finishes = [r for r in records if r["event"] == "finish"]
    latest = get_latest_checkpoint(checkpoint_dir)

    print("=" * 65)
    print(f"Runs started: {len(starts)} | runs finished: {len(finishes)} | latest checkpoint: {latest.name if latest else None}")
    if not finishes:
        print("FAILED: no run finished.")
        return False

    final = finishes[-1]
    ok = (
        len(starts) >= 2
        and starts[0]["resumed_from_step"] == 0
        and final["resumed_from_step"] > 0
        and final["first_global_step"] == final["resumed_from_step"] + 1
        and final["final_global_step"] == MAX_STEPS
        and latest is not None
        and parse_checkpoint_step(latest) == MAX_STEPS
    )
    print(f"First run started at step 0: {starts[0]['resumed_from_step'] == 0}")
    print(f"Final run resumed from step {final['resumed_from_step']}, "
          f"first step completed was {final['first_global_step']}, finished at {final['final_global_step']}")
    print("PHASE 0 RESUME GATE: " + ("PASSED" if ok else "FAILED"))
    print("=" * 65)
    return ok


def run_auto(checkpoint_dir: Path, step_delay: float) -> bool:
    """Kill run 1 abruptly after a complete checkpoint exists, then resume in a new process."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", "run",
           "--checkpoint_dir", str(checkpoint_dir), "--step_delay", str(step_delay)]

    print("--- RUN 1: training until a complete checkpoint exists, then hard kill ---", flush=True)
    proc = subprocess.Popen(cmd)
    deadline = time.time() + 600
    while time.time() < deadline:
        latest = get_latest_checkpoint(checkpoint_dir)
        if latest is not None and parse_checkpoint_step(latest) >= 2 * SAVE_STEPS:
            break
        if proc.poll() is not None:
            print("FAILED: run 1 exited before it could be interrupted.")
            return False
        time.sleep(0.2)
    else:
        proc.kill()
        print("FAILED: no complete checkpoint appeared within 10 minutes.")
        return False

    proc.kill()  # abrupt stop, like a runtime disconnect: no cleanup, no final save
    proc.wait()
    print(f"[Forced stop] latest complete checkpoint: {get_latest_checkpoint(checkpoint_dir).name}", flush=True)

    print("--- RUN 2: new process, should resume ---", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: run 2 exited with code {result.returncode}.")
        return False
    return verify(checkpoint_dir)


def main() -> int:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)

    if args.fresh or args.mode == "auto":
        # Only ever delete a folder this script created, never a real stage checkpoint folder
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not (checkpoint_dir / RUN_LOG).exists():
            print(f"Refusing to delete {checkpoint_dir}: it was not created by this test.")
            return 1
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    if args.mode == "auto":
        return 0 if run_auto(checkpoint_dir, args.step_delay) else 1
    if args.mode == "run":
        train_once(checkpoint_dir, args.step_delay)
        return 0
    return 0 if verify(checkpoint_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
