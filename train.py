#!/usr/bin/env python3
"""Root CLI entrypoint for SFT training (Plan Overview & Day 29).

Provides a top-level executable CLI for training the 7B base model with LoRA adapters,
supporting automatic Google Drive persistence and checkpoint resumption.

Usage:
    python train.py --help
    python train.py --model_name_or_path deepseek-ai/deepseek-math-7b-base
    python train.py --sample_mode --max_steps 5
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_sft import main  # noqa: E402

if __name__ == "__main__":
    main()
