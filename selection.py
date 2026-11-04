"""Root entrypoint and module re-export for targeted example selection (Phase 5)."""

import sys

from src.data.selection import (
    TargetedExample,
    TargetedSelectionReport,
    TargetedSelector,
    load_curated_traces_index,
    load_forbidden_ids,
    load_training_pool,
)

__all__ = [
    "TargetedExample",
    "TargetedSelectionReport",
    "TargetedSelector",
    "load_curated_traces_index",
    "load_forbidden_ids",
    "load_training_pool",
]

if __name__ == "__main__":
    from scripts.run_day34_selection import main

    sys.exit(main())
