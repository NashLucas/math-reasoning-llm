#!/usr/bin/env python3
"""Phase 7 Day 50: Production Serving Engine CLI.

Provides command-line interface to launch the serving engine and interactive
web application for the post-GRPO mathematical reasoning policy.

Usage:
    python scripts/serve_model.py --port 7860
    python scripts/serve_model.py --smoke_test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import build_gradio_app, run_smoke_test  # noqa: E402
from src.run_info import save_run_info  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("serve_model")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 7 Day 50: Production Serving Engine CLI."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface.")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860).")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio share link.")
    parser.add_argument("--use_mock", action="store_true", default=True, help="Use deterministic mock backend.")
    parser.add_argument("--adapter_path", type=str, default="outputs/grpo/adapter_model", help="Path to policy adapter.")
    parser.add_argument("--smoke_test", action="store_true", help="Execute single query verification and exit 0.")
    return parser.parse_args()


def main() -> int:
    """Main CLI entry point."""
    args = parse_args()

    logger.info("========================================================================")
    logger.info("PHASE 7 DAY 50: PRODUCTION SERVING ENGINE & DEMO")
    logger.info("========================================================================")
    logger.info("Host:         %s", args.host)
    logger.info("Port:         %d", args.port)
    logger.info("Adapter:      %s", args.adapter_path)
    logger.info("Mock Mode:    %s", args.use_mock)
    logger.info("Smoke Test:   %s", args.smoke_test)
    logger.info("========================================================================")

    # Record run provenance
    save_run_info(
        stage="serving",
        output_dir="outputs/production",
        config={
            "host": args.host,
            "port": args.port,
            "adapter_path": args.adapter_path,
            "use_mock": args.use_mock,
            "smoke_test": args.smoke_test,
        },
        description="Phase 7 Day 50 Serving Initialization",
    )

    if args.smoke_test:
        return run_smoke_test(use_mock=args.use_mock)

    demo = build_gradio_app(use_mock=args.use_mock, adapter_path=args.adapter_path)
    if demo is not None:
        logger.info(f"Serving application live on {args.host}:{args.port}")
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)
        return 0
    else:
        logger.info("Headless environment: running serving smoke test verification.")
        return run_smoke_test(use_mock=args.use_mock)


if __name__ == "__main__":
    sys.exit(main())
