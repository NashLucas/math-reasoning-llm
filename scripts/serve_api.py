#!/usr/bin/env python3
"""Phase 8 Day 51: Production REST API Server Launcher CLI.

Launches the high-throughput FastAPI service for mathematical reasoning:
- Single query inference (/v1/predict)
- Concurrent batch inference (/v1/predict/batch)
- Health and metrics (/v1/health, /v1/metrics)
- Interactive OpenAPI Swagger documentation at /docs

Usage:
    python scripts/serve_api.py --port 8000
    python scripts/serve_api.py --host 127.0.0.1 --port 8080 --use_mock
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api import app, set_engine  # noqa: E402
from src.serving.engine import ReasoningServingEngine, ServingConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("serve_api")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch production FastAPI serving REST service for mathematical reasoning."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Network host interface")
    parser.add_argument("--port", type=int, default=8000, help="Network port (default: 8000)")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="outputs/grpo/adapter_model",
        help="Path to post-GRPO policy adapter",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        default=True,
        help="Use deterministic mock backend for local testing",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/production",
        help="Directory to save request traffic JSONL logs",
    )
    return parser.parse_args()


def main() -> int:
    import uvicorn

    args = parse_args()

    cfg = ServingConfig(
        use_mock=args.use_mock,
        adapter_path=args.adapter_path,
        log_dir=args.log_dir,
        log_requests=True,
    )
    engine = ReasoningServingEngine(config=cfg)
    set_engine(engine)

    logger.info("========================================================================")
    logger.info("PHASE 8 DAY 51: PRODUCTION REST API SERVER")
    logger.info("========================================================================")
    logger.info("Binding Host:      %s", args.host)
    logger.info("Port:              %d", args.port)
    logger.info("Adapter Path:      %s", args.adapter_path)
    logger.info("Backend Mode:      %s", "Deterministic Mock" if args.use_mock else "CUDA GPU")
    logger.info("API Docs:          http://%s:%d/docs", args.host, args.port)
    logger.info("========================================================================")

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
