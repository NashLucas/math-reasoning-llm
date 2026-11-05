#!/usr/bin/env python3
"""Phase 8 Day 52: Production Traffic Observability & Drift Monitoring CLI.

Parses logs/production/requests.jsonl and computes rolling statistics:
- Answer Contract compliance rate & drift alerts.
- Latency percentiles (mean, p50, p95, p99).
- Token length distribution shifts.
- Decoding regime utilization.

Emits:
- reports/day52_traffic_monitoring_report.md
- reports/day52_traffic_monitoring.json

Usage:
    python scripts/run_traffic_monitor.py
    python scripts/run_traffic_monitor.py --log_file logs/production/requests.jsonl
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

from src.serving.monitor import run_traffic_monitoring  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("traffic_monitor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit production request logs and detect Answer Contract and latency drift."
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=PROJECT_ROOT / "logs" / "production" / "requests.jsonl",
        help="Path to JSONL production traffic log file",
    )
    parser.add_argument(
        "--output_report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "day52_traffic_monitoring_report.md",
        help="Path where markdown report will be written",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "day52_traffic_monitoring.json",
        help="Path where JSON summary will be written",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("========================================================================")
    logger.info("PHASE 8 DAY 52: PRODUCTION TRAFFIC MONITORING & DRIFT DETECTION")
    logger.info("========================================================================")
    logger.info("Traffic Log Path:    %s", args.log_file)
    logger.info("Target Markdown:     %s", args.output_report)
    logger.info("Target JSON Summary: %s", args.output_json)
    logger.info("========================================================================")

    try:
        metrics = run_traffic_monitoring(
            log_file=args.log_file,
            output_report_path=args.output_report,
            output_json_path=args.output_json,
        )
    except Exception as err:
        logger.error("Traffic monitoring failed: %s", err, exc_info=True)
        return 1

    if not args.quiet:
        print("\n" + "=" * 80)
        print("PHASE 8 DAY 52: PRODUCTION TRAFFIC MONITORING SUMMARY")
        print("=" * 80)
        print(f"Timestamp:              {metrics.timestamp}")
        print(f"Total Requests Analyzed:{metrics.total_records}")
        print(f"Contract Compliance:    {metrics.format_compliance_rate * 100:.2f}%")
        print(f"Mean Latency:           {metrics.mean_latency_ms:.2f} ms")
        print(f"p50 / p95 / p99 Latency:{metrics.p50_latency_ms:.2f} / {metrics.p95_latency_ms:.2f} / {metrics.p99_latency_ms:.2f} ms")
        print(f"Mean Prompt/Comp Tokens:{metrics.mean_prompt_tokens:.1f} / {metrics.mean_completion_tokens:.1f} tok")
        print("-" * 80)
        print("Active Alerts:")
        for al in metrics.alerts:
            print(f"  * {al}")
        print("=" * 80)
        print("Reports Generated:")
        print(f"  Markdown: {args.output_report}")
        print(f"  JSON:     {args.output_json}")
        print("=" * 80 + "\n")

    return 0 if not (metrics.contract_alert or metrics.latency_alert) else 1


if __name__ == "__main__":
    sys.exit(main())
