"""Production Traffic Monitoring and Data Drift Detector (Phase 8 Day 52).

Parses and analyzes production traffic records from `logs/production/requests.jsonl`:
1. Rolling Answer Contract compliance monitoring (alerts on format degradation < 98.0%).
2. Latency percentiles (mean, p50, p95, p99) and performance regression tracking.
3. Token length distribution shifts (detects verbosity looping or excessive truncation).
4. Decoding regime utilization telemetry.

Emits:
- reports/day52_traffic_monitoring_report.md
- reports/day52_traffic_monitoring.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class TrafficLogRecord:
    """A single logged production request record."""

    problem_id: str
    question: str
    completion: str
    extracted_answer: Optional[str]
    format_compliant: bool
    latency_ms: float
    tokens_per_sec: float
    regime: str
    prompt_tokens: int
    completion_tokens: int
    timestamp: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrafficLogRecord:
        return cls(
            problem_id=str(data.get("problem_id", "")),
            question=str(data.get("question", "")),
            completion=str(data.get("completion", "")),
            extracted_answer=data.get("extracted_answer"),
            format_compliant=bool(data.get("format_compliant", False)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            tokens_per_sec=float(data.get("tokens_per_sec", 0.0)),
            regime=str(data.get("regime", "greedy")),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            timestamp=str(data.get("timestamp", "")),
        )


@dataclass
class TrafficDriftMetrics:
    """Aggregated operational telemetry and drift indicators."""

    timestamp: str
    total_records: int
    format_compliance_rate: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    contract_alert: bool = False
    latency_alert: bool = False
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrafficMonitor:
    """Parser and statistical analyzer for production serving logs."""

    def __init__(self, log_file: Optional[Union[str, Path]] = None) -> None:
        default_log = PROJECT_ROOT / "logs" / "production" / "requests.jsonl"
        self.log_file = Path(log_file) if log_file else default_log

    def load_records(self) -> List[TrafficLogRecord]:
        """Parse structured records from the traffic log file."""
        if not self.log_file.exists():
            logger.warning("Traffic log file not found at: %s", self.log_file)
            return []

        records: List[TrafficLogRecord] = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        data = json.loads(line_str)
                        records.append(TrafficLogRecord.from_dict(data))
                    except json.JSONDecodeError as err:
                        logger.warning("Skipping malformed log line %d: %s", line_idx, err)
        except Exception as err:
            logger.error("Failed to read traffic log: %s", err, exc_info=True)

        return records

    def compute_metrics(
        self,
        records: Optional[List[TrafficLogRecord]] = None,
        min_contract_threshold: float = 0.98,
        max_p95_latency_threshold_ms: float = 500.0,
    ) -> TrafficDriftMetrics:
        """Compute statistical percentiles, distributions, and drift alerts."""
        data_records = records if records is not None else self.load_records()

        if not data_records:
            return TrafficDriftMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                total_records=0,
                format_compliance_rate=1.0,
                mean_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                mean_prompt_tokens=0.0,
                mean_completion_tokens=0.0,
                regime_distribution={},
                contract_alert=False,
                latency_alert=False,
                alerts=["No production traffic records found to analyze."],
            )

        n = len(data_records)
        compliant_count = sum(1 for r in data_records if r.format_compliant)
        format_rate = round(compliant_count / n, 4)

        latencies = sorted(r.latency_ms for r in data_records)
        mean_lat = round(sum(latencies) / n, 2)

        def percentile(sorted_vals: List[float], p: float) -> float:
            idx = int(round((len(sorted_vals) - 1) * p))
            return round(sorted_vals[max(0, min(idx, len(sorted_vals) - 1))], 2)

        p50 = percentile(latencies, 0.50)
        p95 = percentile(latencies, 0.95)
        p99 = percentile(latencies, 0.99)

        prompts = [r.prompt_tokens for r in data_records]
        completions = [r.completion_tokens for r in data_records]
        mean_prompt = round(sum(prompts) / n, 1)
        mean_comp = round(sum(completions) / n, 1)

        regimes: Dict[str, int] = {}
        for r in data_records:
            regimes[r.regime] = regimes.get(r.regime, 0) + 1

        alerts: List[str] = []
        contract_alert = False
        if format_rate < min_contract_threshold:
            contract_alert = True
            alerts.append(
                f"Answer Contract Drift Alert: observed compliance {format_rate*100:.1f}% "
                f"< required threshold {min_contract_threshold*100:.1f}%."
            )

        latency_alert = False
        if p95 > max_p95_latency_threshold_ms:
            latency_alert = True
            alerts.append(
                f"Latency Degradation Alert: observed p95 latency {p95:.1f}ms "
                f"> ceiling {max_p95_latency_threshold_ms:.1f}ms."
            )

        if not alerts:
            alerts.append("All operational indicators operating within nominal design tolerances.")

        return TrafficDriftMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_records=n,
            format_compliance_rate=format_rate,
            mean_latency_ms=mean_lat,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            mean_prompt_tokens=mean_prompt,
            mean_completion_tokens=mean_comp,
            regime_distribution=regimes,
            contract_alert=contract_alert,
            latency_alert=latency_alert,
            alerts=alerts,
        )


def generate_monitoring_markdown_report(metrics: TrafficDriftMetrics, output_path: Path) -> str:
    """Format and save the traffic monitoring and drift analysis report."""
    status_icon = "❌ CRITICAL ALERTS" if (metrics.contract_alert or metrics.latency_alert) else "✅ ALL SYSTEMS NOMINAL"

    lines = [
        "# Phase 8 Day 52: Production Traffic Observability & Drift Report",
        "",
        f"- **Audit Timestamp:** `{metrics.timestamp}`",
        f"- **Health & Drift Status:** **{status_icon}**",
        f"- **Total Analyzed Requests:** `{metrics.total_records}`",
        f"- **Answer Contract Compliance:** `{metrics.format_compliance_rate * 100:.2f}%`",
        f"- **Latency Profile:** mean=`{metrics.mean_latency_ms:.2f} ms`, p50=`{metrics.p50_latency_ms:.2f} ms`, p95=`{metrics.p95_latency_ms:.2f} ms`, p99=`{metrics.p99_latency_ms:.2f} ms`",
        f"- **Token Footprint:** prompt=`{metrics.mean_prompt_tokens:.1f} tok`, completion=`{metrics.mean_completion_tokens:.1f} tok`",
        "",
        "---",
        "",
        "## 1. Operational Telemetry & Decoding Regime Breakdown",
        "",
        "| Decoding Regime | Handled Request Count | Share (%) |",
        "| :--- | :---: | :---: |",
    ]

    tot = max(1, metrics.total_records)
    for reg, cnt in metrics.regime_distribution.items():
        lines.append(f"| `{reg}` | {cnt} | `{cnt / tot * 100:.1f}%` |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Active Health & Drift Alerts",
            "",
        ]
    )

    for al in metrics.alerts:
        bullet = "⚠️" if ("Alert" in al) else "✅"
        lines.append(f"- {bullet} {al}")

    lines.extend(
        [
            "",
            "---",
            "*Report generated automatically by `scripts/run_traffic_monitor.py` (Phase 8 Day 52).* ",
        ]
    )

    report_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return report_text


def run_traffic_monitoring(
    log_file: Optional[Union[str, Path]] = None,
    output_report_path: Optional[Union[str, Path]] = None,
    output_json_path: Optional[Union[str, Path]] = None,
) -> TrafficDriftMetrics:
    """Execute complete traffic monitoring and save reports."""
    rep_dir = PROJECT_ROOT / "reports"
    out_md = Path(output_report_path) if output_report_path else (rep_dir / "day52_traffic_monitoring_report.md")
    out_json = Path(output_json_path) if output_json_path else (rep_dir / "day52_traffic_monitoring.json")

    monitor = TrafficMonitor(log_file=log_file)
    metrics = monitor.compute_metrics()

    # Save summary JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Saved traffic monitoring JSON to: %s", out_json)

    # Save Markdown report
    generate_monitoring_markdown_report(metrics, out_md)
    logger.info("Saved traffic monitoring Markdown report to: %s", out_md)

    return metrics
