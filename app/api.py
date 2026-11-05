#!/usr/bin/env python3
"""Production FastAPI REST API for Mathematical Reasoning (Phase 8 Day 51).

Provides high-throughput asynchronous REST endpoints for querying the
post-trained DeepSeekMath-7B Post-GRPO Policy model on mathematical reasoning.

Endpoints:
- POST /v1/predict: Single-problem reasoning inference (greedy, sampled, self_consistency).
- POST /v1/predict/batch: Concurrent multi-query batch reasoning.
- GET  /v1/health: Engine health, device/backend status, and adapter provenance.
- GET  /v1/metrics: Operational telemetry (latency, format compliance, throughput, tokens).
- GET  /v1/logs: Streaming/tail of recent production traffic logs.

Usage:
    python app/api.py --port 8000
    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.engine import (  # noqa: E402
    ReasoningServingEngine,
    ServingConfig,
    ServingPrediction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("math_reasoning_api")

# --- Pydantic Data Contracts ---


class PredictionRequest(BaseModel):
    """Schema for a single mathematical reasoning query."""

    question: str = Field(
        ...,
        min_length=1,
        description="Mathematical word problem or reasoning prompt",
        examples=[
            "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins. She sells the remainder for $2 each. How much does she make daily?"
        ],
    )
    regime: str = Field(
        default="greedy",
        pattern=r"^(greedy|sampled|self_consistency)$",
        description="Decoding regime ('greedy', 'sampled', or 'self_consistency')",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (used when regime is 'sampled' or 'self_consistency')",
    )
    num_samples: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Sample count k for majority vote self-consistency (maj@k)",
    )
    problem_id: Optional[str] = Field(
        default=None,
        description="Optional tracking identifier for the problem",
    )


class PredictionResponse(BaseModel):
    """Structured response container complying with the Answer Contract."""

    problem_id: str
    question: str
    completion: str
    extracted_answer: Optional[str]
    format_compliant: bool
    latency_ms: float
    tokens_per_sec: float
    regime: str
    num_samples: int
    prompt_tokens: int
    completion_tokens: int
    all_extracted: List[Optional[str]] = Field(default_factory=list)
    majority_vote_winner: Optional[str] = None
    majority_vote_confidence: float = 1.0
    timestamp: str

    @classmethod
    def from_serving_prediction(cls, pred: ServingPrediction) -> PredictionResponse:
        return cls(
            problem_id=pred.problem_id,
            question=pred.question,
            completion=pred.completion,
            extracted_answer=pred.extracted_answer,
            format_compliant=pred.format_compliant,
            latency_ms=round(pred.latency_ms, 2),
            tokens_per_sec=round(pred.tokens_per_sec, 1),
            regime=pred.regime,
            num_samples=pred.num_samples,
            prompt_tokens=pred.prompt_tokens,
            completion_tokens=pred.completion_tokens,
            all_extracted=pred.all_extracted,
            majority_vote_winner=pred.majority_vote_winner,
            majority_vote_confidence=pred.majority_vote_confidence,
            timestamp=pred.timestamp,
        )


class BatchPredictionRequest(BaseModel):
    """Schema for concurrent multi-query batch reasoning."""

    questions: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of mathematical questions to solve concurrently",
    )
    regime: str = Field(
        default="greedy",
        pattern=r"^(greedy|sampled|self_consistency)$",
        description="Decoding regime applied to all questions in batch",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    num_samples: int = Field(default=8, ge=1, le=32)


class BatchPredictionResponse(BaseModel):
    """Batch prediction results and aggregate throughput telemetry."""

    total_queries: int
    total_elapsed_ms: float
    overall_throughput_tok_s: float
    format_compliance_rate: float
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Operational health status of the serving API."""

    status: str
    version: str = "v1.1"
    uptime_seconds: float
    model_name: str
    adapter_path: Optional[str]
    backend: str
    total_served_queries: int
    timestamp: str


class MetricsResponse(BaseModel):
    """Prometheus-compatible operational metrics report."""

    total_requests: int
    mean_latency_ms: float
    format_compliance_rate: float
    total_completion_tokens: int
    uptime_seconds: float
    regime_distribution: Dict[str, int]
    timestamp: str


# --- Engine Lifecycle & Application Setup ---

START_TIME = time.time()
_engine_instance: Optional[ReasoningServingEngine] = None


def get_engine() -> ReasoningServingEngine:
    """Retrieve or initialize the active ReasoningServingEngine."""
    global _engine_instance
    if _engine_instance is None:
        cfg = ServingConfig(
            use_mock=True,
            adapter_path="outputs/grpo/adapter_model",
            log_dir="logs/production",
            log_requests=True,
        )
        _engine_instance = ReasoningServingEngine(config=cfg)
    return _engine_instance


def set_engine(engine: ReasoningServingEngine) -> None:
    """Explicitly assign an initialized engine instance (useful for testing/runtime config)."""
    global _engine_instance
    _engine_instance = engine


app = FastAPI(
    title="Math Reasoning LLM Production REST API",
    description=(
        "High-performance REST API serving the post-trained DeepSeekMath-7B Post-GRPO Policy "
        "model on GSM8K mathematical reasoning with Answer Contract enforcement."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REST API Endpoints ---


@app.get("/", summary="Root Status", tags=["General"])
def root() -> Dict[str, Any]:
    """Basic root health check and API overview."""
    return {
        "service": "Math Reasoning LLM Production API",
        "version": "v1.1",
        "docs": "/docs",
        "health": "/v1/health",
        "metrics": "/v1/metrics",
    }


@app.get("/v1/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Check API service health, active adapter, and runtime backend."""
    engine = get_engine()
    uptime = time.time() - START_TIME
    backend = "Deterministic Mock Engine" if engine.config.use_mock else "CUDA Hardware Engine"

    return HealthResponse(
        status="HEALTHY",
        version="v1.1",
        uptime_seconds=round(uptime, 2),
        model_name=engine.config.model_name_or_path,
        adapter_path=engine.config.adapter_path,
        backend=backend,
        total_served_queries=len(engine._traffic_history),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Execute mathematical reasoning for a single question.

    Supports greedy decoding, sampled CoT, and majority vote self-consistency (maj@k).
    Guarantees Answer Contract adherence (terminal line '\\n#### <number>').
    """
    engine = get_engine()
    try:
        pred = engine.predict(
            question=request.question,
            regime=request.regime,
            temperature=request.temperature,
            num_samples=request.num_samples,
            problem_id=request.problem_id,
        )
        return PredictionResponse.from_serving_prediction(pred)
    except Exception as err:
        logger.error("Inference failure for question: %s", err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(err)}",
        ) from err


@app.post(
    "/v1/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Execute concurrent batch reasoning over multiple questions."""
    engine = get_engine()
    t0 = time.perf_counter()

    try:
        raw_preds = engine.predict_batch(
            questions=request.questions,
            regime=request.regime,
            temperature=request.temperature,
            num_samples=request.num_samples,
        )
    except Exception as err:
        logger.error("Batch inference failure: %s", err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(err)}",
        ) from err

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    total_tokens = sum(p.completion_tokens for p in raw_preds)
    throughput = (total_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

    compliant_count = sum(1 for p in raw_preds if p.format_compliant)
    format_rate = (compliant_count / len(raw_preds)) if raw_preds else 1.0

    formatted_preds = [PredictionResponse.from_serving_prediction(p) for p in raw_preds]

    return BatchPredictionResponse(
        total_queries=len(raw_preds),
        total_elapsed_ms=round(elapsed_ms, 2),
        overall_throughput_tok_s=round(throughput, 1),
        format_compliance_rate=round(format_rate, 4),
        predictions=formatted_preds,
    )


@app.get("/v1/metrics", response_model=MetricsResponse, tags=["Monitoring"])
def get_metrics() -> MetricsResponse:
    """Retrieve operational telemetry and Answer Contract compliance stats."""
    engine = get_engine()
    stats = engine.get_traffic_stats()
    uptime = time.time() - START_TIME

    # Calculate regime distribution
    regime_dist: Dict[str, int] = {"greedy": 0, "sampled": 0, "self_consistency": 0}
    for p in engine._traffic_history:
        regime_dist[p.regime] = regime_dist.get(p.regime, 0) + 1

    return MetricsResponse(
        total_requests=stats.get("total_requests", 0),
        mean_latency_ms=stats.get("mean_latency_ms", 0.0),
        format_compliance_rate=stats.get("format_compliance_rate", 1.0),
        total_completion_tokens=stats.get("total_completion_tokens", 0),
        uptime_seconds=round(uptime, 2),
        regime_distribution=regime_dist,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/v1/logs", tags=["Monitoring"])
def get_recent_logs(
    limit: int = Query(default=50, ge=1, le=500, description="Max log records to retrieve"),
) -> List[Dict[str, Any]]:
    """Retrieve the most recent production traffic log records."""
    engine = get_engine()
    if not engine.traffic_log_file or not engine.traffic_log_file.exists():
        return []

    try:
        with open(engine.traffic_log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        selected = lines[-limit:]
        return [json.loads(line) for line in selected]
    except Exception as err:
        logger.error("Failed to read traffic log: %s", err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read traffic log: {str(err)}",
        ) from err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Math Reasoning LLM Production REST API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--adapter_path", type=str, default="outputs/grpo/adapter_model", help="Path to LoRA adapter")
    parser.add_argument("--use_mock", action="store_true", default=True, help="Run mock inference engine")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    cfg = ServingConfig(
        use_mock=args.use_mock,
        adapter_path=args.adapter_path,
        log_dir="logs/production",
    )
    set_engine(ReasoningServingEngine(config=cfg))
    logger.info("Starting REST API on %s:%d (mock=%s)...", args.host, args.port, args.use_mock)
    uvicorn.run(app, host=args.host, port=args.port)
