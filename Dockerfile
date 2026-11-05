# Multi-stage production container for Mathematical Reasoning LLM (Phase 8 Day 51)
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn pydantic httpx

# Copy project source and staged models
COPY configs/ ./configs/
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY outputs/ ./outputs/
COPY reports/ ./reports/

# Create persistent logging directory
RUN mkdir -p logs/production

EXPOSE 8000 7860

# Default command: launch production REST API
CMD ["python", "app/api.py", "--host", "0.0.0.0", "--port", "8000"]
