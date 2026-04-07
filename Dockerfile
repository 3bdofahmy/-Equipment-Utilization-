# ============================================================
# Dockerfile  —  Multi-stage, single file for all services
# ============================================================
# Three build targets:
#   base      — shared Python + system deps
#   api       — FastAPI server
#   cv        — CV pipeline (needs OpenCV + GPU libs)
#   analytics — Kafka consumer / analytics worker
#
# Usage (handled automatically by docker-compose.yml):
#   docker build --target api       -t eagle/api .
#   docker build --target cv        -t eagle/cv  .
#   docker build --target analytics -t eagle/analytics .
# ============================================================

# ── base ─────────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies shared by all services
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# ── api ──────────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000

# Wait for postgres + kafka to be ready, then start
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]

# ── cv ───────────────────────────────────────────────────────
FROM base AS cv

# Extra system packages only needed for CV (ffmpeg, X11 stubs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsm6 libxext6 libxrender-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

CMD ["python", "-u", "services/cv_service.py"]

# ── analytics ─────────────────────────────────────────────────
FROM base AS analytics

CMD ["python", "-u", "services/analytics_service.py"]
