"""
api/main.py
────────────
FastAPI application entry point.
Mounts all routers and configures middleware.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.logger import get_logger
from database.connection import check_db_connectivity, start_batching, stop_batching
from inference.registry import ModelRegistry
from tracking.factory import TrackerFactory

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────
    log.info("Starting up …")
    await check_db_connectivity()
    start_batching()
    ModelRegistry.load(settings.inference, settings.classes)
    log.info("API ready")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────
    log.info("Shutting down …")
    await stop_batching()


app = FastAPI(
    title       = "Construction CV Intelligence API",
    description = "Real-time equipment utilization and activity tracking",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.api.cors_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from api.routers import equipment, utilization, detections, stream, model, health  # noqa: E402

app.include_router(health.router,       prefix="/health",      tags=["Health"])
app.include_router(model.router,        prefix="/model",       tags=["Model"])
app.include_router(equipment.router,    prefix="/equipment",   tags=["Equipment"])
app.include_router(utilization.router,  prefix="/utilization", tags=["Utilization"])
app.include_router(detections.router,   prefix="/detections",  tags=["Detections"])
app.include_router(stream.router,       prefix="/stream",      tags=["Stream"])

# ── Serve frontend ────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
