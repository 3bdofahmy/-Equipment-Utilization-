"""
api/routers/health.py
──────────────────────
GET /health — checks database, Kafka, and model status.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from api.schemas import HealthOut
from api.dependencies import get_db
from inference.registry import ModelRegistry

router = APIRouter()


@router.get("", response_model=HealthOut)
async def health_check(db: AsyncSession = Depends(get_db)):
    # Database
    db_status = "ok"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    # Kafka
    kafka_status = "ok"
    try:
        from kafka import KafkaAdminClient
        from core.config import settings
        client = KafkaAdminClient(
            bootstrap_servers=settings.kafka.bootstrap_servers,
            request_timeout_ms=2000,
        )
        client.close()
    except Exception:
        kafka_status = "error"

    # Model
    model_status = ModelRegistry.get_status().get("status", "not_loaded")

    overall = "ok" if all(
        s == "ok" for s in [db_status, kafka_status]
    ) and model_status == "running" else "degraded"

    return HealthOut(
        status   = overall,
        database = db_status,
        kafka    = kafka_status,
        model    = model_status,
    )
