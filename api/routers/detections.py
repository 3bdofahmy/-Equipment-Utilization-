"""
api/routers/detections.py
──────────────────────────
GET /detections
"""

from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from api.schemas import DetectionOut
from api.dependencies import get_db
from database.repository import detection_repo

router = APIRouter()


@router.get("", response_model=list[DetectionOut])
async def list_detections(
    equipment_id: str | None = Query(default=None),
    minutes:      int        = Query(default=10, ge=1, le=1440),
    limit:        int        = Query(default=200, ge=1, le=1000),
    db:           AsyncSession    = Depends(get_db),
):
    now     = datetime.now(timezone.utc)
    from_dt = now - timedelta(minutes=minutes)
    return await detection_repo.query_range(db, from_dt, now, equipment_id, limit)
