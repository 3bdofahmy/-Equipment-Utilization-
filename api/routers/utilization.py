"""
api/routers/utilization.py
───────────────────────────
GET /utilization
GET /utilization/{equipment_id}
GET /utilization/{equipment_id}/history
"""

from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from api.schemas import UtilizationOut, UtilizationHistoryPoint
from api.dependencies import get_db
from database.repository import utilization_repo

router = APIRouter()


@router.get("", response_model=list[UtilizationOut])
async def list_utilization(db: AsyncSession = Depends(get_db)):
    return await utilization_repo.get_all(db)


@router.get("/{equipment_id}", response_model=UtilizationOut)
async def get_utilization(equipment_id: str, db: AsyncSession = Depends(get_db)):
    u = await utilization_repo.get_by_id(db, equipment_id)
    if not u:
        raise HTTPException(status_code=404, detail="Equipment not found")
    return u


@router.get("/{equipment_id}/history", response_model=list[UtilizationHistoryPoint])
async def get_history(
    equipment_id: str,
    minutes:      int     = Query(default=30, ge=1, le=1440),
    limit:        int     = Query(default=200, ge=1, le=1000),
    db:           AsyncSession = Depends(get_db),
):
    now     = datetime.now(timezone.utc)
    from_dt = now - timedelta(minutes=minutes)
    rows    = await utilization_repo.get_history(db, equipment_id, from_dt, now, limit)
    return [
        UtilizationHistoryPoint(
            time              = row.time,
            utilization_state = row.utilization_state,
            activity          = row.activity,
            motion_score      = row.motion_score,
        )
        for row in rows
    ]
