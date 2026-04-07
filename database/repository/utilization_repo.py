"""
database/repository/utilization_repo.py
─────────────────────────────────────────
DB operations for UtilizationSummary — uses SQLAlchemy ORM.
"""

from datetime import datetime, timezone

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Detection, UtilizationSummary
from core.enums import Activity, EquipmentType, UtilizationState


async def get_all(db: AsyncSession) -> list[UtilizationSummary]:
    result = await db.execute(
        select(UtilizationSummary).order_by(UtilizationSummary.updated_at.desc())
    )
    return list(result.scalars().all())


async def get_by_id(db: AsyncSession, equipment_id: str) -> UtilizationSummary | None:
    result = await db.execute(
        select(UtilizationSummary).where(UtilizationSummary.equipment_id == equipment_id)
    )
    return result.scalar_one_or_none()


async def upsert(db: AsyncSession, payload: dict) -> None:
    """Insert or update the utilization summary for one piece of equipment."""
    now = datetime.now(timezone.utc)

    # Normalize enums → string values for the DB
    def _val(v):
        return v.value if hasattr(v, "value") else v

    stmt = (
        insert(UtilizationSummary)
        .values(
            equipment_id       = payload["equipment_id"],
            equipment_type     = _val(payload["equipment_type"]),
            total_active_sec   = payload["active_time_sec"],
            total_inactive_sec = payload["inactive_time_sec"],
            utilization_pct    = payload["utilization_pct"],
            last_activity      = _val(payload["activity"]),
            last_state         = _val(payload["utilization_state"]),
            updated_at         = now,
        )
        .on_conflict_do_update(
            index_elements = ["equipment_id"],
            set_ = {
                "total_active_sec":   payload["active_time_sec"],
                "total_inactive_sec": payload["inactive_time_sec"],
                "utilization_pct":    payload["utilization_pct"],
                "last_activity":      _val(payload["activity"]),
                "last_state":         _val(payload["utilization_state"]),
                "updated_at":         now,
            },
        )
    )
    await db.execute(stmt)
    await db.commit()


async def get_history(
    db:           AsyncSession,
    equipment_id: str,
    from_dt:      datetime,
    to_dt:        datetime,
    limit:        int = 200,
) -> list[Detection]:
    """Pull time-series from detections table for charting."""
    query = (
        select(Detection)
        .where(
            and_(
                Detection.equipment_id == equipment_id,
                Detection.time >= from_dt,
                Detection.time <= to_dt,
            )
        )
        .order_by(Detection.time.asc())
        .limit(limit)
    )
    result = await db.execute(query)
    return list(result.scalars().all())
