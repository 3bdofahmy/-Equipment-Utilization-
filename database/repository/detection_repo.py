"""
database/repository/detection_repo.py
───────────────────────────────────────
DB operations for Detection — uses SQLAlchemy ORM.
"""

from datetime import datetime

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Detection


def insert_detection(payload: dict) -> None:
    """Deprecated direct insert — use the buffer instead."""
    from database.connection import add_detection_to_buffer
    add_detection_to_buffer(payload)


async def query_range(
    db:           AsyncSession,
    from_dt:      datetime,
    to_dt:        datetime,
    equipment_id: str | None = None,
    limit:        int = 500,
) -> list[Detection]:
    query = select(Detection).where(
        and_(
            Detection.time >= from_dt,
            Detection.time <= to_dt,
        )
    )
    if equipment_id:
        query = query.where(Detection.equipment_id == equipment_id)

    query = query.order_by(Detection.time.desc()).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


async def get_activity_counts(
    db:           AsyncSession,
    equipment_id: str,
    from_dt:      datetime,
    to_dt:        datetime,
) -> dict[str, int]:
    query = (
        select(Detection.activity, func.count(Detection.id))
        .where(
            and_(
                Detection.equipment_id == equipment_id,
                Detection.time >= from_dt,
                Detection.time <= to_dt,
            )
        )
        .group_by(Detection.activity)
    )
    result = await db.execute(query)
    return {row[0]: row[1] for row in result}
