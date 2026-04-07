"""
database/repository/equipment_repo.py
───────────────────────────────────────
DB operations for Equipment — uses SQLAlchemy ORM.
"""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Equipment
from core.enums import EquipmentType


async def get_all(db: AsyncSession) -> list[Equipment]:
    result = await db.execute(select(Equipment).order_by(Equipment.first_seen))
    return list(result.scalars().all())


async def get_by_id(db: AsyncSession, equipment_id: str) -> Equipment | None:
    result = await db.execute(
        select(Equipment).where(Equipment.equipment_id == equipment_id)
    )
    return result.scalar_one_or_none()


async def upsert(db: AsyncSession, equipment_id: str, equipment_type: str | EquipmentType) -> None:
    """Insert or update last_seen timestamp."""
    now = datetime.now(timezone.utc)

    # Accept both str and Enum
    eq_type = equipment_type.value if isinstance(equipment_type, EquipmentType) else equipment_type

    stmt = (
        insert(Equipment)
        .values(
            equipment_id   = equipment_id,
            equipment_type = eq_type,
            first_seen     = now,
            last_seen      = now,
        )
        .on_conflict_do_update(
            index_elements = ["equipment_id"],
            set_           = {"last_seen": now},
        )
    )
    await db.execute(stmt)
    await db.commit()
