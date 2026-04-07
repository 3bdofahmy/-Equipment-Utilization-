"""
database/models.py
──────────────────
SQLAlchemy ORM models using DeclarativeBase.
This Base is imported by Alembic's env.py for auto-migration.

All Enum columns use the Python Enum classes from core.enums
so the DB enforces the same values as your code.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    BigInteger, Boolean, Float, ForeignKey,
    Integer, String, Enum as SAEnum, Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import TIMESTAMP

from core.enums import Activity, EquipmentType, UtilizationState


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Base ──────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """
    Single Base for all models.
    Import this in alembic/env.py:
        from database.models import Base
        target_metadata = Base.metadata
    """
    pass


# ── Equipment ─────────────────────────────────────────────────────────────────
class Equipment(Base):
    __tablename__ = "equipment"

    equipment_id:   Mapped[str]      = mapped_column(String, primary_key=True)
    equipment_type: Mapped[str]      = mapped_column(
        SAEnum(EquipmentType, name="equipment_type_enum", create_constraint=True),
        nullable=False,
    )
    first_seen:     Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=_now, nullable=False
    )
    last_seen:      Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    # Relationships
    detections:         list["Detection"]        = relationship("Detection",        back_populates="equipment")
    utilization_summary: "UtilizationSummary | None" = relationship("UtilizationSummary", back_populates="equipment", uselist=False)

    def __repr__(self) -> str:
        return f"<Equipment id={self.equipment_id} type={self.equipment_type}>"


# ── Detection ─────────────────────────────────────────────────────────────────
class Detection(Base):
    __tablename__ = "detections"
    __table_args__ = (
        Index("ix_detections_time",         "time"),
        Index("ix_detections_equipment_id", "equipment_id"),
    )

    id:                Mapped[int]           = mapped_column(Integer, primary_key=True, autoincrement=True)
    time:              Mapped[datetime]      = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    equipment_id:      Mapped[str]           = mapped_column(String, ForeignKey("equipment.equipment_id"), nullable=False)
    equipment_type:    Mapped[str]           = mapped_column(
        SAEnum(EquipmentType, name="equipment_type_enum", create_constraint=True),
        nullable=False,
    )
    utilization_state: Mapped[str]           = mapped_column(
        SAEnum(UtilizationState, name="utilization_state_enum", create_constraint=True),
        nullable=False,
    )
    activity:          Mapped[str]           = mapped_column(
        SAEnum(Activity, name="activity_enum", create_constraint=True),
        nullable=False,
    )
    confidence:        Mapped[float]         = mapped_column(Float,      nullable=False)
    motion_score:      Mapped[float]         = mapped_column(Float,      nullable=False)
    llm_verified:      Mapped[bool]          = mapped_column(Boolean,    default=False)
    bbox_x:            Mapped[int | None]    = mapped_column(Integer,    nullable=True)
    bbox_y:            Mapped[int | None]    = mapped_column(Integer,    nullable=True)
    bbox_w:            Mapped[int | None]    = mapped_column(Integer,    nullable=True)
    bbox_h:            Mapped[int | None]    = mapped_column(Integer,    nullable=True)
    frame_index:       Mapped[int | None]    = mapped_column(BigInteger, nullable=True)
    video_source:      Mapped[str | None]    = mapped_column(String,     nullable=True)

    # Relationship
    equipment: Equipment = relationship("Equipment", back_populates="detections")

    def __repr__(self) -> str:
        return f"<Detection id={self.id} eq={self.equipment_id} state={self.utilization_state}>"


# ── UtilizationSummary ────────────────────────────────────────────────────────
class UtilizationSummary(Base):
    __tablename__ = "utilization_summary"

    equipment_id:       Mapped[str]           = mapped_column(String, ForeignKey("equipment.equipment_id"), primary_key=True)
    equipment_type:     Mapped[str]           = mapped_column(
        SAEnum(EquipmentType, name="equipment_type_enum", create_constraint=True),
        nullable=False,
    )
    total_active_sec:   Mapped[float]         = mapped_column(Float, default=0.0)
    total_inactive_sec: Mapped[float]         = mapped_column(Float, default=0.0)
    utilization_pct:    Mapped[float]         = mapped_column(Float, default=0.0)
    last_activity:      Mapped[str | None]    = mapped_column(
        SAEnum(Activity, name="activity_enum", create_constraint=True),
        nullable=True,
    )
    last_state:         Mapped[str | None]    = mapped_column(
        SAEnum(UtilizationState, name="utilization_state_enum", create_constraint=True),
        nullable=True,
    )
    updated_at:         Mapped[datetime]      = mapped_column(TIMESTAMP(timezone=True), default=_now)

    # Relationship
    equipment: Equipment = relationship("Equipment", back_populates="utilization_summary")

    def __repr__(self) -> str:
        return f"<UtilizationSummary eq={self.equipment_id} pct={self.utilization_pct}>"
