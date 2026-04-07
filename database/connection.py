"""
database/connection.py
───────────────────────
SQLAlchemy async engine + session factory + detection batching.
Migrations are managed by Alembic (see alembic/ folder).
"""

import asyncio
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import settings
from core.logger import get_logger
from database.models import Detection

log = get_logger("database")

# ── Engine ────────────────────────────────────────────────────────────────────
engine: AsyncEngine = create_async_engine(
    settings.database.url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=settings.api.debug,          # SQL logging only in debug mode
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ── FastAPI dependency ────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session; close it when request finishes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ── Health check ──────────────────────────────────────────────────────────────
async def check_db_connectivity() -> bool:
    """
    Verify the DB is reachable.
    Run migrations first:  alembic upgrade head
    """
    try:
        async with engine.begin() as conn:
            await conn.exec_driver_sql("SELECT 1")
        log.info("Database connection OK")
        return True
    except Exception as exc:
        log.error(f"Database connection failed: {exc}")
        return False


# ── Detection batching ────────────────────────────────────────────────────────
BATCH_SIZE         = 500
FLUSH_INTERVAL_SEC = 5.0

_detections_buffer: list[dict] = []
_batching_task:     asyncio.Task | None = None
_buffer_lock        = asyncio.Lock()


def add_detection_to_buffer(payload: dict) -> None:
    """Push a detection dict into the in-memory buffer."""
    _detections_buffer.append(payload)
    if len(_detections_buffer) >= BATCH_SIZE:
        asyncio.create_task(flush_detections())


async def flush_detections() -> None:
    """Flush buffered detections into DB in one INSERT."""
    async with _buffer_lock:
        if not _detections_buffer:
            return
        batch = list(_detections_buffer)
        _detections_buffer.clear()

    try:
        async with AsyncSessionLocal() as session:
            try:
                await session.execute(
                    Detection.__table__.insert(), batch   # bulk insert via ORM table
                )
                await session.commit()
                log.info(f"Flushed {len(batch)} detections to DB")
            except Exception as exc:
                await session.rollback()
                log.error(f"Batch insert failed: {exc}")
    except Exception as exc:
        log.error(f"DB communication error during flush: {exc}")


async def _batch_loop() -> None:
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_SEC)
        await flush_detections()


def start_batching() -> None:
    global _batching_task
    _batching_task = asyncio.create_task(_batch_loop())
    log.info("Detection batching started")


async def stop_batching() -> None:
    global _batching_task
    if _batching_task:
        _batching_task.cancel()
        try:
            await _batching_task
        except asyncio.CancelledError:
            pass
    await flush_detections()
    log.info("Detection batching stopped")
