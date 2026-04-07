"""
services/analytics_service.py
───────────────────────────────
Kafka consumer → aggregation → PostgreSQL.
Runs as a separate process from cv_service.
"""

import time
import asyncio
from collections import defaultdict
from datetime import datetime, timezone

from core.config import settings
from core.logger import get_logger
from streaming.consumer import KafkaConsumerClient
from database.connection import AsyncSessionLocal, check_db_connectivity, start_batching, stop_batching, add_detection_to_buffer
from database.repository import equipment_repo, utilization_repo

log = get_logger("analytics_service")

COMMIT_INTERVAL_SEC  = 5.0
ALERT_IDLE_THRESHOLD = float(settings.database.__class__.__dataclass_fields__.get(
    "ALERT_IDLE_THRESHOLD_SEC", 120
)) if False else 120.0


async def main_async():
    log.info("Analytics Service starting …")
    await check_db_connectivity()
    start_batching()

    consumer    = KafkaConsumerClient(settings.kafka)
    await consumer.connect()
    idle_timers = defaultdict(float)

    try:
        async for msg in consumer.consume_messages():
            eq_id    = msg.get("equipment_id",      "UNKNOWN")
            eq_type  = msg.get("equipment_type",    "unknown")
            state    = msg.get("utilization_state", "INACTIVE")
            activity = msg.get("activity",          "Waiting")

            # Idle alert
            if state == "INACTIVE":
                idle_timers[eq_id] += 1.0 / settings.video.process_fps
                if idle_timers[eq_id] >= ALERT_IDLE_THRESHOLD:
                    log.warning(
                        "ALERT: %s (%s) idle for %.0f seconds",
                        eq_id, eq_type, idle_timers[eq_id],
                    )
            else:
                idle_timers[eq_id] = 0.0

            # Persist
            async with AsyncSessionLocal() as db:
                try:
                    await equipment_repo.upsert(db, eq_id, eq_type)
                    
                    # msg timestamp might need isoformat wrapper normally, but assuming consumer outputs dict
                    add_detection_to_buffer(msg)
                    
                    await utilization_repo.upsert(db, {
                        "equipment_id":      eq_id,
                        "equipment_type":    eq_type,
                        "active_time_sec":   msg.get("active_time_sec",   0),
                        "inactive_time_sec": msg.get("inactive_time_sec", 0),
                        "utilization_pct":   msg.get("utilization_pct",   0),
                        "activity":          activity,
                        "utilization_state": state,
                    })
                except Exception as e:
                    log.warning("DB error: %s", e)
                    await db.rollback()

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        await stop_batching()
        await consumer.close()
        log.info("Analytics Service stopped")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
