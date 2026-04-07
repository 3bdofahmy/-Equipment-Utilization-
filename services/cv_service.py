"""
services/cv_service.py
───────────────────────
Main CV process:
  video → detect → track → motion → activity
       → Kafka (detections + frames)
       → PostgreSQL (using async/buffer)
"""

import base64
import time
import cv2
import asyncio
from datetime import datetime, timezone

from core.config import settings
from core.logger import get_logger
from inference.registry import ModelRegistry
from tracking.factory import TrackerFactory
from pipeline.video_reader import VideoReader
from pipeline.frame_processor import FrameProcessor
from pipeline.annotator import annotate
from streaming.producer import KafkaProducerClient
from streaming.schemas import DetectionPayload, FramePayload
from database.connection import AsyncSessionLocal, check_db_connectivity, start_batching, stop_batching, add_detection_to_buffer
from database.repository import equipment_repo, utilization_repo

log = get_logger("cv_service")

FRAME_JPEG_Q  = 60
DB_COMMIT_EVERY = 30


async def main_async():
    log.info("CV Service starting …")

    # ── Init ──────────────────────────────────────────────────────────────
    await check_db_connectivity()
    start_batching()
    ModelRegistry.load(settings.inference, settings.classes)
    tracker   = TrackerFactory.create(settings.tracking)
    processor = FrameProcessor(settings, tracker)
    producer  = KafkaProducerClient(settings.kafka)
    await producer.connect()

    frame_count = 0

    try:
        with VideoReader(settings.video) as reader:
            for frame_index, frame in reader.frames():

                # ── Process ───────────────────────────────────────────────
                result = processor.process(frame_index, frame)

                # ── Annotate ──────────────────────────────────────────────
                annotated = annotate(result.raw_frame, result.tracks)

                # ── Publish detections ────────────────────────────────────
                async with AsyncSessionLocal() as db:
                    for track in result.tracks:
                        payload = DetectionPayload.from_track(
                            track        = track,
                            frame_index  = frame_index,
                            timestamp    = result.timestamp,
                            video_source = settings.video.source,
                        )
                        await producer.send_detection(payload)

                        # Write to DB
                        try:
                            await equipment_repo.upsert(db, track.track_id, track.class_name)
                            add_detection_to_buffer({
                                "timestamp":         result.timestamp.isoformat() if isinstance(result.timestamp, datetime) else result.timestamp,
                                "equipment_id":      track.track_id,
                                "equipment_type":    track.class_name,
                                "utilization_state": track.utilization_state,
                                "activity":          track.activity,
                                "confidence":        track.confidence,
                                "motion_score":      track.motion_score,
                                "llm_verified":      track.llm_verified,
                                "bbox_x":            track.bbox[0],
                                "bbox_y":            track.bbox[1],
                                "bbox_w":            track.bbox[2],
                                "bbox_h":            track.bbox[3],
                                "frame_index":       frame_index,
                                "video_source":      settings.video.source,
                            })
                            await utilization_repo.upsert(db, {
                                "equipment_id":      track.track_id,
                                "equipment_type":    track.class_name,
                                "active_time_sec":   track.active_time,
                                "inactive_time_sec": track.inactive_time,
                                "utilization_pct":   track.utilization_pct,
                                "activity":          track.activity,
                                "utilization_state": track.utilization_state,
                            })
                        except Exception as e:
                            log.warning("DB write error: %s", e)
                            await db.rollback()

                # ── Publish annotated frame ───────────────────────────────
                _, buf    = cv2.imencode(".jpg", annotated,
                                         [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_Q])
                b64_frame = base64.b64encode(buf.tobytes()).decode()

                frame_payload = FramePayload(
                    timestamp   = result.timestamp,
                    frame_index = frame_index,
                    jpeg_b64    = b64_frame,
                    track_count = len(result.tracks),
                )
                await producer.send_frame(frame_payload)

                # ── Update in-memory frame cache for API ──────────────────
                try:
                    from api.routers.stream import update_latest_frame
                    update_latest_frame({
                        "timestamp":   result.timestamp,
                        "frame_index": frame_index,
                        "jpeg_b64":    b64_frame,
                        "track_count": len(result.tracks),
                    })
                except Exception:
                    pass   # API may not be running in same process

                if frame_count % 100 == 0:
                    log.info("Frame %d | tracks: %d", frame_index, len(result.tracks))
                frame_count += 1

    except KeyboardInterrupt:
        log.info("Interrupted — shutting down …")
    finally:
        await stop_batching()
        await producer.close()
        log.info("CV Service stopped")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
