"""
streaming/producer.py
──────────────────────
Async Kafka producer — publishes detection and frame payloads.
"""

import asyncio
from aiokafka import AIOKafkaProducer
from core.config import KafkaConfig
from core.logger import get_logger

log = get_logger("kafka.producer")


class KafkaProducerClient:

    def __init__(self, config: KafkaConfig):
        self.config    = config
        self._producer = None

    async def connect(self, retries: int = 20, delay: float = 5.0) -> None:
        """Connect to Kafka broker with retries."""
        for attempt in range(1, retries + 1):
            try:
                self._producer = AIOKafkaProducer(
                    bootstrap_servers  = self.config.bootstrap_servers,
                    compression_type   = "gzip",
                )
                await self._producer.start()
                log.info("Kafka producer connected (attempt %d)", attempt)
                return
            except Exception as e:
                if attempt < retries:
                    log.warning("Kafka not ready — retrying in %.0fs (%d/%d): %s",
                               delay, attempt, retries, str(e))
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(f"Cannot connect to Kafka: {e}")

    async def send_detection(self, payload) -> None:
        """Send detection payload to Kafka."""
        if not self._producer:
            raise RuntimeError("Producer not connected. Call connect() first.")
        await self._producer.send_and_wait(
            self.config.topic_detections,
            key    = payload.equipment_id.encode() if isinstance(payload.equipment_id, str) else payload.equipment_id,
            value  = payload.to_json(),
        )

    async def send_frame(self, payload) -> None:
        """Send frame payload to Kafka."""
        if not self._producer:
            raise RuntimeError("Producer not connected. Call connect() first.")
        await self._producer.send_and_wait(
            self.config.topic_frames,
            value = payload.to_json(),
        )

    async def flush(self) -> None:
        """Flush pending messages."""
        if self._producer:
            await self._producer.flush()

    async def close(self) -> None:
        """Close producer connection."""
        if self._producer:
            await self._producer.stop()
            log.info("Kafka producer closed")
