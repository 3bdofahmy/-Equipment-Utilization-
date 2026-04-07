"""
streaming/consumer.py
──────────────────────
Async Kafka consumer — used by analytics_service.
"""

import json
import asyncio
from aiokafka import AIOKafkaConsumer
from core.config import KafkaConfig
from core.logger import get_logger

log = get_logger("kafka.consumer")


class KafkaConsumerClient:

    def __init__(self, config: KafkaConfig):
        self.config    = config
        self._consumer = None

    async def connect(self, retries: int = 20, delay: float = 5.0) -> None:
        """Connect to Kafka broker and subscribe to topic."""
        for attempt in range(1, retries + 1):
            try:
                self._consumer = AIOKafkaConsumer(
                    self.config.topic_detections,
                    bootstrap_servers = self.config.bootstrap_servers,
                    group_id          = self.config.consumer_group,
                    auto_offset_reset = "latest",
                    enable_auto_commit= True,
                )
                await self._consumer.start()
                log.info("Kafka consumer connected (attempt %d)", attempt)
                return
            except Exception as e:
                if attempt < retries:
                    log.warning("Kafka not ready — retrying in %.0fs (%d/%d): %s",
                               delay, attempt, retries, str(e))
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(f"Cannot connect to Kafka: {e}")

    async def consume_messages(self):
        """Async generator — yields decoded message dicts."""
        if not self._consumer:
            raise RuntimeError("Consumer not connected. Call connect() first.")
        async for msg in self._consumer:
            try:
                yield json.loads(msg.value.decode())
            except json.JSONDecodeError:
                log.warning("Failed to decode message: %s", msg.value)
                continue

    async def close(self) -> None:
        """Close consumer connection."""
        if self._consumer:
            await self._consumer.stop()
            log.info("Kafka consumer closed")
