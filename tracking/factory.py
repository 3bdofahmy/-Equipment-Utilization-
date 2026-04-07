"""
tracking/factory.py
────────────────────
TrackerFactory — returns correct tracker based on config.

Usage:
    from tracking.factory import TrackerFactory
    from core.config import settings

    tracker = TrackerFactory.create(settings.tracking)
"""

from core.config import TrackingConfig
from core.logger import get_logger

log = get_logger("tracking.factory")

SUPPORTED_TRACKERS = ("botsort", "bytetrack")


class TrackerFactory:

    @staticmethod
    def create(config: TrackingConfig):
        tracker = config.tracker.lower().strip()

        if tracker not in SUPPORTED_TRACKERS:
            raise ValueError(
                f"Unknown tracker '{tracker}'. "
                f"Choose from: {SUPPORTED_TRACKERS}"
            )

        log.info("Creating tracker: %s", tracker)
        from tracking.tracker import Tracker
        return Tracker(tracker_type=tracker, config=config)
