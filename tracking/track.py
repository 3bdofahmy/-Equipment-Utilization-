"""
tracking/track.py
──────────────────
Track dataclass — one instance per tracked machine.
"""

import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Track:
    track_id:           str
    class_id:           int
    class_name:         str
    bbox:               tuple[int, int, int, int]       # x, y, w, h
    mask:               np.ndarray | None = None        # H x W bool
    confidence:         float = 0.0

    # Motion / activity state
    utilization_state:  str   = "INACTIVE"              # ACTIVE | INACTIVE
    activity:           str   = "Waiting"               # Digging | Swinging/Loading | Dumping | Waiting
    motion_score:       float = 0.0
    llm_verified:       bool  = False

    # Time accounting
    active_time:        float = 0.0
    inactive_time:      float = 0.0
    _last_update:       float = field(default_factory=time.time, repr=False)

    def tick(self) -> None:
        """Accrue time since last update."""
        now = time.time()
        dt  = now - self._last_update
        if self.utilization_state == "ACTIVE":
            self.active_time   += dt
        else:
            self.inactive_time += dt
        self._last_update = now

    @property
    def total_time(self) -> float:
        return self.active_time + self.inactive_time

    @property
    def utilization_pct(self) -> float:
        if self.total_time < 0.01:
            return 0.0
        return round(self.active_time / self.total_time * 100, 2)
