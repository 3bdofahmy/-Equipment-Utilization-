"""
motion/state.py
────────────────
ACTIVE / INACTIVE state machine with hysteresis per track.
Prevents rapid flickering between states.
"""

from collections import deque
from core.enums import UtilizationState


class StateMachine:
    """One instance per track_id."""

    def __init__(self, confirm_frames: int = 3):
        self._confirm = confirm_frames
        self._state: UtilizationState = UtilizationState.INACTIVE
        self._votes: deque[bool] = deque(maxlen=confirm_frames * 2)

    def update(self, is_active: bool) -> UtilizationState:
        self._votes.append(is_active)

        active_votes = sum(self._votes)
        total        = len(self._votes)

        if self._state == UtilizationState.INACTIVE and active_votes >= self._confirm:
            self._state = UtilizationState.ACTIVE
        elif self._state == UtilizationState.ACTIVE and (total - active_votes) >= self._confirm:
            self._state = UtilizationState.INACTIVE

        return self._state

    @property
    def state(self) -> UtilizationState:
        return self._state
