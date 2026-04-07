"""
streaming/schemas.py
─────────────────────
Kafka payload schemas — dataclasses + JSON serialization.
Uses Enum values (.value) for JSON serialization.
"""

import json
from dataclasses import dataclass, asdict

from core.enums import Activity, EquipmentType, UtilizationState


@dataclass
class DetectionPayload:
    timestamp:          str
    frame_index:        int
    video_source:       str
    equipment_id:       str
    equipment_type:     str              # EquipmentType.value
    utilization_state:  str              # UtilizationState.value
    activity:           str              # Activity.value
    confidence:         float
    motion_score:       float
    llm_verified:       bool
    bbox_x:             int
    bbox_y:             int
    bbox_w:             int
    bbox_h:             int
    active_time_sec:    float
    inactive_time_sec:  float
    total_time_sec:     float
    utilization_pct:    float

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_track(
        cls,
        track,
        frame_index:  int,
        timestamp:    str,
        video_source: str,
    ) -> "DetectionPayload":
        # Accept both Enum and raw string from track
        eq_type = track.class_name
        if isinstance(eq_type, EquipmentType):
            eq_type = eq_type.value

        util_state = track.utilization_state
        if isinstance(util_state, UtilizationState):
            util_state = util_state.value

        activity = track.activity
        if isinstance(activity, Activity):
            activity = activity.value

        return cls(
            timestamp         = timestamp,
            frame_index       = frame_index,
            video_source      = video_source,
            equipment_id      = track.track_id,
            equipment_type    = eq_type,
            utilization_state = util_state,
            activity          = activity,
            confidence        = round(track.confidence, 4),
            motion_score      = round(track.motion_score, 4),
            llm_verified      = track.llm_verified,
            bbox_x            = track.bbox[0],
            bbox_y            = track.bbox[1],
            bbox_w            = track.bbox[2],
            bbox_h            = track.bbox[3],
            active_time_sec   = round(track.active_time, 2),
            inactive_time_sec = round(track.inactive_time, 2),
            total_time_sec    = round(track.total_time, 2),
            utilization_pct   = track.utilization_pct,
        )


@dataclass
class FramePayload:
    timestamp:   str
    frame_index: int
    jpeg_b64:    str
    track_count: int

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode()
