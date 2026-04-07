"""
api/schemas.py
───────────────
Pydantic request / response models for all API endpoints.
Enum fields give you validated values and great OpenAPI docs.
"""

from datetime import datetime
from pydantic import BaseModel

from core.enums import Activity, EquipmentType, HealthStatus, UtilizationState


# ── Equipment ─────────────────────────────────────────────────────────────────
class EquipmentOut(BaseModel):
    equipment_id:   str
    equipment_type: EquipmentType
    first_seen:     datetime
    last_seen:      datetime

    model_config = {"from_attributes": True}


# ── Utilization ───────────────────────────────────────────────────────────────
class UtilizationOut(BaseModel):
    equipment_id:       str
    equipment_type:     EquipmentType
    total_active_sec:   float
    total_inactive_sec: float
    utilization_pct:    float
    last_activity:      Activity | None
    last_state:         UtilizationState | None
    updated_at:         datetime

    model_config = {"from_attributes": True}


class UtilizationHistoryPoint(BaseModel):
    time:               datetime
    utilization_state:  UtilizationState
    activity:           Activity
    motion_score:       float


# ── Detections ────────────────────────────────────────────────────────────────
class DetectionOut(BaseModel):
    id:                 int
    time:               datetime
    equipment_id:       str
    equipment_type:     EquipmentType
    utilization_state:  UtilizationState
    activity:           Activity
    confidence:         float
    motion_score:       float
    llm_verified:       bool
    frame_index:        int | None

    model_config = {"from_attributes": True}


# ── Model info ────────────────────────────────────────────────────────────────
class ModelInfoOut(BaseModel):
    model_name:   str
    backend:      str
    weights_file: str
    classes:      dict
    input_size:   int
    task:         str


class ModelPerformanceOut(BaseModel):
    avg_inference_ms:    float
    avg_fps:             float
    frames_processed:    int
    uptime_seconds:      float
    gpu_name:            str | None = None
    gpu_memory_used_mb:  int | None = None
    gpu_memory_total_mb: int | None = None


class ModelStatusOut(BaseModel):
    status:                str
    backend:               str | None
    last_inference_at:     float | None
    detections_last_frame: int


# ── Health ────────────────────────────────────────────────────────────────────
class HealthOut(BaseModel):
    status:   HealthStatus
    database: str
    kafka:    str
    model:    str


# ── Stream ────────────────────────────────────────────────────────────────────
class LatestFrameOut(BaseModel):
    timestamp:   str
    frame_index: int
    jpeg_b64:    str
    track_count: int
