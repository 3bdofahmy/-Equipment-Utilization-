"""
core/config.py
──────────────
Central configuration — all values read from environment variables.
Uses Pydantic Settings for validation and type safety.
Copy .env.example to .env and fill in your values.
"""

from functools import lru_cache
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.enums import InferenceBackend, InferenceDevice, TrackerType, LLMProvider


class InferenceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    backend:     InferenceBackend = Field(InferenceBackend.PYTORCH, alias="INFERENCE_BACKEND")
    model_path:  str              = Field("weights/yolov8s-seg.pt",  alias="MODEL_PATH")
    confidence:  float            = Field(0.35,                       alias="DETECTION_CONF")
    iou_thresh:  float            = Field(0.45,                       alias="DETECTION_IOU")
    input_size:  int              = Field(640,                        alias="MODEL_INPUT_SIZE")
    device:      InferenceDevice  = Field(InferenceDevice.CUDA,       alias="INFERENCE_DEVICE")


class TrackingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    tracker:      TrackerType = Field(TrackerType.BOTSORT, alias="TRACKER")
    track_thresh: float       = Field(0.35,                alias="TRACK_THRESH")
    match_thresh: float       = Field(0.45,                alias="MATCH_THRESH")
    track_buffer: int         = Field(30,                  alias="TRACK_BUFFER")


class MotionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    motion_thresh:        float = Field(0.015, alias="MOTION_THRESH")
    pixel_diff_thresh:    int   = Field(25,    alias="PIXEL_DIFF_THRESH")
    active_confirm_frames: int   = Field(2,     alias="ACTIVE_CONFIRM_FRAMES")
    temporal_window:       int   = Field(5,     alias="TEMPORAL_WINDOW")
    llm_confidence_thresh: float = Field(0.65,  alias="LLM_CONF_THRESH")
    # FIX 2 — truck needs more still frames before flipping to INACTIVE
    truck_still_confirm:   int   = Field(5,     alias="TRUCK_STILL_CONFIRM")
    # FIX 4 — max pixel distance: truck center ↔ dumping arm center
    truck_loading_dist:    float = Field(150.0, alias="TRUCK_LOADING_DIST")


class VideoConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    source:      str = Field("data/input.mp4", alias="VIDEO_SOURCE")
    process_fps: int = Field(10,               alias="PROCESS_FPS")
    output_dir:  str = Field("data/output",    alias="OUTPUT_DIR")


class KafkaConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    bootstrap_servers: str = Field("localhost:9092",       alias="KAFKA_BOOTSTRAP_SERVERS")
    topic_detections:  str = Field("equipment.detections", alias="KAFKA_TOPIC_DETECTIONS")
    topic_frames:      str = Field("equipment.frames",     alias="KAFKA_TOPIC_FRAMES")
    consumer_group:    str = Field("analytics-group",      alias="KAFKA_CONSUMER_GROUP")


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    host:     str = Field("localhost",    alias="DB_HOST")
    port:     int = Field(5432,           alias="DB_PORT")
    name:     str = Field("construction_cv", alias="DB_NAME")
    user:     str = Field("cvuser",       alias="DB_USER")
    password: str = Field("cvpassword",  alias="DB_PASSWORD")

    @computed_field
    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @computed_field
    @property
    def sync_url(self) -> str:
        """Sync URL for Alembic migrations."""
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    provider:    LLMProvider = Field(LLMProvider.OPENAI, alias="LLM_PROVIDER")
    model:       str         = Field("gpt-4o-mini",      alias="LLM_MODEL")
    api_key:     str         = Field("",                 alias="LLM_API_KEY")
    base_url:    str         = Field("",                 alias="LLM_BASE_URL")
    temperature: float       = Field(0.1,                alias="LLM_TEMP")
    max_tokens:  int         = Field(256,                alias="LLM_MAX_TOKENS")
    enabled:     bool        = Field(True,               alias="LLM_ENABLED")


class APIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    host:         str      = Field("0.0.0.0",              alias="API_HOST")
    port:         int      = Field(8000,                   alias="API_PORT")
    debug:        bool     = Field(False,                  alias="API_DEBUG")
    cors_origins: list[str] = Field(["http://localhost:3000"], alias="CORS_ORIGINS")


class AppConfig(BaseSettings):
    """Root config — loads all sub-configs from the same env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    tracking:  TrackingConfig  = Field(default_factory=TrackingConfig)
    motion:    MotionConfig    = Field(default_factory=MotionConfig)
    video:     VideoConfig     = Field(default_factory=VideoConfig)
    kafka:     KafkaConfig     = Field(default_factory=KafkaConfig)
    database:  DatabaseConfig  = Field(default_factory=DatabaseConfig)
    llm:       LLMConfig       = Field(default_factory=LLMConfig)
    api:       APIConfig       = Field(default_factory=APIConfig)


@lru_cache
def get_settings() -> AppConfig:
    """Cached settings singleton — call get_settings() anywhere."""
    return AppConfig()


# Convenient alias
settings = get_settings()
