"""
core/enums.py
─────────────
All Enum definitions for the project.
Using enums instead of raw strings prevents typos and makes autocomplete work.

Usage:
    from core.enums import EquipmentType, UtilizationState, Activity

    state = UtilizationState.ACTIVE
    state.value  # → "ACTIVE"
"""

from enum import Enum


class InferenceBackend(str, Enum):
    PYTORCH   = "pytorch"
    ONNX      = "onnx"
    TENSORRT  = "tensorrt"


class InferenceDevice(str, Enum):
    CUDA = "cuda"
    CPU  = "cpu"


class TrackerType(str, Enum):
    BOTSORT    = "botsort"
    BYTETRACK  = "bytetrack"


class LLMProvider(str, Enum):
    OPENAI  = "openai"
    GEMINI  = "gemini"
    GROQ    = "groq"
    OLLAMA  = "ollama"


class EquipmentType(str, Enum):
    EXCAVATOR     = "excavator"
    EXCAVATOR_ARM = "excavator_arm"
    TRUCK         = "truck"

    @classmethod
    def from_class_id(cls, class_id: int) -> "EquipmentType":
        """Map YOLO class index → EquipmentType."""
        mapping = {
            0: cls.EXCAVATOR,
            1: cls.EXCAVATOR_ARM,
            2: cls.TRUCK,
        }
        return mapping[class_id]

    @classmethod
    def class_map(cls) -> dict[int, str]:
        """Return {class_id: value} dict — used by ModelRegistry."""
        return {
            0: cls.EXCAVATOR.value,
            1: cls.EXCAVATOR_ARM.value,
            2: cls.TRUCK.value,
        }


class UtilizationState(str, Enum):
    ACTIVE   = "ACTIVE"
    INACTIVE = "INACTIVE"


class Activity(str, Enum):
    DIGGING         = "Digging"
    SWINGING_LOADING = "Swinging/Loading"
    DUMPING         = "Dumping"
    TRAVELING       = "Traveling"
    WAITING         = "Waiting"


class MotionMethod(str, Enum):
    MASK    = "mask"
    OPTICAL = "optical"
    BBOX    = "bbox"


class HealthStatus(str, Enum):
    OK       = "ok"
    DEGRADED = "degraded"
    DOWN     = "down"
