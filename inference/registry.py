"""
inference/registry.py
──────────────────────
Singleton that holds the loaded model and tracks
performance counters for the /model/* API endpoints.
"""

import time
from dataclasses import dataclass, field
from core.logger import get_logger

log = get_logger("inference.registry")


@dataclass
class PerformanceStats:
    frames_processed:   int   = 0
    total_inference_ms: float = 0.0
    last_inference_at:  float = 0.0
    last_frame_count:   int   = 0
    started_at:         float = field(default_factory=time.time)

    @property
    def avg_inference_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return round(self.total_inference_ms / self.frames_processed, 2)

    @property
    def avg_fps(self) -> float:
        if self.avg_inference_ms == 0:
            return 0.0
        return round(1000 / self.avg_inference_ms, 1)

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self.started_at, 1)


class ModelRegistry:
    """
    Loaded once at startup. All modules reference this singleton.
    """
    _backend         = None
    _config          = None
    _classes: dict   = {}
    _stats           = PerformanceStats()

    @classmethod
    def load(cls, config, classes: dict[int, str]) -> None:
        from inference.factory import ModelFactory
        cls._config  = config
        cls._classes = classes
        cls._backend = ModelFactory.create(config, classes)
        cls._stats   = PerformanceStats()
        log.info("Model loaded — backend: %s", config.backend)

    @classmethod
    def infer(cls, frame):
        if cls._backend is None:
            raise RuntimeError("ModelRegistry not loaded. Call ModelRegistry.load() first.")
        t0      = time.perf_counter()
        result  = cls._backend.infer(frame)
        elapsed = (time.perf_counter() - t0) * 1000

        cls._stats.frames_processed   += 1
        cls._stats.total_inference_ms += elapsed
        cls._stats.last_inference_at   = time.time()
        cls._stats.last_frame_count    = len(result)
        return result

    @classmethod
    def get_info(cls) -> dict:
        if cls._config is None:
            return {}
        import os
        return {
            "model_name":   os.path.splitext(os.path.basename(cls._config.model_path))[0],
            "backend":      cls._config.backend,
            "weights_file": cls._config.model_path,
            "classes":      cls._classes,
            "input_size":   cls._config.input_size,
            "task":         "segmentation",
        }

    @classmethod
    def get_performance(cls) -> dict:
        s = cls._stats
        info = {
            "avg_inference_ms":  s.avg_inference_ms,
            "avg_fps":           s.avg_fps,
            "frames_processed":  s.frames_processed,
            "uptime_seconds":    s.uptime_seconds,
        }
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=2,
            ).strip().split(",")
            info["gpu_name"]             = out[0].strip()
            info["gpu_memory_used_mb"]   = int(out[1].strip())
            info["gpu_memory_total_mb"]  = int(out[2].strip())
        except Exception:
            pass
        return info

    @classmethod
    def get_status(cls) -> dict:
        import time as t
        return {
            "status":              "running" if cls._backend else "not_loaded",
            "backend":             cls._config.backend if cls._config else None,
            "last_inference_at":   cls._stats.last_inference_at,
            "detections_last_frame": cls._stats.last_frame_count,
        }
