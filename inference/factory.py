"""
inference/factory.py
─────────────────────
ModelFactory — returns the correct backend based on config.

Usage:
    from inference.factory import ModelFactory
    from core.config import settings

    backend = ModelFactory.create(settings.inference, settings.classes)
    detections = backend.infer(frame)
"""

from core.config import InferenceConfig
from core.logger import get_logger

log = get_logger("inference.factory")

SUPPORTED_BACKENDS = ("pytorch", "onnx", "tensorrt")


class ModelFactory:

    @staticmethod
    def create(config: InferenceConfig, classes: dict[int, str]):
        """
        Returns the correct backend instance based on config.backend.

        config.backend values:
            "pytorch"   → PyTorchBackend   (.pt  file)
            "onnx"      → ONNXBackend      (.onnx file)
            "tensorrt"  → TensorRTBackend  (.engine file)
        """
        backend = config.backend.lower().strip()

        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Choose from: {SUPPORTED_BACKENDS}"
            )

        kwargs = dict(
            model_path  = config.model_path,
            confidence  = config.confidence,
            iou_thresh  = config.iou_thresh,
            input_size  = config.input_size,
            device      = config.device,
            classes     = classes,
        )

        log.info("Creating inference backend: %s", backend)

        if backend == "pytorch":
            from inference.backends.pytorch_backend import PyTorchBackend
            return PyTorchBackend(**kwargs)

        if backend == "onnx":
            from inference.backends.onnx_backend import ONNXBackend
            return ONNXBackend(**kwargs)

        if backend == "tensorrt":
            from inference.backends.tensorrt_backend import TensorRTBackend
            return TensorRTBackend(**kwargs)
