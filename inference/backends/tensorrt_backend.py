"""
inference/backends/tensorrt_backend.py
────────────────────────────────────────
TensorRT .engine inference backend.
Uses Ultralytics which wraps TensorRT transparently —
same API as PyTorch backend, ~3x faster.
"""

import numpy as np
from inference.backends.pytorch_backend import Detection
from core.logger import get_logger

log = get_logger("tensorrt_backend")


class TensorRTBackend:
    """
    Ultralytics handles TensorRT loading transparently.
    Just pass the .engine file path — everything else is identical
    to the PyTorch backend.
    """

    def __init__(self, model_path: str, confidence: float, iou_thresh: float,
                 input_size: int, device: str, classes: dict[int, str]):
        from ultralytics import YOLO
        log.info("Loading TensorRT engine: %s", model_path)
        self.model      = YOLO(model_path)
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.input_size = input_size
        self.device     = device
        self.classes    = classes
        log.info("TensorRT backend ready")

    def infer(self, frame: np.ndarray) -> list[Detection]:
        import cv2
        results = self.model(
            frame,
            conf    = self.confidence,
            iou     = self.iou_thresh,
            imgsz   = self.input_size,
            device  = self.device,
            verbose = False,
        )
        return self._parse(results[0], frame.shape)

    def _parse(self, result, frame_shape: tuple) -> list[Detection]:
        import cv2
        detections = []
        fh, fw     = frame_shape[:2]
        boxes      = result.boxes
        masks      = result.masks

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            if cls_id not in self.classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            mask = None
            if masks is not None and i < len(masks.data):
                raw  = masks.data[i].cpu().numpy()
                mask = cv2.resize(raw, (fw, fh)) > 0.5
            detections.append(Detection(
                bbox       = (x1, y1, x2 - x1, y2 - y1),
                class_id   = cls_id,
                class_name = self.classes[cls_id],
                confidence = float(box.conf[0]),
                mask       = mask,
            ))
        return detections

    @staticmethod
    def export(pt_path: str, output_path: str = None) -> str:
        """
        Export a .pt model to TensorRT .engine.
        Run this once before deployment.

            TensorRTBackend.export("weights/yolov8s-seg.pt")
        """
        from ultralytics import YOLO
        log.info("Exporting %s to TensorRT FP16 ...", pt_path)
        model  = YOLO(pt_path)
        result = model.export(format="engine", half=True, device=0)
        log.info("Engine saved: %s", result)
        return result
