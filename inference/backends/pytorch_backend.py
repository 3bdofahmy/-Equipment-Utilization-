"""
inference/backends/pytorch_backend.py
───────────────────────────────────────
YOLOv8 .pt inference backend.
"""

import numpy as np
from dataclasses import dataclass, field
from core.logger import get_logger

log = get_logger("pytorch_backend")


@dataclass
class Detection:
    bbox:       tuple[int, int, int, int]   # x, y, w, h
    class_id:   int
    class_name: str
    confidence: float
    mask:       np.ndarray | None = None    # H x W binary mask


class PyTorchBackend:
    def __init__(self, model_path: str, confidence: float, iou_thresh: float,
                 input_size: int, device: str, classes: dict[int, str]):
        from ultralytics import YOLO
        log.info("Loading PyTorch model: %s", model_path)
        self.model      = YOLO(model_path)
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.input_size = input_size
        self.device     = device
        self.classes    = classes
        log.info("PyTorch backend ready")

    def infer(self, frame: np.ndarray) -> list[Detection]:
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
        detections = []
        fh, fw     = frame_shape[:2]

        boxes = result.boxes
        masks = result.masks

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            if cls_id not in self.classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            mask = None

            if masks is not None and i < len(masks.data):
                raw  = masks.data[i].cpu().numpy()
                import cv2
                mask = cv2.resize(raw, (fw, fh)) > 0.5

            detections.append(Detection(
                bbox       = (x1, y1, x2 - x1, y2 - y1),
                class_id   = cls_id,
                class_name = self.classes[cls_id],
                confidence = float(box.conf[0]),
                mask       = mask,
            ))
        return detections
