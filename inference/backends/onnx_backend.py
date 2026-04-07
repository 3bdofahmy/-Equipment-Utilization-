"""
inference/backends/onnx_backend.py
────────────────────────────────────
ONNX Runtime inference backend.
"""

import cv2
import numpy as np
from inference.backends.pytorch_backend import Detection
from core.logger import get_logger

log = get_logger("onnx_backend")


class ONNXBackend:
    def __init__(self, model_path: str, confidence: float, iou_thresh: float,
                 input_size: int, device: str, classes: dict[int, str]):
        import onnxruntime as ort
        log.info("Loading ONNX model: %s", model_path)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda" else ["CPUExecutionProvider"]
        )
        self.session    = ort.InferenceSession(model_path, providers=providers)
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.input_size = input_size
        self.classes    = classes
        self.input_name = self.session.get_inputs()[0].name
        log.info("ONNX backend ready — provider: %s",
                 self.session.get_providers()[0])

    def infer(self, frame: np.ndarray) -> list[Detection]:
        blob, scale, pad = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs, frame.shape, scale, pad)

    # ── Private ──────────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray):
        h, w  = frame.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (nw, nh))

        pad_h = self.input_size - nh
        pad_w = self.input_size - nw
        top, left = pad_h // 2, pad_w // 2
        padded = cv2.copyMakeBorder(resized, top, pad_h - top,
                                    left, pad_w - left,
                                    cv2.BORDER_CONSTANT, value=114)
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis]
        return blob, scale, (top, left)

    def _postprocess(self, outputs, frame_shape, scale, pad) -> list[Detection]:
        # YOLOv8 ONNX output: [1, 116, 8400] for seg
        # First 4 = xywh, next 80 = class scores, next 32 = mask coefficients
        preds = outputs[0][0].T   # [8400, 116]
        fh, fw = frame_shape[:2]
        detections = []

        scores = preds[:, 4:4 + len(self.classes)]
        class_ids = np.argmax(scores, axis=1)
        confs     = scores[np.arange(len(scores)), class_ids]

        keep = confs >= self.confidence
        preds     = preds[keep]
        class_ids = class_ids[keep]
        confs     = confs[keep]

        if len(preds) == 0:
            return detections

        # xywh → xyxy in original image coords
        boxes = preds[:, :4].copy()
        top, left = pad
        boxes[:, 0] = (boxes[:, 0] - left) / scale
        boxes[:, 1] = (boxes[:, 1] - top)  / scale
        boxes[:, 2] /= scale
        boxes[:, 3] /= scale

        # NMS
        xyxy = np.column_stack([
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
        ])
        indices = cv2.dnn.NMSBoxes(
            xyxy.tolist(), confs.tolist(),
            self.confidence, self.iou_thresh
        )
        if len(indices) == 0:
            return detections

        for idx in indices.flatten():
            cls_id = int(class_ids[idx])
            if cls_id not in self.classes:
                continue
            x1 = max(0, int(xyxy[idx, 0]))
            y1 = max(0, int(xyxy[idx, 1]))
            x2 = min(fw, int(xyxy[idx, 2]))
            y2 = min(fh, int(xyxy[idx, 3]))
            detections.append(Detection(
                bbox       = (x1, y1, x2 - x1, y2 - y1),
                class_id   = cls_id,
                class_name = self.classes[cls_id],
                confidence = float(confs[idx]),
                mask       = None,   # full seg mask decoding omitted for brevity
            ))
        return detections
