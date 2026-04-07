"""
tracking/tracker.py
────────────────────
Wraps Ultralytics BoT-SORT / ByteTrack.
Outputs stable Track objects with persistent IDs.
"""

import numpy as np
from tracking.track import Track
from core.logger import get_logger

log = get_logger("tracker")


class Tracker:
    def __init__(self, tracker_type: str, config):
        """
        tracker_type: "botsort" | "bytetrack"
        config: TrackingConfig
        """
        self.tracker_type = tracker_type.lower()
        self.config       = config
        self._tracks: dict[str, Track] = {}
        self._tracker     = None
        self._init_tracker()

    def _init_tracker(self):
        try:
            from ultralytics.trackers.bot_sort import BOTSORT
            from ultralytics.trackers.byte_tracker import BYTETracker

            class _Args:
                track_high_thresh = self.config.track_thresh
                track_low_thresh  = 0.1
                new_track_thresh  = self.config.track_thresh
                track_buffer      = self.config.track_buffer
                match_thresh      = self.config.match_thresh
                with_reid         = self.tracker_type == "botsort"
                proximity_thresh  = 0.5
                appearance_thresh = 0.25
                cmc_method        = "sparseOptFlow"
                frame_rate        = 10

            args = _Args()
            self._tracker = (
                BOTSORT(args, frame_rate=10)
                if self.tracker_type == "botsort"
                else BYTETracker(args, frame_rate=10)
            )
            log.info("Tracker initialized: %s", self.tracker_type)
        except Exception as e:
            log.warning("Ultralytics tracker init failed: %s — using IoU fallback", e)
            self._tracker = None

    def update(
        self,
        detections: list,      # list[Detection] from inference backend
        frame: np.ndarray,
        classes: dict[int, str],
    ) -> list[Track]:
        """
        Match detections to existing tracks.
        Returns updated list of Track objects.
        """
        if not detections:
            for t in self._tracks.values():
                t.tick()
            return list(self._tracks.values())

        if self._tracker is not None:
            return self._update_ultralytics(detections, frame, classes)
        return self._update_iou(detections, classes)

    def get_track(self, track_id: str) -> Track | None:
        return self._tracks.get(track_id)

    # ── Ultralytics tracker ───────────────────────────────────────────────────

    def _update_ultralytics(self, detections, frame, classes) -> list[Track]:
        import torch

        # Build [x1,y1,x2,y2,conf,cls] tensor
        rows = []
        for d in detections:
            x, y, w, h = d.bbox
            rows.append([x, y, x + w, y + h, d.confidence, d.class_id])

        det_tensor = torch.tensor(rows, dtype=torch.float32)

        try:
            online = self._tracker.update(det_tensor, frame)
        except Exception as e:
            log.warning("Tracker update error: %s", e)
            return self._update_iou(detections, classes)

        matched_ids = set()
        for t in online:
            tid      = f"EQ-{int(t.track_id):04d}"
            cls_id   = int(t.cls) if hasattr(t, "cls") else 0
            cls_name = classes.get(cls_id, "unknown")
            x1, y1, x2, y2 = [int(v) for v in t.tlbr]

            # Find matching detection for mask
            mask = self._match_mask(detections, (x1, y1, x2 - x1, y2 - y1))

            if tid in self._tracks:
                track = self._tracks[tid]
                track.tick()
                track.bbox       = (x1, y1, x2 - x1, y2 - y1)
                track.mask       = mask
                track.confidence = float(t.score) if hasattr(t, "score") else 0.0
            else:
                track = Track(
                    track_id   = tid,
                    class_id   = cls_id,
                    class_name = cls_name,
                    bbox       = (x1, y1, x2 - x1, y2 - y1),
                    mask       = mask,
                    confidence = float(t.score) if hasattr(t, "score") else 0.0,
                )
                self._tracks[tid] = track

            matched_ids.add(tid)

        # Tick unmatched tracks
        for tid, track in self._tracks.items():
            if tid not in matched_ids:
                track.tick()

        return list(self._tracks.values())

    # ── IoU fallback tracker ──────────────────────────────────────────────────

    _id_seq = 0

    def _update_iou(self, detections, classes) -> list[Track]:
        matched_ids = set()

        for det in detections:
            best_id  = self._match_iou(det.bbox)
            if best_id:
                track = self._tracks[best_id]
                track.tick()
                track.bbox       = det.bbox
                track.mask       = det.mask
                track.confidence = det.confidence
            else:
                Tracker._id_seq += 1
                tid   = f"EQ-{Tracker._id_seq:04d}"
                track = Track(
                    track_id   = tid,
                    class_id   = det.class_id,
                    class_name = det.class_name,
                    bbox       = det.bbox,
                    mask       = det.mask,
                    confidence = det.confidence,
                )
                self._tracks[tid] = track
                best_id = tid

            matched_ids.add(best_id)

        for tid, track in self._tracks.items():
            if tid not in matched_ids:
                track.tick()

        return list(self._tracks.values())

    def _match_iou(self, bbox, threshold: float = 0.30) -> str | None:
        best_iou = 0.0
        best_id  = None
        for tid, track in self._tracks.items():
            iou = _iou(bbox, track.bbox)
            if iou > best_iou and iou >= threshold:
                best_iou = iou
                best_id  = tid
        return best_id

    @staticmethod
    def _match_mask(detections, bbox) -> "np.ndarray | None":
        best_iou = 0.0
        best_mask = None
        for d in detections:
            iou = _iou(bbox, d.bbox)
            if iou > best_iou:
                best_iou  = iou
                best_mask = d.mask
        return best_mask


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter    = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union
