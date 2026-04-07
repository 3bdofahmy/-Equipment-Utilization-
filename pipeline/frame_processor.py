"""
pipeline/frame_processor.py
─────────────────────────────
Orchestrates: detect → track → motion → activity → state
Returns a FrameResult — knows nothing about Kafka or DB.

Changes vs original:
  - CameraMotionCompensator runs once per frame (FIX 1)
  - MotionAnalyzer.analyze() receives compensated_diff (FIX 1)
  - MotionAnalyzer created with class_name for truck thresholds (FIX 2)
  - zone_active passed through MotionResult (FIX 3)
  - ActivityClassifier.classify() gets class_name + near_dumping_arm (FIX 4)
  - Arm → excavator association uses IoU + proximity of dumping arms for trucks
"""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from core.config import AppConfig
from core.enums import Activity, EquipmentType, UtilizationState
from core.logger import get_logger
from inference.registry import ModelRegistry
from tracking.track import Track
from motion.analyzer import CameraMotionCompensator, MotionAnalyzer, MotionResult
from motion.activity import ActivityClassifier
from motion.state import StateMachine
from motion.llm_verifier import LLMVerifier

log = get_logger("frame_processor")

# Minimum IoU between arm bbox and excavator bbox to associate them
ARM_EXCAVATOR_IOU = 0.15


@dataclass
class FrameResult:
    frame_index: int
    timestamp:   str
    tracks:      list[Track]
    raw_frame:   np.ndarray


class FrameProcessor:

    def __init__(self, config: AppConfig, tracker):
        self.config  = config
        self.tracker = tracker

        # FIX 1: one global camera compensator
        self._compensator = CameraMotionCompensator()

        # Per-track stateful objects
        self._analyzers:   dict[str, MotionAnalyzer]    = {}
        self._classifiers: dict[str, ActivityClassifier] = {}
        self._states:      dict[str, StateMachine]      = {}
        self._histories:   dict[str, deque]             = {}
        self._class_names: dict[str, str]               = {}

        # LLM verifier (optional)
        self._verifier = LLMVerifier(config.llm) if config.llm.enabled else None

        log.info("FrameProcessor ready")

    def process(self, frame_index: int, frame: np.ndarray) -> FrameResult:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()

        # ── FIX 1: compute compensated diff once for the whole frame ──────
        comp_diff, _ = self._compensator.get_compensated_diff(frame)

        # ── 1. Detection ──────────────────────────────────────────────────
        detections = ModelRegistry.infer(frame)

        # ── 2. Tracking ───────────────────────────────────────────────────
        tracks = self.tracker.update(detections, frame, self.config.classes)

        # ── 3. Motion analysis (pass 1) ───────────────────────────────────
        motion_results: dict[str, MotionResult] = {}
        arm_tracks: list[Track] = []

        for track in tracks:
            tid        = track.track_id
            class_name = track.class_name

            # Lazy-init per-track objects
            if tid not in self._analyzers:
                # FIX 2: pass class_name so truck gets stricter thresholds
                self._analyzers[tid]   = MotionAnalyzer(
                    tid, self.config.motion, class_name=class_name
                )
                self._classifiers[tid] = ActivityClassifier()
                self._states[tid]      = StateMachine(
                    self.config.motion.active_confirm_frames
                )
                self._histories[tid]   = deque(maxlen=20)
                self._class_names[tid] = class_name

            # FIX 1: pass compensated_diff instead of doing per-track diff
            motion = self._analyzers[tid].analyze(
                frame, track.bbox,
                mask             = track.mask,
                compensated_diff = comp_diff,
            )
            motion_results[tid] = motion

            if class_name in (EquipmentType.EXCAVATOR_ARM.value, "excavator_arm"):
                arm_tracks.append(track)

        # ── 4. Find dumping arms (for truck association) ──────────────────
        dumping_arm_bboxes: list[tuple] = []
        for arm in arm_tracks:
            mr = motion_results.get(arm.track_id)
            # Arm is dumping if zone_active["FEET"] is dominant
            if mr and mr.zone_active.get("FEET", False):
                foot_score = mr.zone_scores.get("FEET", 0.0)
                other_max  = max(
                    mr.zone_scores.get("HEAD", 0.0),
                    mr.zone_scores.get("MIDDLE", 0.0),
                )
                if foot_score >= other_max:
                    dumping_arm_bboxes.append(arm.bbox)

        # ── 5. Activity + state (pass 2) ──────────────────────────────────
        for track in tracks:
            tid        = track.track_id
            class_name = self._class_names.get(tid, track.class_name)
            motion     = motion_results[tid]

            # ── Excavator body: use arm's motion result if arm overlaps ───
            if class_name in (EquipmentType.EXCAVATOR.value, "excavator"):
                best_arm_mr = self._best_arm_for_excavator(
                    track, arm_tracks, motion_results
                )
                if best_arm_mr is not None:
                    motion = best_arm_mr   # arm drives excavator state

            # ── Truck: check if near a dumping arm ────────────────────────
            near_dumping = False
            if class_name in (EquipmentType.TRUCK.value, "truck"):
                near_dumping = self._truck_near_dumping(
                    track.bbox, dumping_arm_bboxes
                )

            # FIX 4: pass class_name + near_dumping_arm to classifier
            rule_activity, rule_conf = self._classifiers[tid].classify(
                motion,
                class_name       = class_name,
                near_dumping_arm = near_dumping,
            )

            # Optional LLM verification
            activity     = rule_activity
            llm_verified = False
            if (self._verifier
                    and rule_conf < self.config.motion.llm_confidence_thresh
                    and motion.is_active):
                activity, llm_verified = self._verifier.verify(
                    equipment_type    = class_name,
                    track_id          = tid,
                    rule_prediction   = rule_activity,
                    rule_confidence   = rule_conf,
                    motion_score      = motion.motion_score,
                    zone_scores       = motion.zone_scores,
                    recent_activities = list(self._histories[tid]),
                    frame_index       = frame_index,
                )

            state = self._states[tid].update(motion.is_active)

            track.utilization_state = state
            track.activity          = activity
            track.motion_score      = motion.motion_score
            track.llm_verified      = llm_verified

            self._histories[tid].append(activity)

        return FrameResult(
            frame_index = frame_index,
            timestamp   = ts,
            tracks      = tracks,
            raw_frame   = frame,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        """Compute IoU of two (x, y, w, h) bboxes."""
        ax1, ay1 = a[0], a[1]
        ax2, ay2 = a[0] + a[2], a[1] + a[3]
        bx1, by1 = b[0], b[1]
        bx2, by2 = b[0] + b[2], b[1] + b[3]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center_dist(a: tuple, b: tuple) -> float:
        cx_a = a[0] + a[2] / 2;  cy_a = a[1] + a[3] / 2
        cx_b = b[0] + b[2] / 2;  cy_b = b[1] + b[3] / 2
        return float(np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2))

    def _best_arm_for_excavator(
        self,
        excavator:     Track,
        arm_tracks:    list[Track],
        motion_results: dict[str, MotionResult],
    ) -> MotionResult | None:
        """
        Find the arm track with highest IoU overlap with this excavator.
        Returns its MotionResult if IoU >= threshold, else None.
        """
        best_iou = 0.0
        best_mr  = None
        for arm in arm_tracks:
            iou = self._iou(excavator.bbox, arm.bbox)
            if iou > best_iou:
                best_iou = iou
                best_mr  = motion_results.get(arm.track_id)
        if best_iou >= ARM_EXCAVATOR_IOU:
            return best_mr
        return None

    def _truck_near_dumping(
        self,
        truck_bbox:         tuple,
        dumping_arm_bboxes: list[tuple],
    ) -> bool:
        """True if any dumping arm center is within truck_loading_dist pixels."""
        dist = self.config.motion.truck_loading_dist
        return any(
            self._center_dist(truck_bbox, arm_bbox) <= dist
            for arm_bbox in dumping_arm_bboxes
        )
