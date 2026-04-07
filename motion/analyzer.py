"""
motion/analyzer.py
───────────────────
Motion analysis per track — 4 fixes from the notebook merged in:

  FIX 1 — CameraMotionCompensator
           ORB feature matching + homography warps the previous frame
           to compensate for camera shake before diffing.

  FIX 2 — Truck-specific thresholds
           Trucks use 2x PIXEL_DIFF_THRESH and 2x MOTION_THRESH, plus
           a longer confirm window. Prevents one noisy frame → "Moving".

  FIX 3 — zone_active dict added to MotionResult
           ActivityClassifier needs to know which zones fired.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from collections import deque

from core.config import MotionConfig
from core.enums import EquipmentType, MotionMethod
from core.logger import get_logger

log = get_logger("motion.analyzer")


@dataclass
class MotionResult:
    track_id:     str
    is_active:    bool
    motion_score: float
    method:       MotionMethod = MotionMethod.ZONE
    zone_scores:  dict = field(default_factory=dict)
    zone_active:  dict = field(default_factory=dict)   # FIX 3


# ── FIX 1: Camera Motion Compensator ─────────────────────────────────────────
class CameraMotionCompensator:
    """
    One global instance per FrameProcessor.
    Call once per frame: comp_diff, _ = compensator.get_compensated_diff(frame)
    Then pass comp_diff into every MotionAnalyzer.analyze().
    """

    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None

    def get_compensated_diff(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return None, gray

        compensated_diff: np.ndarray | None = None

        try:
            orb = cv2.ORB_create(500)
            kp1, des1 = orb.detectAndCompute(self._prev_gray, None)
            kp2, des2 = orb.detectAndCompute(gray, None)

            if (des1 is not None and des2 is not None
                    and len(kp1) > 10 and len(kp2) > 10):
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = sorted(bf.match(des1, des2),
                                 key=lambda m: m.distance)[:50]
                if len(matches) >= 4:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
                    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
                    if H is not None:
                        h, w = gray.shape
                        prev_warped = cv2.warpPerspective(
                            self._prev_gray, H, (w, h))
                        compensated_diff = cv2.absdiff(prev_warped, gray)
        except Exception as exc:
            log.debug(f"ORB camera compensation failed: {exc}")

        if compensated_diff is None:
            compensated_diff = cv2.absdiff(self._prev_gray, gray)

        self._prev_gray = gray
        return compensated_diff, gray


# ── MotionAnalyzer ────────────────────────────────────────────────────────────
class MotionAnalyzer:
    """Per-track motion analyzer. One instance per track_id."""

    ZONE_SPLITS = [
        ("HEAD",   0.00, 0.40),
        ("MIDDLE", 0.40, 0.75),
        ("FEET",   0.75, 1.00),
    ]

    def __init__(
        self,
        track_id:   str,
        config:     MotionConfig,
        class_name: str = "",
    ) -> None:
        self.track_id = track_id
        self.config   = config
        # FIX 2: trucks need stricter thresholds
        self.is_truck = (
            class_name in (EquipmentType.TRUCK.value, "truck")
        )
        self._score_history:  deque[float] = deque(maxlen=config.temporal_window)
        self._stable_state    = False
        self._consec_active   = 0
        self._consec_inactive = 0
        self._prev_gray: np.ndarray | None = None

    def analyze(
        self,
        frame:            np.ndarray,
        bbox:             tuple[int, int, int, int],
        mask:             np.ndarray | None = None,
        compensated_diff: np.ndarray | None = None,
    ) -> MotionResult:
        x, y, w, h = bbox
        x1 = max(0, x);           y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            return MotionResult(self.track_id, False, 0.0)

        # Path A: camera-compensated global diff (FIX 1 preferred)
        if compensated_diff is not None:
            roi = compensated_diff[y1:y2, x1:x2]
            if roi.size == 0:
                return MotionResult(self.track_id, False, 0.0)
            return self._zone_diff(roi)

        # Path B: per-track local diff (mask preferred, zone fallback)
        roi  = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return MotionResult(self.track_id, False, 0.0)
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        if mask is not None:
            return self._mask_diff(diff, mask, x1, y1, x2, y2)
        return self._zone_diff(diff)

    # ── Private ───────────────────────────────────────────────────────────────

    def _pix_thresh(self) -> int:
        return self.config.pixel_diff_thresh * (2 if self.is_truck else 1)

    def _mot_thresh(self) -> float:
        return self.config.motion_thresh * (2 if self.is_truck else 1)

    def _confirm_n(self) -> int:
        # FIX 2: trucks need more still frames to flip INACTIVE
        return self.config.active_confirm_frames + (2 if self.is_truck else 0)

    def _zone_diff(self, diff: np.ndarray) -> MotionResult:
        pix = self._pix_thresh()
        mot = self._mot_thresh()
        h   = diff.shape[0]

        zone_scores: dict[str, float] = {}
        zone_active: dict[str, bool]  = {}
        any_active = False

        for name, tf, bf in self.ZONE_SPLITS:
            zone = diff[int(h * tf):int(h * bf), :]
            if zone.size == 0:
                zone_scores[name] = 0.0
                zone_active[name] = False
                continue
            frac = np.sum(zone > pix) / zone.size
            zone_scores[name] = round(float(frac), 4)
            fired = frac >= mot
            zone_active[name] = fired
            if fired:
                any_active = True

        score  = float(max(zone_scores.values())) if zone_scores else 0.0
        stable = self._confirm(any_active)
        self._score_history.append(score)

        return MotionResult(
            track_id     = self.track_id,
            is_active    = stable,
            motion_score = round(float(np.mean(self._score_history)), 4),
            method       = MotionMethod.ZONE,
            zone_scores  = zone_scores,
            zone_active  = zone_active,
        )

    def _mask_diff(
        self, diff: np.ndarray, mask: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> MotionResult:
        pix = self._pix_thresh()
        mot = self._mot_thresh()
        roi_mask = mask[y1:y2, x1:x2]
        if roi_mask.shape != diff.shape:
            roi_mask = cv2.resize(
                roi_mask.astype(np.uint8),
                (diff.shape[1], diff.shape[0]),
            ).astype(bool)
        masked = diff[roi_mask]
        if masked.size == 0:
            return MotionResult(self.track_id, False, 0.0, MotionMethod.MASK)
        frac   = np.sum(masked > pix) / masked.size
        stable = self._confirm(frac >= mot)
        self._score_history.append(float(frac))
        return MotionResult(
            track_id     = self.track_id,
            is_active    = stable,
            motion_score = round(float(np.mean(self._score_history)), 4),
            method       = MotionMethod.MASK,
        )

    def _confirm(self, raw: bool) -> bool:
        n = self._confirm_n()
        if raw:
            self._consec_active   += 1
            self._consec_inactive  = 0
        else:
            self._consec_inactive += 1
            self._consec_active    = 0
        if self._consec_active   >= n: self._stable_state = True
        if self._consec_inactive >= n: self._stable_state = False
        return self._stable_state
