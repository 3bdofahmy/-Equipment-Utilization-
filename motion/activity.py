"""
motion/activity.py
───────────────────
Rule-based activity classifier — FIX 4 from the notebook.

Key fix: priority order in classify() was wrong.
Old code: (head or middle) AND feet → Dumping
  → fired too often because head/middle are slightly active almost always.

New code: feet_score must be the DOMINANT zone to trigger Dumping.
  All other activities are checked first.

Also added class-specific paths:
  - excavator_arm  → full zone logic
  - excavator body → Active / Waiting (arm result injected by FrameProcessor)
  - truck          → Moving / Loading / Waiting
"""

from collections import Counter, deque

from core.enums import Activity, EquipmentType, MotionMethod
from motion.analyzer import MotionResult


ACTIVITIES = list(Activity)


class ActivityClassifier:
    """One instance per track_id."""

    def __init__(self, window: int = 12):
        self._history: deque[Activity] = deque(maxlen=window)

    def classify(
        self,
        result:          MotionResult,
        class_name:      str  = "",
        near_dumping_arm: bool = False,   # for trucks: is a dumping arm nearby?
    ) -> tuple[Activity, float]:
        """
        Returns (Activity, confidence).

        Pass class_name so the classifier can use class-specific logic.
        Pass near_dumping_arm=True when a dumping arm is within proximity
        of the truck (FrameProcessor detects this).
        """

        # ── excavator_arm — full zone logic (FIX 4) ──────────────────────
        if class_name in (EquipmentType.EXCAVATOR_ARM.value, "excavator_arm"):
            return self._classify_arm(result)

        # ── excavator body — driven by arm association in FrameProcessor ──
        if class_name in (EquipmentType.EXCAVATOR.value, "excavator"):
            act = Activity.WAITING if not result.is_active else Activity.WAITING
            # Real state is injected by FrameProcessor via arm IoU association.
            # Fallback here in case no arm is found.
            act = Activity.WAITING
            if result.is_active:
                act = Activity.DIGGING   # generic active label for body
            self._history.append(act)
            return act, 1.0

        # ── truck ─────────────────────────────────────────────────────────
        if class_name in (EquipmentType.TRUCK.value, "truck"):
            return self._classify_truck(result, near_dumping_arm)

        # ── generic fallback ──────────────────────────────────────────────
        act = Activity.WAITING if not result.is_active else Activity.DIGGING
        self._history.append(act)
        return act, 1.0

    # ── Private ───────────────────────────────────────────────────────────────

    def _classify_arm(self, result: MotionResult) -> tuple[Activity, float]:
        """
        FIX 4 — correct priority order for excavator_arm zone classification.

        Priority:
          1. Nothing active           → Waiting
          2. feet_score is the MAX    → Dumping  (bucket moving down)
          3. head + middle both fire  → Digging  (boom + arm extend)
          4. head only                → Swinging/Loading
          5. middle only              → Traveling (arm retract)
          6. feet fires (not dominant)→ Dumping
          7. default                  → Digging
        """
        if not result.is_active:
            self._history.append(Activity.WAITING)
            return Activity.WAITING, 1.0

        z = result.zone_scores
        feet_score   = z.get("FEET",   0.0)
        head_score   = z.get("HEAD",   0.0)
        middle_score = z.get("MIDDLE", 0.0)
        max_score    = max(feet_score, head_score, middle_score)

        # Use zone_active for boolean checks (respects per-class threshold)
        za     = result.zone_active
        head   = za.get("HEAD",   False)
        middle = za.get("MIDDLE", False)
        feet   = za.get("FEET",   False)

        if max_score <= 0:
            act = Activity.WAITING
        elif feet_score == max_score and feet:
            # Bucket is most active zone → Dumping
            act = Activity.DUMPING
        elif head and middle:
            act = Activity.DIGGING
        elif head and not middle:
            act = Activity.SWINGING_LOADING
        elif middle and not head:
            act = Activity.TRAVELING
        elif feet:
            act = Activity.DUMPING
        else:
            act = Activity.DIGGING

        self._history.append(act)

        # Majority vote for temporal stability
        if len(self._history) >= 5:
            counts = Counter(self._history)
            act    = counts.most_common(1)[0][0]
            conf   = counts[act] / len(self._history)
        else:
            conf = 1.0

        return act, round(conf, 2)

    def _classify_truck(
        self, result: MotionResult, near_dumping_arm: bool
    ) -> tuple[Activity, float]:
        if result.is_active:
            act = Activity.TRAVELING       # trucks move, not "digging"
        elif near_dumping_arm:
            act = Activity.DUMPING         # receiving load from arm
        else:
            act = Activity.WAITING

        self._history.append(act)

        # Majority vote — prevents flicker
        if len(self._history) >= 5:
            counts = Counter(self._history)
            act    = counts.most_common(1)[0][0]

        return act, 1.0
