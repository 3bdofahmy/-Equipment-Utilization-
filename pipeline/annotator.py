"""
pipeline/annotator.py
──────────────────────
Draws masks, bounding boxes, state badges, and activity labels on frames.
"""

import cv2
import numpy as np
from tracking.track import Track

STATE_COLORS  = {"ACTIVE": (0, 200, 80), "INACTIVE": (50, 150, 255)}
ACTIVITY_COLORS = {
    "Digging":          (255, 100,   0),
    "Swinging/Loading": (255, 220,   0),
    "Dumping":          (180,  60, 255),
    "Traveling":        ( 50, 200, 255),
    "Waiting":          (150, 150, 150),
}
MASK_ALPHA = 0.35


def annotate(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    out = frame.copy()

    for track in tracks:
        color   = STATE_COLORS.get(track.utilization_state, (200, 200, 200))
        a_color = ACTIVITY_COLORS.get(track.activity, (200, 200, 200))
        x, y, w, h = track.bbox

        # Segmentation mask overlay
        if track.mask is not None:
            overlay = out.copy()
            overlay[track.mask] = color
            cv2.addWeighted(overlay, MASK_ALPHA, out, 1 - MASK_ALPHA, 0, out)

        # Bounding box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        # State badge (top)
        label = f"{track.track_id} | {track.utilization_state}"
        lw    = len(label) * 8 + 6
        cv2.rectangle(out, (x, y - 26), (x + lw, y), color, -1)
        cv2.putText(out, label, (x + 3, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Utilization % (above badge)
        util = f"Util: {track.utilization_pct:.1f}%"
        cv2.putText(out, util, (x + 3, y - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

        # Activity badge (bottom)
        verified = " ✓" if track.llm_verified else ""
        a_label  = f"{track.activity}{verified}"
        aw       = len(a_label) * 9 + 6
        cv2.rectangle(out, (x, y + h), (x + aw, y + h + 22), a_color, -1)
        cv2.putText(out, a_label, (x + 3, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Motion score bar
        bar_w = int(w * min(track.motion_score, 1.0))
        cv2.rectangle(out, (x, y + h + 24), (x + w, y + h + 30), (40, 40, 40), -1)
        cv2.rectangle(out, (x, y + h + 24), (x + bar_w, y + h + 30), color, -1)

    # Timestamp
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cv2.putText(out, ts, (8, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

    return out
