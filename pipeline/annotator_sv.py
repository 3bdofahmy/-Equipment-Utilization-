"""
pipeline/annotator_sv.py
─────────────────────────
Annotation using Supervision library for detections and visualization.
Optional integration alongside existing cv2-based annotator.

Example usage:
    import supervision as sv
    image = cv2.imread("frame.jpg")
    detections = sv.Detections(
        xyxy=[[10, 20, 100, 200], [150, 50, 300, 250]],
        confidence=[0.95, 0.87],
        class_id=[0, 1]
    )
    
    color_annotator = sv.ColorAnnotator()
    annotated_frame = color_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
"""

import cv2
import numpy as np
import supervision as sv
from typing import Optional
from tracking.track import Track


def convert_tracks_to_detections(tracks: list[Track]) -> sv.Detections:
    """
    Convert Track objects to supervision Detections format.
    
    Args:
        tracks: List of Track objects from tracking.track
        
    Returns:
        sv.Detections object with bounding boxes, confidence, and class_id
    """
    if not tracks:
        return sv.Detections.empty()
    
    # Extract bounding boxes in xyxy format
    xyxy = []
    confidence = []
    class_id = []
    
    for track in tracks:
        x, y, w, h = track.bbox
        xyxy.append([x, y, x + w, y + h])
        confidence.append(getattr(track, 'confidence', 0.9))
        # Map utilization state to class_id (ACTIVE=1, INACTIVE=0)
        class_id.append(1 if track.utilization_state == "ACTIVE" else 0)
    
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidence, dtype=np.float32),
        class_id=np.array(class_id, dtype=np.int32),
        tracker_id=np.array([int(t.track_id) for t in tracks], dtype=np.int32)
    )


def annotate_with_supervision(
    frame: np.ndarray,
    tracks: list[Track],
    annotator: Optional[sv.Annotator] = None
) -> np.ndarray:
    """
    Annotate frame using supervision library.
    
    Args:
        frame: Input image/frame (BGR format)
        tracks: List of Track objects to annotate
        annotator: Optional custom annotator (uses BoxAnnotator by default)
        
    Returns:
        Annotated frame
    """
    if annotator is None:
        annotator = sv.BoxAnnotator()
    
    detections = convert_tracks_to_detections(tracks)
    annotated_frame = annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    
    # Add timestamp
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cv2.putText(annotated_frame, ts, (8, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    
    return annotated_frame


# Example annotators from supervision library
class AnnotatorFactory:
    """Factory for creating supervision annotators."""
    
    @staticmethod
    def create_box_annotator(thickness: int = 2, text_scale: float = 0.5) -> sv.BoxAnnotator:
        return sv.BoxAnnotator(thickness=thickness, text_scale=text_scale)
    
    @staticmethod
    def create_color_annotator() -> sv.ColorAnnotator:
        return sv.ColorAnnotator()
    
    @staticmethod
    def create_corner_annotator(thickness: int = 2) -> sv.CornerAnnotator:
        return sv.CornerAnnotator(thickness=thickness)
    
    @staticmethod
    def create_ellipse_annotator(thickness: int = 2) -> sv.EllipseAnnotator:
        return sv.EllipseAnnotator(thickness=thickness)


if __name__ == "__main__":
    # Example usage
    print("Supervision annotator module loaded successfully")
    print("Available annotators:")
    print("  - BoxAnnotator")
    print("  - ColorAnnotator")
    print("  - CornerAnnotator")
    print("  - EllipseAnnotator")
