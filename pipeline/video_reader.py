"""
pipeline/video_reader.py
─────────────────────────
Reads frames from a video file with configurable FPS throttling.
"""

import cv2
import numpy as np
from core.config import VideoConfig
from core.logger import get_logger

log = get_logger("video_reader")


class VideoReader:

    def __init__(self, config: VideoConfig):
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self._skip = 1
        self._frame_index = 0

    def open(self) -> None:
        src = self.config.source
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")

        native_fps     = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._skip     = max(1, int(native_fps / self.config.process_fps))
        total_frames   = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log.info(
            "Video opened: %s | native %.1f FPS | processing every %d frames | total %d frames",
            src, native_fps, self._skip, total_frames,
        )

    def frames(self):
        """
        Generator — yields (frame_index, frame: np.ndarray).
        Automatically loops the video when it ends.
        """
        if self._cap is None:
            self.open()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                log.info("End of video — looping back")
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_index = 0
                continue

            self._frame_index += 1
            if self._frame_index % self._skip != 0:
                continue

            yield self._frame_index, frame

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
            log.info("Video reader closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
