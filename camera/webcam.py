from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class WebcamCapture:
    """Threaded webcam reader that keeps the freshest frame in memory."""

    def __init__(self, width: int, height: int, camera_index: int = 0) -> None:
        self._width = width
        self._height = height
        self._camera_index = camera_index
        self._capture = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._capture.set(cv2.CAP_PROP_FPS, 60)
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        if not self._capture.isOpened():
            raise RuntimeError("Unable to access the webcam.")
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._capture.isOpened():
            self._capture.release()

    def _reader_loop(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def resolution(self) -> Tuple[int, int]:
        return self._width, self._height
