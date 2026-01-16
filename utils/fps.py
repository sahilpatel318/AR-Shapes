from __future__ import annotations

import time
from collections import deque
from typing import Deque


class FPSCounter:
    """Tracks instantaneous and smoothed FPS readings."""

    def __init__(self, average_over: int = 60) -> None:
        self._timestamps: Deque[float] = deque(maxlen=average_over)
        self._last_tick: float = time.time()

    def tick(self) -> float:
        now = time.time()
        self._timestamps.append(now)
        self._last_tick = now
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
