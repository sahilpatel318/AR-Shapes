from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Optional

import numpy as np


class ExponentialSmoother:
    """Scalar/vector exponential moving average."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(np.clip(alpha, 1e-4, 0.999))
        self._state: Optional[np.ndarray] = None

    def update(self, value: Iterable[float]) -> np.ndarray:
        vec = np.asarray(value, dtype=np.float32)
        if self._state is None:
            self._state = vec.copy()
        else:
            self._state = self.alpha * vec + (1.0 - self.alpha) * self._state
        return self._state

    @property
    def value(self) -> Optional[np.ndarray]:
        return None if self._state is None else self._state.copy()


class LandmarkSmoother:
    """Per-landmark position smoothing."""

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._smoothers = [ExponentialSmoother(alpha) for _ in range(21)]

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        if landmarks.shape[0] != 21:
            raise ValueError("MediaPipe Hands should return 21 landmarks.")
        smoothed = [self._smoothers[i].update(landmarks[i]) for i in range(21)]
        return np.stack(smoothed, axis=0)


class TemporalStabilizer:
    """Keeps a rolling buffer of boolean states for stability checks."""

    def __init__(self, window: int) -> None:
        self._window = max(1, window)
        self._buffer: Deque[bool] = deque(maxlen=self._window)

    def push(self, value: bool) -> float:
        self._buffer.append(bool(value))
        return sum(self._buffer) / len(self._buffer)

    def is_stable(self, threshold: float) -> bool:
        if not self._buffer:
            return False
        return sum(self._buffer) / len(self._buffer) >= threshold
