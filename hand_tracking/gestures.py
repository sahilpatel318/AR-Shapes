from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from hand_tracking.detector import HandDetection
from utils.config import GestureConfig

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 7, 11, 15, 19]
FINGER_MCPS = [2, 5, 9, 13, 17]


@dataclass
class GestureResult:
    label: str
    confidence: float


class GestureStabilizer:
    def __init__(self, config: GestureConfig) -> None:
        self._config = config
        self._active_label: str = "sphere"
        self._candidate: Optional[str] = None
        self._frames = 0
        self._cooldown = 0

    def update(self, label: Optional[str]) -> str:
        if self._cooldown > 0:
            self._cooldown -= 1
        if label != self._candidate:
            self._candidate = label
            self._frames = 0
        if label is None:
            return self._active_label
        self._frames += 1
        if (
            self._frames >= self._config.stability_frames
            and self._cooldown == 0
            and label != self._active_label
        ):
            self._active_label = label
            self._cooldown = self._config.cooldown_frames
        return self._active_label


class GestureClassifier:
    """Rule-based gesture recognizer with stability checks."""

    def __init__(self, config: GestureConfig) -> None:
        self._config = config
        self._stabilizer = GestureStabilizer(config)

    def classify(self, detection: Optional[HandDetection]) -> str:
        if detection is None:
            return self._stabilizer.update(None)
        landmarks = detection.landmarks
        thumb_up = self._is_thumb_up(landmarks)
        pinch = self._is_pinch(landmarks)
        palm = self._is_open_palm(landmarks)

        label: Optional[str] = None
        if pinch:
            label = "torus"
        elif thumb_up:
            label = "sphere"
        elif palm:
            label = "cube"
        return self._stabilizer.update(label)

    def _is_thumb_up(self, landmarks: np.ndarray) -> bool:
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        other_fingers = [self._is_finger_extended(landmarks, i) for i in range(1, 5)]
        thumb_vertical = thumb_tip[1] - wrist[1]
        return (
            thumb_vertical < self._config.thumb_up_threshold
            and not any(other_fingers)
            and thumb_tip[1] < thumb_ip[1]
        )

    def _is_pinch(self, landmarks: np.ndarray) -> bool:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        dist = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
        middle_extended = self._is_finger_extended(landmarks, 2)
        return dist < self._config.pinch_threshold and not middle_extended

    def _is_open_palm(self, landmarks: np.ndarray) -> bool:
        extended = [self._is_finger_extended(landmarks, i) for i in range(1, 5)]
        if not all(extended):
            return False
        tips = [landmarks[idx][:2] for idx in FINGER_TIPS[1:]]
        spreads = [np.linalg.norm(tips[i] - tips[i + 1]) for i in range(len(tips) - 1)]
        avg_spread = float(np.mean(spreads)) if spreads else 0.0
        return avg_spread > self._config.palm_spread_threshold

    def _is_finger_extended(self, landmarks: np.ndarray, finger_index: int) -> bool:
        tip = landmarks[FINGER_TIPS[finger_index]]
        pip = landmarks[FINGER_PIPS[finger_index]]
        mcp = landmarks[FINGER_MCPS[finger_index]]
        return np.linalg.norm(tip - mcp) > np.linalg.norm(pip - mcp)
