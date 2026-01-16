from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.config import VisionConfig, project_path
from utils.smoothing import LandmarkSmoother


@dataclass
class HandDetection:
    landmarks: np.ndarray  # (21, 3) normalized coordinates
    handedness: str
    anchor: np.ndarray  # (3,) normalized anchor point


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model() -> Path:
    target = project_path("models", "hand_landmarker.task")
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(MODEL_URL) as response, target.open("wb") as fout:
            shutil.copyfileobj(response, fout)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Failed to download MediaPipe hand landmarker model. Check your internet connection."
        ) from exc
    return target


class HandDetector:
    """MediaPipe Tasks hand detector with landmark smoothing."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        model_path = _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_tracking_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._landmark_smoother = LandmarkSmoother(config.smoothing_alpha)

    def process(self, frame_bgr: np.ndarray) -> Optional[HandDetection]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None

        hand_landmarks = result.hand_landmarks[0]
        coords = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks], dtype=np.float32)
        coords = self._landmark_smoother.update(coords)
        handedness = result.handedness[0][0].category_name.lower()
        anchor = self._anchor_from_landmarks(coords)
        return HandDetection(landmarks=coords, handedness=handedness, anchor=anchor)

    def _anchor_from_landmarks(self, coords: np.ndarray) -> np.ndarray:
        wrist = coords[0]
        index_mcp = coords[5]
        middle_mcp = coords[9]
        anchor = (wrist + index_mcp + middle_mcp) / 3.0
        anchor[2] = np.clip(anchor[2], -0.2, 0.2)
        anchor[1] = np.clip(anchor[1], 0.0, 1.0)
        anchor[0] = np.clip(anchor[0], 0.0, 1.0)
        return anchor

    def close(self) -> None:
        self._landmarker.close()

    def __del__(self) -> None:  # pragma: no cover - cleanup helper
        try:
            self.close()
        except Exception:
            pass
