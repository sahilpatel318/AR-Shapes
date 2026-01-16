from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VisionConfig:
    width: int = 1280
    height: int = 720
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    smoothing_alpha: float = 0.35
    detection_interval: float = 1 / 60.0


@dataclass(frozen=True)
class GestureConfig:
    stability_frames: int = 8
    cooldown_frames: int = 12
    pinch_threshold: float = 0.045
    palm_spread_threshold: float = 0.18
    thumb_up_threshold: float = -0.15


@dataclass(frozen=True)
class RenderConfig:
    window_width: int = 1280
    window_height: int = 720
    particle_count: int = 1800
    particle_size: float = 15.0  # bigger particles
    particle_spring: float = 12.0  # more spring for faster response
    particle_damping: float = 0.78  # less damping for snappier motion
    anchor_smooth: float = 0.12  # less smoothing for faster anchor following
    background_gamma: float = 1.05
    shape_scale: float = 0.45  # larger shape scale
    camera_fov: float = 45.0
    texture_format: int = 3  # RGB by default

    @property
    def aspect(self) -> float:
        return self.window_width / max(1, self.window_height)


@dataclass(frozen=True)
class HUDConfig:
    text_color: Tuple[int, int, int] = (255, 255, 255)
    accent_color: Tuple[int, int, int] = (255, 255, 255)
    text_scale: float = 0.6
    line_thickness: int = 2
    margin: int = 24


class SharedState:
    """Thread-safe bridge between the vision and rendering subsystems."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_rgb: Optional[np.ndarray] = None
        self._anchor_ndc: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._shape: str = "sphere"
        self._shape_changed_at: float = time.time()
        self._updated_at: float = 0.0
        self._shutdown: bool = False

    def update(self, frame_rgb: np.ndarray, anchor_ndc: Tuple[float, float, float], shape: str) -> None:
        frame_copy = np.ascontiguousarray(frame_rgb)
        with self._lock:
            self._frame_rgb = frame_copy
            self._anchor_ndc = anchor_ndc
            if shape != self._shape:
                self._shape = shape
                self._shape_changed_at = time.time()
            self._updated_at = time.time()

    def consume(self) -> Tuple[Optional[np.ndarray], Tuple[float, float, float], str, float]:
        with self._lock:
            if self._frame_rgb is None:
                return None, self._anchor_ndc, self._shape, self._shape_changed_at
            frame = self._frame_rgb.copy()
            anchor = self._anchor_ndc
            shape = self._shape
            changed_at = self._shape_changed_at
        return frame, anchor, shape, changed_at

    def request_shutdown(self) -> None:
        with self._lock:
            self._shutdown = True

    def shutdown_requested(self) -> bool:
        with self._lock:
            return self._shutdown


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the repository root."""
    base = Path(__file__).resolve().parents[1]
    return base.joinpath(*parts)
