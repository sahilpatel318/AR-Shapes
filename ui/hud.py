from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from utils.config import HUDConfig


class HUDOverlay:
    """Draws the minimalist HUD on top of the camera feed."""

    def __init__(self, config: HUDConfig) -> None:
        self._config = config
        self._font = cv2.FONT_HERSHEY_PLAIN

    def apply(
        self,
        frame_bgr: np.ndarray,
        shape_label: str,
        fps: float,
        anchor_norm: Tuple[float, float],
    ) -> np.ndarray:
        frame = frame_bgr
        h, w, _ = frame.shape
        margin = self._config.margin
        text = f"shape :: {shape_label}"
        cv2.putText(
            frame,
            text,
            (margin, margin + 18),
            self._font,
            self._config.text_scale,
            self._config.text_color,
            1,
            cv2.LINE_AA,
        )
        fps_text = f"fps {fps:05.2f}"
        cv2.putText(
            frame,
            fps_text,
            (margin, margin + 38),
            self._font,
            self._config.text_scale * 0.95,
            self._config.text_color,
            1,
            cv2.LINE_AA,
        )
        px = int(anchor_norm[0] * w)
        py = int(anchor_norm[1] * h)
        bracket_size = int(min(w, h) * 0.12)
        self._draw_brackets(frame, px, py, bracket_size)
        return frame

    def _draw_brackets(self, frame: np.ndarray, cx: int, cy: int, size: int) -> None:
        half = size // 2
        thickness = self._config.line_thickness
        color = self._config.accent_color
        offsets = [(-half, -half), (half, -half), (-half, half), (half, half)]
        for ox, oy in offsets:
            x = cx + ox
            y = cy + oy
            if ox < 0:
                x1, x2 = x, x + size // 4
            else:
                x1, x2 = x - size // 4, x
            if oy < 0:
                y1, y2 = y, y + size // 4
            else:
                y1, y2 = y - size // 4, y
            cv2.line(frame, (x1, y), (x2, y), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (x, y1), (x, y2), color, thickness, cv2.LINE_AA)
