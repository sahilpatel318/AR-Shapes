from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def _sphere(count: int) -> np.ndarray:
    idx = np.arange(count, dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (idx / float(count - 1)) * 2.0
    radius = np.sqrt(1.0 - y * y)
    theta = phi * idx
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.stack([x, y, z], axis=1)


def _torus(count: int, major: float = 0.65, minor: float = 0.25) -> np.ndarray:
    rng = np.random.default_rng(19)
    angles = rng.random((count, 2)) * 2.0 * np.pi
    u = angles[:, 0]
    v = angles[:, 1]
    x = (major + minor * np.cos(v)) * np.cos(u)
    y = (major + minor * np.cos(v)) * np.sin(u)
    z = minor * np.sin(v)
    return np.stack([x, y, z], axis=1)


def _cube(count: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    points = rng.uniform(-0.5, 0.5, size=(count, 3))
    face_ids = rng.integers(0, 3, size=count)
    signs = rng.choice([-0.5, 0.5], size=count)
    for axis in range(3):
        mask = face_ids == axis
        points[mask, axis] = signs[mask]
    return points


_SHAPES: Dict[str, Callable[[int], np.ndarray]] = {
    "sphere": _sphere,
    "torus": _torus,
    "cube": _cube,
}


def sample_shape(name: str, count: int) -> np.ndarray:
    if name not in _SHAPES:
        raise ValueError(f"Unsupported shape '{name}'.")
    return _SHAPES[name](count).astype(np.float32)
