from __future__ import annotations

import time
from typing import Dict

import numpy as np

from rendering import shapes
from utils.config import RenderConfig


class ParticleSystem:
    """Simple Euler particle integrator with smooth target morphing."""

    def __init__(self, config: RenderConfig) -> None:
        self._config = config
        self.count = config.particle_count
        self.positions = np.zeros((self.count, 3), dtype=np.float32)
        self.velocities = np.zeros_like(self.positions)
        self._target = np.zeros_like(self.positions)
        self._anchor = np.zeros(3, dtype=np.float32)
        self._shape_offsets = shapes.sample_shape("sphere", self.count) * config.shape_scale
        self._last_time = time.time()
        self._rng = np.random.default_rng(3)
        self._color_phase = self._rng.random(self.count).astype(np.float32)
        self._shape_cache: Dict[str, np.ndarray] = {}
        self.set_shape("sphere", (0.0, 0.0, 0.0))

    def set_shape(self, name: str, anchor_ndc: tuple[float, float, float]) -> None:
        relative = self._shape_cache.get(name)
        if relative is None:
            relative = shapes.sample_shape(name, self.count) * self._config.shape_scale
            self._shape_cache[name] = relative
        self._shape_offsets = relative
        self._anchor = np.array(anchor_ndc, dtype=np.float32)
        self._target = self._shape_offsets + self._anchor

    def move_anchor(self, anchor_ndc: tuple[float, float, float]) -> None:
        anchor = np.array(anchor_ndc, dtype=np.float32)
        self._anchor = anchor
        self._target = self._shape_offsets + anchor

    def step(self, dt: float) -> None:
        if dt <= 0:
            return
        drift = (self._rng.random(self.positions.shape) - 0.5) * 0.002
        self._target = self._shape_offsets + self._anchor
        delta = self._target - self.positions
        accel = delta * self._config.particle_spring
        self.velocities += (accel + drift) * dt
        self.velocities *= self._config.particle_damping
        self.positions += self.velocities * dt

    def color_modulation(self, t: float) -> np.ndarray:
        mod = 0.5 + 0.5 * np.sin(2.0 * np.pi * (self._color_phase + t * 0.2))
        return mod.astype(np.float32)
