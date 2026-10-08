"""Filament ("puff") odor plume model.

A simplified version of the Farrell et al. (2002) filament-based plume:
puffs are released from the source at a fixed rate, advected by a mean wind
plus a slowly meandering cross-wind component, and grow by diffusion.  The
concentration at a point is the sum of Gaussian puffs.  This produces the
intermittent, patchy odor encounters that make plume tracking hard for
insects and robots (cf. Demir et al. 2020).
"""
from __future__ import annotations

import numpy as np


class PuffPlume:
    def __init__(self, source: tuple[float, float, float], wind: tuple[float, float] = (-0.6, 0.0),
                 release_hz: float = 20.0, sigma0: float = 0.04, growth: float = 0.012,
                 meander_amp: float = 0.35, meander_tau: float = 2.0, seed: int = 0, max_age: float = 25.0):
        self.src = np.asarray(source, dtype=float)
        self.wind = np.asarray(wind, dtype=float)
        self.release_hz = release_hz
        self.sigma0, self.growth = sigma0, growth
        self.meander_amp, self.meander_tau = meander_amp, meander_tau
        self.rng = np.random.default_rng(seed)
        self.max_age = max_age
        self.pos = np.zeros((0, 3))
        self.age = np.zeros(0)
        self.t = 0.0
        self._acc = 0.0
        self._meander = 0.0

    def step(self, dt: float) -> None:
        self.t += dt
        # Ornstein-Uhlenbeck cross-wind meander
        self._meander += -self._meander * dt / self.meander_tau + self.meander_amp * np.sqrt(2 * dt / self.meander_tau) * self.rng.normal()
        w = np.linalg.norm(self.wind) + 1e-9
        cross = np.array([-self.wind[1], self.wind[0]]) / w
        vel = self.wind + cross * self._meander
        self._acc += self.release_hz * dt
        n = int(self._acc)
        if n:
            self._acc -= n
            new = np.repeat(self.src[None], n, 0) + self.rng.normal(0, 0.01, (n, 3))
            self.pos = np.vstack([self.pos, new])
            self.age = np.concatenate([self.age, np.zeros(n)])
        if len(self.age):
            self.pos[:, :2] += vel * dt + self.rng.normal(0, 0.03 * np.sqrt(dt), (len(self.age), 2))
            self.pos[:, 2] += self.rng.normal(0, 0.01 * np.sqrt(dt), len(self.age))
            self.age += dt
            keep = self.age < self.max_age
            self.pos, self.age = self.pos[keep], self.age[keep]

    def concentration(self, p: np.ndarray) -> float:
        if not len(self.age):
            return 0.0
        s2 = (self.sigma0 + self.growth * self.age) ** 2
        d2 = np.sum((self.pos - np.asarray(p)[None]) ** 2, axis=1)
        # each puff carries unit mass; normalised so a fresh puff peak ~ 1
        c = np.exp(-d2 / (2 * s2)) * (self.sigma0 ** 2 / s2) ** 1.5
        return float(c.sum())
