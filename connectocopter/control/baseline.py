"""Conventional (non-neural) baseline controller.

Uses exactly the same sensor features and the same low-level controllers as
the connectome controller, but the decision layer is a hand-designed reactive
rule set (Braitenberg-style steering + thresholds), tuned once by hand on
seeds that are *not* used for evaluation (seeds >= 1000 are evaluation seeds).
"""
from __future__ import annotations

import numpy as np

from ..robot.robot import Command


class BaselineController:
    label = "baseline"

    def __init__(self, cruise_ground: float = 0.8, cruise_flight: float = 1.2, k_target: float = 1.6,
                 k_obstacle: float = 1.2, k_optomotor: float = 0.0, loom_thresh: float = 2.0,
                 use_odor: bool = True, seed: int = 0):
        self.cruise_ground, self.cruise_flight = cruise_ground, cruise_flight
        self.k_target, self.k_obstacle, self.k_optomotor = k_target, k_obstacle, k_optomotor
        self.loom_thresh = loom_thresh
        self.use_odor = use_odor
        self.dt = 0.02
        self.feeding = False
        self.rng = np.random.default_rng(seed)
        self._cast_dir = 1.0
        self._cast_n = 0
        self._last_hit = -1e9
        self.t = 0.0
        self.tel = None

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.feeding = False
        self.t = 0.0
        self._cast_dir = 1.0
        self._cast_n = 0
        self._last_hit = -1e9
        if hasattr(self, "_cast_t0"):
            del self._cast_t0

    def step(self, f: dict, mode: str) -> Command:
        self.t += self.dt
        flying = mode in ("flight", "takeoff")
        v = self.cruise_flight if flying else self.cruise_ground
        yaw = 0.0
        # steer toward a visual target, away from expanding obstacles
        if f.get("target_present", 0) > 0:
            yaw += -self.k_target * np.tanh(f["target_az"] / 25.0)
        aL, aR = max(f["approach_L"] - 0.5, 0.0), max(f["approach_R"] - 0.5, 0.0)
        yaw += -self.k_obstacle * (aL - aR)
        if min(aL, aR) > 0.3 and abs(aL - aR) < 0.2:  # obstacle dead ahead: break symmetry (turn left)
            yaw += self.k_obstacle * 0.8
        yaw += self.k_optomotor * (f["hs_L"] - f["hs_R"])
        # odor: moth-inspired surge-and-cast (Kennedy 1983; Belanger & Willis 1996).  While odor is
        # sensed (and 0.5 s after) surge upwind with a small bilateral correction; when the plume is lost,
        # cast crosswind in zigzags of growing width, still biased upwind.
        if self.use_odor and "odor_L" in f:
            c_l, c_r = f["odor_L"], f["odor_R"]
            up = f.get("wind_from", 0.0)  # bearing of the upwind direction (rad, +left)
            if c_l + c_r > 0.05:
                self._last_hit = self.t
                self._cast_n = 0
            since = self.t - self._last_hit
            if since < 0.5:
                yaw = 2.5 * np.tanh(1.5 * up) + 0.8 * np.tanh(4 * (c_l - c_r) / (c_l + c_r + 1e-6))
            elif since < 20.0:
                # cast crosswind in zigzags of growing width, biased 15 deg upwind
                period = 1.5 + 0.5 * self._cast_n
                if not hasattr(self, "_cast_t0") or self.t - self._cast_t0 > period:
                    self._cast_t0, self._cast_dir, self._cast_n = self.t, -self._cast_dir, self._cast_n + 1
                target = up + self._cast_dir * np.deg2rad(75)
                yaw = 2.5 * np.tanh(1.5 * np.arctan2(np.sin(target), np.cos(target)))
                v *= 0.6
            else:
                yaw = 2.5 * np.tanh(1.5 * up)  # long loss: drift upwind and wait for a new encounter
        escape = max(f["loom_L"], f["loom_R"]) > self.loom_thresh or max(f.get("vibration_L", 0), f.get("vibration_R", 0)) > 0.5
        self.feeding = f.get("taste_sugar", 0) > 0.5 and f.get("taste_bitter", 0) < 0.5
        return Command(v_fwd=v, yaw_rate=float(np.clip(yaw, -2.0, 2.0)), escape=bool(escape), halt=self.feeding)
