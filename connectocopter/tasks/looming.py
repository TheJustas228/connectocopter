"""T2 -- Looming escape.

A black ball (r = 0.2 m) is launched at the rolling robot from a random
azimuth within the camera's field of view (+-40 deg) at 2.5-4 m/s, aimed at
the robot's predicted position.  A third of the trials are catch trials (no
ball) that measure spontaneous / false escapes.

Success (ball trials): the ball never comes within r_ball + 0.18 m of the
robot centre.  Reported: escape rate, survival rate, reaction time from
looming onset, time-to-contact remaining at the escape command, false-alarm
rate in catch trials.
"""
from __future__ import annotations

import numpy as np

from ..robot.world import Arena
from ..sensors.suite import Environment
from .base import Task


class LoomingEscape(Task):
    name = "looming_escape"
    description = "Rolling robot must escape (take off) from an approaching black ball"

    def __init__(self, catch: bool | None = None, stimulus: str = "visual"):
        self.catch_forced = catch
        self.stimulus = stimulus  # "visual" ball, or "vibration" (sound / substrate vibration event)
        self.r_ball = 0.2

    def build(self, rng):
        self.catch = bool(rng.random() < 1 / 3) if self.catch_forced is None else self.catch_forced
        self.t_onset = float(rng.uniform(1.5, 3.0))
        self.speed = float(rng.uniform(2.5, 4.0))
        self.az = float(np.deg2rad(rng.uniform(-40, 40)))
        self.dist0 = 6.0
        self.t_contact = self.t_onset + self.dist0 / self.speed
        self.max_time = self.t_contact + 1.5
        cruise = 0.8
        # robot predicted position at contact (rolling straight along +x)
        self.p_contact = np.array([cruise * self.t_contact, 0.0, 0.32])
        d = np.array([np.cos(self.az), np.sin(self.az), 0.0])
        self.p_start = self.p_contact + d * self.dist0
        self.p_start[2] = 0.32
        objs = [] if self.catch or self.stimulus != "visual" else [{"kind": "looming", "pos": (50.0, 50.0, -5.0), "radius": self.r_ball}]
        arena = Arena(size=(14.0, 8.0), center=(6.0, 0.0), objects=objs, robot_pos=(0.0, 0.0, None))
        self.escape_t = None
        self.min_dist = np.inf
        self.hit = False
        return arena, Environment()

    def setup(self, ep):
        self.has_ball = ep.model.nmocap > 0

    def _ball_pos(self, t):
        if t < self.t_onset:
            return np.array([50.0, 50.0, -5.0])
        s = (t - self.t_onset) * self.speed
        v = (self.p_contact - self.p_start) / self.dist0
        return self.p_start + v * s

    def physics_tick(self, ep, t, dt):
        if self.has_ball:
            ep.data.mocap_pos[0] = self._ball_pos(t)

    def pre_step(self, ep, t):
        if self.stimulus == "vibration" and not self.catch:
            on = self.t_onset <= t < self.t_onset + 0.3
            side = 1.0 if self.az > 0 else 0.0  # source on the left -> left JO more
            ep.env.vibration = (float(on) * (0.5 + 0.5 * side), float(on) * (0.5 + 0.5 * (1 - side)))
        if self.escape_t is None and ep.robot.mode != "ground":
            self.escape_t = t
        if self.has_ball and t >= self.t_onset:
            rp = ep.robot.truth()["pos"]
            dist = float(np.linalg.norm(self._ball_pos(t) - rp))
            self.min_dist = min(self.min_dist, dist)
            if dist < self.r_ball + 0.18:
                self.hit = True

    def metrics(self, ep):
        esc = self.escape_t is not None
        m = {"catch_trial": self.catch, "stimulus": self.stimulus, "escaped": esc,
             "azimuth_deg": round(float(np.rad2deg(self.az)), 1), "ball_speed": round(self.speed, 2)}
        if self.catch:
            m["false_escape"] = esc
            m["success"] = not esc
        else:
            m["survived"] = not self.hit
            m["success"] = (not self.hit) if self.stimulus == "visual" else esc
            if esc:
                m["reaction_time_s"] = round(self.escape_t - self.t_onset, 3)
                m["ttc_at_escape_s"] = round(self.t_contact - self.escape_t, 3)
            m["min_distance_m"] = round(self.min_dist, 3) if np.isfinite(self.min_dist) else None
        return m
