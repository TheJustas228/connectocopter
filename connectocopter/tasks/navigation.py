"""Ground navigation tasks.

T1 TargetSeek  -- a glowing beacon is placed 3-5 m away at +-35 deg from the
                 robot's initial heading.  Success: reach within 0.6 m of the
                 beacon post within 25 s.
T3 ObstacleCourse -- a 14 m corridor with randomly placed red/white pillars and
                 a goal beacon at the far end.  Success: reach the goal zone
                 (within 0.8 m of the beacon) within 40 s.  Collisions with
                 pillars and walls are counted.

Metrics: success, time to goal, path efficiency (straight-line distance /
path length), collisions, mean speed, energy.
"""
from __future__ import annotations

import numpy as np

from ..robot.world import Arena
from ..sensors.suite import Environment
from .base import Task


class _GoalTask(Task):
    goal_radius = 0.6

    def setup(self, ep):
        self.t_goal = None
        self.start = ep.robot.truth()["pos"][:2].copy()

    def pre_step(self, ep, t):
        if self.t_goal is None:
            p = ep.robot.truth()["pos"][:2]
            if np.linalg.norm(p - self.goal) < self.goal_radius:
                self.t_goal = t

    def done(self, ep, t):
        return self.t_goal is not None

    def telemetry(self, ep):
        p = ep.robot.truth()["pos"][:2]
        return {"goal_dist": round(float(np.linalg.norm(p - self.goal)), 3)}

    def metrics(self, ep):
        p = ep.robot.truth()["pos"][:2]
        straight = float(np.linalg.norm(self.goal - self.start)) - self.goal_radius
        m = {"success": self.t_goal is not None,
             "final_goal_dist_m": round(float(np.linalg.norm(p - self.goal)), 3)}
        if self.t_goal is not None:
            m["time_to_goal_s"] = round(self.t_goal, 2)
            m["path_efficiency"] = round(min(1.0, straight / max(ep.path_len, 1e-6)), 3)
            m["mean_speed"] = round(ep.path_len / self.t_goal, 3)
        return m


class TargetSeek(_GoalTask):
    name = "target_seek"
    description = "Drive to a glowing beacon placed at a random bearing"
    max_time = 25.0

    def build(self, rng):
        d = rng.uniform(3.0, 5.0)
        az = np.deg2rad(rng.uniform(-35, 35))
        self.goal = np.array([d * np.cos(az), d * np.sin(az)])
        objs = [{"kind": "beacon", "pos": tuple(self.goal)}]
        return Arena(size=(7.0, 7.0), center=(1.5, 0.0), objects=objs), Environment()


class ObstacleCourse(_GoalTask):
    name = "obstacle_course"
    description = "Corridor with random pillars; reach the goal beacon at the far end"
    max_time = 40.0
    goal_radius = 0.8

    def __init__(self, n_pillars: int = 7, length: float = 14.0, half_width: float = 1.6):
        self.n, self.length, self.hw = n_pillars, length, half_width

    def build(self, rng):
        L, hw = self.length, self.hw
        self.goal = np.array([L - 1.0, 0.0])
        objs = []
        xs = np.linspace(2.5, L - 3.0, self.n) + rng.uniform(-0.5, 0.5, self.n)
        for x in xs:
            y = rng.uniform(-hw + 0.4, hw - 0.4)
            r = rng.uniform(0.12, 0.2)
            objs.append({"kind": "cylinder", "pos": (float(x), float(y)), "size": (float(r), 0.6)})
        objs.append({"kind": "beacon", "pos": tuple(self.goal)})
        arena = Arena(size=(L / 2 + 0.5, hw), center=(L / 2 - 0.5, 0.0), objects=objs,
                      robot_pos=(0.0, float(rng.uniform(-0.3, 0.3)), None), robot_yaw=float(rng.uniform(-0.2, 0.2)))
        return arena, Environment()
