"""T6 -- Odor-plume navigation (ground).

An odor source (yellow "fruit") releases a filament plume carried downwind
(wind blows toward -x at 0.6 m/s with meander).  The robot starts 7-8 m
downwind, off-axis.  Two odor sensors sit on the antenna tips 11 cm apart;
the wind sensor reports apparent airflow.

Success: reach within 0.6 m of the source within 60 s.

Important: in the whole-brain LIF model, driving olfactory receptor neurons
(or a single projection neuron) ignites a self-sustaining, brain-wide
runaway state (see docs/brain_interface.md, "olfactory instability"), so
there is no usable connectome olfactory channel.  This task is therefore run
with the baseline controller (cast-and-surge) and, as a documented negative
control, with the connectome controller *with* ORN encoders added.
"""
from __future__ import annotations

import numpy as np

from ..robot.world import Arena
from ..sensors.odor import PuffPlume
from ..sensors.suite import Environment
from .base import Task


class OdorPlume(Task):
    name = "odor_plume"
    description = "Find the odor source by following an intermittent plume upwind"
    max_time = 60.0

    def build(self, rng):
        self.src = np.array([8.0, float(rng.uniform(-0.8, 0.8))])
        self.plume = PuffPlume(source=(self.src[0], self.src[1], 0.08), wind=(-0.6, 0.0), seed=int(rng.integers(1 << 30)))
        start_y = float(rng.uniform(-1.5, 1.5))
        arena = Arena(size=(6.5, 3.5), center=(4.5, 0.0), objects=[{"kind": "source", "pos": tuple(self.src)}],
                      robot_pos=(0.3, start_y, None), robot_yaw=float(rng.uniform(-0.6, 0.6)))
        env = Environment(plume=self.plume, wind=np.array([-0.6, 0.0, 0.0]))
        # pre-fill the plume
        for _ in range(int(15 / 0.05)):
            self.plume.step(0.05)
        self.t_found = None
        self.hits = 0
        return arena, env

    def pre_step(self, ep, t):
        self.plume.step(ep.ctrl_dt)
        p = ep.robot.truth()["pos"][:2]
        if self.t_found is None and np.linalg.norm(p - self.src) < 0.6:
            self.t_found = t
        f = ep.sensors.features
        if f.get("odor_L", 0) + f.get("odor_R", 0) > 0.05:
            self.hits += 1

    def done(self, ep, t):
        return self.t_found is not None

    def telemetry(self, ep):
        f = ep.sensors.features
        return {"odor": round(float(f.get("odor_L", 0) + f.get("odor_R", 0)), 4),
                "puffs": [[round(float(x), 2), round(float(y), 2)] for x, y, _ in self.plume.pos[::6]]}

    def metrics(self, ep):
        p = ep.robot.truth()["pos"][:2]
        return {"success": self.t_found is not None, "time_to_source_s": round(self.t_found, 2) if self.t_found else None,
                "final_dist_m": round(float(np.linalg.norm(p - self.src)), 3), "odor_hit_steps": self.hits}
