"""T5 -- Taste-triggered docking ("feeding" analogue).

The robot rolls down a corridor over three floor pads in random order:
sugar (yellow), bitter (green) and mixed sugar+bitter (orange).  A contact
chemosensor under the chassis reports what the robot is standing on (a real
robot would use e.g. a colour/reflectance or conductivity sensor on a charging
pad; here it is an abstract "taste").

The correct behaviour -- predicted by Shiu et al. (2024) for the fly -- is to
stop ("feed"/dock) on sugar, but not on bitter, and not on the mixed pad
because bitter input suppresses the sugar-evoked MN9 response.

Success: docks (stays halted >= 1.5 s) on the sugar pad and on no other pad.
"""
from __future__ import annotations

import numpy as np

from ..robot.world import Arena
from ..sensors.suite import Environment
from .base import Task


class TasteDock(Task):
    name = "taste_dock"
    description = "Stop to 'feed' on the sugar pad; ignore bitter and mixed pads"
    max_time = 16.0

    def build(self, rng):
        order = list(rng.permutation(["sugar", "bitter", "mixed"]))
        xs = [2.0, 4.6, 7.2]
        self.pads = [{"pos": (x, 0.0), "size": (0.45, 0.9), "taste": t} for x, t in zip(xs, order)]
        objs = [{"kind": "pad", "pos": p["pos"], "size": p["size"], "taste": p["taste"]} for p in self.pads]
        arena = Arena(size=(5.5, 1.6), center=(4.5, 0.0), objects=objs, robot_pos=(0.0, 0.0, None))
        self.halt_start = None
        self.docks = []  # (pad taste, t)
        self.entry_t = {}
        return arena, Environment(pads=self.pads)

    def _pad_at(self, x, y):
        for p in self.pads:
            (px, py), (sx, sy) = p["pos"], p["size"]
            if abs(x - px) <= sx and abs(y - py) <= sy:
                return p["taste"]
        return None

    def pre_step(self, ep, t):
        x, y = ep.robot.truth()["pos"][:2]
        pad = self._pad_at(x, y)
        if pad is not None and pad not in self.entry_t:
            self.entry_t[pad] = t
        v = np.linalg.norm(ep.robot.truth()["vel_w"][:2])
        halted = v < 0.05 and ep.robot.mode == "ground" and t > 0.5
        if halted:
            if self.halt_start is None:
                self.halt_start = (t, pad)
            elif t - self.halt_start[0] >= 1.5 and (not self.docks or self.docks[-1][0] != pad or self.docks[-1][1] != self.halt_start[0]):
                if not any(d[1] == self.halt_start[0] for d in self.docks):
                    self.docks.append((pad, self.halt_start[0]))
        else:
            self.halt_start = None

    def done(self, ep, t):
        return any(d[0] == "sugar" for d in self.docks) and t > self.docks[-1][1] + 2.0

    def telemetry(self, ep):
        x, y = ep.robot.truth()["pos"][:2]
        return {"pad": self._pad_at(x, y)}

    def metrics(self, ep):
        docked = [d[0] for d in self.docks]
        m = {"docked_sugar": "sugar" in docked, "docked_bitter": "bitter" in docked,
             "docked_mixed": "mixed" in docked, "docked_nothing": None in docked,
             "pad_order": [p["taste"] for p in self.pads]}
        m["success"] = m["docked_sugar"] and not (m["docked_bitter"] or m["docked_mixed"] or m["docked_nothing"])
        if m["docked_sugar"] and "sugar" in self.entry_t:
            t_dock = [d[1] for d in self.docks if d[0] == "sugar"][0]
            m["dock_latency_s"] = round(t_dock - self.entry_t["sugar"], 3)
        return m
