"""Flight tasks.

T4 FlightCourse -- take off from the ground (mission command), fly at the
    cruise altitude through a field of tall pillars toward a glowing landing
    beacon, and land next to it (mission command once within 1 m of the
    beacon).  Steering in flight comes from the decision layer (connectome or
    baseline); altitude/attitude are held by the conventional flight
    controller.  Success: landed within 1.0 m of the landing pad centre with
    no collisions.

T7 YawStabilization -- hover with a failed yaw-rate gyro (its z-axis reads 0,
    so the flight controller has no yaw damping) while a disturbance torque
    (rotor-drag asymmetry / gust) acts about the yaw axis.  The only remaining
    yaw-rate information is visual optic flow.  We measure heading drift and
    yaw-rate RMS.  With the connectome controller, horizontal optic flow
    drives HS/H2 cells; their descending output becomes a corrective yaw
    command -- the optomotor response.
"""
from __future__ import annotations

import numpy as np

from ..robot.robot import Command
from ..robot.world import Arena
from ..sensors.suite import Environment
from .base import Task


class FlightCourse(Task):
    name = "flight_course"
    description = "Take off, fly through a pillar field to a landing beacon, land"
    max_time = 45.0

    def __init__(self, n_pillars: int = 6, length: float = 14.0):
        self.n, self.length = n_pillars, length

    def build(self, rng):
        L = self.length
        self.beacon = np.array([L, float(rng.uniform(-1.0, 1.0))])
        self.pad = self.beacon - np.array([0.9, 0.0])
        objs = []
        xs = np.linspace(3.0, L - 3.5, self.n) + rng.uniform(-0.6, 0.6, self.n)
        for x in xs:
            objs.append({"kind": "cylinder", "pos": (float(x), float(rng.uniform(-1.8, 1.8))), "size": (float(rng.uniform(0.12, 0.22)), 1.4)})
        objs.append({"kind": "pad", "pos": tuple(self.pad), "size": (0.5, 0.5), "taste": "plain"})
        objs.append({"kind": "beacon", "pos": tuple(self.beacon), "height": 0.9, "radius": 0.25})
        arena = Arena(size=(L / 2 + 1.5, 3.0), center=(L / 2, 0.0), objects=objs, wall_height=3.5,
                      robot_pos=(0.0, float(rng.uniform(-0.5, 0.5)), None), robot_yaw=float(rng.uniform(-0.25, 0.25)))
        self.land_cmd_t = None
        self.landed_t = None
        self.took_off = False
        return arena, Environment()

    def mission(self, ep, cmd: Command, t: float) -> Command:
        r = ep.robot
        p = r.truth()["pos"][:2]
        if r.mode != "ground":
            self.took_off = True
        if not self.took_off and t > 0.5:
            cmd.takeoff = True  # mission: start of flight
        if r.mode == "flight" and self.land_cmd_t is None and np.linalg.norm(p - self.beacon) < 1.1:
            self.land_cmd_t = t  # mission: land at the beacon
        if self.land_cmd_t is not None:
            cmd.land = True
        if self.took_off and r.mode == "ground":
            cmd.v_fwd, cmd.yaw_rate, cmd.halt = 0.0, 0.0, True
            if self.landed_t is None:
                self.landed_t = t
        return cmd

    def done(self, ep, t):
        return self.landed_t is not None and t > self.landed_t + 0.5

    def telemetry(self, ep):
        p = ep.robot.truth()["pos"][:2]
        return {"beacon_dist": round(float(np.linalg.norm(p - self.beacon)), 3)}

    def metrics(self, ep):
        p = ep.robot.truth()["pos"][:2]
        err = float(np.linalg.norm(p - self.pad))
        landed = self.landed_t is not None
        m = {"landed": landed, "landing_error_m": round(err, 3) if landed else None,
             "flight_time_s": round((self.landed_t or ep.data.time) - 0.5, 2)}
        m["success"] = bool(landed and err <= 1.0 and ep.collisions == 0)
        return m


class YawStabilization(Task):
    name = "yaw_stabilization"
    description = "Hover with a failed yaw gyro under a yaw disturbance torque"
    max_time = 14.0

    def __init__(self, gyro_failure: bool = True, torque: float = 0.03):
        self.gyro_failure = gyro_failure
        self.torque = torque

    def build(self, rng):
        self.t_dist = (4.0, 9.0)
        self.sign = float(rng.choice([-1.0, 1.0]))
        self.torque_eff = self.torque * float(rng.uniform(0.8, 1.2)) * self.sign
        arena = Arena(size=(5.0, 5.0), center=(0.0, 0.0), robot_yaw=float(rng.uniform(-np.pi, np.pi)))
        self.yaw_log = []
        return arena, Environment()

    def setup(self, ep):
        ep.robot.gyro_fail_z = self.gyro_failure

    def mission(self, ep, cmd: Command, t: float) -> Command:
        cmd.takeoff = True
        cmd.v_fwd = 0.0  # hover: decision layer only influences yaw
        cmd.climb_rate = 0.0
        cmd.escape = False
        return cmd

    def physics_tick(self, ep, t, dt):
        if self.t_dist[0] <= t < self.t_dist[1]:
            b = ep.robot.body
            ep.data.xfrc_applied[b, 5] += self.torque_eff

    def pre_step(self, ep, t):
        R = ep.robot.truth()["R"]
        om = ep.robot.truth()["omega_w"]
        self.yaw_log.append((t, float(np.arctan2(R[1, 0], R[0, 0])), float(om[2])))

    def metrics(self, ep):
        a = np.array(self.yaw_log)
        sel = (a[:, 0] >= self.t_dist[0]) & (a[:, 0] < self.t_dist[1] + 2.0)
        yaw = np.unwrap(a[:, 1])
        k0 = np.searchsorted(a[:, 0], self.t_dist[0])
        drift = float(np.rad2deg(abs(yaw[sel][-1] - yaw[k0])))
        rms = float(np.sqrt(np.mean(a[sel, 2] ** 2)))
        m = {"gyro_failure": self.gyro_failure, "heading_drift_deg": round(drift, 1), "yaw_rate_rms": round(rms, 3)}
        m["success"] = drift < 90.0
        return m
