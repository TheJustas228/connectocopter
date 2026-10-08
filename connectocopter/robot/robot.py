"""Robot runtime: state estimate, flight controller, ground controller, modes.

Division of labour (important for honest attribution):

* The **connectome controller** (connectocopter.control.connectome) decides
  *what* to do: forward speed, turn rate, take off / escape, halt, climb.
* This module is the **conventional low-level layer** -- the engineering
  analogue of the fly's ventral nerve cord + muscles.  It turns those
  high-level commands into rotor thrusts and wheel speeds using standard,
  non-neural control (a cascaded geometric flight controller after
  Lee, Leok & McClamroch 2010, and a skid-steer yaw-rate loop).  None of its
  output is attributed to the fly brain.

The same low-level layer is used unchanged by the baseline controller, so the
comparison between controllers isolates the decision layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .model import ROTORS, WHEELS, RobotParams

G = 9.81


@dataclass
class Command:
    """High-level command produced by a decision layer (brain or baseline)."""

    v_fwd: float = 0.0  # m/s along heading (ground or flight)
    yaw_rate: float = 0.0  # rad/s, +CCW (left)
    climb_rate: float = 0.0  # m/s, flight altitude-setpoint rate (+up)
    takeoff: bool = False  # request transition ground -> flight
    escape: bool = False  # request fast escape climb (Giant-Fiber-like)
    land: bool = False  # request flight -> ground
    halt: bool = False  # stop (ground) / hover (flight)


@dataclass
class EstimatorNoise:
    vel: float = 0.03  # m/s
    att_deg: float = 0.5
    alt: float = 0.01  # m
    gyro: float = 0.01  # rad/s


@dataclass
class FlightConfig:
    cruise_alt: float = 1.2
    takeoff_climb: float = 1.0  # m/s
    escape_climb: float = 3.0  # m/s
    escape_alt_gain: float = 1.5  # extra altitude gained in an escape (m)
    land_rate: float = 0.5  # m/s
    max_tilt_deg: float = 30.0
    kv_xy: float = 2.2  # 1/s horizontal velocity loop
    kp_z: float = 5.0
    kd_z: float = 3.5
    att_wn: float = 22.0  # rad/s roll/pitch attitude bandwidth
    yaw_wn: float = 7.0
    rotor_drag: float = 0.10  # N per (m/s) per rotor at hover thrust
    alt_min: float = 0.4
    alt_max: float = 3.0


class Robot:
    MODES = ("ground", "takeoff", "flight", "landing")

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, params: RobotParams = RobotParams(),
                 rng: np.random.Generator | None = None, noise: EstimatorNoise = EstimatorNoise(),
                 flight: FlightConfig = FlightConfig()):
        self.m, self.d, self.p = model, data, params
        self.rng = rng or np.random.default_rng(0)
        self.noise = noise
        self.fc = flight
        self.body = model.body("robot").id
        self.mass = float(model.body_subtreemass[self.body])
        self.J = model.body_inertia[self.body].copy()
        self.rotor_act = np.array([model.actuator(f"thrust_{n}").id for n in ROTORS])
        self.wheel_act = {n: model.actuator(f"drive_{n}").id for n in WHEELS}
        self.sens = {n: (model.sensor(n).adr[0], model.sensor(n).dim[0]) for n in
                     ["gyro", "accel", "quat", "pos", "vel_body", "range_down", "bumper"]}
        self.range_offset = 0.009  # range sensor sits 9 mm below the body origin
        self.ground_range = params.rest_height - self.range_offset
        # allocation: [T, tx, ty, tz] = A @ f
        xy = params.rotor_xy
        self.A = np.vstack([np.ones(4), xy[:, 1], -xy[:, 0], params.rotor_spin * params.k_m])
        self.A_inv = np.linalg.inv(self.A)
        self.mode = "ground"
        self.cmd = Command()
        self.alt_ref = 0.0
        self.yaw_ref = None
        self.escape_until = -1.0
        self.yaw_int = 0.0
        self.gyro_fail_z = False  # sensor-failure experiments
        self.yaw_ff_gain = 0.06  # N m per rad/s of commanded yaw rate when the yaw gyro has failed
        self.energy_j = 0.0
        self.rotor_power_w = 0.0
        self.wheel_power_w = 0.0
        self.mode_log: list[tuple[float, str]] = [(0.0, "ground")]
        self._est = {}
        # whole-body contact sensing (bumper / arm switches) for the retreat reflex
        self.own_geoms = np.array([g for g in range(model.ngeom) if self._root(model.geom_bodyid[g]) == self.body])
        self.floor_geoms = {g for g in range(model.ngeom) if model.geom(g).name == "floor"}
        self.retreat_until = -1.0
        self.retreat_turn = 1.0
        self.reflex_count = 0

    def _root(self, b: int) -> int:
        while self.m.body_parentid[b] != 0:
            b = self.m.body_parentid[b]
        return b

    def contact(self) -> tuple[bool, float]:
        """(touching a non-floor object?, lateral side of contact in body frame: +left)."""
        d = self.d
        own = set(self.own_geoms.tolist())
        R = d.xmat[self.body].reshape(3, 3)
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = c.geom1, c.geom2
            if (a in own) != (b in own) and a not in self.floor_geoms and b not in self.floor_geoms:
                rel = R.T @ (c.pos - d.xpos[self.body])
                return True, float(np.sign(rel[1]) or 1.0)
        return False, 0.0

    # ----------------------------------------------------------------- state
    def _sensor(self, name):
        a, n = self.sens[name]
        return self.d.sensordata[a:a + n].copy()

    def truth(self) -> dict:
        d, b = self.d, self.body
        R = d.xmat[b].reshape(3, 3).copy()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)  # world frame [ang, lin]
        return {"pos": d.xpos[b].copy(), "R": R, "quat": d.xquat[b].copy(), "omega_w": vel[:3], "vel_w": vel[3:]}

    def estimate(self) -> dict:
        """Simulated state estimate (truth + noise) -- stands in for an EKF
        fusing IMU, optical flow, rangefinder (e.g. ArduPilot EKF3)."""
        t = self.truth()
        n = self.noise
        R = t["R"]
        yaw = np.arctan2(R[1, 0], R[0, 0])
        omega_b = self._sensor("gyro") + self.rng.normal(0, n.gyro, 3)
        if self.gyro_fail_z:
            omega_b[2] = 0.0
        # downward rangefinder (along body -z), tilt-compensated -> height of body origin
        rng_raw = self._sensor("range_down")[0]
        alt = (rng_raw + self.range_offset) * max(R[2, 2], 0.5) if rng_raw > 0 else t["pos"][2]
        alt += self.rng.normal(0, n.alt)
        att_noise = np.deg2rad(n.att_deg) * self.rng.normal(0, 1, 3)
        Rn = R @ _rotvec_to_R(att_noise)
        est = {"pos": t["pos"], "R": Rn, "yaw": yaw + att_noise[2], "omega_b": omega_b,
               "vel_w": t["vel_w"] + self.rng.normal(0, n.vel, 3), "alt": alt,
               "bumper": float(self._sensor("bumper")[0])}
        self._est = est
        return est

    # --------------------------------------------------------------- control
    def set_command(self, cmd: Command) -> None:
        self.cmd = cmd

    def _set_mode(self, mode: str) -> None:
        if mode != self.mode:
            self.mode = mode
            self.mode_log.append((float(self.d.time), mode))

    def step_control(self, dt: float) -> None:
        """Run the low-level controllers for one physics step."""
        est = self.estimate()
        c = self.cmd
        t = self.d.time
        if self.mode == "ground":
            if c.takeoff or c.escape:
                self._set_mode("takeoff")
                self.alt_ref = max(est["alt"], self.p.rest_height)
                self.yaw_ref = est["yaw"]
                if c.escape:
                    self.escape_until = t + 1.2
            else:
                self._ground_drive(est, c)
                self._rotors(np.zeros(4))
                return
        self._wheels_stop()
        if self.mode == "takeoff":
            climb = self.fc.escape_climb if t < self.escape_until else self.fc.takeoff_climb
            target = self.fc.cruise_alt + (self.fc.escape_alt_gain if self.escape_until > 0 and t < self.escape_until + 2 else 0)
            self.alt_ref = min(self.alt_ref + climb * dt, target)
            if est["alt"] > target - 0.15:
                self._set_mode("flight")
            self._fly(est, v_fwd=0.0, yaw_rate=0.0, vz_ref=climb)
        elif self.mode == "flight":
            if c.escape and t > self.escape_until:
                self.escape_until = t + 1.0
            if c.land:
                self._set_mode("landing")
            climb = c.climb_rate + (self.fc.escape_climb if t < self.escape_until else 0.0)
            self.alt_ref = float(np.clip(self.alt_ref + climb * dt, self.fc.alt_min, self.fc.alt_max))
            v = 0.0 if c.halt else c.v_fwd
            self._fly(est, v_fwd=v, yaw_rate=c.yaw_rate, vz_ref=climb)
        elif self.mode == "landing":
            self.alt_ref = max(self.alt_ref - self.fc.land_rate * dt, -0.2)
            self._fly(est, v_fwd=0.0, yaw_rate=0.0, vz_ref=-self.fc.land_rate)
            if est["alt"] < self.p.rest_height + 0.02 and abs(est["vel_w"][2]) < 0.25:
                self._set_mode("ground")
                self._rotors(np.zeros(4))
                self.yaw_int = 0.0

    def _ground_drive(self, est: dict, c: Command) -> None:
        """Skid-steer: forward speed + yaw-rate loop on the gyro.

        Includes a low-level *contact retreat reflex* (analogous to local
        ventral-nerve-cord reflexes, not attributed to the brain): on touching
        an obstacle the robot backs up for 0.7 s while turning away from the
        contact side.  Used identically by every decision layer."""
        v = 0.0 if c.halt else c.v_fwd
        r_cmd = 0.0 if c.halt else c.yaw_rate
        t = self.d.time
        if t >= self.retreat_until:
            touching, side = self.contact()
            if touching and not c.halt:
                self.retreat_until = t + 0.7
                self.retreat_turn = -side if side != 0 else 1.0
                self.reflex_count += 1
        if t < self.retreat_until:
            v, r_cmd = -0.45, 1.4 * self.retreat_turn
        half_track = self.p.wheel_y
        r_meas = est["omega_b"][2]
        err = r_cmd - r_meas
        self.yaw_int = float(np.clip(self.yaw_int + err * 0.002, -1.0, 1.0))
        diff = 1.6 * r_cmd * half_track + 0.25 * err + 0.6 * self.yaw_int  # m/s per side
        vl, vr = v - diff, v + diff
        w = self.p.wheel_radius
        for n in WHEELS:
            spd = (vl if n[1] == "L" else vr) / w
            self.d.ctrl[self.wheel_act[n]] = float(np.clip(spd, -self.p.wheel_speed_max, self.p.wheel_speed_max))

    def _wheels_stop(self) -> None:
        for n in WHEELS:
            self.d.ctrl[self.wheel_act[n]] = 0.0

    def _rotors(self, f: np.ndarray) -> None:
        self.d.ctrl[self.rotor_act] = np.clip(f, 0.0, self.p.thrust_max)

    def _fly(self, est: dict, v_fwd: float, yaw_rate: float, vz_ref: float) -> None:
        fc, m = self.fc, self.mass
        R = est["R"]
        yaw = est["yaw"]
        if self.yaw_ref is None:
            self.yaw_ref = yaw
        self.yaw_ref = _wrap(self.yaw_ref + yaw_rate * 0.002)
        # horizontal velocity loop in the world frame
        h = np.array([np.cos(yaw), np.sin(yaw)])
        v_ref = h * v_fwd
        a_xy = fc.kv_xy * (v_ref - est["vel_w"][:2])
        a_z = fc.kp_z * (self.alt_ref - est["alt"]) + fc.kd_z * (vz_ref - est["vel_w"][2])
        F = m * np.array([a_xy[0], a_xy[1], a_z + G])
        # tilt limit
        tmax = np.tan(np.deg2rad(fc.max_tilt_deg))
        hor = np.linalg.norm(F[:2])
        if hor > tmax * F[2]:
            F[:2] *= tmax * F[2] / max(hor, 1e-6)
        F[2] = max(F[2], 0.2 * m * G)
        zb = F / np.linalg.norm(F)
        xc = np.array([np.cos(self.yaw_ref), np.sin(self.yaw_ref), 0.0])
        yb = np.cross(zb, xc)
        yb /= np.linalg.norm(yb)
        xb = np.cross(yb, zb)
        Rd = np.column_stack([xb, yb, zb])
        T = float(F @ R[:, 2])
        eR = 0.5 * _vee(Rd.T @ R - R.T @ Rd)
        omega = est["omega_b"]
        omega_d = np.array([0.0, 0.0, yaw_rate])
        eW = omega - R.T @ Rd @ omega_d
        J = self.J
        kR = np.array([J[0] * fc.att_wn ** 2, J[1] * fc.att_wn ** 2, J[2] * fc.yaw_wn ** 2])
        kW = np.array([2 * 0.8 * fc.att_wn * J[0], 2 * 0.8 * fc.att_wn * J[1], 2 * 0.9 * fc.yaw_wn * J[2]])
        if self.gyro_fail_z:
            # failed yaw gyro: no yaw damping and no heading reference (indoor, no
            # magnetometer).  The yaw command acts as a direct torque command
            # (feed-forward), so the only yaw feedback left is whatever the
            # decision layer derives from vision.
            kR[2] = 0.0
            kW[2] = 0.0
        tau = -kR * eR - kW * eW + np.cross(omega, J * omega)
        if self.gyro_fail_z:
            tau[2] = self.yaw_ff_gain * yaw_rate
        f = self.A_inv @ np.array([T, *tau])
        # saturation: preserve roll/pitch authority by shifting collective
        if f.max() > self.p.thrust_max:
            f -= f.max() - self.p.thrust_max
        self._rotors(f)

    # ---------------------------------------------------------------- physics
    def apply_aero(self) -> None:
        """Rotor drag: horizontal damping proportional to rotor thrust.
        Adds to ``xfrc_applied`` (cleared by the episode every physics step)."""
        b = self.body
        thrust = self.d.actuator_force[self.rotor_act]
        if thrust.sum() <= 1e-3:
            return
        t = self.truth()
        R = t["R"]
        v_b = R.T @ t["vel_w"]
        hover = self.mass * G / 4
        k = self.fc.rotor_drag * thrust.sum() / hover
        f_b = np.array([-k * v_b[0], -k * v_b[1], 0.0])
        self.d.xfrc_applied[b, :3] += R @ f_b

    def update_energy(self, dt: float) -> None:
        """Electrical power estimate: rotors via momentum theory with a
        figure of merit; wheels via torque x speed / efficiency."""
        thrust = np.maximum(self.d.actuator_force[self.rotor_act], 0.0)
        rho, A = 1.225, np.pi * self.p.prop_radius ** 2
        p_ind = np.sum(thrust ** 1.5) / np.sqrt(2 * rho * A)
        self.rotor_power_w = p_ind / (0.60 * 0.80)  # figure of merit 0.6, motor+ESC efficiency 0.8
        wp = 0.0
        for n in WHEELS:
            j = self.m.actuator(f"drive_{n}").trnid[0]
            qv = self.d.qvel[self.m.jnt_dofadr[j]]
            wp += abs(self.d.actuator_force[self.wheel_act[n]] * qv)
        self.wheel_power_w = wp / 0.55 + (0.6 if self.mode == "ground" else 0.0)  # gearmotor eff., driver idle
        self.energy_j += (self.rotor_power_w + self.wheel_power_w) * dt


def _vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _rotvec_to_R(v: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
