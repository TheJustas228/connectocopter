"""Closed-loop episode runner.

Loop (every control step, default 20 ms of simulated time):

    task stimuli -> SensorSuite.read() -> controller.step() -> mission layer
      -> Robot low-level control (500 Hz) -> MuJoCo physics (2 ms steps)

The *mission layer* is task logic that is explicitly not attributed to the
brain (e.g. "take off at t=1 s", "land when over the landing pad").  Every
mission override is logged.
"""
from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..robot.model import RobotParams, WHEELS
from ..robot.robot import Command, Robot
from ..robot.world import build_xml
from ..sensors.suite import SensorSuite


@dataclass
class EpisodeResult:
    task: str
    controller: str
    seed: int
    metrics: dict
    sim_time: float
    wall_time: float
    replay: dict | None = None


class Episode:
    def __init__(self, task, controller, seed: int, record: bool = False, record_fpv_every: int = 5,
                 render: bool = True, params: RobotParams = RobotParams(), fpv_size=(160, 120)):
        self.task, self.ctrl, self.seed = task, controller, seed
        self.record = record
        self.record_fpv_every = record_fpv_every
        self.rng = np.random.default_rng(seed)
        arena, env = task.build(np.random.default_rng(seed))
        self.arena, self.env = arena, env
        self.model = mujoco.MjModel.from_xml_string(build_xml(arena, params))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.robot = Robot(self.model, self.data, params, rng=np.random.default_rng(seed + 7))
        self.sensors = SensorSuite(self.model, self.data, self.robot, env, width=fpv_size[0], height=fpv_size[1], render=render)
        self.ctrl_dt = getattr(controller, "dt", 0.02)
        self.n_phys = int(round(self.ctrl_dt / self.model.opt.timestep))
        robot_body = self.model.body("robot").id
        self.robot_geoms = {g for g in range(self.model.ngeom)
                            if self._root_of(self.model.geom_bodyid[g]) == robot_body}
        self.obstacle_geoms = {g for g in range(self.model.ngeom)
                               if self.model.geom(g).name.startswith(("obst_", "wall_", "gate"))}
        self.collisions = 0
        self._in_contact = False
        self.path_len = 0.0
        self._last_xy = None
        self.frames: list = []
        task.setup(self)

    def _root_of(self, b: int) -> int:
        while self.model.body_parentid[b] != 0:
            b = self.model.body_parentid[b]
        return b

    def _check_contacts(self) -> bool:
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.robot_geoms and g2 in self.obstacle_geoms) or (g2 in self.robot_geoms and g1 in self.obstacle_geoms):
                return True
        return False

    def run(self) -> EpisodeResult:
        m, d, robot, task = self.model, self.data, self.robot, self.task
        dt = m.opt.timestep
        t_wall = time.time()
        step = 0
        if hasattr(self.ctrl, "set_seed"):
            self.ctrl.set_seed(1_000_003 * self.seed + 17)
        if hasattr(self.ctrl, "reset"):
            self.ctrl.reset()
        while True:
            task.pre_step(self, d.time)
            feats = self.sensors.read()
            cmd = self.ctrl.step(feats, robot.mode)
            cmd = task.mission(self, cmd, d.time)
            robot.set_command(cmd)
            contact_now = False
            for _ in range(self.n_phys):
                d.xfrc_applied[:] = 0.0
                task.physics_tick(self, d.time, dt)
                robot.step_control(dt)
                robot.apply_aero()
                mujoco.mj_step(m, d)
                robot.update_energy(dt)
                contact_now |= self._check_contacts()
            if contact_now and not self._in_contact:
                self.collisions += 1
            self._in_contact = contact_now
            xy = robot.truth()["pos"][:2]
            if self._last_xy is not None:
                self.path_len += float(np.linalg.norm(xy - self._last_xy))
            self._last_xy = xy
            if self.record:
                self._record(feats, cmd, step)
            step += 1
            if task.done(self, d.time) or d.time >= task.max_time:
                break
        metrics = task.metrics(self)
        metrics.update({"collisions": self.collisions, "path_length_m": round(self.path_len, 3),
                        "energy_J": round(robot.energy_j, 1), "mode_log": robot.mode_log})
        wall = time.time() - t_wall
        metrics["realtime_factor"] = round(d.time / wall, 3)
        self.sensors.close()
        replay = self._replay_dict(metrics) if self.record else None
        return EpisodeResult(task.name, self.ctrl.label, self.seed, metrics, float(d.time), wall, replay)

    # ---------------------------------------------------------------- record
    def _record(self, feats: dict, cmd: Command, step: int) -> None:
        d, r = self.data, self.robot
        t = r.truth()
        wheel_ang = [float(d.qpos[self.model.jnt_qposadr[self.model.joint(f"wheel_{n}").id]]) for n in WHEELS]
        fr = {
            "t": round(float(d.time), 3),
            "pos": [round(float(v), 4) for v in t["pos"]],
            "quat": [round(float(v), 4) for v in t["quat"]],
            "mode": r.mode,
            "thrust": [round(float(v), 3) for v in d.actuator_force[r.rotor_act]],
            "wheels": [round(v, 3) for v in wheel_ang],
            "mocap": [[round(float(v), 3) for v in p] for p in d.mocap_pos],
            "cmd": {"v": round(cmd.v_fwd, 3), "yaw": round(cmd.yaw_rate, 3), "escape": bool(cmd.escape),
                    "halt": bool(cmd.halt), "takeoff": bool(cmd.takeoff), "land": bool(cmd.land)},
            "feat": {k: round(float(v), 5) for k, v in feats.items()},
            "power_W": round(r.rotor_power_w + r.wheel_power_w, 1),
            "collisions": self.collisions,
        }
        tel = getattr(self.ctrl, "tel", None)
        if tel is not None:
            fr["brain"] = {"in": {k: round(v, 1) for k, v in tel.input_hz.items()},
                           "dn": {k: round(v, 1) for k, v in tel.dn_hz.items()},
                           "steer": round(tel.steer, 2), "gf": tel.gf_spikes,
                           "n_active": tel.n_active, "spikes": tel.total_spikes}
            if tel.active_idx is not None:
                idx = tel.active_idx
                if len(idx) > 3000:
                    idx = np.sort(self.rng.choice(idx, 3000, replace=False))
                fr["brain"]["act"] = base64.b64encode(idx.astype(np.uint32).tobytes()).decode()
        extra = self.task.telemetry(self) if hasattr(self.task, "telemetry") else None
        if extra:
            fr["task"] = extra
        if self.sensors.last_frame is not None and step % self.record_fpv_every == 0:
            import imageio.v3 as iio
            buf = io.BytesIO()
            iio.imwrite(buf, self.sensors.last_frame, extension=".jpg", quality=70)
            fr["fpv"] = base64.b64encode(buf.getvalue()).decode()
        self.frames.append(fr)

    def _replay_dict(self, metrics: dict) -> dict:
        objs = []
        for o in self.arena.objects:
            oo = {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()}
            objs.append(oo)
        return {
            "task": self.task.name, "controller": self.ctrl.label, "seed": self.seed,
            "arena": {"size": list(self.arena.size), "center": list(self.arena.center), "walls": self.arena.walls,
                      "wall_height": self.arena.wall_height, "objects": objs},
            "pads": self.env.pads, "control_dt": self.ctrl_dt, "metrics": metrics, "frames": self.frames,
        }
