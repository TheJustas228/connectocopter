"""All robot sensors -> one feature dictionary per control step.

Feature naming is shared by the connectome encoders (configs/interface.yaml)
and the baseline controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .odor import PuffPlume
from .vision import FlyEye, VisionConfig


@dataclass
class Environment:
    """Non-visual stimuli provided by the task."""

    pads: list = field(default_factory=list)  # dicts: pos, size, taste in {sugar, bitter, mixed}
    wind: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world-frame wind (m/s)
    plume: PuffPlume | None = None
    vibration: tuple[float, float] = (0.0, 0.0)  # current (L, R) vibration amplitude 0..1


class SensorSuite:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot, env: Environment,
                 width: int = 160, height: int = 120, vision: VisionConfig = VisionConfig(), render: bool = True):
        self.m, self.d, self.robot, self.env = model, data, robot, env
        self.renderer = mujoco.Renderer(model, height, width) if render else None
        if self.renderer is not None:  # the robot's camera does not need shadows / mirror passes
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self.eye = FlyEye(vision)
        self.nose = [model.site("nose_L").id, model.site("nose_R").id]
        self.last_frame = None
        self.features: dict = FlyEye._empty()
        # Forward multizone time-of-flight fan (cf. ST VL53L8CX): 25 azimuths x 2 elevations,
        # 4 m range.  Only geom group 0 (obstacles, walls, threats) is sensed; the floor is group 2.
        az = np.deg2rad(np.arange(-60, 61, 5))
        el = np.deg2rad([0.0, 10.0])
        A, E = np.meshgrid(az, el)
        self.tof_az = A.ravel()
        self.tof_dirs_b = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        self.tof_origin_b = np.array([0.095, 0.0, 0.0])
        self.tof_range = 4.0
        self.tof_dist = np.full(len(self.tof_az), self.tof_range)
        self.robot_body = model.body("robot").id

    def _tof(self) -> np.ndarray:
        d = self.d
        t = self.robot.truth()
        R, p = t["R"], t["pos"]
        origin = p + R @ self.tof_origin_b
        n = len(self.tof_az)
        vec = (self.tof_dirs_b @ R.T).ravel()
        geomid = np.zeros(n, dtype=np.int32)
        dist = np.zeros(n)
        mujoco.mj_multiRay(self.m, d, origin, vec, np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), 1,
                           self.robot_body, geomid, dist, None, n, self.tof_range)
        dist[(dist < 0) | (dist > self.tof_range)] = self.tof_range
        self.tof_dist = dist
        return dist

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()

    def read(self) -> dict:
        d = self.d
        f = {}
        if self.renderer is not None:
            self.renderer.update_scene(d, "fpv")
            frame = self.renderer.render()
            self.last_frame = frame
            f.update(self.eye.process(frame, d.time))
        else:
            f.update(FlyEye._empty())
        # approach rate (~1/time-to-contact) per side from the ToF fan and odometry speed
        dist = self._tof()
        t_ = self.robot.truth()
        v_fwd = float((t_["R"].T @ t_["vel_w"])[0])
        appr = np.cos(self.tof_az) * max(v_fwd, 0.3) / np.maximum(dist, 0.05)
        f["approach_L"] = float(appr[self.tof_az >= 0].max())
        f["approach_R"] = float(appr[self.tof_az <= 0].max())
        f["tof_min"] = float(dist.min())
        # LC10a drive: target in left/right hemifield (centred target drives both equally)
        pres, az = f["target_present"], f["target_az"]
        f["target_drive_L"] = pres * float(np.clip(0.5 - az / 30.0, 0.0, 1.0))
        f["target_drive_R"] = pres * float(np.clip(0.5 + az / 30.0, 0.0, 1.0))
        # contact chemosensation: pad under the chassis while on the ground
        t = self.robot.truth()
        on_ground = self.robot.mode == "ground"
        sugar = bitter = 0.0
        if on_ground:
            x, y = t["pos"][:2]
            for p in self.env.pads:
                (px, py), (sx, sy) = p["pos"][:2], p.get("size", (0.35, 0.35))
                if abs(x - px) <= sx and abs(y - py) <= sy:
                    sugar += 1.0 if p["taste"] in ("sugar", "mixed") else 0.0
                    bitter += 1.0 if p["taste"] in ("bitter", "mixed") else 0.0
        f["taste_sugar"], f["taste_bitter"] = min(sugar, 1.0), min(bitter, 1.0)
        # apparent airflow at the antennae (wind minus own velocity), body frame
        R = t["R"]
        air_b = R.T @ (self.env.wind - t["vel_w"])
        lateral = air_b[1]  # +ve: air moving toward the robot's left (wind from the right)
        speed = float(np.linalg.norm(air_b[:2]))
        f["wind_speed"] = speed
        f["wind_side"] = float(np.clip(-lateral, -3, 3))  # +ve: wind coming from the left
        # antennal deflection proxy: the windward antenna is deflected more
        f["wind_L"] = max(-lateral, 0.0) + 0.1 * speed
        f["wind_R"] = max(lateral, 0.0) + 0.1 * speed
        f["vibration_L"], f["vibration_R"] = self.env.vibration
        f["bumper"] = float(self.robot._est.get("bumper", 0.0)) if self.robot._est else 0.0
        # odor at the two nose sites
        if self.env.plume is not None:
            f["odor_L"] = self.env.plume.concentration(d.site_xpos[self.nose[0]])
            f["odor_R"] = self.env.plume.concentration(d.site_xpos[self.nose[1]])
        self.features = f
        return f
