"""World builder: arena + robot -> MuJoCo model.

Arenas are described by simple Python dicts (built by the task modules) so a
task fully determines its geometry from a random seed.  Supported objects:

* ``box`` / ``cylinder`` obstacles (static, collidable)
* ``beacon``: emissive sphere on a post (visual target / light source)
* ``pad``: flat coloured floor patch (taste pads: sugar, bitter, mixed)
* ``looming``: a dark sphere on a mocap body, moved by the task (threat)
* ``source``: odor source marker (visual only; the plume is simulated separately)
* ``gate``: flight gate (two posts + top bar)
* walls around the arena with high-contrast textures (for optic flow)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import RobotParams, actuators_xml, robot_assets_xml, robot_body_xml, sensors_xml

PAD_COLORS = {"sugar": "1.0 0.80 0.15 1", "bitter": "0.25 0.75 0.30 1", "mixed": "0.95 0.45 0.10 1", "plain": "0.6 0.6 0.6 1"}


@dataclass
class Arena:
    size: tuple[float, float] = (12.0, 6.0)  # half extents (x, y) of the floor
    center: tuple[float, float] = (0.0, 0.0)
    wall_height: float = 2.5
    walls: bool = True
    objects: list[dict] = field(default_factory=list)
    robot_pos: tuple[float, float, float | None] = (0.0, 0.0, None)
    robot_yaw: float = 0.0
    floor_texture: str = "grid"  # "grid" or "noise"


def _arena_assets() -> str:
    return """
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.68 0.85" rgb2="0.95 0.95 0.97" width="512" height="3072"/>
    <texture name="grid_tex" type="2d" builtin="checker" rgb1="0.30 0.32 0.35" rgb2="0.42 0.44 0.47" width="512" height="512" mark="edge" markrgb="0.2 0.2 0.22"/>
    <material name="grid" texture="grid_tex" texrepeat="2 2" texuniform="true" reflectance="0.02"/>
    <texture name="noise_tex" type="2d" builtin="flat" rgb1="0.4 0.42 0.44" rgb2="0.2 0.2 0.2" mark="random" random="0.25" markrgb="0.15 0.15 0.15" width="1024" height="1024"/>
    <material name="noise" texture="noise_tex" texrepeat="4 4" texuniform="true"/>
    <texture name="wall_tex" type="2d" builtin="checker" rgb1="0.92 0.92 0.90" rgb2="0.42 0.44 0.48" width="256" height="256"/>
    <material name="wall" texture="wall_tex" texrepeat="1 1" texuniform="true"/>
    <texture name="obst_tex" type="2d" builtin="checker" rgb1="0.75 0.22 0.18" rgb2="0.95 0.92 0.88" width="64" height="64"/>
    <material name="obstacle" texture="obst_tex" texrepeat="3 3" texuniform="true"/>
    <material name="beacon" rgba="1.0 0.95 0.55 1" emission="1.0"/>
    <material name="post" rgba="0.35 0.35 0.38 1"/>
    <material name="looming" rgba="0.01 0.01 0.012 1" specular="0" shininess="0"/>
    <material name="source" rgba="0.95 0.85 0.20 1" emission="0.2"/>
    <material name="gate" rgba="0.15 0.65 0.95 1" emission="0.15"/>
    """


def _object_xml(i: int, o: dict) -> str:
    kind = o["kind"]
    x, y = o["pos"][:2]
    if kind == "box":
        sx, sy, sz = o["size"]
        return (f'<geom name="obst_{i}" type="box" pos="{x:.3f} {y:.3f} {sz:.3f}" size="{sx:.3f} {sy:.3f} {sz:.3f}" '
                f'euler="0 0 {o.get("yaw", 0):.3f}" material="obstacle"/>')
    if kind == "cylinder":
        r, h = o["size"]
        return f'<geom name="obst_{i}" type="cylinder" pos="{x:.3f} {y:.3f} {h:.3f}" size="{r:.3f} {h:.3f}" material="obstacle"/>'
    if kind == "beacon":
        h = o.get("height", 0.35)
        r = o.get("radius", 0.18)
        return (f'<geom name="post_{i}" type="cylinder" pos="{x:.3f} {y:.3f} {h / 2:.3f}" size="0.02 {h / 2:.3f}" material="post" contype="0" conaffinity="0"/>'
                f'<geom name="beacon_{i}" type="sphere" pos="{x:.3f} {y:.3f} {h + r:.3f}" size="{r:.3f}" material="beacon" contype="0" conaffinity="0"/>'
                f'<light name="beacon_light_{i}" pos="{x:.3f} {y:.3f} {h + r:.3f}" diffuse="0.6 0.55 0.3" attenuation="1 0.4 0.2" castshadow="false"/>')
    if kind == "pad":
        sx, sy = o.get("size", (0.35, 0.35))
        rgba = PAD_COLORS[o["taste"]]
        return (f'<geom name="pad_{i}" type="box" pos="{x:.3f} {y:.3f} 0.001" size="{sx:.3f} {sy:.3f} 0.001" rgba="{rgba}" '
                f'contype="0" conaffinity="0"/>')
    if kind == "looming":
        r = o.get("radius", 0.25)
        z = o["pos"][2] if len(o["pos"]) > 2 else 0.4
        return (f'<body name="looming_{i}" mocap="true" pos="{x:.3f} {y:.3f} {z:.3f}">'
                f'<geom name="loom_{i}" type="sphere" size="{r:.3f}" material="looming" contype="0" conaffinity="0"/></body>')
    if kind == "source":
        return (f'<geom name="source_{i}" type="ellipsoid" pos="{x:.3f} {y:.3f} 0.08" size="0.10 0.06 0.08" material="source" '
                f'contype="0" conaffinity="0"/>')
    if kind == "gate":
        w, h = o.get("width", 1.2), o.get("height", 1.6)
        yaw = o.get("yaw", 0.0)
        c, s = np.cos(yaw), np.sin(yaw)
        px, py = -s * w / 2, c * w / 2
        return (f'<geom name="gateL_{i}" type="cylinder" pos="{x + px:.3f} {y + py:.3f} {h / 2:.3f}" size="0.04 {h / 2:.3f}" material="gate"/>'
                f'<geom name="gateR_{i}" type="cylinder" pos="{x - px:.3f} {y - py:.3f} {h / 2:.3f}" size="0.04 {h / 2:.3f}" material="gate"/>'
                f'<geom name="gateT_{i}" type="box" pos="{x:.3f} {y:.3f} {h:.3f}" size="0.04 {w / 2:.3f} 0.04" euler="0 0 {yaw:.3f}" material="gate"/>')
    raise ValueError(f"unknown object kind {kind}")


def build_xml(arena: Arena, p: RobotParams = RobotParams(), timestep: float = 0.002) -> str:
    hx, hy = arena.size
    cx, cy = arena.center
    walls = ""
    if arena.walls:
        t, h = 0.05, arena.wall_height
        walls = "\n".join([
            f'<geom name="wall_N" type="box" pos="{cx:.3f} {cy + hy + t:.3f} {h / 2:.3f}" size="{hx + 2 * t:.3f} {t} {h / 2:.3f}" material="wall"/>',
            f'<geom name="wall_S" type="box" pos="{cx:.3f} {cy - hy - t:.3f} {h / 2:.3f}" size="{hx + 2 * t:.3f} {t} {h / 2:.3f}" material="wall"/>',
            f'<geom name="wall_E" type="box" pos="{cx + hx + t:.3f} {cy:.3f} {h / 2:.3f}" size="{t} {hy:.3f} {h / 2:.3f}" material="wall"/>',
            f'<geom name="wall_W" type="box" pos="{cx - hx - t:.3f} {cy:.3f} {h / 2:.3f}" size="{t} {hy:.3f} {h / 2:.3f}" material="wall"/>',
        ])
    objs = "\n".join(_object_xml(i, o) for i, o in enumerate(arena.objects))
    return f"""
<mujoco model="connectocopter_world">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" integrator="implicitfast" density="1.225" viscosity="1.8e-5"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="50"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.42 0.42 0.45" diffuse="0.40 0.40 0.40" specular="0.1 0.1 0.1"/>
    <map znear="0.01" zfar="200" haze="0.15"/>
  </visual>
  <asset>
    {_arena_assets()}
    {robot_assets_xml()}
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 10" dir="-0.3 -0.2 -1" diffuse="0.7 0.7 0.68" specular="0.2 0.2 0.2" castshadow="true"/>
    <geom name="floor" type="plane" pos="{cx:.3f} {cy:.3f} 0" size="{hx:.3f} {hy:.3f} 0.1" material="{arena.floor_texture}" friction="0.9 0.01 0.001" group="2"/>
    {walls}
    {objs}
    <camera name="chase_static" mode="targetbody" target="robot" pos="{cx - hx * 0.6:.2f} {cy - hy * 0.9:.2f} 3.0"/>
    <camera name="overhead" pos="{cx:.2f} {cy:.2f} {max(hx, hy) * 1.9:.2f}" xyaxes="1 0 0 0 1 0" fovy="60"/>
    {robot_body_xml(p, arena.robot_pos, arena.robot_yaw)}
  </worldbody>
  <actuator>
    {actuators_xml(p)}
  </actuator>
  <sensor>
    {sensors_xml()}
  </sensor>
</mujoco>
"""
