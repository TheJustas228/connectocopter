"""MJCF model of the Connectocopter hybrid FPV quadcopter / skid-steer rover.

Geometry and masses follow the hardware proposal in docs/hardware.md
(7-inch X frame, 295 mm motor-to-motor diagonal, ~1.25 kg all-up weight).

Physics modelling choices (see docs/robot.md):

* Rotors: thrust ``f_i`` acts along the rotor axis at each motor site, with a
  reaction (drag) torque ``s_i * k_m * f_i`` about the same axis.  Motor/prop
  spin-up lag is a first-order filter (MuJoCo ``dyntype="filter"``).
  Not modelled: blade flapping, ground effect, prop wash on the wheels,
  battery sag (documented limitations).
* Wheels: four independently powered wheels on hinge joints with velocity
  servos (gearmotor + encoder + speed loop) and a torque limit.  Steering is
  skid-steer (left/right wheel speed difference); there is no steering joint.
* Body drag: MuJoCo's inertia-box fluid model (air density 1.225 kg/m^3)
  plus a rotor-drag term applied in ``Robot`` (horizontal velocity damping
  proportional to thrust, the dominant drag effect for multirotors).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RobotParams:
    # frame
    diag: float = 0.295  # motor-to-motor diagonal (m), 7-inch class
    prop_radius: float = 0.089  # 7" propeller
    # masses (kg) -- see docs/hardware.md for the matching bill of materials
    m_frame: float = 0.205
    m_motor: float = 0.062  # per motor incl. prop and hardware
    m_battery: float = 0.330
    m_compute: float = 0.115  # Jetson Orin NX module + carrier + heatsink
    m_avionics: float = 0.045  # FC + 4-in-1 ESC + receiver + VTX
    m_sensors: float = 0.040  # cameras, flow/range sensor, antennae sensors
    m_wheel_unit: float = 0.034  # per wheel: tyre/hub + gearmotor + bracket
    m_misc: float = 0.040  # wiring, straps, fasteners
    # propulsion
    thrust_max: float = 16.0  # N per rotor (conservative for 2807 1300KV / 7x3.5 / 6S)
    k_m: float = 0.012  # yaw torque per newton of thrust (m)
    motor_tau: float = 0.035  # s, spin-up time constant
    # wheels
    wheel_radius: float = 0.040
    wheel_halfwidth: float = 0.009
    wheel_x: float = 0.062  # axle x offset (front/back)
    wheel_y: float = 0.088  # track half-width
    axle_z: float = -0.046  # axle height below body origin
    wheel_kv: float = 0.02  # velocity servo gain (N m per rad/s)
    wheel_torque_max: float = 0.25  # N m (gearmotor stall-limited)
    wheel_speed_max: float = 60.0  # rad/s (~2.4 m/s)
    # camera
    cam_uptilt_deg: float = 20.0
    cam_fovy_deg: float = 80.0

    @property
    def arm_len(self) -> float:
        return self.diag / 2

    @property
    def total_mass(self) -> float:
        return (self.m_frame + 4 * self.m_motor + self.m_battery + self.m_compute + self.m_avionics
                + self.m_sensors + 4 * self.m_wheel_unit + self.m_misc)

    @property
    def rotor_xy(self) -> np.ndarray:
        """Rotor positions [FL, FR, RR, RL] in the body frame (x fwd, y left)."""
        a = self.arm_len / np.sqrt(2)
        return np.array([[a, a], [a, -a], [-a, -a], [-a, a]])

    @property
    def rotor_spin(self) -> np.ndarray:
        """Sign of the reaction torque about +z for [FL, FR, RR, RL]."""
        return np.array([1.0, -1.0, 1.0, -1.0])

    @property
    def rest_height(self) -> float:
        return -self.axle_z + self.wheel_radius


ROTORS = ["FL", "FR", "RR", "RL"]
WHEELS = ["FL", "FR", "RL", "RR"]


def _f(x) -> str:
    return " ".join(f"{v:.6g}" for v in np.atleast_1d(x))


def robot_body_xml(p: RobotParams, pos=(0.0, 0.0, None), yaw: float = 0.0) -> str:
    z0 = p.rest_height + 0.002 if pos[2] is None else pos[2]
    q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    L = p.arm_len
    parts = [f'<body name="robot" pos="{pos[0]:.4f} {pos[1]:.4f} {z0:.4f}" quat="{_f(q)}">',
             '<freejoint name="root"/>',
             '<site name="imu" pos="0 0 0.01" size="0.004" rgba="0 0 0 0"/>',
             # --- frame: bottom plate, top plate, standoffs
             f'<geom name="plate_bot" type="box" size="0.078 0.042 0.002" pos="0 0 0" mass="{p.m_frame * 0.35:.4f}" material="carbon"/>',
             f'<geom name="plate_top" type="box" size="0.060 0.034 0.0015" pos="0 0 0.032" mass="{p.m_frame * 0.15:.4f}" material="carbon"/>']
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(f'<geom type="cylinder" size="0.0035 0.015" pos="{0.052 * sx:.3f} {0.026 * sy:.3f} 0.016" '
                         f'mass="0.003" material="alu" contype="0" conaffinity="0"/>')
    # --- arms, motors, prop discs, rotor sites
    for i, (name, (x, y)) in enumerate(zip(ROTORS, p.rotor_xy)):
        ang = np.arctan2(y, x)
        parts.append(f'<geom name="arm_{name}" type="box" size="{L / 2:.4f} 0.0115 0.003" '
                     f'pos="{x / 2:.4f} {y / 2:.4f} 0.0005" euler="0 0 {ang:.5f}" mass="{p.m_frame * 0.12:.4f}" material="carbon"/>')
        parts.append(f'<geom name="motor_{name}" type="cylinder" size="0.0175 0.0105" pos="{x:.4f} {y:.4f} 0.0135" '
                     f'mass="{p.m_motor:.4f}" material="motor"/>')
        parts.append(f'<geom name="bell_{name}" type="cylinder" size="0.012 0.004" pos="{x:.4f} {y:.4f} 0.027" '
                     f'mass="0" material="motor_top" contype="0" conaffinity="0"/>')
        parts.append(f'<geom name="prop_{name}" type="cylinder" size="{p.prop_radius:.4f} 0.001" pos="{x:.4f} {y:.4f} 0.031" '
                     f'mass="0" material="prop_disc_{"cw" if p.rotor_spin[i] > 0 else "ccw"}" contype="0" conaffinity="0" group="1"/>')
        parts.append(f'<site name="rotor_{name}" pos="{x:.4f} {y:.4f} 0.031" size="0.005" rgba="0 0 0 0"/>')
    # --- avionics stack, compute, battery
    parts += [
        f'<geom name="compute" type="box" size="0.034 0.028 0.009" pos="-0.008 0 0.0115" mass="{p.m_compute:.4f}" material="pcb"/>',
        f'<geom name="heatsink" type="box" size="0.026 0.022 0.003" pos="-0.008 0 0.0235" mass="0" material="alu" contype="0" conaffinity="0"/>',
        f'<geom name="avionics" type="box" size="0.018 0.018 0.004" pos="0.040 0 0.008" mass="{p.m_avionics:.4f}" material="pcb_blue"/>',
        f'<geom name="battery" type="box" size="0.056 0.019 0.0225" pos="-0.004 0 0.0565" mass="{p.m_battery:.4f}" material="battery"/>',
        f'<geom name="strap" type="box" size="0.006 0.0195 0.023" pos="0.02 0 0.0565" mass="0" material="strap" contype="0" conaffinity="0"/>',
        f'<geom name="strap2" type="box" size="0.006 0.0195 0.023" pos="-0.03 0 0.0565" mass="0" material="strap" contype="0" conaffinity="0"/>',
        f'<geom name="misc" type="box" size="0.03 0.02 0.004" pos="-0.045 0 0.006" mass="{p.m_misc:.4f}" rgba="0 0 0 0" contype="0" conaffinity="0"/>',
    ]
    # --- FPV camera (uptilted), with housing and lens
    tilt = np.deg2rad(p.cam_uptilt_deg)
    parts += [
        f'<body name="cam_mount" pos="0.074 0 0.017" euler="0 {-tilt:.5f} 0">',
        f'<geom name="cam_housing" type="box" size="0.010 0.0105 0.0105" mass="{p.m_sensors * 0.3:.4f}" material="cam"/>',
        '<geom name="cam_lens" type="cylinder" size="0.0065 0.004" pos="0.012 0 0" euler="0 1.5708 0" mass="0" material="lens" contype="0" conaffinity="0"/>',
        # MuJoCo cameras look along -z with +y up: rotate so it looks along +x
        f'<camera name="fpv" pos="0.016 0 0" xyaxes="0 -1 0 0 0 1" fovy="{p.cam_fovy_deg:.1f}"/>',
        '</body>',
    ]
    # --- antennae: odor / airflow sensor stalks (fly-inspired, functional)
    for side, sy in (("L", 1), ("R", -1)):
        parts.append(f'<geom name="antenna_{side}" type="capsule" fromto="0.06 {0.012 * sy:.3f} 0.034 0.105 {0.055 * sy:.3f} 0.07" '
                     f'size="0.0018" mass="{p.m_sensors * 0.05:.4f}" material="antenna" contype="0" conaffinity="0"/>')
        parts.append(f'<geom name="antenna_tip_{side}" type="sphere" size="0.006" pos="0.105 {0.055 * sy:.3f} 0.07" '
                     f'mass="0" material="sensor_tip" contype="0" conaffinity="0"/>')
        parts.append(f'<site name="nose_{side}" pos="0.105 {0.055 * sy:.3f} 0.07" size="0.003" rgba="0 0 0 0"/>')
    # --- downward optical-flow / range sensor, front bumper (touch)
    parts += [
        f'<geom name="flow_sensor" type="box" size="0.009 0.009 0.003" pos="0.02 0 -0.005" mass="{p.m_sensors * 0.2:.4f}" material="pcb_blue" contype="0" conaffinity="0"/>',
        '<site name="range_down" pos="0.02 0 -0.009" size="0.003" zaxis="0 0 -1" rgba="0 0 0 0"/>',
        '<geom name="bumper" type="capsule" fromto="0.085 0.03 -0.01 0.085 -0.03 -0.01" size="0.006" mass="0.004" material="bumper"/>',
        '<site name="bumper_site" type="box" pos="0.089 0 -0.01" size="0.012 0.04 0.012" rgba="0 0 0 0"/>',
        f'<geom name="sensors_misc" type="sphere" size="0.005" pos="0 0 0.0" mass="{p.m_sensors * 0.4:.4f}" rgba="0 0 0 0" contype="0" conaffinity="0"/>',
    ]
    # --- wheels: bracket + gearmotor on the chassis, wheel on a hinge
    for name in WHEELS:
        sx = 1 if name[0] == "F" else -1
        sy = 1 if name[1] == "L" else -1
        wx, wy, wz = sx * p.wheel_x, sy * p.wheel_y, p.axle_z
        m_tyre = p.m_wheel_unit * 0.45
        m_mount = p.m_wheel_unit - m_tyre
        parts.append(f'<geom name="strut_{name}" type="box" size="0.004 0.004 {(-wz) / 2:.4f}" pos="{wx:.4f} {sy * 0.050:.4f} {wz / 2:.4f}" '
                     f'mass="{m_mount * 0.3:.4f}" material="alu"/>')
        parts.append(f'<geom name="gearmotor_{name}" type="cylinder" size="0.0062 0.0175" pos="{wx:.4f} {sy * 0.0655:.4f} {wz:.4f}" '
                     f'euler="1.5708 0 0" mass="{m_mount * 0.7:.4f}" material="gearmotor"/>')
        parts.append(f'<body name="wheel_{name}" pos="{wx:.4f} {wy:.4f} {wz:.4f}">')
        parts.append(f'<joint name="wheel_{name}" type="hinge" axis="0 1 0" damping="0.0004" armature="0.0001"/>')
        parts.append(f'<geom name="tyre_{name}" type="cylinder" size="{p.wheel_radius:.4f} {p.wheel_halfwidth:.4f}" euler="1.5708 0 0" '
                     f'mass="{m_tyre * 0.8:.4f}" material="tyre" friction="0.9 0.01 0.001" solref="0.008 1" condim="3"/>')
        parts.append(f'<geom name="hub_{name}" type="cylinder" size="{p.wheel_radius * 0.55:.4f} {p.wheel_halfwidth + 0.0008:.4f}" '
                     f'euler="1.5708 0 0" mass="{m_tyre * 0.2:.4f}" material="hub" contype="0" conaffinity="0"/>')
        parts.append(f'<geom name="spoke_{name}" type="box" size="{p.wheel_radius * 0.5:.4f} {p.wheel_halfwidth + 0.0012:.4f} 0.003" '
                     f'mass="0" material="hub_dark" contype="0" conaffinity="0"/>')
        parts.append('</body>')
    parts.append("</body>")
    return "\n".join(parts)


def robot_assets_xml() -> str:
    return """
    <texture name="carbon_tex" type="2d" builtin="checker" rgb1="0.09 0.09 0.10" rgb2="0.13 0.13 0.14" width="64" height="64"/>
    <material name="carbon" texture="carbon_tex" texrepeat="12 12" specular="0.4" shininess="0.6" reflectance="0.05"/>
    <material name="alu" rgba="0.72 0.74 0.78 1" specular="0.8" shininess="0.8"/>
    <material name="motor" rgba="0.12 0.12 0.13 1" specular="0.6" shininess="0.7"/>
    <material name="motor_top" rgba="0.95 0.45 0.08 1" specular="0.7" shininess="0.8"/>
    <material name="prop_disc_cw" rgba="0.95 0.55 0.12 0.28"/>
    <material name="prop_disc_ccw" rgba="0.2 0.75 0.95 0.28"/>
    <material name="pcb" rgba="0.08 0.35 0.15 1"/>
    <material name="pcb_blue" rgba="0.1 0.2 0.55 1"/>
    <material name="battery" rgba="0.9 0.78 0.12 1" specular="0.3"/>
    <material name="strap" rgba="0.08 0.08 0.08 1"/>
    <material name="cam" rgba="0.15 0.15 0.16 1"/>
    <material name="lens" rgba="0.05 0.1 0.25 1" specular="1" shininess="1" reflectance="0.3"/>
    <material name="antenna" rgba="0.15 0.15 0.15 1"/>
    <material name="sensor_tip" rgba="0.85 0.15 0.15 1" emission="0.3"/>
    <material name="bumper" rgba="0.95 0.45 0.08 1"/>
    <material name="gearmotor" rgba="0.75 0.75 0.78 1" specular="0.8"/>
    <material name="tyre" rgba="0.06 0.06 0.06 1" specular="0.1"/>
    <material name="hub" rgba="0.95 0.45 0.08 1"/>
    <material name="hub_dark" rgba="0.25 0.25 0.27 1"/>
    """


def actuators_xml(p: RobotParams) -> str:
    out = []
    for i, name in enumerate(ROTORS):
        out.append(f'<general name="thrust_{name}" site="rotor_{name}" gear="0 0 1 0 0 {p.rotor_spin[i] * p.k_m:.5f}" '
                   f'dyntype="filter" dynprm="{p.motor_tau}" gainprm="1" ctrlrange="0 {p.thrust_max}"/>')
    for name in WHEELS:
        out.append(f'<velocity name="drive_{name}" joint="wheel_{name}" kv="{p.wheel_kv}" '
                   f'ctrlrange="{-p.wheel_speed_max} {p.wheel_speed_max}" forcerange="{-p.wheel_torque_max} {p.wheel_torque_max}"/>')
    return "\n".join(out)


def sensors_xml() -> str:
    return """
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
    <framequat name="quat" objtype="site" objname="imu"/>
    <framepos name="pos" objtype="site" objname="imu"/>
    <velocimeter name="vel_body" site="imu"/>
    <rangefinder name="range_down" site="range_down"/>
    <touch name="bumper" site="bumper_site"/>
    <jointvel name="wheel_FL" joint="wheel_FL"/>
    <jointvel name="wheel_FR" joint="wheel_FR"/>
    <jointvel name="wheel_RL" joint="wheel_RL"/>
    <jointvel name="wheel_RR" joint="wheel_RR"/>
    """
