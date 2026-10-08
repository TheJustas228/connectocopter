#!/usr/bin/env python3
"""Export static assets for the browser viewer (web/assets/).

* robot.json   -- every visual geom of the simulated robot, read from the
                  compiled MuJoCo model (type, size, local pose, colour), so the
                  browser model is the simulated model.
* brain.bin    -- FlyWire v783 neuron positions (a representative supervoxel
                  per neuron from the annotation table, given in 4x4x40 nm
                  voxels and converted to nm here), int16 in units of 50 nm, x/y/z
                  interleaved, in model-index order (CC-BY 4.0: Dorkenwald et al.
                  2024; Schlegel et al. 2024).
* brain.json   -- metadata: count, super-class codes per neuron (uint8 in
                  brain_class.bin), and the neuron indices of every encoder /
                  decoder population used by the interface (for highlighting).
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import connectocopter  # noqa: F401
from connectocopter.brain.populations import resolve
from connectocopter.control.connectome import get_connectome, load_interface
from connectocopter.robot.model import RobotParams
from connectocopter.robot.world import Arena, build_xml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "assets"

GEOM_TYPES = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
              int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
              int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid"}


def export_robot() -> None:
    p = RobotParams()
    m = mujoco.MjModel.from_xml_string(build_xml(Arena(size=(2, 2), walls=False), p))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    root = m.body("robot").id
    parts = []
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        bb = b
        while m.body_parentid[bb] != 0:
            bb = m.body_parentid[bb]
        if bb != root:
            continue
        t = GEOM_TYPES.get(int(m.geom_type[g]))
        if t is None:
            continue
        rgba = m.geom_rgba[g].copy()
        mat = m.geom_matid[g]
        if mat >= 0:
            rgba = m.mat_rgba[mat].copy()
        if rgba[3] == 0:
            continue
        name = m.geom(g).name
        body_name = m.body(b).name
        parts.append({
            "name": name, "body": body_name, "type": t,
            "size": [round(float(v), 5) for v in m.geom_size[g]],
            "pos": [round(float(v), 5) for v in m.geom_pos[g]],
            "quat": [round(float(v), 5) for v in m.geom_quat[g]],
            "rgba": [round(float(v), 3) for v in rgba],
            "emission": round(float(m.mat_emission[mat]), 3) if mat >= 0 else 0.0,
        })
    bodies = {}
    for name in ["wheel_FL", "wheel_FR", "wheel_RL", "wheel_RR", "cam_mount"]:
        bid = m.body(name).id
        bodies[name] = {"pos": [round(float(v), 5) for v in m.body_pos[bid]],
                        "quat": [round(float(v), 5) for v in m.body_quat[bid]]}
    out = {"parts": parts, "bodies": bodies, "rotor_xy": p.rotor_xy.round(5).tolist(),
           "rotor_spin": p.rotor_spin.tolist(), "prop_radius": p.prop_radius, "mass_kg": round(p.total_mass, 3)}
    (OUT / "robot.json").write_text(json.dumps(out))
    print(f"robot.json: {len(parts)} geoms")


def export_brain() -> None:
    c = get_connectome("783")
    ann = c.annotations
    xyz = ann[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float) * np.array([4.0, 4.0, 40.0])  # voxels -> nm
    ok = np.isfinite(xyz).all(1)
    center = np.nanmedian(xyz[ok], 0)
    q = np.zeros_like(xyz)
    q[ok] = (xyz[ok] - center) / 50.0  # units of 50 nm -> fits int16 (+-1.6 mm)
    q = np.clip(np.round(q), -32767, 32767).astype(np.int16)
    q[~ok] = 0
    (OUT / "brain.bin").write_bytes(q.tobytes())
    classes = ["central", "optic", "sensory", "visual_projection", "ascending", "descending", "sensory_ascending",
               "visual_centrifugal", "motor", "endocrine"]
    code = np.array([classes.index(s) if s in classes else 255 for s in ann.super_class.fillna("").to_numpy()], dtype=np.uint8)
    (OUT / "brain_class.bin").write_bytes(code.tobytes())
    cfg = load_interface()
    pops = {}
    for e in cfg["encoders"]:
        pops[e["name"]] = {"role": "input", "idx": resolve(c, e["neurons"]).tolist(), "evidence": e.get("evidence", "")}
    side = ann.side.to_numpy()
    st = cfg["decoders"]["steering"]
    for t in list(st["ipsiversive"]) + list(st["contraversive"]):
        idx = c.select(cell_type=t)
        for s, lab in [("left", "L"), ("right", "R")]:
            pops[f"{t}_{lab}"] = {"role": "output", "idx": idx[side[idx] == s].tolist()}
    for name, key in [("GF", "escape"), ("MDN", "reverse"), ("MN9", "feeding_halt"), ("DNg02", "flight_power")]:
        idx = resolve(c, cfg["decoders"][key]["neurons"])
        for s, lab in [("left", "L"), ("right", "R")]:
            pops[f"{name}_{lab}"] = {"role": "output", "idx": idx[side[idx] == s].tolist()}
    meta = {"n": int(c.n_neurons), "units_nm": 50, "classes": classes, "populations": pops,
            "n_connections": int(c.n_connections), "n_synapses": int(c.n_synapses),
            "attribution": "FlyWire v783 (Dorkenwald et al. 2024; Schlegel et al. 2024), CC-BY 4.0"}
    (OUT / "brain.json").write_text(json.dumps(meta))
    print(f"brain.bin: {c.n_neurons} neurons, {len(pops)} populations")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    export_robot()
    export_brain()
