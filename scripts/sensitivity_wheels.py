#!/usr/bin/env python3
"""Sensitivity check: does ground behaviour depend on the simulator's generous
wheel actuators?  Re-runs the rolling tasks with the limits of the gearmotor in
the hardware proposal (Pololu 100:1 HPCB 12 V: 0.127 N m stall torque,
330 rpm = 34.6 rad/s no-load) on the first 10 evaluation seeds and compares with
the main benchmark (same seeds, simulator defaults 0.25 N m / 60 rad/s).

Output: results/sensitivity_wheels.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import connectocopter  # noqa: E402,F401
from connectocopter.control.connectome import ConnectomeController  # noqa: E402
from connectocopter.robot.model import RobotParams  # noqa: E402
from connectocopter.sim.episode import Episode  # noqa: E402
from connectocopter.tasks.navigation import ObstacleCourse, TargetSeek  # noqa: E402
from connectocopter.tasks.taste import TasteDock  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = list(range(1000, 1010))
REAL = RobotParams(wheel_torque_max=0.127, wheel_speed_max=34.6)


def main():
    ctrl = ConnectomeController(seed=0)
    out = {"params": {"wheel_torque_max": REAL.wheel_torque_max, "wheel_speed_max": REAL.wheel_speed_max}, "tasks": {}}
    for T, key in [(ObstacleCourse, "obstacle_course"), (TargetSeek, "target_seek"), (TasteDock, "taste_dock")]:
        raw = ROOT / "results" / "benchmarks" / "raw" / f"{key}__connectome.jsonl"
        ref = {json.loads(l)["seed"]: json.loads(l)["metrics"] for l in raw.read_text().splitlines()}
        rows = []
        for s in SEEDS:
            m = Episode(T(), ctrl, s, params=REAL).run().metrics
            r = ref.get(s, {})
            rows.append({"seed": s, "success_real_wheels": bool(m["success"]), "success_sim_default": bool(r.get("success")),
                         "time_real": m.get("time_to_goal_s"), "time_default": r.get("time_to_goal_s"),
                         "collisions_real": m.get("collisions"), "collisions_default": r.get("collisions")})
            print(key, s, rows[-1], flush=True)
        out["tasks"][key] = {
            "success_real_wheels": sum(x["success_real_wheels"] for x in rows),
            "success_sim_default": sum(x["success_sim_default"] for x in rows), "n": len(rows), "episodes": rows}
    (ROOT / "results" / "sensitivity_wheels.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("success_real_wheels", "success_sim_default", "n")} for k, v in out["tasks"].items()}, indent=1))


if __name__ == "__main__":
    main()
