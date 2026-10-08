"""Command-line interface.

    connectocopter tasks                                   # list tasks
    connectocopter run obstacle_course --controller connectome --seed 7 --record
    connectocopter serve --port 8765                        # viewer + live simulations

``run`` executes one closed-loop episode headless and prints its metrics; with
``--record`` it also writes a replay to web/replays/ and adds it to the
viewer's episode list.  ``serve`` hosts the browser viewer and a WebSocket
endpoint that streams a brand-new simulation into the browser as it runs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "looming_escape": ("connectocopter.tasks.looming", "LoomingEscape", {"catch": False}),
    "vibration_escape": ("connectocopter.tasks.looming", "LoomingEscape", {"catch": False, "stimulus": "vibration"}),
    "target_seek": ("connectocopter.tasks.navigation", "TargetSeek", {}),
    "obstacle_course": ("connectocopter.tasks.navigation", "ObstacleCourse", {}),
    "flight_course": ("connectocopter.tasks.flight", "FlightCourse", {}),
    "taste_dock": ("connectocopter.tasks.taste", "TasteDock", {}),
    "yaw_stabilization": ("connectocopter.tasks.flight", "YawStabilization", {}),
    "odor_plume": ("connectocopter.tasks.odor", "OdorPlume", {}),
}


def make_task(name: str):
    import importlib

    mod, cls, kw = TASKS[name]
    return getattr(importlib.import_module(mod), cls)(**kw)


def make_controller(kind: str, record_activity: bool = False):
    if kind == "baseline":
        from .control.baseline import BaselineController

        return BaselineController()
    from .control.connectome import ConnectomeController

    return ConnectomeController(seed=0, record_activity=record_activity)


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def cmd_run(a) -> None:
    from .sim.episode import Episode

    ctrl = make_controller(a.controller, record_activity=a.record)
    res = Episode(make_task(a.task), ctrl, a.seed, record=a.record, record_fpv_every=3).run()
    m = {k: v for k, v in res.metrics.items()}
    print(json.dumps({"task": a.task, "controller": a.controller, "seed": a.seed, "sim_time_s": round(res.sim_time, 2),
                      "wall_time_s": round(res.wall_time, 2), "metrics": m}, indent=1, default=_json_default))
    if a.record:
        out = ROOT / "web" / "replays"
        out.mkdir(parents=True, exist_ok=True)
        name = f"run_{a.task}_{a.controller}_{a.seed}.json.gz"
        r = res.replay
        r["task_title"] = f"{a.task} (your run)"
        payload = json.dumps(r, separators=(",", ":"), default=_json_default).encode()
        (out / name).write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
        idx_p = out / "index.json"
        idx = json.loads(idx_p.read_text()) if idx_p.exists() else []
        idx = [e for e in idx if e["file"] != name]
        idx.append({"file": name, "label": f"Your run: {a.task}, {a.controller}, seed {a.seed}", "task": a.task,
                    "controller": a.controller, "seed": a.seed, "success": bool(m.get("success"))})
        idx_p.write_text(json.dumps(idx, indent=1))
        print(f"replay written: web/replays/{name}  (open the viewer with: connectocopter serve)", file=sys.stderr)


def cmd_serve(a) -> None:
    from .server import serve

    serve(a.host, a.port)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="connectocopter", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks", help="list tasks")
    r = sub.add_parser("run", help="run one closed-loop episode")
    r.add_argument("task", choices=sorted(TASKS))
    r.add_argument("--controller", choices=["connectome", "baseline"], default="connectome")
    r.add_argument("--seed", type=int, default=2000)
    r.add_argument("--record", action="store_true", help="write a replay for the browser viewer")
    s = sub.add_parser("serve", help="serve the browser viewer (with live simulations)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    if a.cmd == "tasks":
        for k in sorted(TASKS):
            print(k)
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "serve":
        cmd_serve(a)


if __name__ == "__main__":
    main()
