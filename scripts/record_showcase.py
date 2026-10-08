#!/usr/bin/env python3
"""Record replay files for the browser viewer (web/replays/).

Showcase episodes use seeds 2000+ (separate from tuning seeds 0-99 and
evaluation seeds 1000+).  They illustrate behaviour; the quantitative claims
come from scripts/run_benchmarks.py.  Failures are kept and labelled.

    python scripts/record_showcase.py            # record everything in SHOWCASE
    python scripts/record_showcase.py --only looming_escape
"""
from __future__ import annotations

import argparse
import gzip
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import connectocopter  # noqa: E402,F401
from connectocopter.control.baseline import BaselineController  # noqa: E402
from connectocopter.control.connectome import ConnectomeController  # noqa: E402
from connectocopter.sim.episode import Episode  # noqa: E402
from connectocopter.tasks.flight import FlightCourse, YawStabilization  # noqa: E402
from connectocopter.tasks.looming import LoomingEscape  # noqa: E402
from connectocopter.tasks.navigation import ObstacleCourse, TargetSeek  # noqa: E402
from connectocopter.tasks.odor import OdorPlume  # noqa: E402
from connectocopter.tasks.taste import TasteDock  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "replays"

# (file stem, title, task factory, controller kind, seed)
SHOWCASE = [
    ("flight_course_connectome", "Take off, weave through pillars, land at the beacon", lambda: FlightCourse(), "connectome", 2001),
    ("looming_escape_connectome", "Looming ball triggers a Giant Fiber escape takeoff", lambda: LoomingEscape(catch=False), "connectome", 2002),
    ("obstacle_course_connectome", "Roll through a pillar corridor to the beacon", lambda: ObstacleCourse(), "connectome", 2003),
    ("taste_dock_connectome", "Stop to 'feed' on sugar; ignore bitter and mixed pads", lambda: TasteDock(), "connectome", 2004),
    ("target_seek_connectome", "Turn toward and drive to a glowing beacon", lambda: TargetSeek(), "connectome", 2005),
    ("yaw_stabilization_connectome", "Yaw gyro failed: optic flow -> HS/H2 -> DNp15 holds heading", lambda: YawStabilization(), "connectome", 2006),
    ("vibration_escape_connectome", "Vibration via Johnston's organ triggers escape", lambda: LoomingEscape(catch=False, stimulus="vibration"), "connectome", 2007),
    ("obstacle_course_baseline", "Baseline (hand-written rules) on the same corridor", lambda: ObstacleCourse(), "baseline", 2003),
    ("odor_plume_baseline", "Baseline cast-and-surge plume tracking (no connectome olfaction)", lambda: OdorPlume(), "baseline", 2008),
]


def note_for(m: dict, task: str) -> str:
    if task == "looming_escape":
        if m.get("reaction_time_s") is not None and m.get("stimulus") == "vibration":
            return f"Escape takeoff {m['reaction_time_s']:.2f} s after vibration onset."
        if m.get("reaction_time_s") is not None:
            return f"Escape {m['reaction_time_s']:.2f} s after looming onset, {m['ttc_at_escape_s']:.2f} s before contact."
    if task == "flight_course" and m.get("landing_error_m") is not None:
        return f"Landed {m['landing_error_m']:.2f} m from the pad centre; {m['collisions']} collisions."
    if task in ("obstacle_course", "target_seek") and m.get("time_to_goal_s") is not None:
        return f"Reached the beacon in {m['time_to_goal_s']:.1f} s; {m['collisions']} collisions."
    if task == "taste_dock":
        return f"Pad order: {', '.join(m['pad_order'])}. Docked on sugar: {m['docked_sugar']}; on bitter: {m['docked_bitter']}; on mixed: {m['docked_mixed']}."
    if task == "yaw_stabilization":
        return f"Heading drift {m['heading_drift_deg']:.0f} deg over the 5 s disturbance (+2 s)."
    if task == "odor_plume" and m.get("time_to_source_s") is not None:
        return f"Reached the odor source in {m['time_to_source_s']:.1f} s."
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    brain = None
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    for stem, title, tf, kind, seed in SHOWCASE:
        if a.only and not any(o in stem for o in a.only):
            continue
        if kind == "connectome":
            if brain is None:
                brain = ConnectomeController(seed=0, record_activity=True)
            ctrl = brain
        else:
            ctrl = BaselineController()
        ep = Episode(tf(), ctrl, seed, record=True, record_fpv_every=3)
        res = ep.run()
        r = res.replay
        m = {k: (v.item() if hasattr(v, "item") else v) for k, v in res.metrics.items()}
        if "pad_order" in m:
            m["pad_order"] = [str(x) for x in m["pad_order"]]
        r["metrics"] = m
        r["task_title"] = title
        r["outcome_note"] = note_for(m, r["task"])
        f = f"{stem}.json.gz"
        payload = json.dumps(r, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
        (OUT / f).write_bytes(gzip.compress(payload.encode(), compresslevel=9, mtime=0))
        size = (OUT / f).stat().st_size / 1e6
        label = f"{title} ({kind}{'' if m['success'] else ', failed'})"
        index = [e for e in index if e["file"] != f] + [{"file": f, "label": label, "task": r["task"], "controller": kind,
                                                         "seed": seed, "success": bool(m["success"])}]
        print(f"{f}: success={m['success']} {r['outcome_note']} [{size:.1f} MB]", flush=True)
    order = [s[0] + ".json.gz" for s in SHOWCASE]
    index.sort(key=lambda e: order.index(e["file"]) if e["file"] in order else 99)
    index_path.write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
