#!/usr/bin/env python3
"""Run the Connectocopter benchmark suite (configs/benchmarks.yaml).

Every (controller variant, task) pair is a job; each job runs N episodes on
fixed evaluation seeds (seed_start ... seed_start+N-1) and appends one JSON
line per episode to results/benchmarks/raw/<task>__<controller>.jsonl, so an
interrupted run resumes where it stopped.  ``--summarize`` aggregates into
results/benchmarks/summary.json and summary.md.

    python scripts/run_benchmarks.py --workers 2          # run everything
    python scripts/run_benchmarks.py --only-tasks taste_dock --quick
    python scripts/run_benchmarks.py --summarize
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "benchmarks"
RAW = OUT / "raw"

ODOR_ENCODERS = [
    {"name": f"odor_{t}_{s}", "neurons": {"cell_type": f"ORN_{t}", "side": side}, "feature": f"odor_{s}",
     "offset": 0.05, "scale": 0.5, "max_hz": 80, "evidence": "negative control (olfactory runaway)"}
    for t in ["DM1", "DM4", "VA2"] for s, side in [("L", "left"), ("R", "right")]
]


def load_cfg():
    return yaml.safe_load((ROOT / "configs" / "benchmarks.yaml").read_text())


def make_task(spec):
    mod, cls = spec["task"].rsplit(".", 1)
    T = getattr(importlib.import_module(f"connectocopter.tasks.{mod}"), cls)
    return lambda: T(**spec.get("kwargs", {}))


def make_controller(name, spec):
    if spec["kind"] == "baseline":
        from connectocopter.control.baseline import BaselineController
        c = BaselineController(**spec.get("kwargs", {}))
        c.label = name
        return c
    from connectocopter.control.connectome import ConnectomeController, get_connectome
    kw = {"seed": 0, "label": name, "calibrate": spec.get("calibrate", True)}
    if "silence" in spec:
        kw["silence"] = spec["silence"]
    if "disable_encoders" in spec:
        kw["disable_encoders"] = spec["disable_encoders"]
    if spec.get("extra_encoders") == "odor":
        kw["extra_encoders"] = ODOR_ENCODERS
    if "rewire_seed" in spec:
        from connectocopter.brain.nulls import rewire_preserving_degree_and_sign
        idx, w = rewire_preserving_degree_and_sign(get_connectome("783"), seed=spec["rewire_seed"])
        kw["indices"], kw["weights"] = idx, w
    return ConnectomeController(**kw)


def jobs(cfg, only_tasks=None, only_ctrls=None):
    out = []
    for cname, cspec in cfg["controllers"].items():
        if only_ctrls and cname not in only_ctrls:
            continue
        tasks = list(cfg["tasks"]) if cspec["tasks"] == "all" else cspec["tasks"]
        for t in tasks:
            if only_tasks and t not in only_tasks:
                continue
            out.append((cname, t))
    return out


def run_job(args):
    cname, tname, quick = args
    import warnings

    warnings.filterwarnings("ignore")
    import connectocopter  # noqa: F401  (GL setup)
    from connectocopter.sim.episode import Episode

    cfg = load_cfg()
    cspec, tspec = cfg["controllers"][cname], cfg["tasks"][tname]
    n = cspec.get("n_override", tspec["n"])
    if quick:
        n = min(n, 3)
    path = RAW / f"{tname}__{cname}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            done.add(json.loads(line)["seed"])
    seeds = [cfg["seed_start"] + i for i in range(n) if cfg["seed_start"] + i not in done]
    if not seeds:
        return cname, tname, 0
    ctrl = make_controller(cname, cspec)
    task_factory = make_task(tspec)
    for s in seeds:
        res = Episode(task_factory(), ctrl, s).run()
        m = {k: (v.item() if hasattr(v, "item") else v) for k, v in res.metrics.items()}
        rec = {"task": tname, "controller": cname, "seed": s, "sim_time": round(res.sim_time, 3),
               "wall_time": round(res.wall_time, 3), "metrics": m}
        with path.open("a") as f:
            f.write(json.dumps(rec, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        print(f"[{cname} | {tname}] seed {s}: success={m.get('success')}", flush=True)
    return cname, tname, len(seeds)


# --------------------------------------------------------------------- summary
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


KEY_METRICS = {
    "looming_escape": ["reaction_time_s", "ttc_at_escape_s", "min_distance_m"],
    "vibration_escape": ["reaction_time_s"],
    "target_seek": ["time_to_goal_s", "path_efficiency", "collisions", "energy_J"],
    "obstacle_course": ["time_to_goal_s", "path_efficiency", "collisions", "energy_J"],
    "flight_course": ["landing_error_m", "flight_time_s", "collisions", "energy_J"],
    "taste_dock": ["dock_latency_s"],
    "yaw_stabilization": ["heading_drift_deg", "yaw_rate_rms"],
    "odor_plume": ["time_to_source_s", "final_dist_m"],
}


def summarize():
    rows = {}
    for p in sorted(RAW.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            rows.setdefault((r["task"], r["controller"]), []).append(r)
    summary = {}
    for (t, c), rs in sorted(rows.items()):
        ms = [r["metrics"] for r in rs]
        if t == "looming_escape":
            ball = [m for m in ms if not m.get("catch_trial")]
            catch = [m for m in ms if m.get("catch_trial")]
            k = sum(bool(m["success"]) for m in ball)
            ent = {"n": len(ball), "success": k, "success_rate": round(k / max(len(ball), 1), 3),
                   "ci95": wilson(k, len(ball)), "n_catch": len(catch),
                   "false_escapes": sum(bool(m.get("false_escape")) for m in catch)}
        else:
            if t == "vibration_escape":
                ms = [m for m in ms if not m.get("catch_trial")]
            k = sum(bool(m["success"]) for m in ms)
            ent = {"n": len(ms), "success": k, "success_rate": round(k / max(len(ms), 1), 3), "ci95": wilson(k, len(ms))}
        for key in KEY_METRICS.get(t, []):
            vals = [m[key] for m in ms if m.get(key) is not None and not (isinstance(m.get(key), float) and math.isnan(m[key]))]
            if vals:
                ent[key] = {"mean": round(float(np.mean(vals)), 3), "sd": round(float(np.std(vals)), 3), "n": len(vals)}
        if t == "obstacle_course":  # reaching the goal does not require zero contacts; report both
            kc = sum(bool(m["success"]) and m.get("collisions", 0) == 0 for m in ms)
            ent["collision_free_success"] = kc
            ent["collision_free_ci95"] = wilson(kc, len(ms))
        if t == "taste_dock":
            for key in ["docked_sugar", "docked_bitter", "docked_mixed"]:
                ent[key + "_rate"] = round(float(np.mean([bool(m[key]) for m in ms])), 3)
        rt = [r["sim_time"] / r["wall_time"] for r in rs if r["wall_time"] > 0]
        ent["realtime_factor_median"] = round(float(np.median(rt)), 2)
        summary.setdefault(t, {})[c] = ent
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        meta["torch"] = torch.__version__
        import mujoco
        meta["mujoco"] = mujoco.__version__
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=1))
    lines = ["# Benchmark summary (evaluation seeds >= 1000)\n", f"Generated {meta['generated']} on {meta.get('gpu', '?')}.\n",
             "Success: k/n episodes with a Wilson 95% CI.  Other metrics: mean ± SD over the episodes where "
             "they are defined (e.g. time-to-goal only for successful runs).  Looming escape counts ball trials "
             "only; catch trials (no ball) are reported as false escapes.  RT× = median simulated / wall-clock time.\n"]
    for t, ctrls in summary.items():
        lines.append(f"\n## {t}\n")
        keys = KEY_METRICS.get(t, [])
        hdr = "| controller | n | success | 95% CI | " + " | ".join(keys) + " | extra |"
        lines += [hdr, "|" + "---|" * (5 + len(keys))]
        for c, e in ctrls.items():
            cells = [f"{e[k]['mean']} ± {e[k]['sd']}" if k in e else "–" for k in keys]
            extra = []
            if "collision_free_success" in e:
                lo, hi = e["collision_free_ci95"]
                extra.append(f"collision-free {e['collision_free_success']}/{e['n']} ({lo:.2f}–{hi:.2f})")
            if "false_escapes" in e:
                extra.append(f"false escapes {e['false_escapes']}/{e['n_catch']}")
            for kk in ["docked_sugar_rate", "docked_bitter_rate", "docked_mixed_rate"]:
                if kk in e:
                    extra.append(f"{kk.replace('_rate', '')} {e[kk]:.2f}")
            extra.append(f"RT×{e['realtime_factor_median']}")
            lines.append(f"| {c} | {e['n']} | {e['success_rate']:.2f} | {e['ci95'][0]:.2f}–{e['ci95'][1]:.2f} | " + " | ".join(cells) + " | " + "; ".join(extra) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only-tasks", nargs="*")
    ap.add_argument("--only-controllers", nargs="*")
    ap.add_argument("--quick", action="store_true", help="3 episodes per job (smoke test)")
    ap.add_argument("--summarize", action="store_true", help="only aggregate existing results")
    a = ap.parse_args()
    if not a.summarize:
        RAW.mkdir(parents=True, exist_ok=True)
        js = [(c, t, a.quick) for c, t in jobs(load_cfg(), a.only_tasks, a.only_controllers)]
        # longest jobs first
        order = {"obstacle_course": 0, "flight_course": 1, "yaw_stabilization": 2, "odor_plume": 3}
        js.sort(key=lambda j: order.get(j[1], 9))
        print(f"{len(js)} jobs, {a.workers} workers")
        t0 = time.time()
        if a.workers <= 1:
            for j in js:
                run_job(j)
        else:
            with mp.get_context("spawn").Pool(a.workers) as pool:
                for c, t, n in pool.imap_unordered(run_job, js):
                    print(f"== done {c} | {t}: {n} new episodes ({time.time() - t0:.0f} s)", flush=True)
    summarize()


if __name__ == "__main__":
    main()
