"""Decoder tuning sweep for the obstacle course on TUNING seeds (0-9).
Evaluation seeds (>= 1000) are never used here.  Output: results/tuning/course_decoder.json"""
import copy, json, sys, time
from pathlib import Path
import numpy as np, connectocopter, warnings; warnings.filterwarnings("ignore")
from connectocopter.sim.episode import Episode
from connectocopter.tasks.navigation import ObstacleCourse
from connectocopter.control.connectome import ConnectomeController, load_interface

base = load_interface()
variants = {
    "saccade_off": {"saccade": {"enabled": False}},
    "saccade_thr0.18": {},
    "saccade_thr0.35": {"saccade": {"threshold": 0.35}},
    "saccade_off_gain2.5": {"saccade": {"enabled": False}, "gain_ground": 2.5},
}
out = {}
for name, mod in variants.items():
    cfg = copy.deepcopy(base)
    st = cfg["decoders"]["steering"]
    for k, v in mod.items():
        if isinstance(v, dict):
            st[k].update(v)
        else:
            st[k] = v
    ctrl = ConnectomeController(cfg=cfg, seed=0)
    res = []
    for seed in range(10):
        m = Episode(ObstacleCourse(), ctrl, seed).run().metrics
        res.append({"seed": seed, "success": m["success"], "collisions": m["collisions"]})
    sr = np.mean([r["success"] for r in res]); col = np.mean([r["collisions"] for r in res])
    out[name] = {"success_rate": sr, "mean_collisions": col, "runs": res}
    print(name, sr, col, flush=True)
Path("results/tuning").mkdir(parents=True, exist_ok=True)
Path("results/tuning/course_decoder.json").write_text(json.dumps(out, indent=1))
