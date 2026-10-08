#!/usr/bin/env python3
"""Choose the yaw-task baseline's optomotor gain on tuning seeds 0-2 (never evaluation seeds).

Output: results/tuning/optomotor_baseline_gain.json
"""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import connectocopter  # noqa: F401
from connectocopter.control.baseline import BaselineController
from connectocopter.sim.episode import Episode
from connectocopter.tasks.flight import YawStabilization

GAINS = [0, 25, 50, 100, 150, 200, 300, 450, 650, 900, 1300, 1800]
rows = {}
for k in GAINS:
    c = BaselineController(k_optomotor=float(k))
    ms = [Episode(YawStabilization(), c, s).run().metrics for s in range(3)]
    rows[k] = {"success": sum(m["success"] for m in ms),
               "mean_abs_drift_deg": round(sum(abs(m["heading_drift_deg"]) for m in ms) / 3, 1)}
    print(k, rows[k], flush=True)
best = min(GAINS, key=lambda k: (-rows[k]["success"], rows[k]["mean_abs_drift_deg"]))
out = Path(__file__).resolve().parents[2] / "results" / "tuning" / "optomotor_baseline_gain.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"seeds": [0, 1, 2], "gains": rows, "chosen": best}, indent=1))
print("chosen", best)
