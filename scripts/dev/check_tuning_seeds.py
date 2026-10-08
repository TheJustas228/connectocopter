"""Re-check the frozen interface on TUNING seeds after an engine change."""
import numpy as np, connectocopter, warnings; warnings.filterwarnings('ignore')
from connectocopter.sim.episode import Episode
from connectocopter.tasks.flight import FlightCourse, YawStabilization
from connectocopter.tasks.navigation import ObstacleCourse, TargetSeek
from connectocopter.tasks.looming import LoomingEscape
from connectocopter.control.connectome import ConnectomeController
c = ConnectomeController(seed=0)
print("calibrated gains", {k: round(v, 2) for k, v in c.side_gain.items() if v != 1}, flush=True)
for T, n in [(FlightCourse, 4), (ObstacleCourse, 8), (TargetSeek, 4), (YawStabilization, 3)]:
    out = [Episode(T(), c, s).run().metrics for s in range(n)]
    print(T.name, "success", sum(m["success"] for m in out), "/", n, "collisions", [m.get("collisions") for m in out], flush=True)
out = [Episode(LoomingEscape(catch=False), c, s).run().metrics for s in range(4)]
print("looming", [(m["success"], m.get("reaction_time_s")) for m in out], flush=True)
