import numpy as np, connectocopter, warnings; warnings.filterwarnings('ignore')
from connectocopter.sim.episode import Episode
from connectocopter.tasks.flight import FlightCourse
from connectocopter.tasks.navigation import ObstacleCourse
from connectocopter.control.connectome import ConnectomeController, load_interface
cfg = load_interface()
for e in cfg["encoders"]:
    if e["name"].startswith("obstacle_"):
        e["offset"], e["scale"] = 0.3, 1.5
c = ConnectomeController(cfg=cfg, seed=0)
for T, n in [(FlightCourse, 4), (ObstacleCourse, 8)]:
    out = [Episode(T(), c, s).run().metrics for s in range(n)]
    print(T.name, "success", sum(m["success"] for m in out), "/", n, "collisions", [m["collisions"] for m in out], flush=True)
