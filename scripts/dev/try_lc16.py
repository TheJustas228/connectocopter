import copy, numpy as np, connectocopter, warnings; warnings.filterwarnings('ignore')
from connectocopter.sim.episode import Episode
from connectocopter.tasks.navigation import ObstacleCourse
from connectocopter.control.connectome import ConnectomeController, load_interface
for off, sc in [(0.3, 1.0), (0.3, 1.5)]:
    cfg = load_interface()
    for e in cfg["encoders"]:
        if e["name"].startswith("obstacle_"):
            e["offset"], e["scale"] = off, sc
    c = ConnectomeController(cfg=cfg, seed=0)
    out = [Episode(ObstacleCourse(), c, s).run().metrics for s in range(8)]
    print(off, sc, "success", sum(m["success"] for m in out), "/8 collisions", sum(m["collisions"] for m in out), flush=True)
