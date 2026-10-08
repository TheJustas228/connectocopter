# Connectocopter

A whole-brain spiking model of the fruit fly — the FlyWire connectome, **138,639 neurons and 15.1 M connections** — reads the sensors of a simulated FPV quadcopter-rover and drives its wheels and rotors in closed loop.

![viewer](docs/img/viewer_hero.gif)

**Status: work in progress.** The simulation, brain engine, browser viewer and documentation are done. The final benchmark run on evaluation seeds is in progress, and its results table and the rest of this README are still to be written.

- Brain: exact GPU port of the Shiu et al. (2024) LIF model (validated against the original Brian2 code, r = 0.9996), ~10x real time on an RTX 3070 — [docs/brain_interface.md](docs/brain_interface.md)
- Robot, sensors, low-level control — [docs/robot.md](docs/robot.md); fly → robot capability matrix — [docs/capability_matrix.md](docs/capability_matrix.md)
- Hardware proposal and feasibility — [docs/hardware.md](docs/hardware.md); decisions — [docs/decisions.md](docs/decisions.md); limitations — [docs/limitations.md](docs/limitations.md); sources — [docs/sources.md](docs/sources.md)

![architecture](docs/img/architecture.svg)

## Quick start

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"            # + a CUDA build of torch for the GPU engine
python scripts/download_data.py       # FlyWire data (CC-BY 4.0), not redistributed
connectocopter run looming_escape --record
connectocopter serve                  # browser viewer at http://127.0.0.1:8765
```
