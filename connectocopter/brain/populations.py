"""Named neuron populations defined by FlyWire annotation queries.

A population spec is a mapping of annotation column -> value(s), optionally
with a list of explicit ``root_ids``.  Several specs can be combined with
``any:`` (union).  Example YAML::

    looming_L:
      any:
        - {cell_type: LPLC2, side: left}
        - {cell_type: LC4, side: left}
    MN9:
      cell_type: CB0701       # proboscis motor neuron 9 (Shiu et al. 2024)

Population definitions live in ``configs/neurons.yaml`` so that every
sensor->neuron and neuron->motor mapping is explicit and editable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .connectome import Connectome

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def resolve(c: Connectome, spec) -> np.ndarray:
    """Model indices for a population spec (sorted, unique)."""
    if isinstance(spec, dict) and "any" in spec:
        parts = [resolve(c, s) for s in spec["any"]]
        return np.unique(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)
    spec = dict(spec)
    root_ids = spec.pop("root_ids", None)
    idx = []
    if root_ids:
        idx.append(c.idx(root_ids))
    if spec:
        idx.append(c.select(**spec))
    return np.unique(np.concatenate(idx)) if idx else np.array([], dtype=np.int64)


def load_population_specs(path: Path | None = None) -> dict:
    path = path or CONFIG_DIR / "neurons.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_all(c: Connectome, specs: dict) -> dict[str, np.ndarray]:
    out = {}
    for name, spec in specs.items():
        idx = resolve(c, spec)
        if len(idx) == 0:
            raise ValueError(f"population '{name}' resolved to zero neurons: {spec}")
        out[name] = idx
    return out
