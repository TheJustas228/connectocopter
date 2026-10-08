"""Loading the FlyWire connectome into compact arrays.

The connectivity tables come from Shiu et al. (2024)'s packaging of FlyWire
(`Connectivity_<v>.parquet`, `Completeness_<v>.csv`).  Each row is one
pre->post neuron pair with

* ``Connectivity``              number of synapses between the pair,
* ``Excitatory``                +1 (acetylcholine) or -1 (GABA / glutamate),
* ``Excitatory x Connectivity`` signed synapse count (what the model uses).

Neurotransmitter signs follow Shiu et al.: ACh is excitatory, GABA and
glutamate are inhibitory (glutamate acts via GluCl in much of the fly brain),
and the few modulatory types are treated as excitatory.  The resulting signed
counts are stored as a CSR matrix ``W[post, pre]`` so that ``W @ spikes``
gives the signed number of synapses activated on each neuron.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = DATA_DIR / "cache"


@dataclass
class Connectome:
    version: str
    root_ids: np.ndarray  # (N,) int64 FlyWire root IDs, model index order
    indptr: np.ndarray  # CSR over postsynaptic rows
    indices: np.ndarray  # presynaptic column indices
    weights: np.ndarray  # signed synapse counts (int32)
    annotations: pd.DataFrame | None = None  # indexed by model index (783 only)
    _id_to_idx: dict = field(default_factory=dict, repr=False)

    @property
    def n_neurons(self) -> int:
        return len(self.root_ids)

    @property
    def n_connections(self) -> int:
        return len(self.weights)

    @property
    def n_synapses(self) -> int:
        return int(np.abs(self.weights).sum())

    def idx(self, root_ids) -> np.ndarray:
        """Model indices for FlyWire root IDs (raises KeyError if absent)."""
        if not self._id_to_idx:
            self._id_to_idx = {int(r): i for i, r in enumerate(self.root_ids)}
        return np.array([self._id_to_idx[int(r)] for r in root_ids], dtype=np.int64)

    def select(self, **filters) -> np.ndarray:
        """Model indices of neurons whose annotation columns match ``filters``.

        Values may be a scalar or a list; ``cell_type="DNa02", side="left"``.
        """
        if self.annotations is None:
            raise RuntimeError("annotations are only available for v783")
        mask = np.ones(len(self.annotations), dtype=bool)
        for col, val in filters.items():
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            mask &= self.annotations[col].isin(vals).to_numpy()
        return np.flatnonzero(mask)


def _load_tables(version: str) -> tuple[np.ndarray, pd.DataFrame]:
    comp = pd.read_csv(DATA_DIR / f"Completeness_{version}.csv", index_col=0)
    con = pd.read_parquet(
        DATA_DIR / f"Connectivity_{version}.parquet",
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    return comp.index.to_numpy(dtype=np.int64), con


def _build_csr(n: int, con: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    post = con["Postsynaptic_Index"].to_numpy(np.int64)
    pre = con["Presynaptic_Index"].to_numpy(np.int64)
    w = con["Excitatory x Connectivity"].to_numpy(np.int32)
    order = np.lexsort((pre, post))
    post, pre, w = post[order], pre[order], w[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, post + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, pre.astype(np.int32), w


def _load_annotations(root_ids: np.ndarray) -> pd.DataFrame:
    ann = pd.read_csv(DATA_DIR / "flywire_annotations_783.tsv", sep="\t", low_memory=False)
    ann = ann.drop_duplicates("root_id").set_index("root_id")
    cols = [
        "flow", "super_class", "cell_class", "cell_sub_class", "cell_type",
        "hemibrain_type", "top_nt", "side", "nerve", "pos_x", "pos_y", "pos_z",
    ]
    out = ann.reindex(root_ids)[cols].reset_index(names="root_id")
    return out


def load_connectome(version: str = "783", annotations: bool = True) -> Connectome:
    """Load (and cache) the FlyWire connectome as CSR arrays."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"csr_{version}.npz"
    if cache.exists():
        z = np.load(cache)
        root_ids, indptr, indices, weights = z["root_ids"], z["indptr"], z["indices"], z["weights"]
    else:
        root_ids, con = _load_tables(version)
        indptr, indices, weights = _build_csr(len(root_ids), con)
        np.savez(cache, root_ids=root_ids, indptr=indptr, indices=indices, weights=weights)
    ann = None
    if annotations and version == "783":
        ann_cache = CACHE_DIR / "annotations_783.parquet"
        if ann_cache.exists():
            ann = pd.read_parquet(ann_cache)
        else:
            ann = _load_annotations(root_ids)
            ann.to_parquet(ann_cache)
    return Connectome(version, root_ids, indptr, indices, weights, ann)
