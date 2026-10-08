"""Null-model connectomes for ablations."""
from __future__ import annotations

import numpy as np

from .connectome import Connectome


def rewire_preserving_degree_and_sign(c: Connectome, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Degree- and sign-preserving rewiring.

    Every edge keeps its postsynaptic neuron (so each neuron's in-degree is
    unchanged) while the (presynaptic neuron, signed synapse count) pairs are
    randomly permuted across all edges (so each neuron's out-degree, total
    output synapses and transmitter sign are unchanged).  Who-connects-to-whom
    is destroyed.  Returns (indices, weights) in the CSR layout of ``c``.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(c.indices))
    return c.indices[perm].copy(), c.weights[perm].copy()
