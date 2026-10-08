"""Tests for the whole-brain LIF engine (connectocopter.brain.lif)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from connectocopter.brain.connectome import Connectome
from connectocopter.brain.lif import LIFBrain, LIFParams

HAS_DATA = __import__("pathlib").Path(__file__).resolve().parents[1].joinpath("data", "Connectivity_783.parquet").exists()
CUDA = torch.cuda.is_available()


def chain(nsyn: int) -> Connectome:
    """0 -> 1 -> 2 feed-forward chain with ``nsyn`` synapses per connection."""
    return Connectome("chain", np.array([0, 1, 2]), np.array([0, 0, 1, 2]), np.array([0, 1], dtype=np.int32),
                      np.array([nsyn, nsyn], dtype=np.int32))


def test_single_spike_delivered_after_exact_delay():
    """A forced spike of neuron 0 at step 0 reaches neuron 1 at step 18
    (t_dly = 1.8 ms at dt = 0.1 ms) and not earlier."""
    c = chain(200)
    b = LIFBrain(c, device="cpu")
    b.v[0, 0] = 0.0  # above threshold -> spikes at step 0
    g1 = []
    for _ in range(25):
        b.run_steps(1)
        g1.append(float(b.g[0, 1]))
    first = next(i for i, g in enumerate(g1) if g != 0.0)
    assert first == 18
    assert g1[18] == pytest.approx(200 * LIFParams().w_syn, rel=1e-6)


def test_input_discarded_while_refractory():
    """Brian2 semantics: synaptic input arriving during the refractory period
    (v, g are '(unless refractory)') is discarded, not accumulated."""
    c = chain(30)
    b = LIFBrain(c, device="cpu")
    b.v[0, 1] = 0.0  # neuron 1 spikes at step 0 -> refractory steps 1..21
    b.run_steps(1)
    b.pending[(b.pos + 5) % b.D, 0, 1] = 30.0  # input arriving during refractoriness
    b.run_steps(10)
    assert float(b.g[0, 1]) == 0.0


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_gpu_matches_cpu_reference_bit_exact():
    """With deterministic drive, the fused Triton path and the plain torch path
    produce identical spike counts on a random sparse network."""
    rng = np.random.default_rng(0)
    n, k = 3000, 60
    pre = rng.integers(0, n, n * k)
    post = np.repeat(np.arange(n), k)
    order = np.lexsort((pre, post))
    w = rng.choice([-3, -1, 1, 2, 4, 8], n * k).astype(np.int32)
    indptr = np.arange(0, n * k + 1, k)
    c = Connectome("rand", np.arange(n), indptr, pre[order].astype(np.int32), w[order])
    drive = np.arange(50)
    out = {}
    for dev in ["cpu", "cuda"]:
        b = LIFBrain(c, device=dev, input_idx=drive)
        for _ in range(30):
            b.v[:, torch.as_tensor(drive)] = 0.0
            b.run_steps(37)
        out[dev] = b.read_counts().cpu()
    assert torch.equal(out["cpu"], out["cuda"])
    assert out["cpu"].sum() > 50 * 30


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_seed_determinism():
    c = chain(30)
    res = []
    for _ in range(2):
        b = LIFBrain(c, device="cuda", seed=11, input_idx=np.array([0]))
        b.set_rates([0], 300.0)
        b.run_ms(2000)
        res.append(b.read_counts().cpu())
    assert torch.equal(res[0], res[1])


def test_poisson_rate_and_downstream_rate_match_brian2_reference():
    """Chain 0->1 with 30 synapses, neuron 0 driven at 300 Hz.  Reference values
    from the original Brian2 model (6 seeds x 20 s): neuron 0 ~ 291 Hz,
    neuron 1 = 35.5 +- 0.4 Hz."""
    b = LIFBrain(chain(30), batch=8, device="cuda" if CUDA else "cpu", seed=1, input_idx=np.array([0]))
    b.set_rates([0], 300.0)
    b.run_ms(20000 if CUDA else 5000)
    secs = (20.0 if CUDA else 5.0)
    r = b.read_counts().mean(0).cpu().numpy() / secs
    assert 280 < r[0] < 300
    assert abs(r[1] - 35.5) < (1.5 if CUDA else 3.0)


@pytest.mark.skipif(not (HAS_DATA and CUDA), reason="needs FlyWire data and CUDA")
def test_full_brain_sugar_to_mn9():
    """Sugar GRNs drive the proboscis motor neuron MN9 (Shiu et al. 2024)."""
    from connectocopter.brain.connectome import load_connectome

    c = load_connectome("783")
    sugar = c.select(cell_sub_class="sugar")
    mn9 = c.select(cell_type="CB0701")
    b = LIFBrain(c, batch=4, seed=3, input_idx=sugar)
    b.set_rates(sugar, 100.0)
    b.run_ms(1000)
    rate = b.read_counts().mean(0).cpu().numpy()
    assert rate[mn9].max() > 10.0
