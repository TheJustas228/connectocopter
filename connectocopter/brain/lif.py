"""Whole-brain leaky integrate-and-fire (LIF) model on GPU.

This is a re-implementation of the model of Shiu et al. (2024, Nature,
"A Drosophila computational brain model reveals sensorimotor processing";
code: github.com/philshiu/Drosophila_brain_model, MIT).  Equations and
parameters are taken verbatim from that model:

    dv/dt = (v_0 - v + g) / t_mbr        (unless refractory)
    dg/dt = -g / tau                     (unless refractory)
    spike when v > v_th  ->  v = v_rst, g = 0, refractory for t_rfc
    presynaptic spike  ->  g_post += w_syn * (signed synapse count)  after t_dly

    v_0 = v_rst = -52 mV, v_th = -45 mV, t_mbr = 20 ms, tau = 5 ms,
    t_rfc = 2.2 ms, t_dly = 1.8 ms, w_syn = 0.275 mV (the single free parameter)

External drive ("activation") is a Poisson process per input neuron that adds
f_poi * w_syn = 68.75 mV to v (enough to force a spike), exactly like the
Brian2 ``PoissonInput`` used in the original; driven neurons have no
refractory period, as in the original.

Integration scheme
------------------
Brian2 integrates the linear ODEs exactly (``method='linear'``) at
dt = 0.1 ms with the per-step order *state update -> threshold -> synaptic
delivery / Poisson input -> reset*.  We keep that order, including Brian2's
"conditional write" semantics: because v and g are declared
``(unless refractory)``, synaptic and Poisson input arriving while a neuron is
refractory (or in the step it spikes) is discarded, not accumulated.  Every synapse has
the same delay D = t_dly / dt = 18 steps, so spikes emitted inside a block of
K <= D steps cannot influence that same block.  Each block therefore runs as

1. one fused kernel that advances every neuron K steps, consuming synaptic
   input from a D-slot ring buffer and appending spikes to an event list;
2. one event-driven kernel that scatters each spike's outgoing synapses into
   the ring-buffer slot D steps in the future.

This is an exact reordering of the per-step algorithm, not an approximation.
Synaptic input is accumulated as integer synapse counts in float32, which is
exact in any summation order, so a run is bit-reproducible for a given seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .connectome import Connectome

try:  # Triton is optional; CPU / no-Triton falls back to plain torch
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


@dataclass(frozen=True)
class LIFParams:
    v_0: float = -52.0  # mV, resting potential
    v_rst: float = -52.0  # mV, reset potential
    v_th: float = -45.0  # mV, spike threshold
    t_mbr: float = 20.0  # ms, membrane time constant
    tau: float = 5.0  # ms, synaptic time constant
    t_rfc: float = 2.2  # ms, refractory period
    t_dly: float = 1.8  # ms, synaptic delay
    w_syn: float = 0.275  # mV per synapse (free parameter, Shiu et al.)
    f_poi: float = 250.0  # Poisson input weight = f_poi * w_syn
    dt: float = 0.1  # ms, integration step (Brian2 default)


if _HAVE_TRITON:

    @triton.jit
    def _lif_block_kernel(
        v_ptr, g_ptr, ref_ptr, rfc_ptr, rate_ptr, txmask_ptr, pend_ptr, cnt_ptr,
        ev_idx_ptr, ev_k_ptr, ev_count_ptr,
        n_total, n_neurons, n_steps, pos, n_slots, seed, step0,
        em, es, coef_a, v_0, v_th, v_rst, w_syn, w_poi, p_scale,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_total
        j = offs % n_neurons
        v = tl.load(v_ptr + offs, mask=mask, other=0.0)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        ref = tl.load(ref_ptr + offs, mask=mask, other=0)
        rfc = tl.load(rfc_ptr + j, mask=mask, other=0)
        tx = tl.load(txmask_ptr + j, mask=mask, other=0) > 0
        p = tl.load(rate_ptr + offs, mask=mask, other=0.0) * p_scale
        offs64 = offs.to(tl.int64)
        nspk = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(n_steps):
            # 1) state update: exact solution of the linear ODEs over dt
            nr = ref <= 0
            u = v - v_0
            a = g * coef_a
            v = tl.where(nr, v_0 + (u - a) * em + a * es, v)
            g = tl.where(nr, g * es, g)
            # 2) threshold
            spk = (v > v_th) & nr & mask
            # 3) synaptic delivery from the ring buffer + Poisson drive.  As in
            #    Brian2, v and g are "(unless refractory)", so writes to them
            #    from synapses / PoissonInput are discarded while the neuron is
            #    refractory -- including the step in which it just spiked.
            slot = ((pos + k) % n_slots).to(tl.int64)
            pp = pend_ptr + slot * n_total + offs64
            syn = tl.load(pp, mask=mask, other=0.0)
            # load and store may be mapped to different threads: order them
            tl.debug_barrier()
            tl.store(pp, tl.zeros([BLOCK], dtype=tl.float32), mask=mask)
            live = nr & (~spk)
            g = g + tl.where(live, syn * w_syn, 0.0)
            r = tl.rand(seed, (step0 + k) * n_total + offs64)
            v = v + tl.where(live & (r < p), w_poi, 0.0)
            # 4) reset
            v = tl.where(spk, v_rst, v)
            g = tl.where(spk, 0.0, g)
            ref = tl.where(spk, rfc - 1, ref - 1)
            ref = tl.maximum(ref, -1)
            nspk += spk.to(tl.float32)
            # 5) append transmitting spikes to the event list
            s_i = (spk & tx).to(tl.int32)
            n_new = tl.sum(s_i, axis=0)
            if n_new > 0:
                base = tl.atomic_add(ev_count_ptr, n_new)
                local = tl.cumsum(s_i, axis=0) - s_i
                tl.store(ev_idx_ptr + base + local, offs, mask=s_i > 0)
                tl.store(ev_k_ptr + base + local, tl.zeros([BLOCK], dtype=tl.int32) + k, mask=s_i > 0)
        tl.store(v_ptr + offs, v, mask=mask)
        tl.store(g_ptr + offs, g, mask=mask)
        tl.store(ref_ptr + offs, ref, mask=mask)
        c = tl.load(cnt_ptr + offs, mask=mask, other=0.0)
        tl.store(cnt_ptr + offs, c + nspk, mask=mask)

    @triton.jit
    def _deliver_kernel(
        ev_count_ptr, ev_idx_ptr, ev_k_ptr, colptr_ptr, post_ptr, w_ptr, pend_ptr,
        n_neurons, n_total, pos, n_slots, n_prog,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        count = tl.load(ev_count_ptr)
        for i in range(pid, count, n_prog):
            gidx = tl.load(ev_idx_ptr + i)
            k = tl.load(ev_k_ptr + i)
            b = gidx // n_neurons
            j = gidx - b * n_neurons
            slot = ((pos + k) % n_slots).to(tl.int64)
            base = slot * n_total + b.to(tl.int64) * n_neurons
            e0 = tl.load(colptr_ptr + j)
            e1 = tl.load(colptr_ptr + j + 1)
            for e in range(e0, e1, BLOCK):
                eo = e + tl.arange(0, BLOCK)
                m = eo < e1
                post = tl.load(post_ptr + eo, mask=m, other=0)
                w = tl.load(w_ptr + eo, mask=m, other=0.0)
                tl.atomic_add(pend_ptr + base + post, w, mask=m)


class LIFBrain:
    """Batched whole-brain LIF simulation with Poisson sensory drive.

    Parameters
    ----------
    connectome : Connectome
    batch : number of independent copies simulated in parallel (trials)
    device : "cuda" or "cpu"
    seed : RNG seed for the Poisson inputs (fully determines a GPU run)
    input_idx : neurons that may receive Poisson drive; as in the original
        model these have no refractory period.
    silence_idx : neurons whose output synapses are removed (in-silico silencing,
        identical to setting their outgoing weights to zero in Shiu et al.)
    weights : optional replacement signed synapse counts (same CSR layout),
        used for connectome-shuffle ablations.
    """

    def __init__(
        self,
        connectome: Connectome,
        params: LIFParams = LIFParams(),
        batch: int = 1,
        device: str | None = None,
        seed: int = 0,
        input_idx: np.ndarray | None = None,
        silence_idx: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ):
        self.c = connectome
        self.p = params
        self.B = batch
        self.N = connectome.n_neurons
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = int(seed)
        self.use_triton = _HAVE_TRITON and self.device.type == "cuda"

        self.D = int(round(params.t_dly / params.dt))
        n_rfc = int(round(params.t_rfc / params.dt))
        w = np.asarray(connectome.weights if weights is None else weights, dtype=np.float32)
        pre = np.asarray(connectome.indices if indices is None else indices, dtype=np.int64)
        # Outgoing-edge (presynaptic-major, CSC) layout for event-driven delivery.
        post = np.repeat(np.arange(self.N, dtype=np.int64), np.diff(connectome.indptr))
        order = np.argsort(pre, kind="stable")
        colptr = np.zeros(self.N + 1, dtype=np.int64)
        np.add.at(colptr, pre + 1, 1)
        np.cumsum(colptr, out=colptr)
        self.colptr = torch.as_tensor(colptr, device=self.device)
        self.out_post = torch.as_tensor(post[order].astype(np.int32), device=self.device)
        self.out_w = torch.as_tensor(w[order], device=self.device)

        rfc = torch.full((self.N,), n_rfc, dtype=torch.int32)
        if input_idx is not None and len(input_idx):
            rfc[torch.as_tensor(np.asarray(input_idx), dtype=torch.long)] = 0
        self.rfc = rfc.to(self.device)
        tx = torch.ones(self.N, dtype=torch.int32)
        if silence_idx is not None and len(silence_idx):
            tx[torch.as_tensor(np.asarray(silence_idx), dtype=torch.long)] = 0
        self.txmask = tx.to(self.device)

        p = params
        self.em = math.exp(-p.dt / p.t_mbr)
        self.es = math.exp(-p.dt / p.tau)
        self.coef_a = p.tau / (p.tau - p.t_mbr)
        self.w_poi = p.f_poi * p.w_syn
        # Event list capacity: non-driven neurons fire at most once per block
        # (refractory 22 steps > block 18 steps); driven neurons up to D times.
        n_in = 0 if input_idx is None else len(input_idx)
        self.ev_cap = self.B * (self.N + n_in * self.D) + 1024
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        B, N, D, dev = self.B, self.N, self.D, self.device
        self.v = torch.full((B, N), self.p.v_0, device=dev)
        self.g = torch.zeros((B, N), device=dev)
        self.ref = torch.full((B, N), -1, dtype=torch.int32, device=dev)
        self.rates = torch.zeros((B, N), device=dev)  # Poisson drive, Hz
        # ring buffer of synaptic input (signed synapse counts) for the next D steps
        self.pending = torch.zeros((D, B, N), device=dev)
        self.pos = 0
        self.step = 0
        self.counts = torch.zeros((B, N), device=dev)  # spikes since last read
        self.ev_idx = torch.zeros(self.ev_cap, dtype=torch.int32, device=dev)
        self.ev_k = torch.zeros(self.ev_cap, dtype=torch.int32, device=dev)
        self.ev_count = torch.zeros(1, dtype=torch.int32, device=dev)
        self._gen = None

    def set_rates(self, idx, hz, batch: int | slice = slice(None)) -> None:
        """Set Poisson drive (Hz) for neurons ``idx`` (other rates unchanged)."""
        idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=self.device)
        self.rates[batch, idx_t] = torch.as_tensor(hz, dtype=torch.float32, device=self.device)

    def clear_rates(self) -> None:
        self.rates.zero_()

    # ------------------------------------------------------------ simulation
    def run_steps(self, n_steps: int) -> None:
        """Advance the network by ``n_steps`` x dt."""
        done = 0
        while done < n_steps:
            k = min(self.D, n_steps - done)
            if self.use_triton:
                self._block_triton(k)
            else:
                self._block_torch(k)
            self.pos = (self.pos + k) % self.D
            self.step += k
            done += k

    def run_ms(self, ms: float) -> None:
        self.run_steps(int(round(ms / self.p.dt)))

    def _block_triton(self, k: int) -> None:
        n_total = self.B * self.N
        self.ev_count.zero_()
        BLOCK = 256
        _lif_block_kernel[(triton.cdiv(n_total, BLOCK),)](
            self.v, self.g, self.ref, self.rfc, self.rates, self.txmask, self.pending, self.counts,
            self.ev_idx, self.ev_k, self.ev_count,
            n_total, self.N, k, self.pos, self.D, self.seed, self.step,
            self.em, self.es, self.coef_a, self.p.v_0, self.p.v_th, self.p.v_rst,
            self.p.w_syn, self.w_poi, self.p.dt * 1e-3,
            BLOCK=BLOCK,
        )
        n_prog = 2048
        _deliver_kernel[(n_prog,)](
            self.ev_count, self.ev_idx, self.ev_k, self.colptr, self.out_post, self.out_w, self.pending,
            self.N, n_total, self.pos, self.D, n_prog,
            BLOCK=64,
        )

    def _block_torch(self, k: int) -> None:
        """Reference implementation for CPU (same algorithm and order of
        operations; uses torch's RNG, so individual spike trains differ from
        the GPU path for the same seed while the statistics are identical)."""
        p, D, B, N = self.p, self.D, self.B, self.N
        if self._gen is None:
            self._gen = torch.Generator(device=self.device).manual_seed(self.seed)
        prob = self.rates * (p.dt * 1e-3)
        v, g, ref = self.v, self.g, self.ref
        spikes = torch.zeros((k, B, N), dtype=torch.bool, device=self.device)
        for i in range(k):
            slot = (self.pos + i) % D
            nr = ref <= 0
            u = v - p.v_0
            a = g * self.coef_a
            v = torch.where(nr, p.v_0 + (u - a) * self.em + a * self.es, v)
            g = torch.where(nr, g * self.es, g)
            spk = (v > p.v_th) & nr
            live = nr & ~spk  # Brian2 conditional write: no input while refractory
            g = g + torch.where(live, self.pending[slot] * p.w_syn, 0.0)
            self.pending[slot] = 0
            r = torch.rand((B, N), generator=self._gen, device=self.device)
            v = v + torch.where(live & (r < prob), self.w_poi, 0.0)
            v = torch.where(spk, torch.full_like(v, p.v_rst), v)
            g = torch.where(spk, torch.zeros_like(g), g)
            ref = torch.where(spk, self.rfc - 1, ref - 1).clamp_min(-1)
            spikes[i] = spk
        self.v, self.g, self.ref = v, g, ref
        self.counts += spikes.sum(0)
        ev = (spikes & (self.txmask > 0)).nonzero(as_tuple=False)  # (k, b, pre)
        if ev.numel() == 0:
            return
        pre = ev[:, 2]
        start = self.colptr[pre]
        cnt = self.colptr[pre + 1] - start
        total = int(cnt.sum())
        if total == 0:
            return
        rep = torch.repeat_interleave(torch.arange(len(pre), device=self.device), cnt)
        e = start[rep] + torch.arange(total, device=self.device) - (torch.cumsum(cnt, 0) - cnt)[rep]
        slot = (self.pos + ev[rep, 0]) % D
        flat = (slot * B + ev[rep, 1]) * N + self.out_post[e].long()
        self.pending.view(-1).index_add_(0, flat, self.out_w[e])

    # ---------------------------------------------------------------- readout
    def read_counts(self, reset: bool = True) -> torch.Tensor:
        """Spike counts per neuron since the previous read, shape [B, N]."""
        out = self.counts.clone()
        if reset:
            self.counts.zero_()
        return out

    @property
    def time_ms(self) -> float:
        return self.step * self.p.dt
