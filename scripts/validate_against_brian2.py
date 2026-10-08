#!/usr/bin/env python3
"""Reproduce Shiu et al. (2024) with the ORIGINAL Brian2 code and compare
against Connectocopter's GPU LIF port.

Experiment: Poisson activation of the right-hemisphere sugar gustatory
receptor neurons (the 21 FlyWire v630 IDs used in the original example
notebook) at several rates; 30 trials x 1 s each, FlyWire v630, as in the
paper's Fig. 1.  We compare per-neuron firing rates over the whole brain and
the response of the proboscis motor neuron MN9 (the paper's readout).

Steps
-----
1. ``--brian2``  runs the unmodified ``model.py`` from
   github.com/philshiu/Drosophila_brain_model (pinned commit, MIT) and stores
   spike rates in results/validation/brian2_*.parquet  (CPU; ~3 GB RAM per
   worker, ~1 min per 1-s trial; use ``--n-proc`` to fit your memory).
2. ``--port``    runs the same experiments with connectocopter.brain.lif.
3. ``--compare`` writes results/validation/summary.json and a figure.

Requires ``python scripts/download_data.py --with-630``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "validation"
EXT = DATA / "external" / "Drosophila_brain_model"
SHIU_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"

# Right-hemisphere sugar GRNs (FlyWire v630 root IDs) from the original example.ipynb
SUGAR_R_630 = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9_630 = [720575940660219265, 720575940645521262]  # left, right (from figures.ipynb)
RATES_HZ = [50, 100, 150, 200]
T_RUN_MS = 1000
N_RUN = 30


def rates_from_spikes(df: pd.DataFrame, n_run: int, t_run_s: float) -> pd.Series:
    return df.groupby("flywire_id").size() / (n_run * t_run_s)


def run_brian2(n_proc: int) -> None:
    if not EXT.exists():
        EXT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "https://github.com/philshiu/Drosophila_brain_model.git", str(EXT)])
        subprocess.check_call(["git", "-C", str(EXT), "checkout", SHIU_COMMIT])
    sys.path.insert(0, str(EXT))
    from brian2 import Hz, ms  # noqa: E402
    from model import default_params, run_exp  # noqa: E402  (original code)

    (OUT / "brian2_raw").mkdir(parents=True, exist_ok=True)  # run_exp does not create it
    for hz in RATES_HZ:
        params = dict(default_params)
        params["r_poi"] = hz * Hz
        params["t_run"] = T_RUN_MS * ms
        params["n_run"] = N_RUN
        t0 = time.time()
        run_exp(
            exp_name=f"sugarR_{hz}Hz", neu_exc=SUGAR_R_630, path_res=OUT / "brian2_raw",
            path_comp=DATA / "Completeness_630.csv", path_con=DATA / "Connectivity_630.parquet",
            params=params, n_proc=n_proc, force_overwrite=False,
        )
        df = pd.read_parquet(OUT / "brian2_raw" / f"sugarR_{hz}Hz.parquet")
        r = rates_from_spikes(df, N_RUN, T_RUN_MS / 1000)
        r.rename("rate_hz").to_frame().to_parquet(OUT / f"brian2_sugarR_{hz}Hz.parquet")
        print(f"brian2 {hz} Hz done in {time.time() - t0:.0f} s; {len(r)} active neurons")


def run_port() -> None:
    import torch

    from connectocopter.brain.connectome import load_connectome
    from connectocopter.brain.lif import LIFBrain

    OUT.mkdir(parents=True, exist_ok=True)
    c = load_connectome("630", annotations=False)
    sugar = c.idx(SUGAR_R_630)
    timing = {}
    for hz in RATES_HZ:
        t0 = time.time()
        brain = LIFBrain(c, batch=N_RUN, seed=hz, input_idx=sugar)
        brain.set_rates(sugar, float(hz))
        brain.run_ms(T_RUN_MS)
        counts = brain.read_counts().cpu().numpy()  # [N_RUN, N]
        if brain.device.type == "cuda":
            torch.cuda.synchronize()
        timing[hz] = time.time() - t0
        rate = counts.sum(0) / (N_RUN * T_RUN_MS / 1000)
        nz = np.flatnonzero(rate)
        s = pd.Series(rate[nz], index=c.root_ids[nz], name="rate_hz")
        s.index.name = "flywire_id"
        s.to_frame().to_parquet(OUT / f"port_sugarR_{hz}Hz.parquet")
        print(f"port {hz} Hz: {len(nz)} active neurons, {timing[hz]:.1f} s for {N_RUN} x {T_RUN_MS} ms")
    (OUT / "port_timing.json").write_text(json.dumps({"seconds_per_condition": timing, "device": str(brain.device)}, indent=2))


def compare() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {"experiment": "sugarR (21 right sugar GRNs, FlyWire v630), 30 trials x 1 s", "conditions": {}}
    rates = [hz for hz in RATES_HZ if (OUT / f"brian2_sugarR_{hz}Hz.parquet").exists()]
    fig, axes = plt.subplots(1, len(rates) + 1, figsize=(4 * (len(rates) + 1), 4))
    mn9_b, mn9_p = [], []
    for ax, hz in zip(axes, rates):
        b = pd.read_parquet(OUT / f"brian2_sugarR_{hz}Hz.parquet")["rate_hz"]
        p = pd.read_parquet(OUT / f"port_sugarR_{hz}Hz.parquet")["rate_hz"]
        both = pd.concat([b.rename("brian2"), p.rename("port")], axis=1).fillna(0.0)
        # exclude the directly driven neurons (their rate is set by the stimulus)
        both = both.drop(index=[i for i in SUGAR_R_630 if i in both.index])
        r = float(np.corrcoef(both.brian2, both.port)[0, 1])
        act_b, act_p = set(both.index[both.brian2 >= 1]), set(both.index[both.port >= 1])
        jacc = len(act_b & act_p) / max(1, len(act_b | act_p))
        mb = float(both.brian2.reindex(MN9_630).fillna(0).mean())
        mp = float(both.port.reindex(MN9_630).fillna(0).mean())
        mn9_b.append(mb)
        mn9_p.append(mp)
        summary["conditions"][f"{hz}Hz"] = {
            "pearson_r_rates": round(r, 4),
            "n_active_brian2(>=1Hz)": len(act_b),
            "n_active_port(>=1Hz)": len(act_p),
            "jaccard_active_sets": round(jacc, 4),
            "mean_abs_rate_diff_hz": round(float((both.brian2 - both.port).abs().mean()), 4),
            "MN9_rate_brian2_hz": round(mb, 2),
            "MN9_rate_port_hz": round(mp, 2),
        }
        lim = max(both.max().max(), 1) * 1.05
        ax.scatter(both.brian2, both.port, s=6, alpha=0.5)
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Brian2 original (Hz)", ylabel="GPU port (Hz)",
               title=f"sugar GRNs @ {hz} Hz\nr = {r:.3f}, n = {len(both)}")
    ax = axes[-1]
    ax.plot(rates, mn9_b, "o-", label="Brian2 original")
    ax.plot(rates, mn9_p, "s--", label="GPU port")
    ax.set(xlabel="sugar GRN drive (Hz)", ylabel="MN9 rate (Hz)", title="Proboscis motor neuron MN9")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "brian2_vs_port.png", dpi=130)
    t = OUT / "port_timing.json"
    if t.exists():
        summary["port_timing"] = json.loads(t.read_text())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brian2", action="store_true")
    ap.add_argument("--port", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--n-proc", type=int, default=3,
                    help="parallel Brian2 workers; each needs ~3 GB RAM (16 workers OOM a 16 GB machine)")
    a = ap.parse_args()
    if not (a.brian2 or a.port or a.compare):
        a.brian2 = a.port = a.compare = True
    if a.brian2:
        run_brian2(a.n_proc)
    if a.port:
        run_port()
    if a.compare:
        compare()


if __name__ == "__main__":
    main()
