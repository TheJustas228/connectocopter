#!/usr/bin/env python3
"""Pathway atlas: which descending / motor neurons does each sensory
population drive in the whole-brain LIF model?

For every candidate input population (left and right separately) we apply
Poisson activation (default 100 Hz, as in Shiu et al. 2024) for 1 s across
several independent trials and record the firing rate of every descending
neuron (1,303 DNs), every motor neuron and a few named outputs.  The result
is the evidence base for the sensor->neuron->actuator mapping used by the
robot (see docs/brain_interface.md).

Outputs: results/pathways/atlas.json and results/pathways/atlas.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from connectocopter.brain.connectome import load_connectome
from connectocopter.brain.lif import LIFBrain

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "pathways"

# Candidate input populations (annotation queries); each probed per side.
PROBES = {
    # vision: looming / object / motion
    "LPLC2 (looming)": {"cell_type": "LPLC2"},
    "LC4 (looming)": {"cell_type": "LC4"},
    "LPLC1 (looming, lateral)": {"cell_type": "LPLC1"},
    "LC16 (frontal looming)": {"cell_type": "LC16"},
    "LC10a (small object)": {"cell_type": "LC10a"},
    "LC9": {"cell_type": "LC9"},
    "LC11 (small object)": {"cell_type": "LC11"},
    "HS cells (yaw optic flow)": {"cell_type": ["HSN", "HSE", "HSS"]},
    "H2 (back-to-front flow)": {"cell_type": "H2"},
    "VS cells (vertical flow)": {"cell_type": ["VS1", "VS2", "VS3", "VS4", "VS5", "VS6", "VS7", "VS8"]},
    "R7 photoreceptors": {"cell_type": "R7"},
    "R8 photoreceptors": {"cell_type": "R8"},
    "ocellar photoreceptors": {"cell_sub_class": "ocellar"},
    # chemosensation
    "ORN DM1 (vinegar)": {"cell_type": "ORN_DM1"},
    "ORN DM4 (vinegar)": {"cell_type": "ORN_DM4"},
    "ORN VA2": {"cell_type": "ORN_VA2"},
    "ORN DM2": {"cell_type": "ORN_DM2"},
    "ORN DA2 (geosmin, aversive)": {"cell_type": "ORN_DA2"},
    "sugar GRNs": {"cell_sub_class": "sugar"},
    "bitter GRNs": {"cell_sub_class": "bitter"},
    "water GRNs": {"cell_sub_class": "water"},
    # mechanosensation / thermo / hygro
    "JO wind/gravity (JO-CE)": {"cell_sub_class": "wind_gravity"},
    "JO auditory (JO-AB)": {"cell_sub_class": "auditory"},
    "head bristles": {"cell_sub_class": "head bristle"},
    "heating TRNs": {"cell_sub_class": "heating"},
    "cooling TRNs": {"cell_sub_class": "cooling"},
    "humid HRNs": {"cell_sub_class": "humid"},
    "dry HRNs": {"cell_sub_class": "dry"},
    # walking command neurons (to identify their downstream DN populations)
    "P9 / DNp09 (forward walk)": {"cell_type": "DNp09"},
    "MDN (backward walk)": {"cell_type": "MDN"},
}

# Outputs of special interest (literature-identified)
NAMED_OUTPUTS = {
    "DNp01 (Giant Fiber, escape takeoff)": {"cell_type": "DNp01"},
    "DNp02 (looming escape)": {"cell_type": "DNp02"},
    "DNp04": {"cell_type": "DNp04"},
    "DNp06": {"cell_type": "DNp06"},
    "DNp11": {"cell_type": "DNp11"},
    "DNa01 (steering)": {"cell_type": "DNa01"},
    "DNa02 (steering)": {"cell_type": "DNa02"},
    "DNb05 (steering)": {"cell_type": "DNb05"},
    "DNg13 (steering)": {"cell_type": "DNg13"},
    "DNb06 (contraversive steering)": {"cell_type": "DNb06"},
    "DNp09 (P9 forward)": {"cell_type": "DNp09"},
    "MDN (backward)": {"cell_type": "MDN"},
    "DNg02 (wing power, population)": {"cell_type": ["DNg02_a", "DNg02_b", "DNg02_c", "DNg02_d", "DNg02_e", "DNg02_f", "DNg02_g", "DNg02_h"]},
    "MN9 / CB0701 (proboscis)": {"cell_type": "CB0701"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=100.0, help="Poisson drive (Hz)")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--ms", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    c = load_connectome("783")
    ann = c.annotations
    dn_idx = np.flatnonzero((ann.super_class.isin(["descending", "motor"])).to_numpy())
    named = {k: c.select(**v) for k, v in NAMED_OUTPUTS.items()}

    atlas = {"config": vars(a), "connectome": "FlyWire v783 (Shiu et al. packaging)", "probes": {}}
    t_all = time.time()
    for pname, spec in PROBES.items():
        for side in ["left", "right"]:
            idx = c.select(**spec, side=side)
            if len(idx) == 0:
                continue
            brain = LIFBrain(c, batch=a.trials, seed=a.seed, input_idx=idx)
            brain.set_rates(idx, a.rate)
            brain.run_ms(a.ms)
            rate = (brain.read_counts().mean(0) / (a.ms / 1000)).cpu().numpy()
            # named outputs, per side
            nm = {}
            for k, ni in named.items():
                sides = ann.side.to_numpy()[ni]
                nm[k] = {s: round(float(rate[ni[sides == s]].mean()), 2) for s in ["left", "right"] if (sides == s).any()}
            # top responding DNs / MNs
            r_dn = rate[dn_idx]
            order = np.argsort(-r_dn)[:15]
            top = [
                {"cell_type": str(ann.cell_type.iat[dn_idx[i]]), "side": str(ann.side.iat[dn_idx[i]]),
                 "super_class": str(ann.super_class.iat[dn_idx[i]]), "rate_hz": round(float(r_dn[i]), 2)}
                for i in order if r_dn[i] > 0.5
            ]
            n_active = int((rate > 0.5).sum())
            atlas["probes"][f"{pname} | {side}"] = {
                "n_input_neurons": int(len(idx)), "n_active_neurons": n_active,
                "named_outputs": nm, "top_descending_or_motor": top,
            }
            print(f"{pname:32s} {side:5s} n={len(idx):5d} active={n_active:6d}  top: "
                  + ", ".join(f"{t['cell_type']}{t['side'][0]}:{t['rate_hz']:.0f}" for t in top[:5]))
            del brain
            torch.cuda.empty_cache()
    atlas["wall_time_s"] = round(time.time() - t_all, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "atlas.json").write_text(json.dumps(atlas, indent=1))

    # markdown summary
    lines = [f"# Pathway atlas (Poisson {a.rate:.0f} Hz, {a.trials} trials x {a.ms:.0f} ms, FlyWire v783)\n",
             "Rates in Hz, mean over trials; L/R = soma side of the output neuron.\n",
             "| Input (side) | n | active | GF DNp01 L/R | DNa02 L/R | DNa01 L/R | DNp09 L/R | MDN L/R | DNg02 L/R | MN9 L/R | top DN/MN |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    def lr(d):
        return f"{d.get('left', 0):.0f}/{d.get('right', 0):.0f}"
    for k, v in atlas["probes"].items():
        n = v["named_outputs"]
        top = ", ".join(f"{t['cell_type']}{t['side'][0]} {t['rate_hz']:.0f}" for t in v["top_descending_or_motor"][:4])
        lines.append(f"| {k} | {v['n_input_neurons']} | {v['n_active_neurons']} | {lr(n['DNp01 (Giant Fiber, escape takeoff)'])} | "
                     f"{lr(n['DNa02 (steering)'])} | {lr(n['DNa01 (steering)'])} | {lr(n['DNp09 (P9 forward)'])} | "
                     f"{lr(n['MDN (backward)'])} | {lr(n['DNg02 (wing power, population)'])} | {lr(n['MN9 / CB0701 (proboscis)'])} | {top} |")
    (OUT / "atlas.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'atlas.md'} in {atlas['wall_time_s']} s")


if __name__ == "__main__":
    main()
