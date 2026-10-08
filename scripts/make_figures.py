#!/usr/bin/env python3
"""Result figures for the README (docs/img/results_*.png) from
results/benchmarks/summary.json and the raw per-episode files.

Colour roles (validated with the dataviz palette validator on the #121a26
surface): connectome = blue, baseline = orange, every ablation = neutral grey
(identified by its direct label, never by colour alone).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "results" / "benchmarks"
IMG = ROOT / "docs" / "img"

SURFACE, INK, MUTED, GRID = "#121a26", "#d7dee8", "#8a96a8", "#263345"
COL = {"connectome": "#3987e5", "baseline": "#d95926", "baseline_optomotor": "#d95926"}  # the yaw task's baseline
ABL = "#7b8799"
LABEL = {
    "connectome": "Connectome (FlyWire LIF)", "baseline": "Baseline (hand-written rules)",
    "baseline_optomotor": "Baseline + engineered optomotor",
    "rewired_connectome": "Rewired connectome", "connectome_uncalibrated": "Connectome, no L/R calibration",
    "silence_giant_fiber": "Connectome, Giant Fiber silenced", "silence_LC10a": "Connectome, LC10a silenced",
    "silence_DNp15": "Connectome, DNp15 silenced", "no_optic_flow_input": "Connectome, no HS/H2 input",
    "no_bitter_input": "Connectome, no bitter input", "baseline_no_optomotor": "Baseline, no optomotor term",
    "connectome_with_ORN_input": "Connectome + ORN input (neg. control)",
}
TASK_TITLE = {
    "looming_escape": "Looming escape (survive the ball)", "vibration_escape": "Vibration escape (take off)",
    "target_seek": "Drive to a beacon", "obstacle_course": "Rolling obstacle course (goal reached)",
    "flight_course": "Flight course + landing", "taste_dock": "Taste docking (sugar only)",
    "yaw_stabilization": "Hold heading, yaw gyro failed", "odor_plume": "Odor source (plume)",
}
ORDER = ["connectome", "baseline", "baseline_optomotor", "connectome_uncalibrated", "rewired_connectome", "silence_giant_fiber", "silence_LC10a",
         "silence_DNp15", "no_optic_flow_input", "no_bitter_input", "baseline_no_optomotor", "connectome_with_ORN_input"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
        "text.color": INK, "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "font.family": "DejaVu Sans",
    })


def color(c):
    return COL.get(c, ABL)


def success_figure(summary):
    tasks = [t for t in TASK_TITLE if t in summary]
    n = len(tasks)
    cols = 2
    rows = int(np.ceil(n / cols))
    heights = [max(len(summary[t]) for t in tasks[i * cols:(i + 1) * cols]) for i in range(rows)]
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 0.55 * sum(heights) + 1.2 * rows),
                             gridspec_kw={"height_ratios": heights})
    axes = np.atleast_2d(axes)
    for k, t in enumerate(tasks):
        ax = axes[k // cols, k % cols]
        ctrls = [c for c in ORDER if c in summary[t]]
        rowmax = heights[k // cols]
        y = rowmax - 1 - np.arange(len(ctrls))  # top-aligned, constant bar thickness across panels
        for yi, c in zip(y, ctrls):
            e = summary[t][c]
            lo, hi = e["ci95"]
            ax.barh(yi, e["success_rate"], height=0.56, color=color(c), zorder=2)
            ax.plot([lo, hi], [yi, yi], color=INK, lw=1.2, zorder=3)
            ax.plot([lo, lo], [yi - 0.12, yi + 0.12], color=INK, lw=1.2, zorder=3)
            ax.plot([hi, hi], [yi - 0.12, yi + 0.12], color=INK, lw=1.2, zorder=3)
            ax.text(1.03, yi, f"{e['success']}/{e['n']}", va="center", ha="left", color=INK, fontsize=9.5)
        ax.set_yticks(y, [LABEL.get(c, c) for c in ctrls], fontsize=9.5)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.6, rowmax - 0.4)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_title(TASK_TITLE[t], loc="left", color=INK)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.suptitle("Task success on evaluation seeds (bars) with 95% Wilson intervals (whiskers)", x=0.01, ha="left",
                 color=INK, fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.97, 0.97))
    fig.savefig(IMG / "results_success.png", dpi=130)
    plt.close(fig)


def load_raw(task, ctrl):
    p = B / "raw" / f"{task}__{ctrl}.jsonl"
    return [json.loads(l)["metrics"] for l in p.read_text().splitlines()] if p.exists() else []


def strip(ax, rows, key, xlabel, logx=False):
    for yi, (c, ms) in enumerate(rows[::-1]):
        vals = np.array([m[key] for m in ms if m.get(key) is not None], float)
        if not len(vals):
            ax.text(0.02, yi, "no events", transform=ax.get_yaxis_transform(), va="center", color=MUTED, fontsize=9)
            continue
        jit = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.3
        ax.scatter(vals, yi + jit, s=34, color=color(c), edgecolor=SURFACE, linewidth=1.2, zorder=3)
        med = np.median(vals)
        ax.plot([med, med], [yi - 0.3, yi + 0.3], color=INK, lw=2, zorder=4)
    ax.set_yticks(range(len(rows)), [LABEL.get(c, c) for c, _ in rows[::-1]], fontsize=9.5)
    ax.set_xlabel(xlabel)
    if logx:
        ax.set_xscale("log")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def detail_figure():
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.4))
    axes = axes.ravel()
    rows = [(c, [m for m in load_raw("looming_escape", c) if not m.get("catch_trial")]) for c in ["connectome", "baseline", "rewired_connectome", "silence_giant_fiber"]]
    rows = [r for r in rows if r[1]]
    strip(axes[0], rows, "ttc_at_escape_s", "time-to-contact left when escaping (s)  ·  higher = earlier")
    axes[0].set_title("Looming escape: margin at take-off", loc="left")
    rows = [(c, load_raw("yaw_stabilization", c)) for c in ["connectome", "baseline_optomotor", "no_optic_flow_input", "silence_DNp15", "baseline_no_optomotor", "connectome_uncalibrated", "rewired_connectome"]]
    rows = [r for r in rows if r[1]]
    strip(axes[1], rows, "heading_drift_deg", "heading drift over 7 s (deg, log scale)  ·  lower = better", logx=True)
    axes[1].axvline(90, color=MUTED, lw=1, ls="--", zorder=1)
    axes[1].set_title("Yaw gyro failed: heading drift (dashed = 90° limit)", loc="left")
    rows = [(c, [m for m in load_raw("flight_course", c) if m.get("landed")]) for c in ["connectome", "baseline", "connectome_uncalibrated", "rewired_connectome"]]
    rows = [r for r in rows if r[1]]
    strip(axes[2], rows, "landing_error_m", "landing error (m)  ·  lower = better  ·  landed runs only")
    axes[2].set_title("Flight course: landing accuracy", loc="left")
    rows = [(c, load_raw("obstacle_course", c)) for c in ["connectome", "baseline", "connectome_uncalibrated", "rewired_connectome"]]
    rows = [r for r in rows if r[1]]
    for c, ms in rows:
        for m in ms:
            m["contacts_p1"] = m.get("collisions", 0) + 1
    strip(axes[3], rows, "contacts_p1", "contacts per run + 1 (log scale)  ·  1 = collision-free", logx=True)
    axes[3].set_title("Obstacle course: contacts with pillars and walls", loc="left")
    fig.text(0.01, 0.01, "Dots = episodes (evaluation seeds); white tick = median.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(IMG / "results_details.png", dpi=130)
    plt.close(fig)


def main():
    style()
    summary = json.loads((B / "summary.json").read_text())["summary"]
    IMG.mkdir(parents=True, exist_ok=True)
    success_figure(summary)
    detail_figure()
    print("figures written")


if __name__ == "__main__":
    main()
