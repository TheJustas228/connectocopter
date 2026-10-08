#!/usr/bin/env python3
"""Real-world feasibility of the proposed Connectocopter hardware.

Reads configs/hardware_bom.yaml and writes docs/hardware.md (tables and
calculations) and results/feasibility.json.  All propulsion numbers use the
motor manufacturer's published 6S thrust/power table, not the simulator's
momentum-theory power model (which is ~30% optimistic, see docs/limitations.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
G = 9.81


def hover_power_per_motor(thrust_g: float, tt: dict) -> tuple[float, float]:
    """Electrical power (W) and throttle (%) for a given thrust per motor."""
    T = np.array(tt["thrust_g"], float)
    P = np.array(tt["power_w"], float)
    thr = np.array(tt["throttle_pct"], float)
    if thrust_g >= T[0]:
        return float(np.interp(thrust_g, T, P)), float(np.interp(thrust_g, T, thr))
    # below the table: power-law fit through the three lowest points
    b, loga = np.polyfit(np.log(T[:3]), np.log(P[:3]), 1)
    p = float(np.exp(loga) * thrust_g ** b)
    # throttle extrapolation (thrust ~ throttle^2 for a fixed-pitch prop)
    t = float(thr[0] * np.sqrt(thrust_g / T[0]))
    return p, t


def main() -> None:
    bom = yaml.safe_load((ROOT / "configs" / "hardware_bom.yaml").read_text())
    parts = bom["parts"]
    tt = bom["thrust_table"]
    mass = sum(p["qty"] * p["mass_g"] for p in parts)
    cost = sum(p["qty"] * p["price_usd"] for p in parts)
    cost_verified = sum(p["qty"] * p["price_usd"] for p in parts if p["verified"])
    mass_unverified = sum(p["qty"] * p["mass_g"] for p in parts if not p["verified"])
    avionics_w = sum(p["qty"] * (p["power_w"] or 0) for p in parts if p["name"] != "Wheel gearmotors")
    max_thrust_g = 4 * tt["thrust_g"][-1]

    def flight(auw_g, energy_wh, usable=0.8):
        p_m, thr = hover_power_per_motor(auw_g / 4, tt)
        p_tot = 4 * p_m + avionics_w
        return {"auw_g": round(auw_g), "thrust_per_motor_g": round(auw_g / 4), "hover_throttle_pct": round(thr, 1),
                "prop_power_w": round(4 * p_m), "total_power_w": round(p_tot),
                "hover_minutes": round(60 * usable * energy_wh / p_tot, 1),
                "thrust_to_weight": round(max_thrust_g / auw_g, 2),
                "hover_current_a": round(p_tot / 22.2, 1)}

    def rolling(auw_g, energy_wh, v=0.8, crr=0.03, eta=0.40, usable=0.8):
        p_mech = crr * auw_g / 1000 * G * v
        p_wheels = p_mech / eta + 4 * 0.08 * 10.0  # + gearmotor no-load losses
        p_tot = p_wheels + avionics_w
        hours = usable * energy_wh / p_tot
        return {"speed_m_s": v, "wheel_power_w": round(p_wheels, 1), "total_power_w": round(p_tot, 1),
                "minutes": round(60 * hours), "range_km": round(hours * 3600 * v / 1000, 1)}

    lipo_wh = 48.8
    alt = bom["alt_battery"]
    lipo_mass = next(p["mass_g"] for p in parts if p["name"] == "Battery")
    auw_lipo = mass
    auw_liion = mass - lipo_mass + alt["mass_g"]
    res = {
        "checked": bom["checked"],
        "total_mass_g": round(mass), "mass_from_estimates_g": round(mass_unverified),
        "total_cost_usd": round(cost), "cost_verified_items_usd": round(cost_verified),
        "avionics_power_w": round(avionics_w, 1),
        "max_static_thrust_g": max_thrust_g,
        "lipo": {"flight": flight(auw_lipo, lipo_wh), "rolling": rolling(auw_lipo, lipo_wh)},
        "liion": {"flight": flight(auw_liion, alt["energy_wh"]), "rolling": rolling(auw_liion, alt["energy_wh"])},
        "simulated_mass_kg": 1.171,
        "simulated_thrust_limit_per_rotor_g": round(16.0 / G * 1000),
    }
    # energy per metre: flying at 1.2 m/s vs rolling at 0.8 m/s (LiPo build)
    f, r = res["lipo"]["flight"], res["lipo"]["rolling"]
    res["energy_per_m_J"] = {"flying_1.2m_s": round(f["total_power_w"] / 1.2, 1), "rolling_0.8m_s": round(r["total_power_w"] / 0.8, 1)}
    # compute: scale the measured RTX 3070 block time by memory bandwidth ratio (448 vs 102 GB/s)
    res["compute_estimate"] = {
        "measured_rtx3070_ms_per_1.8ms_block": 0.16,
        "bandwidth_ratio": round(448 / 102.4, 2),
        "orin_nx_estimate_ms_per_block": round(0.16 * 448 / 102.4, 2),
        "realtime_factor_estimate": round(1.8 / (0.16 * 448 / 102.4), 2),
        "model_memory_mb": round(15.1e6 * 8 / 1e6),
        "note": "estimate only; not measured on Jetson hardware",
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "feasibility.json").write_text(json.dumps(res, indent=1))

    # ---------------------------------------------------------------- markdown
    L = []
    L.append("# Real-world hardware proposal and feasibility\n")
    L.append("> **Status: design study only.** Nothing was bought or built. Prices and specifications were checked on "
             f"{bom['checked']} on the linked pages; rows marked *estimate* are engineering estimates without a "
             "verified product page. Generated by `scripts/feasibility.py` from `configs/hardware_bom.yaml`.\n")
    L.append("## Bill of materials\n")
    L.append("| Part | Maker / model | Qty | Mass each (g) | Price each (USD) | Power (W) | Interface | Category | Why | Source |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in parts:
        src = f"[link]({p['url']})" if p.get("url") else "–"
        if p.get("price_url"):
            src += f" · [price]({p['price_url']})"
        flag = "" if p["verified"] else " *(estimate)*"
        pw = "see table" if p["power_w"] is None else f"{p['power_w']:g}"
        L.append(f"| {p['name']} | {p['maker']} — {p['model']}{flag} | {p['qty']} | {p['mass_g']:g} | {p['price_usd']:.2f} | {pw} | "
                 f"{p['interface']} | {p['category']} | {p['why']} | {src} |")
    L.append(f"\n**Total: {res['total_mass_g']} g, about ${res['total_cost_usd']:,}** "
             f"(verified-price items ${res['cost_verified_items_usd']:,}; {res['mass_from_estimates_g']} g of the mass is estimated). "
             "Category A = commercially available and directly suitable; B = needs custom integration or fabrication.\n")
    L.append("The simulated robot (`connectocopter/robot/model.py`) weighs 1.171 kg; this build comes to "
             f"{res['total_mass_g'] / 1000:.3f} kg, so the simulation is within "
             f"{abs(res['total_mass_g'] / 1000 - 1.171) / 1.171 * 100:.0f}% of the proposed hardware. The simulator uses a 295 mm "
             "motor-to-motor diagonal; the proposed frame has a 327 mm wheelbase (7.5-inch class), slightly increasing inertia.\n")
    L.append("## Propulsion and endurance\n")
    L.append("Motor data: Hobbywing XRotor 2807 1300KV on 6S with HQ 7x4x3 (manufacturer table):\n")
    L.append("| Throttle (%) | " + " | ".join(str(t) for t in tt["throttle_pct"]) + " |")
    L.append("|---|" + "---|" * len(tt["throttle_pct"]))
    L.append("| Thrust (g) | " + " | ".join(str(t) for t in tt["thrust_g"]) + " |")
    L.append("| Power (W) | " + " | ".join(str(t) for t in tt["power_w"]) + " |\n")
    for key, label, wh in [("lipo", "6S 2200 mAh LiPo (48.8 Wh)", lipo_wh), ("liion", f"{alt['name']} ({alt['energy_wh']} Wh, *estimate*)", alt["energy_wh"])]:
        f, r = res[key]["flight"], res[key]["rolling"]
        L.append(f"**{label}** — all-up weight {f['auw_g']} g\n")
        L.append(f"- Thrust-to-weight: **{f['thrust_to_weight']}** ({max_thrust_g} g max static thrust); hover at ~{f['hover_throttle_pct']}% throttle, {f['thrust_per_motor_g']} g per motor.")
        L.append(f"- Hover power: ~{f['prop_power_w']} W propulsion + {res['avionics_power_w']} W avionics/compute = **{f['total_power_w']} W** (~{f['hover_current_a']} A).")
        L.append(f"- Flight endurance (80% usable energy): **~{f['hover_minutes']} min** hover; slow cruise is similar (±20%).")
        L.append(f"- Rolling at {r['speed_m_s']} m/s: ~{r['wheel_power_w']} W for the wheels + avionics = **{r['total_power_w']} W** "
                 f"→ **~{r['minutes']} min, ~{r['range_km']} km** per charge.\n")
    e = res["energy_per_m_J"]
    L.append(f"Energy per metre travelled (LiPo build): **{e['flying_1.2m_s']} J/m flying** vs **{e['rolling_0.8m_s']} J/m rolling** — "
             f"rolling is ~{e['flying_1.2m_s'] / e['rolling_0.8m_s']:.0f}x cheaper, which is the case for a hybrid body. "
             "Most of the rolling power is the Jetson, not the wheels.\n")
    L.append("Assumptions and uncertainty: hover power below 40% throttle is extrapolated with a power-law fit (P ∝ T^1.3) to the "
             "manufacturer table; rolling resistance coefficient 0.03 (silicone tyres on hard floor) and 40% gearmotor+driver "
             "efficiency; 80% usable battery energy; no wind, no battery sag. Treat endurance figures as ±25%.\n")
    L.append("The simulator limits each rotor to 16 N (1.63 kgf), 58% of this motor's maximum thrust, so simulated flight "
             "behaviour does not rely on thrust the hardware cannot deliver. The wheels do not make the flight design "
             f"implausible: wheels, gearmotors, struts, drivers and regulator add ~{4*10 + 2*20 + 2*2 + 2*1 + 4 + 24 + 4} g "
             f"(~{(4*10 + 2*20 + 2*2 + 2*1 + 4 + 24 + 4) / res['total_mass_g'] * 100:.0f}% of all-up weight) and the "
             "thrust-to-weight ratio stays above 8. The simulated wheel torque limit (0.25 N m) is higher than the chosen "
             "gearmotor's 0.13 N m stall torque; see the sensitivity check in docs/experiments.md.\n")
    c = res["compute_estimate"]
    L.append("## Compute: can the whole brain fly on board?\n")
    L.append(f"On an RTX 3070 the full 138,639-neuron model advances one 1.8 ms block in {c['measured_rtx3070_ms_per_1.8ms_block']} ms "
             "(≈10x real time, measured). The engine is memory-bandwidth and launch-latency bound; scaling by memory bandwidth "
             f"(448 vs 102 GB/s) gives ≈{c['orin_nx_estimate_ms_per_block']} ms per block on a Jetson Orin NX 16GB, i.e. "
             f"**≈{c['realtime_factor_estimate']}x real time — plausible but unverified on hardware**. The connectivity needs "
             f"~{c['model_memory_mb']} MB of GPU memory, well within 16 GB. The sensor pipeline (EMD vision on a 40x30 "
             "lattice, ToF fan) is negligible next to the brain. A reduced circuit (only the pathways actually used) would "
             "run on far smaller hardware, but would no longer be a whole-brain controller.\n")
    L.append("## What is easy, what needs custom work, what remains hard\n")
    L.append("**Commercially available and directly suitable (A):** frame, motors, props, 4-in-1 ESC, H7 flight controller "
             "running ArduPilot, ELRS receiver, Jetson Orin NX, global-shutter camera, VL53L8CX ToF sensors, optical-flow/range "
             "sensor, BME688 gas sensors, micro gearmotors, wheels, motor drivers, LiPo battery.\n")
    L.append("**Needs custom integration or fabrication (B):** wheel struts doubling as landing gear; a 10 V supply for the "
             "wheel drivers; a lightweight Jetson cooler; mounting the camera and ToF fan with vibration isolation; the "
             "MAVLink bridge that sends the brain's high-level commands (forward speed, yaw rate, take-off/escape, halt) to "
             "ArduPilot GUIDED mode and runs the skid-steer wheel loop on the Jetson; the pad sensor (spectral sensor or "
             "electrical contacts on a real charging dock).\n")
    L.append("**Difficult or unsupported in a practical robot (C):**\n")
    L.append("- *Odor plume tracking*: MOX/VOC sensors respond in ~1–10 s versus milliseconds for fly olfactory receptor neurons, "
             "rotor downwash destroys plume structure, and — independently of hardware — the whole-brain LIF model has no stable "
             "olfactory channel (runaway activity; docs/brain_interface.md).")
    L.append("- *Wind / airflow sensing in flight*: the propellers' own downwash dominates any antenna-like airflow sensor.")
    L.append("- *Fly-like vision*: one camera covers ~110° instead of a fly's near-panoramic field; flicker fusion and "
             "polarisation vision are not reproduced. An event camera (e.g. Prophesee GENX320) would be the closest analogue.")
    L.append("- *Escape timing*: a fly's Giant Fiber escape completes in milliseconds; a 1.2 kg quadrotor needs ~0.3–0.5 s to "
             "gain 1 m, so only slow threats can be escaped (the simulator reproduces this physics).")
    L.append("- *Real-time whole-brain execution on board*: estimated feasible on Orin NX but untested; power (15–25 W) dominates "
             "the rolling energy budget.")
    L.append("- *Safety*: open 7-inch propellers at ~2.8 kg max thrust each are dangerous; prop guards conflict with the wheel "
             "layout and add mass. Any physical test would need a safety pilot, geofence and kill switch.\n")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "hardware.md").write_text("\n".join(L) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
