"""Connectome controller: sensors -> identified neurons -> whole-brain LIF ->
descending neurons -> high-level command.

The mapping is defined in configs/interface.yaml.  The brain itself is the
unmodified Shiu et al. (2024) LIF model of the full FlyWire v783 connectome
(138,639 neurons, 15.1 M connections) running in connectocopter.brain.lif.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import hashlib
import json

import numpy as np
import torch
import yaml

from ..brain.connectome import Connectome, load_connectome
from ..brain.lif import ENGINE_VERSION, LIFBrain
from ..brain.populations import resolve
from ..robot.robot import Command

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "interface.yaml"
CALIB_DIR = Path(__file__).resolve().parents[2] / "results" / "calibration"

_CONNECTOME_CACHE: dict[str, Connectome] = {}


def get_connectome(version: str = "783") -> Connectome:
    if version not in _CONNECTOME_CACHE:
        _CONNECTOME_CACHE[version] = load_connectome(version)
    return _CONNECTOME_CACHE[version]


def load_interface(path: Path | None = None) -> dict:
    with open(path or CONFIG) as f:
        return yaml.safe_load(f)


@dataclass
class BrainTelemetry:
    input_hz: dict = field(default_factory=dict)
    dn_hz: dict = field(default_factory=dict)  # filtered rates of readout populations (L/R)
    steer: float = 0.0
    gf_spikes: int = 0
    mn9_hz: float = 0.0
    mdn_hz: float = 0.0
    n_active: int = 0
    total_spikes: int = 0
    active_idx: np.ndarray | None = None  # neurons that spiked this step (for visualisation)


class ConnectomeController:
    """Closed-loop controller driven by the whole-brain LIF model."""

    label = "connectome"

    def __init__(self, cfg: dict | None = None, seed: int = 0, device: str | None = None,
                 silence: dict | None = None, weights: np.ndarray | None = None,
                 disable_encoders: list[str] | None = None, record_activity: bool = False,
                 connectome: Connectome | None = None, calibrate: bool = True, label: str | None = None,
                 extra_encoders: list[dict] | None = None, indices: np.ndarray | None = None):
        import copy as _copy
        self.cfg = _copy.deepcopy(cfg or load_interface())
        if extra_encoders:
            self.cfg["encoders"] = self.cfg["encoders"] + list(extra_encoders)
        self.c = connectome or get_connectome(self.cfg["brain"]["connectome"])
        ann = self.c.annotations
        self.dt = float(self.cfg["brain"]["control_dt"])
        self.tau = float(self.cfg["brain"]["rate_tau"])
        self.record_activity = record_activity
        disabled = set(disable_encoders or [])
        self.encoders = [e for e in self.cfg["encoders"] if e["name"] not in disabled]
        self.enc_idx = {e["name"]: resolve(self.c, e["neurons"]) for e in self.encoders}
        input_idx = np.unique(np.concatenate(list(self.enc_idx.values())))
        silence_idx = None
        if silence:
            silence_idx = np.unique(np.concatenate([resolve(self.c, s) for s in silence.values()]))
        self.silence = silence or {}
        self.brain = LIFBrain(self.c, batch=1, seed=seed, device=device, input_idx=input_idx,
                              silence_idx=silence_idx, weights=weights, indices=indices)
        self.silence_idx = silence_idx if silence_idx is not None else np.array([], dtype=np.int64)
        dec = self.cfg["decoders"]
        side = ann.side.to_numpy()

        def lr(cell_type):
            idx = self.c.select(cell_type=cell_type)
            return idx[side[idx] == "left"], idx[side[idx] == "right"]

        st = dec["steering"]
        self.steer_types = {t: (w, lr(t)) for t, w in st["ipsiversive"].items()}
        self.steer_types.update({t: (-w, lr(t)) for t, w in st["contraversive"].items()})
        self.gf_idx = resolve(self.c, dec["escape"]["neurons"])
        self.mdn_idx = resolve(self.c, dec["reverse"]["neurons"])
        self.mn9_idx = resolve(self.c, dec["feeding_halt"]["neurons"])
        self.dng02_idx = resolve(self.c, dec["flight_power"]["neurons"])
        # monitored neurons -> filtered rates
        groups = {}
        for t, (_, (l, r)) in self.steer_types.items():
            groups[f"{t}_L"], groups[f"{t}_R"] = l, r
        for name, idx in [("GF", self.gf_idx), ("MDN", self.mdn_idx), ("MN9", self.mn9_idx), ("DNg02", self.dng02_idx)]:
            groups[f"{name}_L"] = idx[side[idx] == "left"]
            groups[f"{name}_R"] = idx[side[idx] == "right"]
        self.groups = {k: v for k, v in groups.items() if len(v)}
        self.mon_idx = np.unique(np.concatenate(list(self.groups.values())))
        self._pos = {k: np.searchsorted(self.mon_idx, v) for k, v in self.groups.items()}
        self.mon_t = torch.as_tensor(self.mon_idx, device=self.brain.device)
        # silenced read-out neurons cannot drive their (VNC) targets: decode them as silent
        self.mon_live = (~np.isin(self.mon_idx, self.silence_idx)).astype(float)
        self.rate = np.zeros(len(self.mon_idx))
        self.drive = self.cfg["internal_drive"]
        self.tel = BrainTelemetry()
        self.feeding = False
        if label:
            self.label = label
        self.side_gain = {e["name"]: 1.0 for e in self.encoders}
        self._seed = seed
        self._weights_tag = "shuffled" if (weights is not None or indices is not None) else "connectome"
        self.steer_rate = np.zeros(len(self.mon_idx))
        self._sacc_I = 0.0
        self._sacc_until = -1.0
        self._sacc_dir = 0.0
        self._sacc_ready = 0.0
        self.t = 0.0
        self.saccades = 0
        self.calibrated = False
        if calibrate:
            self.calibrate_symmetry()
            self.calibrated = True

    # ------------------------------------------------------------------
    def set_seed(self, seed: int) -> None:
        """Seed the brain's Poisson input noise (one seed per episode)."""
        self.brain.seed = int(seed)

    def reset(self) -> None:
        self.brain.reset()
        self.rate[:] = 0
        self.steer_rate[:] = 0
        self.feeding = False
        self._sacc_I, self._sacc_until, self._sacc_dir, self._sacc_ready, self.t = 0.0, -1.0, 0.0, 0.0, 0.0
        self.saccades = 0
        self._last_yaw = 0.0

    def _encode(self, feats: dict) -> dict:
        hz = {}
        ec = self.cfg.get("efference_copy", {})
        ec_set = set(ec.get("encoders", []))
        suppress = ec.get("suppress_during_saccade", False) and self.t < self._sacc_until + ec.get("hold", 0.0)
        k_ec = ec.get("subtract_gain", 0.0)
        yaw = getattr(self, "_last_yaw", 0.0)
        for e in self.encoders:
            f = feats.get(e["feature"], 0.0)
            if e["name"] in ec_set:
                if suppress:
                    f = -1e9  # silenced
                elif k_ec:
                    # turning left (yaw>0) adds front-to-back flow on the right, back-to-front on the left
                    sgn = {"hs_L": 1.0, "hs_R": -1.0, "h2_L": -1.0, "h2_R": 1.0}.get(e["feature"], 0.0)
                    f = f + sgn * k_ec * yaw
            r = e["max_hz"] * float(np.clip(self.side_gain[e["name"]] * (f - e["offset"]) / e["scale"], 0.0, 1.0))
            hz[e["name"]] = r
            self.brain.set_rates(self.enc_idx[e["name"]], r)
        return hz

    # --------------------------------------------------------- calibration
    def _calib_key(self) -> str:
        enc = json.dumps([(e["name"], e["neurons"], e["feature"]) for e in self.encoders], sort_keys=True, default=str)
        dec = json.dumps(self.cfg["decoders"]["steering"], sort_keys=True)
        sil = json.dumps(self.silence, sort_keys=True, default=str)
        h = hashlib.sha1((enc + dec + sil + self._weights_tag + ENGINE_VERSION).encode()).hexdigest()[:12]
        return h

    def _mean_steer(self, feats: dict, seconds: float, seed: int) -> float:
        self.brain.seed = seed
        self.reset()
        vals = []
        for i in range(int(seconds / self.dt)):
            self.step(feats, "ground")
            if i * self.dt > 0.3:
                vals.append(self.tel.steer)
        return float(np.mean(vals))

    def calibrate_symmetry(self, pairs=None, level: float = 0.6, seconds: float = 4.0, iters: int = 8,
                           use_cache: bool = True) -> dict:
        """Find per-side input gains so that symmetric bilateral stimulation of
        each steering-relevant pair yields zero mean steering.  The FlyWire
        brain is one individual and is not left/right symmetric; without this
        the robot has a turning bias (reported as an ablation)."""
        if pairs is None:
            pairs = [("obstacle_L", "obstacle_R"), ("yawflow_HS_L", "yawflow_HS_R"),
                     ("yawflow_H2_L", "yawflow_H2_R"), ("target_L", "target_R")]
        names = {e["name"]: e for e in self.encoders}
        pairs = [p for p in pairs if p[0] in names and p[1] in names]
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        cache = CALIB_DIR / f"symmetry_{self._calib_key()}.json"
        if use_cache and cache.exists():
            gains = json.loads(cache.read_text())["gains"]
            self.side_gain.update(gains)
            return gains
        seed0 = self.brain.seed
        report = {}
        for l, r in pairs:
            el, er = names[l], names[r]
            feats = {el["feature"]: el["offset"] + level * el["scale"], er["feature"]: er["offset"] + level * er["scale"]}
            lo, hi = -1.2, 1.2  # log gain ratio rho: gain_L = e^{rho/2}, gain_R = e^{-rho/2}
            for _ in range(iters):
                rho = 0.5 * (lo + hi)
                self.side_gain[l], self.side_gain[r] = float(np.exp(rho / 2)), float(np.exp(-rho / 2))
                s_ = np.mean([self._mean_steer(feats, seconds, 9000 + k) for k in range(2)])
                # steer > 0 means turning left; left inputs drive (mostly) ipsilateral or contralateral
                # turns depending on the pathway, so bisect on the sign of the measured effect
                if self._pair_sign(l, r, feats, seconds) * s_ > 0:
                    hi = rho
                else:
                    lo = rho
            report[l], report[r] = self.side_gain[l], self.side_gain[r]
        self.brain.seed = seed0
        self.reset()
        cache.write_text(json.dumps({"gains": report, "level": level, "seconds": seconds}, indent=1))
        return report

    def _pair_sign(self, l: str, r: str, feats: dict, seconds: float) -> float:
        """+1 if raising the left input turns the robot left (ipsiversive), -1 if right."""
        if not hasattr(self, "_pair_sign_cache"):
            self._pair_sign_cache = {}
        if (l, r) not in self._pair_sign_cache:
            names = {e["name"]: e for e in self.encoders}
            g0 = dict(self.side_gain)
            self.side_gain[l], self.side_gain[r] = 1.0, 0.0
            s_left = self._mean_steer(feats, 2.0, 8000)
            self.side_gain.update(g0)
            self._pair_sign_cache[(l, r)] = 1.0 if s_left > 0 else -1.0
        return self._pair_sign_cache[(l, r)]

    def step(self, feats: dict, mode: str) -> Command:
        """One closed-loop step: encode -> run brain for control_dt -> decode."""
        tel = self.tel
        tel.input_hz = self._encode(feats)
        self.brain.run_ms(self.dt * 1000.0)
        counts = self.brain.read_counts()[0]
        mon = counts[self.mon_t].float().cpu().numpy() * self.mon_live
        tel.total_spikes = int(counts.sum().item())
        if self.record_activity:
            act = torch.nonzero(counts > 0).flatten()
            tel.active_idx = act.cpu().numpy().astype(np.int32)
            tel.n_active = int(len(tel.active_idx))
        else:
            tel.n_active = int((counts > 0).sum().item())
        inst = mon / self.dt
        a = self.dt / (self.tau + self.dt)
        self.rate += a * (inst - self.rate)
        st = self.cfg["decoders"]["steering"]
        a_s = self.dt / (st.get("rate_tau", self.tau) + self.dt)
        self.steer_rate += a_s * (inst - self.steer_rate)
        g = {k: float(self.rate[p].mean()) for k, p in self._pos.items()}
        gs = {k: float(self.steer_rate[p].mean()) for k, p in self._pos.items()}
        tel.dn_hz = g
        self.t += self.dt
        # ---- decoders
        steer = sum(w * (gs.get(f"{t}_L", 0.0) - gs.get(f"{t}_R", 0.0)) for t, (w, _) in self.steer_types.items())
        tel.steer = float(steer)
        gain = st["gain_flight"] if mode in ("flight", "takeoff") else st["gain_ground"]
        yaw_rate = gain * float(np.tanh(steer / st["norm_hz"]))
        sc = st.get("saccade", {})
        if sc.get("enabled", False):
            self._sacc_I += self.dt * (steer / st["norm_hz"] - self._sacc_I / sc["tau"])
            if self.t < self._sacc_until:
                yaw_rate = self._sacc_dir * sc["rate"]
            elif self.t >= self._sacc_ready and abs(self._sacc_I) > sc["threshold"]:
                self._sacc_dir = float(np.sign(self._sacc_I))
                self._sacc_until = self.t + sc["duration"]
                self._sacc_ready = self._sacc_until + sc["refractory"]
                self._sacc_I = 0.0
                self.saccades += 1
                yaw_rate = self._sacc_dir * sc["rate"]
        tel.gf_spikes = int(mon[self._pos["GF_L"]].sum() + mon[self._pos["GF_R"]].sum())
        escape = tel.gf_spikes >= self.cfg["decoders"]["escape"]["spikes_to_trigger"]
        tel.mdn_hz = 0.5 * (g.get("MDN_L", 0) + g.get("MDN_R", 0))
        rev = self.cfg["decoders"]["reverse"]
        v_back = rev["gain"] * min(tel.mdn_hz / rev["norm_hz"], 1.0)
        tel.mn9_hz = max(g.get("MN9_L", 0), g.get("MN9_R", 0))
        self.feeding = tel.mn9_hz > self.cfg["decoders"]["feeding_halt"]["threshold_hz"]
        cruise = self.drive["cruise_speed_flight"] if mode in ("flight", "takeoff") else self.drive["cruise_speed_ground"]
        self._last_yaw = yaw_rate
        return Command(v_fwd=cruise - v_back, yaw_rate=yaw_rate, escape=escape, halt=self.feeding)
