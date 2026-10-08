"""Fly-inspired early vision on the FPV camera image.

This module turns camera frames into the *hemifield features* that drive
identified visual neurons in the brain model.  It is a functional
approximation of fly early vision, not a connectome simulation:

* Ommatidial sampling: the RGB frame is converted to luminance and
  block-averaged to a coarse lattice (default 40 x 30 "ommatidia", ~2.5 deg
  each for an 98 x 80 deg camera; real Drosophila ommatidia are ~5 deg apart
  over a ~270 deg field per eye -- the camera sees only the frontal field).
* Photoreceptor adaptation: a first-order temporal high-pass (cf. the
  transient lamina L1/L2 responses).
* Motion: Hassenstein-Reichardt correlators between neighbouring lattice
  points (the classic model of fly motion vision, implemented in T4/T5)
  for horizontal and vertical motion.
* HS-like signal (per side): mean front-to-back horizontal motion in that
  hemifield (HS cells' preferred direction); H2-like: back-to-front.
* VS-like: mean vertical motion per side.
* Expansion (LC16/LPLC2-like): outward motion relative to the image centre
  (left half: leftward + outward-vertical; right half: rightward + ...).
* Looming of dark objects (LPLC2/LC4-like): relative growth rate of dark
  blobs per hemifield (a 1/tau estimate) gated by their angular size.
* Target (LC10a-like): a bright, saturated "beacon" blob: presence, azimuth
  and angular size.
* Ocellar-like: mean luminance of the upper image per side.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VisionConfig:
    grid_w: int = 40
    grid_h: int = 30
    tau_lp: float = 0.035  # s, EMD delay filter (fly ~ 35 ms)
    tau_hp: float = 0.25  # s, photoreceptor adaptation high-pass
    dark_thresh: float = 0.07  # absolute luminance for "dark object" pixels
    tau_dark: float = 0.06  # s, smoothing of dark-area estimate for looming
    target_lum: float = 0.88  # beacon luminance threshold
    target_sat_max: float = 0.40  # beacon: bright, low-saturation core
    hfov_deg: float = 98.0


class FlyEye:
    def __init__(self, cfg: VisionConfig = VisionConfig()):
        self.cfg = cfg
        self.prev_lp = None
        self.prev_in = None
        self.hp_state = None
        self.prev_dark = None
        self.t_prev = None
        self.last = {}

    def reset(self) -> None:
        self.__init__(self.cfg)

    def _lattice(self, rgb: np.ndarray) -> np.ndarray:
        c = self.cfg
        img = rgb.astype(np.float32) / 255.0
        lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        H, W = lum.shape
        bh, bw = H // c.grid_h, W // c.grid_w
        lum = lum[: bh * c.grid_h, : bw * c.grid_w]
        return lum.reshape(c.grid_h, bh, c.grid_w, bw).mean(axis=(1, 3))

    def process(self, rgb: np.ndarray, t: float) -> dict:
        c = self.cfg
        x = self._lattice(rgb)
        if self.t_prev is None:
            self.t_prev, self.prev_in, self.prev_lp, self.hp_state = t, x, x.copy(), np.zeros_like(x)
            self.prev_dark = self._dark_areas(x)
            self.last = self._empty()
            return self.last
        dt = max(t - self.t_prev, 1e-3)
        a_lp = dt / (c.tau_lp + dt)
        a_hp = c.tau_hp / (c.tau_hp + dt)
        # photoreceptor adaptation (high-pass), keeps sign of contrast change
        self.hp_state = a_hp * (self.hp_state + x - self.prev_in)
        hp = self.hp_state + 0.3 * (x - x.mean())  # retain some sustained contrast
        lp = self.prev_lp + a_lp * (hp - self.prev_lp)
        # Hassenstein-Reichardt correlators: positive = rightward / downward
        emd_h = lp[:, :-1] * hp[:, 1:] - hp[:, :-1] * lp[:, 1:]
        emd_v = lp[:-1, :] * hp[1:, :] - hp[:-1, :] * lp[1:, :]
        self.prev_lp, self.prev_in, self.t_prev = lp, x, t

        W = emd_h.shape[1]
        half = W // 2
        # front-to-back on the left half = leftward (negative); right half = rightward
        ftb_L = float(np.mean(-emd_h[:, :half]))
        ftb_R = float(np.mean(emd_h[:, half:]))
        Hh = emd_v.shape[0] // 2
        # vertical: positive = downward image motion (e.g. climbing / pitching up)
        v_L = float(np.mean(emd_v[:, : emd_v.shape[1] // 2]))
        v_R = float(np.mean(emd_v[:, emd_v.shape[1] // 2 :]))
        # expansion: outward horizontal + outward vertical motion, per hemifield
        out_v = np.concatenate([-emd_v[:Hh], emd_v[Hh:]], axis=0)
        exp_L = float(np.mean(np.maximum(-emd_h[:, :half], 0)) + 0.5 * np.mean(np.maximum(out_v[:, : out_v.shape[1] // 2], 0)))
        exp_R = float(np.mean(np.maximum(emd_h[:, half:], 0)) + 0.5 * np.mean(np.maximum(out_v[:, out_v.shape[1] // 2 :], 0)))
        # dark-object looming (relative growth rate per hemifield)
        dark_raw = self._dark_areas(x)
        a_d = dt / (c.tau_dark + dt)
        dark = self.prev_dark + a_d * (dark_raw - self.prev_dark)
        # relative growth rate of dark area ~ 2/tau for an approaching object
        loom = np.clip((dark - self.prev_dark) / dt / np.maximum(dark, 0.004), 0.0, 30.0)
        loom = loom * (dark > 0.003)
        self.prev_dark = dark
        tgt = self._target(rgb)
        upper = x[: c.grid_h // 3]
        out = {
            "hs_L": ftb_L, "hs_R": ftb_R,  # front-to-back (HS preferred direction)
            "h2_L": -ftb_L, "h2_R": -ftb_R,  # back-to-front (H2 preferred direction)
            "vs_L": v_L, "vs_R": v_R,
            "exp_L": exp_L, "exp_R": exp_R,
            "dark_L": float(dark[0]), "dark_R": float(dark[1]),
            "loom_L": float(loom[0]), "loom_R": float(loom[1]),
            "ocellar_L": float(upper[:, : c.grid_w // 2].mean()), "ocellar_R": float(upper[:, c.grid_w // 2 :].mean()),
            **tgt,
        }
        self.last = out
        return out

    def _dark_areas(self, x: np.ndarray) -> np.ndarray:
        dark = x < self.cfg.dark_thresh
        half = x.shape[1] // 2
        return np.array([dark[:, :half].mean(), dark[:, half:].mean()])

    def _target(self, rgb: np.ndarray) -> dict:
        c = self.cfg
        img = rgb.astype(np.float32) / 255.0
        mx, mn = img.max(-1), img.min(-1)
        sat = (mx - mn) / np.maximum(mx, 1e-3)
        mask = (mx > c.target_lum) & (sat < c.target_sat_max) & (img[..., 2] < img[..., 0] + 0.05)
        frac = float(mask.mean())
        if frac < 2e-4:
            return {"target_present": 0.0, "target_az": 0.0, "target_size": 0.0}
        cols = np.nonzero(mask)[1]
        W = rgb.shape[1]
        az = (cols.mean() / (W - 1) - 0.5) * c.hfov_deg  # deg, + = right
        return {"target_present": 1.0, "target_az": float(az), "target_size": frac}

    @staticmethod
    def _empty() -> dict:
        keys = ["hs_L", "hs_R", "h2_L", "h2_R", "vs_L", "vs_R", "exp_L", "exp_R", "dark_L", "dark_R",
                "loom_L", "loom_R", "ocellar_L", "ocellar_R", "target_present", "target_az", "target_size"]
        return {k: 0.0 for k in keys}
