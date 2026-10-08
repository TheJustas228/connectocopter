#!/usr/bin/env python3
"""Render README media from recorded simulation output.

Every image and clip is produced by MuJoCo from the robot/world state stored
in a replay file (web/replays/*.json.gz) -- no re-simulation, no editing of
the trajectory.  Outputs:

* docs/img/robot_closeup.png, docs/img/robot_flight.png  -- still renders
* docs/img/<episode>.gif   -- short chase-camera clips with an FPV inset and a
                              small HUD of the neural signals for that moment
* docs/media/<episode>.mp4 -- the same clips at higher quality

    python scripts/make_media.py                 # everything
    python scripts/make_media.py --only looming  # one clip
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import connectocopter  # noqa: F401  (GL setup)
from connectocopter.robot.model import WHEELS
from connectocopter.robot.world import Arena, build_xml

ROOT = Path(__file__).resolve().parent.parent
REPLAYS = ROOT / "web" / "replays"
IMG = ROOT / "docs" / "img"
MEDIA = ROOT / "docs" / "media"


def font(size, bold=False):
    for p in [f"/usr/share/fonts/truetype/lato/Lato-{'Bold' if bold else 'Regular'}.ttf",
              f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load(name):
    return json.loads(gzip.decompress((REPLAYS / name).read_bytes()))


def world_from_replay(r):
    a = r["arena"]
    objs = [{k: (tuple(v) if isinstance(v, list) else v) for k, v in o.items()} for o in a["objects"]]
    ar = Arena(size=tuple(a["size"]), center=tuple(a["center"]), walls=a["walls"], wall_height=a["wall_height"], objects=objs)
    m = mujoco.MjModel.from_xml_string(build_xml(ar))
    return m, mujoco.MjData(m)


def set_state(m, d, f):
    jr = m.joint("root")
    d.qpos[jr.qposadr[0]:jr.qposadr[0] + 3] = f["pos"]
    d.qpos[jr.qposadr[0] + 3:jr.qposadr[0] + 7] = f["quat"]
    for n, ang in zip(WHEELS, f["wheels"]):
        d.qpos[m.jnt_qposadr[m.joint(f"wheel_{n}").id]] = ang
    for i, p in enumerate(f.get("mocap", [])):
        d.mocap_pos[i] = p
    mujoco.mj_forward(m, d)


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


HUD_KEYS = {
    "looming_escape": [("LPLC2+LC4 L", "in", "looming_L"), ("LPLC2+LC4 R", "in", "looming_R"), ("Giant Fiber L", "dn", "GF_L"), ("Giant Fiber R", "dn", "GF_R")],
    "vibration_escape": [("JO-A/B L", "in", "vibration_L"), ("JO-A/B R", "in", "vibration_R"), ("Giant Fiber L", "dn", "GF_L"), ("Giant Fiber R", "dn", "GF_R")],
    "obstacle_course": [("LC16 L", "in", "obstacle_L"), ("LC16 R", "in", "obstacle_R"), ("DNa02 L", "dn", "DNa02_L"), ("DNa02 R", "dn", "DNa02_R")],
    "flight_course": [("LC10a L", "in", "target_L"), ("LC10a R", "in", "target_R"), ("LC16 L", "in", "obstacle_L"), ("LC16 R", "in", "obstacle_R")],
    "target_seek": [("LC10a L", "in", "target_L"), ("LC10a R", "in", "target_R"), ("DNa02 L", "dn", "DNa02_L"), ("DNa02 R", "dn", "DNa02_R")],
    "taste_dock": [("sugar GRNs", "in", "taste_sugar"), ("bitter GRNs", "in", "taste_bitter"), ("MN9 L", "dn", "MN9_L"), ("MN9 R", "dn", "MN9_R")],
    "yaw_stabilization": [("HS L", "in", "yawflow_HS_L"), ("HS R", "in", "yawflow_HS_R"), ("DNp15 L", "dn", "DNp15_L"), ("DNp15 R", "dn", "DNp15_R")],
}


def hud(img, f, r, title):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    dr.rectangle([0, 0, W, 44], fill=(18, 26, 38, 215))
    dr.text((14, 10), title, font=font(19, True), fill=(215, 222, 232))
    mode = f["mode"]
    col = (242, 169, 59) if mode == "ground" else (143, 184, 232)
    tw = dr.textlength(mode, font=font(17, True))
    dr.rounded_rectangle([W - tw - 34, 9, W - 12, 35], radius=12, outline=col, width=2, fill=(18, 26, 38, 200))
    dr.text((W - tw - 23, 11), mode, font=font(17, True), fill=col)
    dr.text((14, H - 30), f"t = {f['t']:.2f} s   ·   {r['controller']} controller   ·   recorded simulation output", font=font(15), fill=(200, 208, 220))
    b = f.get("brain")
    keys = HUD_KEYS.get(r["task"] if not (r["task"] == "looming_escape" and r.get("metrics", {}).get("stimulus") == "vibration") else "vibration_escape", [])
    if b and keys:
        x0, y0 = 14, 58
        dr.rectangle([x0 - 6, y0 - 6, x0 + 260, y0 + 26 * len(keys) + 4], fill=(18, 26, 38, 200))
        for i, (lab, kind, key) in enumerate(keys):
            v = (b["in"] if kind == "in" else b["dn"]).get(key, 0.0)
            c = (143, 184, 232) if kind == "in" else (99, 230, 140)
            y = y0 + 26 * i
            dr.text((x0, y), lab, font=font(14), fill=(200, 208, 220))
            dr.rectangle([x0 + 110, y + 3, x0 + 110 + 110, y + 17], fill=(38, 51, 69, 255))
            dr.rectangle([x0 + 110, y + 3, x0 + 110 + int(110 * min(v / 150.0, 1.0)), y + 17], fill=c + (255,))
            dr.text((x0 + 226, y), f"{v:.0f}", font=font(14), fill=(215, 222, 232))
        cmd = f["cmd"]
        txt = "ESCAPE" if cmd["escape"] else ("HALT (feed)" if cmd["halt"] else f"yaw {cmd['yaw']:+.2f} rad/s")
        dr.text((x0, y0 + 26 * len(keys) + 8), f"→ {txt}", font=font(15, True), fill=(242, 169, 59))
    return im


def fpv_for(r, i):
    for k in range(i, max(-1, i - 40), -1):
        if r["frames"][k].get("fpv"):
            return Image.open(io.BytesIO(base64.b64decode(r["frames"][k]["fpv"])))
    return None


def vis_option():
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False  # hide the sensor ray (visualisation only)
    return opt


def render_clip(name, title, t0=0.0, t1=None, size=(960, 544), every=2, gif_width=480, gif_fps=10, dist=None,
                elev=-16.0, az_off=-25.0):
    r = load(name)
    m, d = world_from_replay(r)
    ren = mujoco.Renderer(m, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    frames = r["frames"]
    t1 = t1 if t1 is not None else frames[-1]["t"]
    sel = [i for i, f in enumerate(frames) if t0 <= f["t"] <= t1][::every]
    out = []
    yaw_s = None
    look = None
    for i in sel:
        f = frames[i]
        set_state(m, d, f)
        y = yaw_of(f["quat"])
        if yaw_s is None:
            yaw_s = y
        yaw_s += np.arctan2(np.sin(y - yaw_s), np.cos(y - yaw_s)) * 0.12
        p = np.array(f["pos"])
        look = p if look is None else look + 0.25 * (p - look)
        cam.lookat[:] = look + np.array([0, 0, 0.05])
        cam.azimuth = np.rad2deg(yaw_s) + 180 + az_off
        cam.elevation = elev
        cam.distance = dist or (1.25 if f["mode"] == "ground" else 1.9)
        ren.update_scene(d, cam, vis_option())
        img = hud(ren.render(), f, r, title)
        fp = fpv_for(r, i)
        if fp is not None:
            fp = fp.resize((240, 180))
            img.paste(fp, (size[0] - 252, size[1] - 222))
            ImageDraw.Draw(img).rectangle([size[0] - 253, size[1] - 223, size[0] - 12, size[1] - 42], outline=(143, 184, 232), width=2)
        out.append(np.asarray(img))
    ren.close()
    stem = name.replace(".json.gz", "")
    MEDIA.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / (r["control_dt"] * every)
    iio.imwrite(MEDIA / f"{stem}.mp4", np.stack(out), fps=fps, codec="libx264", quality=7, macro_block_size=8)
    # GIF: downscale and subsample to keep the file small
    step = max(1, int(round(fps / gif_fps)))
    g = [np.asarray(Image.fromarray(o).resize((gif_width, int(gif_width * size[1] / size[0])), Image.LANCZOS)) for o in out[::step]]
    pal = [Image.fromarray(x).convert("P", palette=Image.ADAPTIVE, colors=96) for x in g]
    pal[0].save(IMG / f"{stem}.gif", save_all=True, append_images=pal[1:], duration=int(1000 / gif_fps), loop=0, optimize=True)
    print(f"{stem}: {len(out)} frames -> {MEDIA / (stem + '.mp4')} ({(MEDIA / (stem + '.mp4')).stat().st_size / 1e6:.1f} MB), "
          f"gif {(IMG / (stem + '.gif')).stat().st_size / 1e6:.1f} MB")


def stills():
    r = load("taste_dock_connectome.json.gz")
    m, d = world_from_replay(r)
    set_state(m, d, r["frames"][10])
    ren = mujoco.Renderer(m, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array(r["frames"][10]["pos"]) + np.array([0, 0, 0.0])
    cam.azimuth, cam.elevation, cam.distance = 140, -20, 0.62
    ren.update_scene(d, cam, vis_option())
    Image.fromarray(ren.render()).save(IMG / "robot_closeup.png")
    r = load("flight_course_connectome.json.gz")
    m, d = world_from_replay(r)
    k = next(i for i, f in enumerate(r["frames"]) if f["mode"] == "flight" and f["t"] > 4.0)
    set_state(m, d, r["frames"][k])
    ren2 = mujoco.Renderer(m, 720, 1280)
    cam.lookat[:] = r["frames"][k]["pos"]
    cam.azimuth, cam.elevation, cam.distance = 200, -12, 1.4
    ren2.update_scene(d, cam, vis_option())
    Image.fromarray(ren2.render()).save(IMG / "robot_flight.png")
    print("stills written")


CLIPS = [
    ("looming_escape_connectome.json.gz", "Looming ball → LPLC2/LC4 → Giant Fiber → escape take-off", {"t0": 1.5, "dist": 3.2, "elev": -10, "az_off": -150}),
    ("flight_course_connectome.json.gz", "Flight course: LC10a beacon tracking + LC16 avoidance", {"t0": 0.0, "t1": 12.0}),
    ("obstacle_course_connectome.json.gz", "Rolling: LC16 → contralateral DNa01/DNa02 steer around pillars", {"t0": 2.0, "t1": 12.0}),
    ("taste_dock_connectome.json.gz", "Taste: sugar GRNs → MN9 → stop and 'feed'; bitter vetoes", {}),
    ("target_seek_connectome.json.gz", "Beacon → LC10a → ipsilateral DNa02 → turn toward it", {}),
    ("yaw_stabilization_connectome.json.gz", "Yaw gyro failed: HS/H2 → DNp15 optomotor response", {"t0": 3.0, "t1": 11.0}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    if not a.only:
        stills()
    for name, title, kw in CLIPS:
        if a.only and not any(o in name for o in a.only):
            continue
        if (REPLAYS / name).exists():
            render_clip(name, title, **kw)


if __name__ == "__main__":
    main()
