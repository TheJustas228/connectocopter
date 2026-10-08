#!/usr/bin/env python3
"""Capture screenshots and an animated GIF of the browser viewer (README media).

Serves web/ on a local port, drives the viewer in headless Chromium through
its ``window.connectocopter`` hook, and saves:
  docs/img/viewer_<episode>.png  and  docs/img/viewer_hero.gif
Every frame is a recorded simulation frame rendered by the viewer.
"""
from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]

SHOTS = [  # (replay file, time in s, output name)
    ("looming_escape_connectome.json.gz", 3.86, "viewer_looming"),
    ("taste_dock_connectome.json.gz", 5.9, "viewer_taste"),
    ("flight_course_connectome.json.gz", 6.0, "viewer_flight"),
    ("obstacle_course_connectome.json.gz", 6.8, "viewer_obstacle"),
]
HERO = ("looming_escape_connectome.json.gz", 3.0, 5.0, 3)  # file, t0, t1, frame step


def serve(port: int):
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "web"))
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main(port: int = 8799):
    srv = serve(port)
    IMG.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(args=ARGS)
        pg = b.new_page(viewport={"width": 1440, "height": 1560}, device_scale_factor=1)
        pg.goto(f"http://127.0.0.1:{port}/index.html")
        pg.wait_for_function("window.connectocopter && window.connectocopter.frameCount() > 0", timeout=60000)
        for f, t, name in SHOTS:
            pg.evaluate(f"window.connectocopter.load('{f}')")
            pg.wait_for_timeout(2500)
            i = int(round(t / 0.02))
            for k in range(max(0, i - 12), i + 1):  # replay a few frames so activity traces build up
                pg.evaluate(f"window.connectocopter.showFrame({k})")
                pg.wait_for_timeout(60)
            pg.wait_for_timeout(1200)
            pg.screenshot(path=str(IMG / f"{name}.png"), full_page=True)
            print("saved", name)
        f, t0, t1, step = HERO
        pg.evaluate(f"window.connectocopter.load('{f}')")
        pg.wait_for_timeout(2500)
        frames = []
        for k in range(int(t0 / 0.02), int(t1 / 0.02), step):
            pg.evaluate(f"window.connectocopter.showFrame({k})")
            pg.wait_for_timeout(250)
            path = IMG / "_hero_tmp.png"
            pg.screenshot(path=str(path), clip={"x": 0, "y": 160, "width": 1440, "height": 840})
            frames.append(Image.open(path).convert("RGB").resize((960, 560), Image.LANCZOS))
        (IMG / "_hero_tmp.png").unlink(missing_ok=True)
        pal = [fr.convert("P", palette=Image.ADAPTIVE, colors=128) for fr in frames]
        pal[0].save(IMG / "viewer_hero.gif", save_all=True, append_images=pal[1:], duration=120, loop=0, optimize=True)
        print("saved viewer_hero.gif", len(frames), "frames", round((IMG / "viewer_hero.gif").stat().st_size / 1e6, 1), "MB")
        b.close()
    srv.shutdown()


if __name__ == "__main__":
    main()
