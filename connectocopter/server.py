"""Local web server: the static viewer plus live simulations over WebSocket.

GET  /                      -> web/index.html (and the rest of web/)
GET  /api/info              -> {"live": true, "tasks": [...]}
WS   /ws/live?task=..&controller=..&seed=..
        first message: {"type": "header", ...scene...}
        then:          {"type": "frame", ...}  (one per 20 ms control step, as fast as simulated)
        finally:       {"type": "done", "metrics": {...}}

Simulations run in a worker thread; one at a time per process (the brain
holds GPU state).  This is for local use only -- do not expose it publicly.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

# imported at module level: FastAPI resolves (string) annotations from module globals
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
_lock = threading.Lock()
_ctrl_cache: dict = {}


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def build_app():
    from .cli import TASKS, make_controller, make_task

    app = FastAPI(title="Connectocopter")

    @app.get("/api/info")
    def info():
        return {"live": True, "tasks": sorted(TASKS), "controllers": ["connectome", "baseline"]}

    @app.websocket("/ws/live")
    async def live(ws: WebSocket):
        await ws.accept()
        q = ws.query_params
        task = q.get("task", "obstacle_course")
        kind = q.get("controller", "connectome")
        seed = int(q.get("seed", "2100"))
        if task not in TASKS or kind not in ("connectome", "baseline"):
            await ws.send_text(json.dumps({"type": "error", "message": "unknown task or controller"}))
            await ws.close()
            return
        if not _lock.acquire(blocking=False):
            await ws.send_text(json.dumps({"type": "error", "message": "a simulation is already running; try again when it ends"}))
            await ws.close()
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def worker():
            from .sim.episode import Episode

            try:
                if kind not in _ctrl_cache:
                    _ctrl_cache[kind] = make_controller(kind, record_activity=True)
                ctrl = _ctrl_cache[kind]

                def on_frame(fr):
                    if stop.is_set():
                        raise KeyboardInterrupt
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "frame", **fr})

                ep = Episode(make_task(task), ctrl, seed, on_frame=on_frame, record_fpv_every=3)
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "header", **ep.header()})
                res = ep.run()
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "metrics": res.metrics})
            except KeyboardInterrupt:
                pass
            except Exception as e:  # report to the browser instead of dying silently
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        try:
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                await ws.send_text(json.dumps(msg, separators=(",", ":"), default=_json_default))
        except WebSocketDisconnect:
            stop.set()
        finally:
            stop.set()
            th.join(timeout=30)
            _lock.release()
            try:
                await ws.close()
            except Exception:
                pass

    app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
    return app


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    print(f"Connectocopter viewer: http://{host}:{port}/   (live simulations at ?live=<task>)")
    uvicorn.run(build_app(), host=host, port=port, log_level="warning")
