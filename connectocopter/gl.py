"""Pick a working OpenGL backend for MuJoCo offscreen rendering.

* WSL2 (WSLg): GLFW on Mesa.  We deliberately keep Mesa's llvmpipe CPU
  rasteriser: MuJoCo issues many small draw calls and the d3d12 (GPU) path
  was 6-13x slower in our measurements (FPV 160x120: 5 ms vs 68 ms;
  720p with shadows: 44 ms vs 281 ms on an RTX 3070).
* Linux with an NVIDIA/EGL driver: EGL (headless).
* Anything else: leave MuJoCo's default (GLFW, needs a display) or set
  ``MUJOCO_GL=osmesa`` yourself for pure software rendering.

User-provided ``MUJOCO_GL`` / ``GALLIUM_DRIVER`` always take precedence.
"""
from __future__ import annotations

import os
from pathlib import Path


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def configure_gl() -> str:
    if "MUJOCO_GL" not in os.environ:
        if is_wsl():
            os.environ["MUJOCO_GL"] = "glfw"
        elif Path("/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0").exists():
            os.environ["MUJOCO_GL"] = "egl"
    return os.environ.get("MUJOCO_GL", "default")
