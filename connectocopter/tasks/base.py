"""Task interface.  A task builds its arena from a seed, injects stimuli,
optionally issues mission-level commands, and scores the episode."""
from __future__ import annotations

from ..robot.robot import Command


class Task:
    name = "task"
    max_time = 30.0
    description = ""

    def build(self, rng):  # -> (Arena, Environment)
        raise NotImplementedError

    def setup(self, ep) -> None:
        pass

    def pre_step(self, ep, t: float) -> None:
        """Called once per control step before sensing (move stimuli etc.)."""

    def physics_tick(self, ep, t: float, dt: float) -> None:
        """Called every physics step (smooth stimulus motion)."""

    def mission(self, ep, cmd: Command, t: float) -> Command:
        """Mission-level overrides (not attributed to the brain); default none."""
        return cmd

    def done(self, ep, t: float) -> bool:
        return False

    def metrics(self, ep) -> dict:
        return {}

    def telemetry(self, ep) -> dict | None:
        return None
