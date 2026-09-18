"""Lightweight session state: goal + plan + recent trajectory. No long-term memory."""
from __future__ import annotations

from collections import deque

from ..core.models import ActionResult, GoalState, PlayerState
from .planner import Plan


class Session:
    def __init__(self, goal: GoalState, *, recent_events_max: int = 10, trajectory_max: int = 20):
        self.goal = goal
        self.running = True
        self.step = 0
        self.plan: Plan | None = None
        self._events: deque[str] = deque(maxlen=recent_events_max)
        self._trajectory: deque[tuple[int, int, int]] = deque(maxlen=trajectory_max)

    @property
    def recent_events(self) -> list[str]:
        return list(self._events)

    def record_result(self, result: ActionResult) -> None:
        for e in result.events:
            self._events.append(e)

    def record_position(self, player: PlayerState | None) -> None:
        if player is not None:
            self._trajectory.append((player.x, player.y, player.map_id))

    def trajectory_summary(self) -> str:
        if not self._trajectory:
            return "no movement recorded yet"
        pts = list(self._trajectory)
        maps = sorted({m for _, _, m in pts})
        xs = [x for x, _, _ in pts]
        ys = [y for _, y, _ in pts]
        uniq = len(set(pts))
        last = pts[-1]
        return (
            f"visited {uniq} unique tiles across maps {maps}; "
            f"x range {min(xs)}-{max(xs)}, y range {min(ys)}-{max(ys)}; "
            f"now at {last[:2]} on map {last[2]}; recent events: {self.recent_events[-6:]}"
        )

    @property
    def summary(self) -> str:
        return f"step={self.step} goal={self.goal.current!r} recent={self.recent_events[-3:]}"
