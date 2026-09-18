"""Deterministic pathing toward a target tile, over what WorldMap knows.

The decider chooses WHERE to go (a Poké Ball, an NPC, an exit); this turns that
into the next concrete button action, so the agent never bumps a wall trying to
reach something it can already see. One primitive action is returned per call, so
the caller stays in control and re-grounds on real position after each step.

Passability: a cell is walkable unless WorldMap has recorded it as a WALL (from a
real blocked move). Unknown cells are treated as tentatively passable — if that
turns out wrong, the controller reports `blocked`, the wall gets recorded, and the
next call routes around it. This is how you head toward something in view.
"""
from __future__ import annotations

from collections import deque

from ..core.models import Direction, InteractAction, MoveAction, PlayerState
from .world_map import DELTA, WALL, WorldMap

_DELTA_TO_DIR = {v: k for k, v in DELTA.items()}


def _dir_between(fx: int, fy: int, tx: int, ty: int) -> Direction | None:
    """Direction from (fx,fy) to an orthogonally-adjacent (tx,ty), else None."""
    return _DELTA_TO_DIR.get((tx - fx, ty - fy))


class Navigator:
    def __init__(self, world: WorldMap):
        self.world = world

    def _passable(self, map_id: int, xy: tuple[int, int], blocked: frozenset | set | None = None) -> bool:
        if blocked and xy in blocked:
            return False  # an object/NPC stands here — can't walk onto or through it
        return self.world.tiles[map_id].get(xy) != WALL

    def _bfs_first_step(
        self, map_id: int, start: tuple[int, int], goals: set[tuple[int, int]],
        blocked: frozenset | set | None = None, max_nodes: int = 4000,
    ) -> Direction | None:
        """First move on a shortest path (over passable tiles) from start to any goal."""
        if start in goals:
            return None
        seen = {start}
        q: deque[tuple[tuple[int, int], Direction | None]] = deque([(start, None)])
        while q and len(seen) < max_nodes:
            (x, y), first = q.popleft()
            for d, (dx, dy) in DELTA.items():
                nxt = (x + dx, y + dy)
                if nxt in seen or not self._passable(map_id, nxt, blocked):
                    continue
                step = first or d
                if nxt in goals:
                    return step
                seen.add(nxt)
                q.append((nxt, step))
        return None

    def step_toward(
        self, player: PlayerState | None, target: dict, blocked: frozenset | set | None = None
    ) -> tuple[MoveAction | InteractAction | None, bool]:
        """Next primitive action toward `target` and whether we've ARRIVED.

        target = {"x", "y", "interact": bool}. Returns:
          (InteractAction(), True)  — adjacent & facing an object: press A now
          (MoveAction(...), False)  — turn to face, or step along the path
          (None, True)              — standing on an exit tile (arrived, no press)
          (None, False)             — no known route to the target
        """
        if player is None:
            return None, False
        m = player.map_id
        tx, ty = int(target["x"]), int(target["y"])
        interact = bool(target.get("interact", True))
        here = (player.x, player.y)

        # never treat the target tile itself as an obstacle for reaching it
        blk = frozenset(b for b in (blocked or ()) if b != (tx, ty))

        if not interact:  # exit: stand on the tile itself
            if here == (tx, ty):
                return None, True
            d = self._bfs_first_step(m, here, {(tx, ty)}, blk)
            return (MoveAction(direction=d), False) if d else (None, False)

        # object/NPC: be on an orthogonal neighbor, facing the target, then press A
        face = _dir_between(player.x, player.y, tx, ty)
        if face is not None:  # already adjacent
            if player.facing == face.value:
                return InteractAction(), True
            return MoveAction(direction=face), False  # turn in place (blocked move just faces)

        # approach only from a neighbor the player can actually STAND on (not another
        # object/NPC, not a wall) — otherwise it tries to walk onto the thing beside it.
        approach = {
            (tx + dx, ty + dy)
            for dx, dy in DELTA.values()
            if self._passable(m, (tx + dx, ty + dy), blk)
        }
        d = self._bfs_first_step(m, here, approach, blk)
        return (MoveAction(direction=d), False) if d else (None, False)
