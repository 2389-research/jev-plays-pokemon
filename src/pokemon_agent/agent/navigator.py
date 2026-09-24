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


_LEDGE_DIR = {"ledge_s": Direction.SOUTH, "ledge_w": Direction.WEST, "ledge_e": Direction.EAST}


def ledge_hops(terrain: dict) -> dict[tuple[int, int], list[tuple[Direction, tuple[int, int]]]]:
    """One-way ledge edges from a map's semantic terrain: {take-off cell: [(hop direction, landing)]}.
    Stepping onto a ledge tile in its direction hops the player two cells (the ledge tile itself is
    never stood on); the reverse is impossible."""
    out: dict[tuple[int, int], list] = {}
    for (x, y), cls in terrain.items():
        d = _LEDGE_DIR.get(cls)
        if d is None:
            continue
        dx, dy = DELTA[d]
        out.setdefault((x - dx, y - dy), []).append((d, (x + dx, y + dy)))
    return out


class Navigator:
    def __init__(self, world: WorldMap):
        self.world = world

    def _passable(self, map_id: int, xy: tuple[int, int], blocked: frozenset | set | None = None) -> bool:
        if blocked and xy in blocked:
            return False  # an object/NPC stands here — can't walk onto or through it
        # When the full map extent is known (a collision ingest), out-of-bounds cells are NOT
        # passable — otherwise BFS routes off the map edge through unmarked border cells.
        bounds = getattr(self.world, "bounds", {}).get(map_id)
        if bounds is not None:
            w, h = bounds
            if not (0 <= xy[0] < w and 0 <= xy[1] < h):
                return False
        return self.world.tiles[map_id].get(xy) != WALL

    def _cuts(self, map_id: int) -> set:
        return (getattr(self.world, "cut_edges", None) or {}).get(map_id, set())

    def _bfs_first_step(
        self, map_id: int, start: tuple[int, int], goals: set[tuple[int, int]],
        blocked: frozenset | set | None = None, max_nodes: int = 4000,
    ) -> Direction | None:
        """First move on a shortest path (over passable tiles, plus one-way LEDGE hops) from start to
        any goal. Sets ``self.first_is_hop`` when that first move is a planned ledge hop."""
        self.first_is_hop = False
        if start in goals:
            return None
        hops = ledge_hops(self.world.terrain.get(map_id) or {})
        seen = {start}
        q: deque[tuple[tuple[int, int], Direction | None, bool]] = deque([(start, None, False)])
        while q and len(seen) < max_nodes:
            (x, y), first, hop0 = q.popleft()
            moves = [(d, (x + dx, y + dy), False) for d, (dx, dy) in DELTA.items()]
            moves += [(d, land, True) for d, land in hops.get((x, y), [])]
            for d, nxt, is_hop in moves:
                if nxt in seen or not self._passable(map_id, nxt, blocked):
                    continue
                if not is_hop and frozenset({(x, y), nxt}) in self._cuts(map_id):
                    continue      # an elevation edge: both cells walkable, the step between them isn't
                step, h = (first, hop0) if first else (d, is_hop)
                if nxt in goals:
                    self.first_is_hop = h
                    return step
                seen.add(nxt)
                q.append((nxt, step, h))
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

        # A tile we must STAND ON is never exempt from occupancy: someone standing there makes it
        # unreachable, not one step away (runs/sleeves-cerulean: 15+ walks into a trainer on Misty's
        # front tile). Only an interaction target (the person/object itself) may be in ``blocked``.
        self.goal_blocked = False
        if interact:
            blk = frozenset(b for b in (blocked or ()) if b != (tx, ty))
        else:
            blk = frozenset(blocked or ())

        if not interact:  # exit: stand on the tile itself
            if here == (tx, ty):
                return None, True
            if (tx, ty) in blk:
                self.goal_blocked = True
                return None, False
            d = self._bfs_first_step(m, here, {(tx, ty)}, blk)
            return (MoveAction(direction=d), False) if d else (None, False)

        # object/NPC: be on an orthogonal neighbor, facing the target, then press A
        face = _dir_between(player.x, player.y, tx, ty)
        if face is not None:  # already adjacent
            if player.facing == face.value:
                return InteractAction(), True
            return MoveAction(direction=face), False  # turn in place (blocked move just faces)

        # COUNTER TALK: an NPC behind a real COUNTER tile is reached from 2 tiles away in a
        # straight line (talk over the counter). Only over actual counter cells (from RAM's
        # tileset talk-over tiles) — never a generic wall.
        cface = self._counter_face(player, tx, ty)
        if cface is not None:
            if player.facing == cface.value:
                return InteractAction(), True
            return MoveAction(direction=cface), False

        # approach set: distance-1 walkable neighbors, PLUS distance-2 tiles across a counter.
        approach = {
            (tx + dx, ty + dy)
            for dx, dy in DELTA.values()
            if self._passable(m, (tx + dx, ty + dy), blk)
        }
        for dx, dy in DELTA.values():
            mid, far = (tx + dx, ty + dy), (tx + 2 * dx, ty + 2 * dy)
            if mid in self._counters(m) and self._passable(m, far, blk):
                approach.add(far)
        d = self._bfs_first_step(m, here, approach, blk)
        return (MoveAction(direction=d), False) if d else (None, False)

    def _counters(self, map_id: int) -> set:
        return getattr(self.world, "counters", {}).get(map_id, set())

    def _counter_face(self, player: PlayerState, tx: int, ty: int):
        """Direction to face to talk to (tx,ty) ACROSS a counter: target exactly 2 tiles away
        in a straight line with a real COUNTER cell between. Else None."""
        dx, dy = tx - player.x, ty - player.y
        if (abs(dx), dy) == (2, 0):
            step = (1 if dx > 0 else -1, 0)
        elif (dx, abs(dy)) == (0, 2):
            step = (0, 1 if dy > 0 else -1)
        else:
            return None
        mid = (player.x + step[0], player.y + step[1])
        if mid not in self._counters(player.map_id):
            return None
        return _DELTA_TO_DIR.get(step)
