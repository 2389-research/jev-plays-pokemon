"""Where to stand to interact with a person, an object or a door — geometry with one right answer.

The model names WHO/WHAT and WHY; this decides WHERE TO STAND (spec 2026-09-24-interaction-routing).
runs/sleeves-cerulean: a beaten trainer stood on Misty's front tile and the executor walked into him
15+ times while her open side was one step away, because every pathfinder treated an occupied goal as
free. Here a side qualifies only if it is ``standable``, and when none is reachable the caller gets a
report that names who is standing where instead of a doomed move.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..core.models import Direction
from .navigator import ledge_hops

Tile = tuple[int, int]

# the tile you stand on to face the target in each direction, one step away (two across a counter)
_BACK = {Direction.NORTH: (0, 1), Direction.SOUTH: (0, -1), Direction.EAST: (-1, 0), Direction.WEST: (1, 0)}
_SIDE_NAME = {Direction.NORTH: "south", Direction.SOUTH: "north", Direction.EAST: "west", Direction.WEST: "east"}


@dataclass
class Side:
    face: Direction           # direction to face from the stand tile
    stand: Tile
    counter: bool = False     # talked to across a counter tile
    status: str = "open"      # open | occupied | wall | door | cut
    by: str | None = None     # who stands there (status == occupied)
    dist: int | None = None   # walking steps from the player (open sides only; None = unreachable)

    @property
    def name(self) -> str:
        """Which side of the target this is (a north-facing stand tile is the target's south side)."""
        return _SIDE_NAME[self.face]


def why_not(tile: Tile, *, walkable, occupied, warps=(), target_warp: bool = False) -> str | None:
    """None when ``tile`` can be stood on, else the reason: walkable terrain, no sprite on it, and not a
    door (stepping on a door warps you off the map) unless that door is the target."""
    if tile in occupied:
        return "occupied"
    if tile in warps and not target_warp:
        return "door"
    if tile not in walkable:
        return "wall"
    return None


def standable(tile: Tile, *, walkable, occupied, warps=(), target_warp: bool = False) -> bool:
    """One predicate for every chosen destination (see ``why_not``)."""
    return why_not(tile, walkable=walkable, occupied=occupied, warps=warps, target_warp=target_warp) is None


def sides(target: Tile, *, counters=(), face: str | None = None) -> list[Side]:
    """The tiles an interaction with ``target`` can be done from: its 4 neighbours, or the tile 2 away
    in a straight line when the neighbour is a counter (nurse, clerk). ``face`` limits it to the one
    side the game accepts (Bill's PC is used facing north)."""
    tx, ty = target
    out = []
    for d, (dx, dy) in _BACK.items():
        if face and d.value != face:
            continue
        near = (tx + dx, ty + dy)
        if near in counters:
            out.append(Side(d, (tx + 2 * dx, ty + 2 * dy), counter=True))
        else:
            out.append(Side(d, near))
    return out


def distances(start: Tile, *, walkable, blocked=(), cuts=(), terrain=None, ledge_ok=None) -> dict[Tile, int]:
    """Walking steps from ``start`` to every reachable tile (sprites/doors in ``blocked`` are
    obstacles, elevation cuts respected, ledges one-way)."""
    hops = ledge_hops(terrain or {}, ledge_ok)
    dist, q = {start: 0}, deque([start])
    while q:
        c = q.popleft()
        x, y = c
        nxt = [((x + 1, y), False), ((x - 1, y), False), ((x, y + 1), False), ((x, y - 1), False)]
        nxt += [(land, True) for _d, land in hops.get(c, [])]
        for n, hop in nxt:
            if n in dist or n not in walkable or n in blocked:
                continue
            if not hop and frozenset({c, n}) in cuts:
                continue
            dist[n] = dist[c] + 1
            q.append(n)
    return dist


def plan(player: Tile, target: Tile, *, walkable, occupied: dict[Tile, str], warps=(), counters=(),
         cuts=(), terrain=None, face: str | None = None, ledge_ok=None) -> list[Side]:
    """Every side of ``target`` with its status and walking distance; open reachable sides first,
    nearest first. The caller stands on ``result[0]`` when its ``dist`` is not None."""
    others = {xy: who for xy, who in occupied.items() if xy != tuple(target)}
    dist = distances(tuple(player), walkable=walkable, blocked=set(others) | (set(warps) - {tuple(player)}),
                     cuts=cuts, terrain=terrain, ledge_ok=ledge_ok)
    out = []
    for s in sides(tuple(target), counters=counters, face=face):
        s.status = why_not(s.stand, walkable=walkable, occupied=others, warps=warps) or "open"
        if s.status == "open" and not s.counter and frozenset({s.stand, tuple(target)}) in cuts:
            s.status = "cut"                       # a ledge/elevation edge between you and them
        if s.status == "occupied":
            s.by = others.get(s.stand)
        if s.status == "open":
            s.dist = dist.get(s.stand)
        out.append(s)
    out.sort(key=lambda s: (s.dist is None, s.dist if s.dist is not None else 0))
    return out


def report(name: str, target: Tile, player: Tile, planned: list[Side]) -> str:
    """Why no side of the target can be reached, e.g. "Misty (4,2): south (4,3) occupied by
    Cooltrainer F; east (5,2) is walkable but has no path from where you are (7,9) — a separate area;
    north/west wall"."""
    parts, walls = [], []
    for s in planned:
        if s.status == "occupied":
            parts.append(f"{s.name} ({s.stand[0]},{s.stand[1]}) occupied by {s.by or 'someone'}")
        elif s.status == "open":
            parts.append(f"{s.name} ({s.stand[0]},{s.stand[1]}) is walkable but has no path from where you are "
                         f"({player[0]},{player[1]}) — a separate area")
        elif s.status == "door":
            parts.append(f"{s.name} ({s.stand[0]},{s.stand[1]}) is a door")
        elif s.status == "cut":
            parts.append(f"{s.name} ({s.stand[0]},{s.stand[1]}) is across a ledge")
        else:
            walls.append(s.name)
    if walls:
        parts.append("/".join(walls) + " wall")
    return f"{name} ({target[0]},{target[1]}): " + "; ".join(parts)
