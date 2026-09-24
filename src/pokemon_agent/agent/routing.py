"""Weighted routing under a selectable POLICY (the deterministic executor of a routing objective).

The middle model (LunaRoute) picks WHERE to go (a target tile); Jev picks the POLICY; this module
executes it — a weighted Dijkstra over the RAM collision + terrain map that returns the next step.
Policies differ only in their per-tile cost, so the model's choice of objective changes the route
while execution stays deterministic and optimal for that objective:

  * ``shortest``    — every walkable tile costs 1 (plain shortest path).
  * ``dodge-grass`` — grass costs a lot: avoid wild-encounter tiles (low HP / just delivering).
  * ``farm-exp``    — grass is the cheap tile: deliberately route THROUGH grass to grind.
"""
from __future__ import annotations

import heapq

from ..core.models import Direction
from .world_map import DELTA, WALL

POLICIES = ("shortest", "dodge-grass", "farm-exp")

_STEP = {(0, -1): Direction.NORTH, (0, 1): Direction.SOUTH, (-1, 0): Direction.WEST, (1, 0): Direction.EAST}


def tile_cost(cls: str, policy: str) -> float:
    """Per-tile step cost for a terrain class under a routing policy."""
    if policy == "dodge-grass":
        return 12.0 if cls == "grass" else 1.0
    if policy == "farm-exp":
        return 1.0 if cls == "grass" else 4.0   # make grass the cheap tile to prefer
    return 1.0                                    # shortest: uniform


def policy_first_step(world, map_id: int, start: tuple[int, int], goal: tuple[int, int],
                      policy: str = "shortest", avoid: set | None = None) -> Direction | None:
    """The first step of the least-COST route from ``start`` to ``goal`` under ``policy`` (weighted
    Dijkstra over the world model's walkable + terrain grids). ``avoid`` cells (NPCs / off-route
    doors) are impassable. Returns a Direction, or None if unreachable / no map ingested."""
    bounds = world.bounds.get(map_id)
    if bounds is None or start == goal:
        return None
    w, h = bounds
    tiles = world.tiles.get(map_id, {})
    terr = world.terrain.get(map_id, {})
    avoid = avoid or set()
    cuts = (getattr(world, "cut_edges", None) or {}).get(map_id, set())
    dist = {start: 0.0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    pq: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    while pq:
        d, c = heapq.heappop(pq)
        if c == goal:
            break
        if d > dist.get(c, 1e18):
            continue
        for (dx, dy) in _STEP:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < w and 0 <= n[1] < h):
                continue
            if tiles.get(n) == WALL or n in avoid:
                continue
            if frozenset({c, n}) in cuts:
                continue      # elevation edge (tile-pair collision)
            nd = d + tile_cost(terr.get(n, "floor"), policy)
            if nd < dist.get(n, 1e18):
                dist[n] = nd
                prev[n] = c
                heapq.heappush(pq, (nd, n))
    if goal not in prev:
        return None
    # walk the predecessor chain back to the first step out of `start`
    cur = goal
    while prev.get(cur) != start:
        cur = prev[cur]
        if cur not in prev:      # defensive: broken chain
            return None
    return _STEP.get((cur[0] - start[0], cur[1] - start[1]))


def grass_nearby(world, map_id: int, center: tuple[int, int], radius: int = 4) -> bool:
    """Whether any grass tile sits within ``radius`` of ``center`` — so policy choice is only
    offered when grass is actually a factor for this leg."""
    terr = world.terrain.get(map_id, {})
    cx, cy = center
    return any(cls == "grass" and abs(x - cx) + abs(y - cy) <= radius
               for (x, y), cls in terr.items())


_REVERSE = {Direction.NORTH: Direction.SOUTH, Direction.SOUTH: Direction.NORTH,
            Direction.EAST: Direction.WEST, Direction.WEST: Direction.EAST}
_ORDER = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)


def grind_step(world, map_id: int, pos: tuple[int, int], last_dir: Direction | None,
               avoid: set | None = None, blocked: set | None = None, anywhere: bool = False) -> Direction | None:
    """Grind in place: the next step that keeps us walking through grass (every grass step rolls a
    wild encounter). On grass: keep going straight while the next tile is grass, else turn onto a
    grass neighbor (not back where we came from), else reverse — so we sweep the patch instead of
    ping-ponging two tiles. Off grass: the first step toward the nearest reachable grass tile.
    None when the map has no reachable grass (the caller wedges the step so L1 grinds elsewhere).
    ``blocked`` = direction values the executor forbids from ``pos`` (ledges — never hop one).
    ``anywhere`` = this map rolls encounters on every walkable tile (caves, towers): pace in place."""
    tiles = world.tiles.get(map_id, {})
    terr = world.terrain.get(map_id, {})
    avoid = avoid or set()
    if anywhere:   # a cave / tower: every walkable tile rolls an encounter, so pace wherever we are
        grass = {c for c, v in tiles.items() if v != WALL and c not in avoid}
    else:
        grass = {c for c, cls in terr.items() if cls == "grass" and tiles.get(c) != WALL and c not in avoid}
    if not grass:
        return None

    blocked = {getattr(b, "value", b) for b in (blocked or set())}

    def nb(d):
        return (pos[0] + DELTA[d][0], pos[1] + DELTA[d][1])

    cuts = (getattr(world, "cut_edges", None) or {}).get(map_id, set())

    def ok(d):
        return d.value not in blocked and frozenset({pos, nb(d)}) not in cuts

    if pos in grass:
        if last_dir is not None and ok(last_dir) and nb(last_dir) in grass:
            return last_dir
        back = _REVERSE.get(last_dir) if last_dir is not None else None
        turns = [d for d in _ORDER if d != back and ok(d) and nb(d) in grass]
        if turns:
            return turns[0]
        if back is not None and ok(back) and nb(back) in grass:
            return back
        # a lone grass tile: step off onto any walkable neighbor (we'll come straight back)
        return next((d for d in _ORDER if ok(d) and tiles.get(nb(d)) not in (None, WALL)
                     and nb(d) not in avoid), None)

    for g in sorted(grass, key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]))[:12]:
        d = policy_first_step(world, map_id, pos, g, "shortest", avoid)
        if d is not None and ok(d):
            return d
    return None


def route_blockers(walkable: set, start: tuple[int, int], goal: tuple[int, int], occupied: dict,
                   cuts: set | None = None) -> list[tuple[str, tuple[int, int]]] | None:
    """Why a route fails: ``[]`` if ``goal`` is reachable around the ``occupied`` cells (NPCs/objects,
    {cell: name}); ``None`` if it's unreachable even ignoring them (walls, not objects); otherwise the
    objects standing in the way — those on the shortest object-free path plus their adjacent objects
    (e.g. both fossils filling Mt. Moon B2F's two-wide corridor). L1 decides what to do about them."""
    cuts = cuts or set()
    walk = set(walkable) | {goal}

    def bfs(block):
        prev = {start: None}
        q = [start]
        while q:
            c = q.pop(0)
            if c == goal:
                path, n = [], c
                while n is not None:
                    path.append(n)
                    n = prev[n]
                return path
            for dx, dy in _STEP:
                n = (c[0] + dx, c[1] + dy)
                if n in walk and n not in prev and n not in block and frozenset({c, n}) not in cuts:
                    prev[n] = c
                    q.append(n)
        return None

    if bfs(set(occupied) - {goal}) is not None:
        return []
    free = bfs(set())
    if free is None:
        return None
    hit = [c for c in free if c in occupied and c != goal]
    frontier, seen = list(hit), set(hit)
    while frontier:
        c = frontier.pop()
        for dx, dy in _STEP:
            n = (c[0] + dx, c[1] + dy)
            if n in occupied and n not in seen:
                seen.add(n)
                frontier.append(n)
    # narrow to what actually closes the route: the smallest adjacent group of these objects whose
    # removal (all other objects kept) opens it — e.g. both fossils, not the Super Nerd you can walk around
    groups, left = [], set(seen)
    while left:
        g, stack = set(), [left.pop()]
        while stack:
            c = stack.pop()
            g.add(c)
            for dx, dy in _STEP:
                n = (c[0] + dx, c[1] + dy)
                if n in left:
                    left.discard(n)
                    stack.append(n)
        groups.append(g)
    opening = [g for g in groups if bfs(set(occupied) - g - {goal}) is not None]
    chosen = min(opening, key=len) if opening else seen
    return sorted(((occupied[c], c) for c in chosen), key=lambda t: (t[1][1], t[1][0]))
