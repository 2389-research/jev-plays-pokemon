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
from .world_map import WALL

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
