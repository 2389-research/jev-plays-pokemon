"""Which DIRECTION connects one map to the next, from the ripped Kanto connection graph.

Map-edge connections (walking off a map's edge into the adjacent map) are not warps, so they aren't
in RAM's warp table. The world graph carries every overworld connection with its direction (ripped
from the pokered disassembly), so this works anywhere in Kanto — no hand-written corridor.
"""
from __future__ import annotations

from .constants import MAP_NAMES_RAW

# name -> id (MAP_NAMES_RAW is id -> name)
_NAME_TO_ID = {v: k for k, v in MAP_NAMES_RAW.items()} if isinstance(MAP_NAMES_RAW, dict) else {
    n: i for i, n in enumerate(MAP_NAMES_RAW)
}


def _mid(name: str) -> int | None:
    return _NAME_TO_ID.get(name)


def next_direction(graph, current_map: int, goal_map: int | None) -> str | None:
    """The direction to walk from `current_map` toward `goal_map` (the world graph's ripped
    connection directions). None if already there / no known route / direction unknown."""
    if goal_map is None or current_map == goal_map:
        return None
    return graph.route_direction(current_map, goal_map) or None   # the ripped Kanto connection graph


def _reaches(graph, frm: int, goal: int) -> bool:
    r = graph.route(frm, goal)
    return bool(r)
