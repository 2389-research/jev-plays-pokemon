"""Static corridor knowledge: which DIRECTION connects one map to the next.

Map-edge connections (walking off a map's edge into the adjacent map) are not warps,
so they aren't in RAM's warp table — the agent can't otherwise know that Route 1 is
NORTH of Pallet. This is exactly the kind of fixed fact the spec says to inject. We
encode the Pallet→Pewter corridor directions and use the world graph to pick the next
map, then look up which way to walk toward it.
"""
from __future__ import annotations

from .constants import MAP_NAMES_RAW

# name -> id (MAP_NAMES_RAW is id -> name)
_NAME_TO_ID = {v: k for k, v in MAP_NAMES_RAW.items()} if isinstance(MAP_NAMES_RAW, dict) else {
    n: i for i, n in enumerate(MAP_NAMES_RAW)
}


def _mid(name: str) -> int | None:
    return _NAME_TO_ID.get(name)


PEWTER = _mid("Pewter City")

# directed edges along the corridor, with the walking direction from -> to.
_CORRIDOR = [
    ("Pallet Town", "Route 1", "north"),
    ("Route 1", "Viridian City", "north"),
    ("Viridian City", "Route 2", "north"),
    ("Route 2", "Viridian Forest", "north"),
    ("Viridian Forest", "Pewter City", "north"),
]

CONNECTION_DIR: dict[tuple[int, int], str] = {}
_OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
for _a, _b, _d in _CORRIDOR:
    ia, ib = _mid(_a), _mid(_b)
    if ia is not None and ib is not None:
        CONNECTION_DIR[(ia, ib)] = _d
        CONNECTION_DIR[(ib, ia)] = _OPP[_d]


def next_direction(graph, current_map: int, goal_map: int | None) -> str | None:
    """The direction to walk from `current_map` toward `goal_map`. Uses the world
    graph's ripped connection directions (authoritative), falling back to the small
    hand table for any edge the graph doesn't carry a direction for. None if already
    there / no known route / direction unknown."""
    if goal_map is None or current_map == goal_map:
        return None
    d = graph.route_direction(current_map, goal_map)  # from the ripped pokered graph
    if d:
        return d
    route = graph.route(current_map, goal_map)         # fallback: hand table on the next hop
    if route and len(route) >= 2:
        return CONNECTION_DIR.get((current_map, route[1]))
    return None


def _reaches(graph, frm: int, goal: int) -> bool:
    r = graph.route(frm, goal)
    return bool(r)
