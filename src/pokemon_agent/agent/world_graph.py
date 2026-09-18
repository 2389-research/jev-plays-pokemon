"""Cross-map world graph: navigation OVER maps, not within one.

`WorldMap`/`Navigator` handle tile-level pathing inside a single map. This is the
layer above them: a directed graph whose NODES are map ids and whose EDGES are
warps — "step onto exit tile (x, y) on map A and you arrive on map B". Once the
agent observes a map's RAM `exits` (each {x, y, dest_map, dest_name}), every warp
on that map becomes a known edge, so "route to Pewter" collapses to a sequence of
maps plus which exit tile to take on each hop.

Edge tiles:
  * A warp observed from RAM carries its exact exit tile (x, y).
  * A *seeded* adjacency (see `seeded_kanto_graph`) records that two maps connect
    without yet knowing the tile — stored as tile ``None`` (UNKNOWN_TILE).
  * A real observation (`add_warp` / `observe_exits`) always refines/overwrites an
    unknown-tile edge; a bare adjacency never clobbers a known tile.

`next_hop` therefore returns the next map to head to even when its exit tile is
still unknown (tile ``None``) — the caller can then fall back to on-map
exploration to find the door, and a later `observe_exits` fills the tile in.
"""
from __future__ import annotations

from collections import deque

from ..games.pokemon_red.constants import MAP_NAMES_RAW

# Tile value for an edge whose exit coordinate is not yet known (seeded adjacency).
UNKNOWN_TILE: None = None

Tile = tuple[int, int] | None


class WorldGraph:
    """Directed graph of map connectivity. Nodes are map ids (int); an edge
    ``from_map -> to_map`` carries the exit tile on ``from_map`` that leads there
    (or ``None`` if only the adjacency is known)."""

    def __init__(self) -> None:
        # edges[from_map][to_map] = (x, y) | None
        self.edges: dict[int, dict[int, Tile]] = {}
        # dirs[(from_map, to_map)] = "north"|"south"|"east"|"west" (map-edge connections)
        self.dirs: dict[tuple[int, int], str] = {}
        # names from a seed that could not be resolved to a map id (diagnostics)
        self.unresolved: list[str] = []

    # --- construction -----------------------------------------------------
    def _set_edge(self, from_map: int, to_map: int, tile: Tile) -> None:
        d = self.edges.setdefault(from_map, {})
        if tile is not None:
            d[to_map] = tile  # a known exit tile always wins / refines
        else:
            d.setdefault(to_map, None)  # adjacency: never clobber a known tile
        self.edges.setdefault(to_map, self.edges.get(to_map, {}))  # ensure node exists

    def add_warp(self, from_map: int, x: int, y: int, to_map: int) -> None:
        """Record a directed warp: exit tile (x, y) on ``from_map`` -> ``to_map``."""
        self._set_edge(from_map, to_map, (int(x), int(y)))

    def add_connection(self, from_map: int, direction: str, to_map: int) -> None:
        """Record a map-edge connection: walk `direction` off `from_map` -> `to_map`.
        (Edge connections aren't RAM warps, so they carry no exit tile — the direction
        is what the agent needs to head that way.)"""
        self._set_edge(from_map, to_map, None)
        self.dirs[(from_map, to_map)] = direction

    def direction_between(self, from_map: int, to_map: int) -> str | None:
        """The compass direction to walk from `from_map` to a directly-connected map."""
        return self.dirs.get((from_map, to_map))

    def route_direction(self, from_map: int, goal_map: int) -> str | None:
        """The direction to walk NOW to progress from `from_map` toward `goal_map`."""
        hop = self.next_hop(from_map, goal_map)
        return self.dirs.get((from_map, hop[0])) if hop else None

    def add_adjacency(self, a: int, b: int) -> None:
        """Record a bidirectional connection a<->b with the exit tiles unknown.

        Used for seeding known map layout before the tiles have been observed.
        Does not overwrite any exit tile already known in either direction."""
        self._set_edge(a, b, None)
        self._set_edge(b, a, None)

    def observe_exits(self, map_id: int, exits: list[dict]) -> None:
        """Ingest a map's RAM exits (each {'x','y','dest_map', ...}) as warp edges."""
        for e in exits or []:
            dest = e.get("dest_map")
            if dest is None:
                continue
            self.add_warp(map_id, e["x"], e["y"], int(dest))

    # --- queries ----------------------------------------------------------
    def neighbors(self, map_id: int) -> set[int]:
        """Map ids directly reachable via one warp from ``map_id``."""
        return set(self.edges.get(map_id, {}).keys())

    def route(self, from_map: int, to_map: int) -> list[int] | None:
        """Shortest sequence of map ids from ``from_map`` to ``to_map`` (BFS),
        inclusive of both endpoints. ``[from_map]`` if equal; ``None`` if there is
        no known path."""
        if from_map == to_map:
            return [from_map]
        prev: dict[int, int] = {from_map: from_map}
        q: deque[int] = deque([from_map])
        while q:
            cur = q.popleft()
            for nxt in self.edges.get(cur, {}):
                if nxt in prev:
                    continue
                prev[nxt] = cur
                if nxt == to_map:
                    path = [to_map]
                    while path[-1] != from_map:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                q.append(nxt)
        return None

    def next_hop(self, from_map: int, to_map: int) -> tuple[int, Tile] | None:
        """The next map to head to plus the exit tile on ``from_map`` that leads
        there. Returns ``None`` when already at the destination or no route exists.

        The tile may be ``None`` (edge known only as an adjacency); the map hop is
        still returned so the caller can navigate toward it and discover the door."""
        path = self.route(from_map, to_map)
        if path is None or len(path) < 2:
            return None
        nxt = path[1]
        tile = self.edges.get(from_map, {}).get(nxt)
        return (nxt, tile)

    # --- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        """JSON-serializable snapshot (map-id keys as strings, tiles as lists)."""
        return {
            "edges": {
                str(f): {
                    str(t): (list(tile) if tile is not None else None)
                    for t, tile in dests.items()
                }
                for f, dests in self.edges.items()
            },
            "dirs": {f"{a},{b}": d for (a, b), d in self.dirs.items()},
            "unresolved": list(self.unresolved),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorldGraph":
        g = cls()
        for f, dests in (d.get("edges") or {}).items():
            fm = int(f)
            bucket = g.edges.setdefault(fm, {})
            for t, tile in (dests or {}).items():
                bucket[int(t)] = (int(tile[0]), int(tile[1])) if tile is not None else None
        for key, direction in (d.get("dirs") or {}).items():
            a, b = key.split(",")
            g.dirs[(int(a), int(b))] = direction
        g.unresolved = list(d.get("unresolved") or [])
        return g


# --- seed ----------------------------------------------------------------
# name->id lookup, reverse of the disassembly-generated MAP_NAMES_RAW.
_NAME_TO_ID = {name: mid for mid, name in MAP_NAMES_RAW.items()}

# The Pallet -> Pewter corridor as ordered name adjacencies. Route 2 is a single
# map that touches both the south (Viridian) and north (Pewter/forest) ends, so it
# appears in several pairs. Gate buildings sit between Route 2 and Viridian Forest.
_KANTO_CORRIDOR: list[tuple[str, str]] = [
    ("Pallet Town", "Route 1"),
    ("Route 1", "Viridian City"),
    ("Viridian City", "Route 2"),
    ("Route 2", "Viridian Forest South Gate"),
    ("Viridian Forest South Gate", "Viridian Forest"),
    ("Viridian Forest", "Viridian Forest North Gate"),
    ("Viridian Forest North Gate", "Route 2"),
    ("Route 2", "Pewter City"),
]


def full_kanto_graph() -> WorldGraph:
    """The whole Kanto map-connection graph, ripped from the pokered disassembly
    (map_graph_data.CONNECTIONS): every overworld map-edge connection with its
    direction. This is the agent's authoritative regional map for routing."""
    from ..games.pokemon_red.map_graph_data import CONNECTIONS

    g = WorldGraph()
    for from_id, direction, to_id in CONNECTIONS:
        g.add_connection(from_id, direction, to_id)
    return g


def seeded_kanto_graph() -> WorldGraph:
    """A `WorldGraph` pre-populated with the Pallet Town -> Pewter City map
    adjacency (exit tiles unknown, to be refined by `observe_exits`).

    Map ids are resolved by name against `MAP_NAMES_RAW`; a name not present in
    that table is skipped and recorded in ``graph.unresolved`` rather than guessed.
    """
    g = WorldGraph()
    for a_name, b_name in _KANTO_CORRIDOR:
        a, b = _NAME_TO_ID.get(a_name), _NAME_TO_ID.get(b_name)
        for name, mid in ((a_name, a), (b_name, b)):
            if mid is None and name not in g.unresolved:
                g.unresolved.append(name)
        if a is None or b is None:
            continue
        g.add_adjacency(a, b)
    return g
