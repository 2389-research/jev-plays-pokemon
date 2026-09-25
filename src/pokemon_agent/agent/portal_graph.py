"""PortalGraph — the navigation interface L1 (strategy) and L2 (routing) use to read the map.

The graph is ripped offline from ground-truth game data (see ``scripts/rip_portals.py``): nodes are
PORTALS (door/edge/warp tiles, each with a destination + a compass direction), and two portals on the
same map are walk-reachable iff they sit in the same collision component. Routing BFSes over portals,
so cross-map paths (e.g. crossing Viridian Forest to reach Pewter) fall out of the geometry.

Split of concerns:
  * L2 (per-step routing): ``component_at`` to locate the player, then ``next_portal`` for the portal
    to head toward, or ``render_view`` for the scoped natural-language map the L2 model reads and picks.
  * L1 (strategy): ``route`` / ``reachable_maps`` for "can I get there and roughly how"; the semantic
    "what/where/why" lives in Orrery, not here.

Coordinates are handles for the pathfinder; the model reasons over NAMES + compass direction, never
raw coordinates (that is the documented LLM failure mode).
"""
from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parents[1] / "games" / "pokemon_red" / "portal_graph.json"
_CARDINAL = {"north": "NORTH", "south": "SOUTH", "east": "EAST", "west": "WEST"}
_DELTA = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _friendly(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", name).replace("1 F", "1F")


class PortalGraph:
    """Read-only view over the ripped portal graph. Load once (cache) and share."""

    def __init__(self, maps: dict, portals: dict, version: str | None = None):
        self.maps = {int(k): v for k, v in maps.items()}
        self.portals = portals  # id -> portal dict
        self.version = version
        self._grids: dict[int, dict[tuple[int, int], int]] = {}
        # portals the agent OBSERVED it couldn't get through (a sprite in the way, pushed back) ->
        # {portal_id: why}. Set by the loop from memory; never story knowledge baked into the data.
        self.blocked: dict[str, str] = {}
        # Cut-tree crossings ("cut" portals) are routed only when the party can cut (set by the loop:
        # someone knows Cut and the Cascade Badge is held) — the graph never assumes an ability
        self.can_cut: bool = False
        self._by_map: dict[int, list[dict]] = {}
        for p in portals.values():
            self._by_map.setdefault(p["map"], []).append(p)

    def add_cut_trees(self, trees: dict) -> None:
        """Each tree that separates two static components becomes a pair of "cut" portals (one per
        direction) standing on the tree tile, approached from that side (scripts/gen_cut_trees.py)."""
        for mid, ts in trees.items():
            mid = int(mid)
            if mid not in self.maps:
                continue
            for t in ts:
                sides = {int(c): tuple(a) for c, a in (t.get("sides") or {}).items()}
                for c1, a1 in sides.items():
                    for c2 in sides:
                        if c2 == c1:
                            continue
                        pid = f"{self.maps[mid]['name'].lower()}:cut_{t['x']}_{t['y']}_{c1}_{c2}"
                        p = {"id": pid, "map": mid, "coord": [t["x"], t["y"]], "approach": list(a1), "kind": "cut",
                             "dest_map": mid, "dest_component": c2, "component": c1, "direction": None,
                             "label": f"{self.map_name(mid)} Cut tree at ({t['x']},{t['y']})"}
                        self.portals[pid] = p
                        self._by_map.setdefault(mid, []).append(p)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PortalGraph":
        d = json.loads(Path(path or _DEFAULT).read_text())
        pg = cls(d["maps"], d["portals"], d.get("version"))
        trees = Path(path or _DEFAULT).with_name("cut_trees.json")
        if trees.exists():
            pg.add_cut_trees(json.loads(trees.read_text()))
        return pg

    # --- names -----------------------------------------------------------
    def map_name(self, mid) -> str:
        m = self.maps.get(int(mid)) if mid is not None else None
        return _friendly(m["name"]) if m else f"Map {mid}"

    def portals_on(self, map_id: int) -> list[dict]:
        return self._by_map.get(int(map_id), [])

    def exits_from(self, map_id: int, comp: int, *, allow_blocked: bool = False) -> list[dict]:
        """Portals in this component that lead somewhere (a walk-reachable exit set). Portals observed
        to be blocked and elevators (dynamic destinations) are not routed unless ``allow_blocked``."""
        return [p for p in self.portals_on(map_id)
                if p["component"] == comp and p["dest_map"] is not None and p["kind"] != "elevator"
                and (p["kind"] != "cut" or self.can_cut) and not p.get("inaccessible")
                and (allow_blocked or p["id"] not in self.blocked)]

    # --- static routing over (map, component) nodes ----------------------
    def _comps_of(self, mid: int) -> set[int]:
        return {p["component"] for p in self.portals_on(mid)}

    def _dest_nodes(self, portal: dict) -> list[tuple[int, int]]:
        dp = self.portals.get(portal.get("dest_portal") or "")
        dm = portal["dest_map"]
        if dm is None:
            return []
        if portal.get("dest_component") is not None:   # edges / ledges land in a known component
            return [(dm, portal["dest_component"])]
        if dp is not None:
            return [(dm, dp["component"])]
        return [(dm, c) for c in self._comps_of(dm)]  # unknown landing component -> any

    def route(self, from_map: int, from_comp, to_map: int) -> list[dict] | None:
        """Ordered list of portals to traverse from (from_map, from_comp) to any component of
        ``to_map`` — the shortest portal path. ``from_comp`` may be a set: every component the player
        can walk to right now (see ``components_reachable``). ``None`` if unreachable, ``[]`` if there."""
        comps = sorted(from_comp) if isinstance(from_comp, (set, frozenset, list, tuple)) else [from_comp]
        if int(from_map) == int(to_map):
            return []
        starts = [(int(from_map), int(c)) for c in comps if c is not None]
        prev: dict[tuple[int, int], tuple[tuple[int, int], dict] | None] = {s: None for s in starts}
        q = deque(starts)
        goal = None
        while q:
            node = q.popleft()
            if node[0] == int(to_map):
                goal = node
                break
            mid, comp = node
            for p in self.exits_from(mid, comp):
                for t in self._dest_nodes(p):
                    if t not in prev:
                        prev[t] = (node, p)
                        q.append(t)
        if goal is None:
            return None
        chain = []
        node = goal
        while prev[node] is not None:
            parent, portal = prev[node]
            chain.append(portal)
            node = parent
        chain.reverse()
        return chain

    def next_portal(self, from_map: int, from_comp: int, to_map: int) -> dict | None:
        """The single portal to head toward next on the way to ``to_map`` (for L2 / the servo)."""
        r = self.route(from_map, from_comp, to_map)
        return r[0] if r else None

    def reachable_maps(self, from_map: int, from_comp: int) -> set[int]:
        """Every map reachable on foot + through portals from here (L1 sanity: is X reachable)."""
        start = (int(from_map), int(from_comp))
        seen = {start}
        q = deque([start])
        maps = {start[0]}
        while q:
            mid, comp = q.popleft()
            for p in self.exits_from(mid, comp):
                for t in self._dest_nodes(p):
                    if t not in seen:
                        seen.add(t); maps.add(t[0]); q.append(t)
        return maps

    # --- live: locate the player from the current collision map ----------
    def components_reachable(self, map_id: int, x: int, y: int, walkable: set[tuple[int, int]]) -> set[int]:
        """Every static component the player can walk to NOW: flood-fill the LIVE walkable set from
        (x, y) (elevation cuts respected) and collect the static component of each cell and portal it
        touches. The rip's components are fixed, but the map isn't — a cut tree joins two of them
        (runs/sleeves-vermilion: after Cut the gym door's component stayed "unreachable" and L1 recut the
        tree 4 times believing it had regrown)."""
        grid = self._grid(map_id)
        portal_comp: dict[tuple[int, int], int] = {}
        for p in self.portals_on(map_id):
            if p["kind"] in ("ledge", "cut"):
                continue
            portal_comp[tuple(p["coord"])] = p["component"]
            if p.get("approach"):
                portal_comp[tuple(p["approach"])] = p["component"]
            else:
                for dx, dy in _DELTA:
                    portal_comp.setdefault((p["coord"][0] + dx, p["coord"][1] + dy), p["component"])
        start = (int(x), int(y))
        found: set[int] = set()
        for c in (grid.get(start), portal_comp.get(start)):
            if c is not None:
                found.add(c)
        if start not in walkable:
            return found
        cuts = self.cut_edges(map_id)
        seen, q = {start}, deque([start])
        while q:
            c = q.popleft()
            for comp in (grid.get(c), portal_comp.get(c)):
                if comp is not None:
                    found.add(comp)
            cx, cy = c
            for dx, dy in _DELTA:
                nb = (cx + dx, cy + dy)
                if nb in walkable and nb not in seen and frozenset({c, nb}) not in cuts:
                    seen.add(nb)
                    q.append(nb)
        return found

    def component_at(self, map_id: int, x: int, y: int, walkable: set[tuple[int, int]]) -> int | None:
        """Which static component the player stands in, found by flood-filling the LIVE walkable set
        from (x, y) until it reaches a known portal tile (or a portal's approach tile). Uses live
        collision so it stays correct even if map state changed (e.g. a cut tree). The shipped static
        grid is consulted first: it encodes tile-pair (elevation) cuts a live flood fill can't see."""
        grid = self._grid(map_id)
        if (int(x), int(y)) in grid:
            return grid[(int(x), int(y))]
        portal_comp = {}
        for p in self.portals_on(map_id):
            if p["kind"] in ("ledge", "cut"):
                continue
            portal_comp[tuple(p["coord"])] = p["component"]
            if p.get("approach"):
                portal_comp[tuple(p["approach"])] = p["component"]
            else:
                for dx, dy in _DELTA:                   # doors sit on non-walkable tiles: index approaches
                    portal_comp.setdefault((p["coord"][0] + dx, p["coord"][1] + dy), p["component"])
        start = (int(x), int(y))
        if start not in walkable:
            return portal_comp.get(start)
        seen = {start}
        q = deque([start])
        while q:
            c = q.popleft()
            if c in portal_comp:
                return portal_comp[c]
            cx, cy = c
            for dx, dy in _DELTA:
                nb = (cx + dx, cy + dy)
                if nb in walkable and nb not in seen:
                    seen.add(nb); q.append(nb)
        return None

    def cut_edges(self, map_id: int) -> set[frozenset]:
        """Elevation (tile-pair) edges on this map: pairs of adjacent walkable cells you can't step
        between. Path-finders must not cross them (RAM walkability can't see them)."""
        enc = (self.maps.get(int(map_id)) or {}).get("cuts") or ""
        out: set[frozenset] = set()
        for tok in enc.split(";") if enc else []:
            x, y, d = tok.split(",")
            x, y = int(x), int(y)
            out.add(frozenset({(x, y), (x + 1, y) if d == "E" else (x, y + 1)}))
        return out

    def _grid(self, map_id: int) -> dict[tuple[int, int], int]:
        """Decode (once) the map's run-length static component grid ('.' = not walkable)."""
        mid = int(map_id)
        if mid not in self._grids:
            cells: dict[tuple[int, int], int] = {}
            enc = (self.maps.get(mid) or {}).get("grid") or ""
            for y, row in enumerate(enc.split("|") if enc else []):
                x = 0
                for run in row.split(","):
                    v, n = run.split("*")
                    n = int(n)
                    if v != ".":
                        for i in range(n):
                            cells[(x + i, y)] = int(v)
                    x += n
            self._grids[mid] = cells
        return self._grids[mid]

    # --- the scoped natural-language view L2's model reads ---------------
    def render_view(self, map_id: int, comp: int, goal_map: int) -> str:
        """Directed connectivity as names + compass direction (no coordinates). This is the exact
        representation validated in scripts/eval_portal_view.py (both models 5/5)."""
        lines = [f"YOU ARE: {self.map_name(map_id)}."]
        lines.append("\nEXITS FROM HERE (the compass direction you travel to take each):")
        seen_exit = {}
        for p in self.exits_from(map_id, comp):
            if p["kind"] == "ledge":
                continue
            seen_exit.setdefault((p.get("direction"), p["dest_map"]), p)
        for (d, dm), p in seen_exit.items():
            way = f"go {_CARDINAL[d]}" if d in _CARDINAL else "go inside"
            lines.append(f"  - {way}  ->  {self.map_name(dm)}")
        lines.append("\nHOW AREAS CONNECT (direction of travel between them):")
        seen = set()
        frontier = {int(map_id)}
        for _ in range(3):
            nxt = set()
            for m in sorted(frontier):
                if m in seen:
                    continue
                links = {}
                for p in self.portals_on(m):
                    if p["kind"] == "ledge" or p["id"] in self.blocked:
                        continue
                    if p["dest_map"] is not None and p.get("direction") in _CARDINAL:
                        links[(p["direction"], p["dest_map"])] = None
                if links:
                    lines.append(f"  {self.map_name(m)}:  " +
                                 ";  ".join(f"{_CARDINAL[d]} -> {self.map_name(dm)}" for (d, dm) in links))
                    seen.add(m)
                nxt.update(dm for (_, dm) in links)
            frontier = nxt - seen
        lines.append(f"\nGOAL: reach {self.map_name(goal_map)}.")
        return "\n".join(lines)
