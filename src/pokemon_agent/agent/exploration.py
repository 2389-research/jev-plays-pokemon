"""Exploration: what on this map the agent hasn't tried yet, and a monitor for "not getting anywhere".

A player who is stuck goes and looks: enters the house they haven't entered, talks to the person they
haven't talked to, reads the sign. ``unexplored`` lists exactly those things for the current map
(doors / edges leading to maps never visited, people not talked to, objects not used) — information,
not a solution; L1 decides whether to explore and may name a preference. ``pick_target`` chooses the
next one for the deterministic ``explore`` executor.

``StallMonitor`` is the deterministic half of the critic: it tracks progress (new maps, new tiles
walked, new things heard, items / badges / levels, goals met, steps completed) and reports how long
it has been since any. Moving isn't progress — bouncing between two maps visits no new tiles.
"""
from __future__ import annotations

import json
import re

from ..games.pokemon_red.map_objects import objects_on


def unexplored(pg, map_id: int, comp: int | None, *, visited_maps: set[int], npcs: list[dict],
               used_tiles: set[tuple[int, int, int]], skip: set[str] | None = None) -> list[dict]:
    """Things on ``map_id`` not yet explored, each {"kind", "label", "x", "y", ...}, limited to the
    player's walkable component ``comp`` when known (the rest can't be reached on foot from here)."""
    skip = skip or set()
    out: list[dict] = []
    comps = None if comp is None else (set(comp) if isinstance(comp, (set, frozenset, list, tuple)) else {comp})
    grid = pg._grid(map_id) if (pg is not None and comps and map_id in pg.maps) else None

    def reachable(x: int, y: int) -> bool:
        """Standing next to it (or on it) is inside the player's walkable area."""
        if grid is None:
            return True
        return any(grid.get(c) in comps for c in ((x, y), (x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)))
    if pg is not None and map_id in pg.maps:
        for p in pg.portals_on(map_id):
            if p["kind"] not in ("warp", "edge") or p["dest_map"] is None or p["dest_map"] in visited_maps:
                continue
            if comps and p["component"] not in comps:
                continue
            key = f"portal:{p['id']}"
            if key in skip:
                continue
            what = "door" if p["kind"] == "warp" else f"{p.get('direction')} edge"
            out.append({"kind": "portal", "key": key, "portal": p["id"], "x": p["coord"][0], "y": p["coord"][1],
                        "label": f"{what} to {pg.map_name(p['dest_map'])} (never visited) at "
                                 f"({p['coord'][0]},{p['coord'][1]})", "edge_dir": p.get("direction")
                        if p["kind"] == "edge" else None})
    for n in npcs:
        if n.get("kind") == "item" or n.get("talked_to") or "x" not in n or not reachable(int(n["x"]), int(n["y"])):
            continue
        key = f"npc:{map_id}:{n.get('slot', (n['x'], n['y']))}"
        if key in skip:
            continue
        out.append({"kind": "npc", "key": key, "x": int(n["x"]), "y": int(n["y"]), "sprite": n.get("sprite"),
                    "label": f"{n.get('sprite')} at ({n['x']},{n['y']}) (not talked to)"})
    for o in objects_on(map_id):
        key = f"obj:{map_id}:{o['x']}:{o['y']}"
        if (map_id, o["x"], o["y"]) in used_tiles or key in skip or not reachable(o["x"], o["y"]):
            continue
        out.append({"kind": "object", "key": key, "x": o["x"], "y": o["y"], "name": o["name"], "face": o.get("face"),
                    "label": f"{o['name']} at ({o['x']},{o['y']}) (not checked)"})
    return out


def pick_target(cands: list[dict], player_xy: tuple[int, int], prefer: str | None = None,
                choose=None) -> dict | None:
    """Which unexplored thing to try next. L1 is asked to copy an UNEXPLORED_HERE entry (or its
    coordinates) as its preference, so this is normally an exact lookup: 1) coordinates in ``prefer``
    (a person who wandered: within 2 tiles), 2) a candidate whose label contains ``prefer`` (or vice
    versa), 3) ``choose(prefer, labels) -> index | None`` — a model reading free text ("the house by the
    Rocket") against the candidates; then the nearest (new places before people and objects)."""
    if not cands:
        return None
    rank = {"portal": 0, "npc": 1, "object": 2}

    def nearest(pool):
        return min(pool, key=lambda c: (abs(c["x"] - player_xy[0]) + abs(c["y"] - player_xy[1]), rank[c["kind"]]))
    if prefer:
        m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", prefer)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            exact = [c for c in cands if (c["x"], c["y"]) == (x, y)]
            near = [c for c in cands if c["kind"] == "npc" and abs(c["x"] - x) + abs(c["y"] - y) <= 2]
            if exact or near:
                return (exact or near)[0]
        low = prefer.strip().lower()
        sub = [c for c in cands if low in c["label"].lower() or c["label"].lower() in low]
        if sub:
            return nearest(sub)
        if choose is not None:
            try:
                idx = choose(prefer, [c["label"] for c in cands])
            except Exception:
                idx = None
            if isinstance(idx, int) and 0 <= idx < len(cands):
                return cands[idx]
    return nearest(cands)


CHOOSE_SYSTEM = """An agent playing Pokémon Red wants to explore. PREFERENCE says (in its own words) what
it wants to try first; CANDIDATES are the unexplored things actually on this map. Pick the candidate
that best matches the preference, or null if none does.
Return ONLY JSON: {"index": <0-based index or null>}"""


def llm_chooser(provider):
    """``choose(prefer, labels) -> index | None`` backed by a (fast) chat provider."""
    if provider is None:
        return None
    from ..providers.parsing import strip_fences

    def choose(prefer: str, labels: list[str]):
        raw, _lat, _usage = provider.chat_json(CHOOSE_SYSTEM, {"preference": prefer,
                                                              "candidates": list(enumerate(labels))})
        idx = json.loads(strip_fences(raw)).get("index")
        return int(idx) if idx is not None else None
    return choose


class StallMonitor:
    """Steps since the agent last made ANY progress."""

    def __init__(self, threshold: int = 150):
        self.threshold = threshold
        self.last_progress = 0
        self._sig = None
        self._tiles: set[tuple[int, int, int]] = set()

    def observe(self, step: int, *, pos: tuple[int, int, int] | None, signature: tuple) -> bool:
        """``signature`` = everything that counts as progress other than new tiles (maps visited,
        things heard, items, badges, levels, goals met, steps done). Returns True on progress."""
        progressed = False
        if pos is not None and pos not in self._tiles:
            if len(self._tiles) < 200_000:
                self._tiles.add(pos)
            progressed = True
        if signature != self._sig:
            progressed = progressed or self._sig is not None
            self._sig = signature
        if progressed:
            self.last_progress = step
        return progressed

    def stalled_for(self, step: int) -> int:
        return max(0, step - self.last_progress)

    def stalled(self, step: int) -> bool:
        return self.stalled_for(step) >= self.threshold
