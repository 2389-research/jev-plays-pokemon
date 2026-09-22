#!/usr/bin/env python3
"""Static portal-graph ripper for Pokémon Red (Pallet -> Pewter / Brock corridor).

ALL map data is parsed from the pokered disassembly (``/tmp/pokered`` by default):
map constants, per-map headers (tileset + edge connections), per-map warp events,
the raw ``.blk`` block grids, the tileset blocksets (``.bst``) and the per-tileset
collision tile-id lists. NO emulator traversal and NO save-state reading is used to
BUILD the graph. Save states are used ONLY to VALIDATE that our static collision
decode reproduces the RAM reader's walkable set (``read_collision_map``).

Outputs:
  * a portal graph (warp portals + edge portals, with connected-component tags)
  * ``scripts/out/portals_corridor.json`` — the corridor subgraph, for inspection
  * two hard checks printed as PASS/FAIL:
      1. COLLISION VALIDATION: static walkable set == RAM walkable set, per map.
      2. ROUTING PROOF: route(Forest 51 -> Pewter 2) goes via the NORTH gate (47),
         and Route 2's SOUTH component cannot reach Pewter without the forest detour.

The pure logic (component splitting + routing) is also covered by
``tests/unit/test_portal_rip.py``.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import deque
from pathlib import Path
from statistics import median_low

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
POKERED = Path(os.environ.get("POKERED", "/tmp/pokered"))
REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "scripts" / "out"
ROM = REPO / "roms" / "pokemon_red.gb"

# Corridor map ids (Pallet -> Pewter / Brock)
CORRIDOR_IDS = {0, 1, 2, 12, 13, 47, 50, 51}
# Maps we have (or want to) validate collision against RAM. id -> is required.
VALIDATE_IDS = [0, 13, 51, 1, 50]  # PALLET, ROUTE_2, FOREST, VIRIDIAN_CITY, SOUTH_GATE

LAST_MAP = 0xFF


# --------------------------------------------------------------------------- #
# 1. pokered static parsers
# --------------------------------------------------------------------------- #
def parse_map_constants() -> dict[str, tuple[int, int, int]]:
    """``constants/map_constants.asm`` -> {CONST_NAME: (map_id, W_blocks, H_blocks)}.

    The map id is the running index under ``const_def`` and is confirmed by the
    trailing ``; $XX`` hex comment.
    """
    out: dict[str, tuple[int, int, int]] = {}
    text = (POKERED / "constants" / "map_constants.asm").read_text()
    rx = re.compile(
        r"map_const\s+(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*;\s*\$([0-9A-Fa-f]+)"
    )
    for m in rx.finditer(text):
        name, w, h, hexid = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        out[name] = (int(hexid, 16), w, h)
    return out


def parse_tilesets() -> dict[str, dict]:
    """Map each tileset CONST (e.g. ``OVERWORLD``, ``FOREST_GATE``) to its
    blockset file and its collision tile-id set.

    Three pokered files are joined by tileset ORDER / CamelCase name:
      * ``constants/tileset_constants.asm`` — ordered CONST names (id = order).
      * ``data/tilesets/tileset_headers.asm`` — same order, CamelCase names
        (``tileset Overworld`` -> ``Overworld_Block`` / ``Overworld_Coll``).
      * ``gfx/tilesets.asm`` — ``<Camel>_Block:: INCBIN gfx/blocksets/<file>.bst``
        (labels may stack: several share one INCBIN).
      * ``data/tilesets/collision_tile_ids.asm`` — ``<Camel>_Coll::`` -> coll tiles
        (labels may be shared, e.g. ``Mart_Coll:: Pokecenter_Coll::``).
    """
    # ordered CONST names
    const_names: list[str] = []
    for line in (POKERED / "constants" / "tileset_constants.asm").read_text().splitlines():
        m = re.match(r"\s*const\s+(\w+)", line)
        if m:
            const_names.append(m.group(1))

    # ordered CamelCase names from the tileset header table
    camel_names: list[str] = []
    for line in (POKERED / "data" / "tilesets" / "tileset_headers.asm").read_text().splitlines():
        m = re.match(r"\s*tileset\s+(\w+)\s*,", line)
        if m:
            camel_names.append(m.group(1))
    assert len(const_names) == len(camel_names), (len(const_names), len(camel_names))

    # CamelBlock label -> bst file (stacked labels share the following INCBIN)
    block_to_bst: dict[str, str] = {}
    pending: list[str] = []
    for line in (POKERED / "gfx" / "tilesets.asm").read_text().splitlines():
        lbl = re.match(r"\s*(\w+)_Block::", line)
        if lbl:
            pending.append(lbl.group(1))
        inc = re.search(r'INCBIN\s+"gfx/blocksets/(\w+)\.bst"', line)
        if inc:
            for camel in pending:
                block_to_bst[camel] = inc.group(1)
            pending = []

    # CamelColl label -> coll set (stacked labels share the following coll_tiles)
    coll_to_set: dict[str, set[int]] = {}
    pending = []
    for line in (POKERED / "data" / "tilesets" / "collision_tile_ids.asm").read_text().splitlines():
        for lbl in re.finditer(r"(\w+)_Coll::", line):
            pending.append(lbl.group(1))
        ct = re.match(r"\s*coll_tiles\s+(.*)", line)
        if ct and pending:
            ids: set[int] = set()
            for tok in ct.group(1).split(";")[0].split(","):
                tok = tok.strip()
                m = re.match(r"\$([0-9A-Fa-f]+)", tok)
                if m:
                    ids.add(int(m.group(1), 16))
            if ids:  # ignore the "unused" empty coll_tiles line
                for camel in pending:
                    coll_to_set[camel] = ids
                pending = []

    out: dict[str, dict] = {}
    for const, camel in zip(const_names, camel_names):
        out[const] = {
            "camel": camel,
            "bst": block_to_bst[camel],
            "coll": coll_to_set[camel],
        }
    return out


def parse_header(map_name: str) -> dict:
    """``data/maps/headers/<Name>.asm`` -> tileset const + edge connections."""
    text = (POKERED / "data" / "maps" / "headers" / f"{map_name}.asm").read_text()
    mh = re.search(r"map_header\s+(\w+)\s*,\s*(\w+)\s*,\s*(\w+)", text)
    tileset = mh.group(3)
    conns = []
    for m in re.finditer(
        r"connection\s+(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*(-?\d+)", text
    ):
        conns.append({
            "dir": m.group(1),
            "target_name": m.group(2),
            "target_const": m.group(3),
            "offset": int(m.group(4)),
        })
    return {"tileset": tileset, "connections": conns}


def parse_warps(map_name: str) -> list[dict]:
    """``data/maps/objects/<Name>.asm`` -> ordered warp events (1-indexed slots).

    Only the block between ``def_warp_events`` and the next ``def_*`` is parsed, so
    the trailing ``warp_to`` back-pointers and object/bg events are ignored.
    """
    text = (POKERED / "data" / "maps" / "objects" / f"{map_name}.asm").read_text()
    lines = text.splitlines()
    warps: list[dict] = []
    in_block = False
    for line in lines:
        s = line.strip()
        if s.startswith("def_warp_events"):
            in_block = True
            continue
        if in_block and s.startswith("def_") and not s.startswith("def_warp_events"):
            break
        if in_block:
            m = re.match(r"warp_event\s+(\d+)\s*,\s*(\d+)\s*,\s*(\w+)\s*,\s*(\w+)", s)
            if m:
                warps.append({
                    "slot": len(warps) + 1,  # 1-indexed, matches dest_warp_id
                    "x": int(m.group(1)),
                    "y": int(m.group(2)),
                    "dest_const": m.group(3),
                    "dest_warp": m.group(4),
                })
    return warps


def load_blk(map_name: str, w: int, h: int) -> list[int]:
    data = (POKERED / "maps" / f"{map_name}.blk").read_bytes()
    assert len(data) == w * h, f"{map_name}.blk is {len(data)} bytes, expected {w*h}"
    return list(data)


def load_bst(bst_file: str) -> list[list[int]]:
    """``gfx/blocksets/<file>.bst`` -> list of 16-tile blocks."""
    data = (POKERED / "gfx" / "blocksets" / f"{bst_file}.bst").read_bytes()
    assert len(data) % 16 == 0
    return [list(data[i * 16:(i + 1) * 16]) for i in range(len(data) // 16)]


# --------------------------------------------------------------------------- #
# 2. Collision decode  (mirrors map_reader.read_collision_map exactly)
# --------------------------------------------------------------------------- #
def decode_walkable(blk: list[int], blockset: list[list[int]], collset: set[int],
                    w_blocks: int, h_blocks: int) -> set[tuple[int, int]]:
    """Static equivalent of ``read_collision_map(emu)["walkable"]``.

    For each block, its 16 tiles are ``blockset[block_id]``. The collision-relevant
    tile of player-cell (cr, cc) in {0,1} is tile index ``(cr*2+1)*4 + (cc*2)``, at
    map-local cell ``(bx*2+cc, by*2+cr)``; the cell is walkable iff that tile id is in
    the tileset's collision set. (Exactly the decode in ``map_reader.py``.)
    """
    walkable: set[tuple[int, int]] = set()
    for by in range(h_blocks):
        for bx in range(w_blocks):
            bid = blk[by * w_blocks + bx]
            tiles = blockset[bid]
            for cr in (0, 1):
                for cc in (0, 1):
                    t = tiles[(cr * 2 + 1) * 4 + (cc * 2)]
                    if t in collset:
                        walkable.add((bx * 2 + cc, by * 2 + cr))
    return walkable


# --------------------------------------------------------------------------- #
# 3. Connected components (4-connectivity over the walkable set)  [pure]
# --------------------------------------------------------------------------- #
def connected_components(walkable: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Label each walkable cell with a component id. Deterministic: components are
    numbered in ascending (y, x) order of their first-seen cell."""
    comp: dict[tuple[int, int], int] = {}
    next_id = 0
    for start in sorted(walkable, key=lambda c: (c[1], c[0])):
        if start in comp:
            continue
        cid = next_id
        next_id += 1
        stack = [start]
        comp[start] = cid
        while stack:
            x, y = stack.pop()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in walkable and (nx, ny) not in comp:
                    comp[(nx, ny)] = cid
                    stack.append((nx, ny))
    return comp


# --------------------------------------------------------------------------- #
# 4. Portal graph builder
# --------------------------------------------------------------------------- #
class MapData:
    def __init__(self, name: str, const: str, map_id: int, w_blocks: int, h_blocks: int,
                 tileset: str, connections: list[dict], warps: list[dict],
                 walkable: set, comp: dict):
        self.name = name
        self.const = const
        self.id = map_id
        self.w_blocks = w_blocks
        self.h_blocks = h_blocks
        self.width = w_blocks * 2   # tile/cell dims
        self.height = h_blocks * 2
        self.tileset = tileset
        self.connections = connections
        self.warps = warps
        self.walkable = walkable
        self.comp = comp

    def cell_component(self, x: int, y: int) -> int | None:
        """Component of (x,y); if the exact cell is non-walkable (e.g. a doorway on
        the wall) fall back to the smallest-id adjacent walkable component (the
        'approach' tile the player actually stands on)."""
        if (x, y) in self.comp:
            return self.comp[(x, y)]
        neigh = [self.comp[c] for c in
                 ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if c in self.comp]
        return min(neigh) if neigh else None


def load_map(map_name: str, consts: dict, tilesets: dict) -> MapData:
    header = parse_header(map_name)
    tileset_const = header["tileset"]
    # find this map's const by parsing the map_header line
    mh = re.search(
        r"map_header\s+\w+\s*,\s*(\w+)\s*,",
        (POKERED / "data" / "maps" / "headers" / f"{map_name}.asm").read_text(),
    )
    const = mh.group(1)
    map_id, w_blocks, h_blocks = consts[const]
    blk = load_blk(map_name, w_blocks, h_blocks)
    ts = tilesets[tileset_const]
    blockset = load_bst(ts["bst"])
    walkable = decode_walkable(blk, blockset, ts["coll"], w_blocks, h_blocks)
    comp = connected_components(walkable)
    warps = parse_warps(map_name)
    return MapData(map_name, const, map_id, w_blocks, h_blocks, tileset_const,
                   header["connections"], warps, walkable, comp)


# Name<->const<->id helpers built once from the header set.
def build_name_index() -> tuple[dict[str, str], dict[str, str]]:
    """Return (const->map_name, map_name->const) for every header file."""
    const_to_name: dict[str, str] = {}
    name_to_const: dict[str, str] = {}
    hdr_dir = POKERED / "data" / "maps" / "headers"
    for f in hdr_dir.glob("*.asm"):
        mh = re.search(r"map_header\s+(\w+)\s*,\s*(\w+)\s*,", f.read_text())
        if mh:
            const_to_name[mh.group(2)] = mh.group(1)
            name_to_const[mh.group(1)] = mh.group(2)
    return const_to_name, name_to_const


_OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}


def build_portal_graph(map_names: list[str]) -> dict:
    """Rip a portal graph for the given maps. Returns
    ``{"maps": {id: MapData}, "portals": {id: portal_dict}}``."""
    consts = parse_map_constants()
    tilesets = parse_tilesets()
    const_to_name, name_to_const = build_name_index()

    maps: dict[int, MapData] = {}
    for nm in map_names:
        md = load_map(nm, consts, tilesets)
        maps[md.id] = md

    id_to_const = {consts[c][0]: c for c in consts}

    # ---- global warp index: (dest_const, dest_slot) -> list of source (map_id, slot)
    incoming: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for md in maps.values():
        for wp in md.warps:
            try:
                dslot = int(wp["dest_warp"])
            except ValueError:
                dslot = None
            if wp["dest_const"] != "LAST_MAP" and dslot is not None:
                incoming.setdefault((wp["dest_const"], dslot), []).append((md.id, wp["slot"]))

    portals: dict[str, dict] = {}

    def pid_warp(md: MapData, slot: int) -> str:
        return f"{md.name.lower()}:warp{slot}"

    def pid_edge(md: MapData, direction: str, comp_id: int) -> str:
        return f"{md.name.lower()}:edge_{direction}_c{comp_id}"

    # ---- warp portals ------------------------------------------------------
    for md in maps.values():
        for wp in md.warps:
            x, y, slot = wp["x"], wp["y"], wp["slot"]
            comp_id = md.cell_component(x, y)
            dest_const = wp["dest_const"]
            note = None
            if dest_const == "LAST_MAP":
                dest_id, dest_slot, note = resolve_last_map(md, wp, incoming, consts)
            else:
                dest_id = consts[dest_const][0]
                try:
                    dest_slot = int(wp["dest_warp"])
                except ValueError:
                    dest_slot = None
            dest_name = const_to_name.get(id_to_const.get(dest_id, ""), None) if dest_id is not None else None
            dest_portal = None
            if dest_name is not None and dest_slot is not None:
                dest_portal = f"{dest_name.lower()}:warp{dest_slot}"
            portals[pid_warp(md, slot)] = {
                "id": pid_warp(md, slot),
                "map": md.id,
                "coord": [x, y],
                "kind": "warp",
                "dest_map": dest_id,
                "dest_warp": dest_slot,
                "dest_portal": dest_portal,
                "component": comp_id,
                "label": f"{md.name} warp {slot} -> {dest_const}",
                "note": note,
            }

    # ---- edge portals (one per walkable component on the border) -----------
    for md in maps.values():
        for conn in md.connections:
            direction = conn["dir"]
            border_cells = border_walkable(md, direction)
            by_comp: dict[int, list[tuple[int, int]]] = {}
            for c in border_cells:
                by_comp.setdefault(md.comp[c], []).append(c)
            dest_id = consts[conn["target_const"]][0]
            for comp_id, cells in sorted(by_comp.items()):
                rep = representative(cells, direction)
                pid = pid_edge(md, direction, comp_id)
                portals[pid] = {
                    "id": pid,
                    "map": md.id,
                    "coord": [rep[0], rep[1]],
                    "kind": "edge",
                    "dest_map": dest_id,
                    "dest_warp": None,
                    "dest_portal": None,   # paired below
                    "component": comp_id,
                    "label": f"{md.name} {direction} edge -> {conn['target_name']}",
                    "direction": direction,
                }

    # ---- pair edge portals to their reverse edge on the neighbour ----------
    for p in list(portals.values()):
        if p["kind"] != "edge":
            continue
        a_id, b_id = p["map"], p["dest_map"]
        opp = _OPPOSITE[p["direction"]]
        cands = [q for q in portals.values()
                 if q["kind"] == "edge" and q["map"] == b_id
                 and q["dest_map"] == a_id and q.get("direction") == opp]
        if len(cands) == 1:
            p["dest_portal"] = cands[0]["id"]
        elif len(cands) > 1:
            # multi-component border: pick the reverse edge nearest along the seam
            axis = 0 if p["direction"] in ("north", "south") else 1
            p["dest_portal"] = min(
                cands, key=lambda q: abs(q["coord"][axis] - p["coord"][axis])
            )["id"]

    return {"maps": maps, "portals": portals}


def border_walkable(md: MapData, direction: str) -> list[tuple[int, int]]:
    if direction == "north":
        return [c for c in md.walkable if c[1] == 0]
    if direction == "south":
        return [c for c in md.walkable if c[1] == md.height - 1]
    if direction == "west":
        return [c for c in md.walkable if c[0] == 0]
    if direction == "east":
        return [c for c in md.walkable if c[0] == md.width - 1]
    return []


def representative(cells: list[tuple[int, int]], direction: str) -> tuple[int, int]:
    """Deterministic representative walkable tile of a border-segment: the median
    along the border axis."""
    if direction in ("north", "south"):
        xs = sorted(c[0] for c in cells)
        mx = median_low(xs)
        return (mx, cells[0][1])
    ys = sorted(c[1] for c in cells)
    my = median_low(ys)
    return (cells[0][0], my)


def resolve_last_map(md: MapData, wp: dict, incoming: dict, consts: dict):
    """Resolve a ``LAST_MAP`` (0xFF, 'return to the map you came from') warp to a
    concrete overworld/route map, statically.

    Rule: a two-sided gate G exposes its return doors as LAST_MAP warps. A neighbour
    map M reaches G through an explicit ``warp_event ... G, k`` that targets warp
    SLOT k inside G. That slot k is one of G's own warps; if slot k is itself the
    LAST_MAP warp (co-located return door), then stepping on it returns to M. So a
    LAST_MAP warp at slot s resolves to the neighbour whose incoming warp targets s.
    (Geometry check: y==0 doors are the north side, y==maxY the south side; a
    neighbour that enters at the north door returns to the map physically north.)
    """
    slot = wp["slot"]
    try:
        dest_slot = int(wp["dest_warp"])
    except ValueError:
        dest_slot = None

    # Primary: a neighbour whose incoming warp targets THIS exact slot.
    srcs = incoming.get((md.const, slot), [])
    if len(srcs) == 1:
        return srcs[0][0], dest_slot, (
            f"LAST_MAP -> map {srcs[0][0]} (incoming warp targets slot {slot})")

    # Fallback: paired return doors (x differs, same side) share one neighbour.
    # Resolve by SIDE geometry — the neighbour whose incoming warp targets ANY slot
    # on the same side (same y-row) as this door. (y==0 = north side returns to the
    # map physically north; y==maxY = south side returns to the map to the south.)
    slot_y = {w["slot"]: w["y"] for w in md.warps}
    same_side = [
        src for (dest_const, tslot), lst in incoming.items()
        if dest_const == md.const and slot_y.get(tslot) == wp["y"]
        for src in lst
    ]
    side = "north" if wp["y"] <= md.height // 2 else "south"
    if len(srcs) > 1:
        return srcs[0][0], dest_slot, (
            f"LAST_MAP ambiguous {srcs}; side={side}, took {srcs[0][0]}")
    if same_side:
        src_id = same_side[0][0]
        return src_id, dest_slot, (
            f"LAST_MAP -> map {src_id} (paired {side}-side door; twin slot entered)")
    return None, dest_slot, "LAST_MAP unresolved (no incoming warp on this side)"


# --------------------------------------------------------------------------- #
# 5. Routing  (BFS over portals)  [pure]
# --------------------------------------------------------------------------- #
def route(portals: dict, from_map: int, from_component: int, to_map: int) -> list[str] | None:
    """BFS over portals. Two portals on the same map are mutually reachable iff same
    component (walk_reachable). A portal connects to its paired ``dest_portal`` on the
    other map. Success = arriving on ``to_map``. Returns the ordered list of portal
    ids traversed, or None."""
    # start frontier: portals on from_map reachable on foot from from_component
    start = [p["id"] for p in portals.values()
             if p["map"] == from_map and p["component"] == from_component]
    if from_map == to_map:
        return []  # already there
    # BFS
    q: deque[tuple[str, list[str]]] = deque()
    seen: set[str] = set()
    for pid in start:
        q.append((pid, [pid]))
        seen.add(pid)
    while q:
        pid, path = q.popleft()
        p = portals[pid]
        dest = p.get("dest_portal")
        if p["dest_map"] == to_map:
            return path  # stepping through this portal lands on the target map
        if dest is None or dest not in portals:
            continue
        dp = portals[dest]
        # arrived on dest map, in dp's component -> can walk to same-component portals
        arrivals = [q2["id"] for q2 in portals.values()
                    if q2["map"] == dp["map"] and q2["component"] == dp["component"]]
        for a in arrivals:
            if a not in seen:
                seen.add(a)
                q.append((a, path + [a]))
    return None


# --------------------------------------------------------------------------- #
# 6. Validation checks
# --------------------------------------------------------------------------- #
def find_states() -> dict[int, Path]:
    """Newest ``runs/*/states/map<ID>_*.state`` per map id."""
    states: dict[int, tuple[float, Path]] = {}
    for p in (REPO / "runs").glob("*/states/map*_*.state"):
        m = re.match(r"map(\d+)_", p.name)
        if not m:
            continue
        mid = int(m.group(1))
        mt = p.stat().st_mtime
        if mid not in states or mt > states[mid][0]:
            states[mid] = (mt, p)
    return {k: v[1] for k, v in states.items()}


def validate_collision(graph: dict) -> dict[int, dict]:
    """For each validate-id we have a save state for, load it, decode RAM walkable,
    and diff against the static walkable. Returns per-map result dict."""
    sys.path.insert(0, str(REPO / "src"))
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.games.pokemon_red.map_reader import read_collision_map

    states = find_states()
    maps = graph["maps"]
    results: dict[int, dict] = {}
    for mid in VALIDATE_IDS:
        if mid not in states:
            results[mid] = {"status": "NO_STATE"}
            continue
        emu = PyBoyEmulator(str(ROM), window="null")
        emu.load_state(states[mid])
        emu.tick(4)
        ram = read_collision_map(emu)
        emu.stop() if hasattr(emu, "stop") else None
        if ram is None:
            results[mid] = {"status": "RAM_NONE"}
            continue
        ram_walk = ram["walkable"]
        static_walk = maps[mid].walkable if mid in maps else set()
        only_ram = ram_walk - static_walk
        only_static = static_walk - ram_walk
        results[mid] = {
            "status": "MATCH" if not only_ram and not only_static else "DIFF",
            "ram_map_id": ram["map_id"],
            "ram_cells": len(ram_walk),
            "static_cells": len(static_walk),
            "only_ram": sorted(only_ram)[:20],
            "only_static": sorted(only_static)[:20],
            "diff_count": len(only_ram) + len(only_static),
        }
    return results


# --------------------------------------------------------------------------- #
# 7. Main
# --------------------------------------------------------------------------- #
CORRIDOR_NAMES = [
    "PalletTown", "ViridianCity", "PewterCity", "Route1", "Route2",
    "ViridianForestNorthGate", "ViridianForestSouthGate", "ViridianForest",
]


def main() -> int:
    print("=" * 70)
    print("STATIC PORTAL-GRAPH RIP — Pallet -> Pewter (Brock) corridor")
    print("source:", POKERED)
    print("=" * 70)

    graph = build_portal_graph(CORRIDOR_NAMES)
    portals = graph["portals"]
    maps = graph["maps"]

    print(f"\nRipped {len(maps)} maps, {len(portals)} portals.")
    for mid in sorted(maps):
        md = maps[mid]
        ncomp = len(set(md.comp.values())) if md.comp else 0
        print(f"  map {mid:2d} {md.name:26s} {md.width}x{md.height} cells, "
              f"{len(md.walkable):4d} walkable, {ncomp} component(s), tileset {md.tileset}")

    # ---- emit corridor JSON ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "maps": {
            str(mid): {
                "id": mid, "name": md.name, "const": md.const,
                "width": md.width, "height": md.height, "tileset": md.tileset,
                "n_components": len(set(md.comp.values())) if md.comp else 0,
                "walkable_count": len(md.walkable),
            } for mid, md in maps.items()
        },
        "portals": {pid: {k: v for k, v in p.items()} for pid, p in portals.items()},
    }
    (OUT_DIR / "portals_corridor.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_DIR / 'portals_corridor.json'}")

    # ============================ CHECK 1 ============================ #
    print("\n" + "=" * 70)
    print("CHECK 1 — COLLISION VALIDATION (static decode == RAM read_collision_map)")
    print("=" * 70)
    all_pass = True
    try:
        vres = validate_collision(graph)
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR running validation: {e!r}")
        vres = {}
        all_pass = False
    for mid in VALIDATE_IDS:
        r = vres.get(mid, {"status": "MISSING"})
        name = maps[mid].name if mid in maps else "?"
        if r["status"] == "MATCH":
            print(f"  map {mid:2d} {name:26s} PASS  "
                  f"({r['static_cells']} cells, exact match)")
        elif r["status"] == "DIFF":
            all_pass = False
            print(f"  map {mid:2d} {name:26s} FAIL  "
                  f"static={r['static_cells']} ram={r['ram_cells']} "
                  f"diff={r['diff_count']}")
            print(f"        only_ram(≤20)={r['only_ram']}")
            print(f"        only_static(≤20)={r['only_static']}")
        else:
            all_pass = False
            print(f"  map {mid:2d} {name:26s} {r['status']}")
    print(f"\n  CHECK 1: {'PASS' if all_pass else 'FAIL'}")

    # ============================ CHECK 2 ============================ #
    print("\n" + "=" * 70)
    print("CHECK 2 — ROUTING PROOF (Forest -> Pewter via NORTH gate; Route2-south detours)")
    print("=" * 70)
    check2 = True

    # Forest is a single component; find it.
    forest = maps[51]
    forest_comp = next(iter(set(forest.comp.values())))
    fp = route(portals, 51, forest_comp, 2)
    print("\n  route(VIRIDIAN_FOREST 51 -> PEWTER_CITY 2):")
    if fp is None:
        print("    None  <-- FAIL")
        check2 = False
    else:
        for pid in fp:
            p = portals[pid]
            print(f"    {pid:42s} map {p['map']:2d} {p['kind']:4s} "
                  f"-> map {p['dest_map']}")
        uses_north = any("northgate" in pid for pid in fp)
        uses_south = any("southgate" in pid for pid in fp)
        print(f"    uses NORTH gate: {uses_north} ; uses SOUTH gate: {uses_south}")
        if not (uses_north and not uses_south):
            print("    <-- FAIL: expected via NORTH gate only")
            check2 = False

    # Route2 split: south component (contains south-gate warp (3,43)) vs north.
    route2 = maps[13]
    comp_at = lambda x, y: route2.cell_component(x, y)  # noqa: E731
    south_comp = comp_at(3, 43)
    north_comp = comp_at(3, 11)
    print(f"\n  Route2 components: warp(3,11)->NorthGate in comp {north_comp}; "
          f"warp(3,43)->SouthGate in comp {south_comp}")
    if south_comp == north_comp:
        print("    <-- FAIL: Route2 did not split into two components")
        check2 = False

    rp = route(portals, 13, south_comp, 2)
    print("\n  route(ROUTE_2 13 south-component -> PEWTER_CITY 2):")
    if rp is None:
        print("    None  <-- FAIL")
        check2 = False
    else:
        for pid in rp:
            print(f"    {pid}")
        via_forest = any(pid.startswith("viridianforest:") for pid in rp)
        print(f"    detours through the forest: {via_forest}")
        if not via_forest:
            print("    <-- FAIL: south Route2 reached Pewter without the forest detour")
            check2 = False

    print(f"\n  CHECK 2: {'PASS' if check2 else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"OVERALL: CHECK1={'PASS' if all_pass else 'FAIL'}  "
          f"CHECK2={'PASS' if check2 else 'FAIL'}")
    print("=" * 70)
    return 0 if (all_pass and check2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
