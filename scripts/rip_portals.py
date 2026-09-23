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


LOAD_NOTES: dict[str, str] = {}


def load_blk(map_name: str, w: int, h: int) -> list[int]:
    data = list((POKERED / "maps" / f"{map_name}.blk").read_bytes())
    if len(data) < w * h:      # K4: e.g. UndergroundPathNorthSouth ships 92 of 96 bytes
        LOAD_NOTES[map_name] = f"blk short by {w * h - len(data)} bytes (padded)"
        data += [data[-1] if data else 0] * (w * h - len(data))
    assert len(data) == w * h, f"{map_name}.blk is {len(data)} bytes, expected {w*h}"
    return data


def load_bst(bst_file: str) -> list[list[int]]:
    """``gfx/blocksets/<file>.bst`` -> list of 16-tile blocks."""
    data = (POKERED / "gfx" / "blocksets" / f"{bst_file}.bst").read_bytes()
    assert len(data) % 16 == 0
    return [list(data[i * 16:(i + 1) * 16]) for i in range(len(data) // 16)]


# --------------------------------------------------------------------------- #
# 2. Collision decode  (mirrors map_reader.read_collision_map exactly)
# --------------------------------------------------------------------------- #
def decode_tiles(blk: list[int], blockset: list[list[int]], w_blocks: int, h_blocks: int
                 ) -> dict[tuple[int, int], int]:
    """The collision-relevant tile id of EVERY cell (the tile the game tests: bottom-left of the
    cell's 2x2 quarter) — needed for ledges and tile-pair (elevation) collisions."""
    tiles: dict[tuple[int, int], int] = {}
    for by in range(h_blocks):
        for bx in range(w_blocks):
            t16 = blockset[blk[by * w_blocks + bx]]
            for cr in (0, 1):
                for cc in (0, 1):
                    tiles[(bx * 2 + cc, by * 2 + cr)] = t16[(cr * 2 + 1) * 4 + (cc * 2)]
    return tiles


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
def connected_components(walkable: set[tuple[int, int]], blocked=None) -> dict[tuple[int, int], int]:
    """Label each walkable cell with a component id. Deterministic: components are
    numbered in ascending (y, x) order of their first-seen cell. ``blocked(a, b)`` (optional) forbids
    a step between two walkable cells (tile-pair / elevation collisions, K7)."""
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
                    if blocked is not None and blocked((x, y), (nx, ny)):
                        continue
                    comp[(nx, ny)] = cid
                    stack.append((nx, ny))
    return comp


def parse_pair_collisions() -> dict[str, set[frozenset]]:
    """``data/tilesets/pair_collision_tile_ids.asm`` (land table) -> {TILESET: {frozenset(t1, t2)}}:
    the player may not step between those two tiles (either direction) — cave/forest elevation."""
    out: dict[str, set[frozenset]] = {}
    text = (POKERED / "data" / "tilesets" / "pair_collision_tile_ids.asm").read_text()
    land = text.split("TilePairCollisionsWater")[0]
    for m in re.finditer(r"db\s+(\w+)\s*,\s*\$([0-9A-Fa-f]+)\s*,\s*\$([0-9A-Fa-f]+)", land):
        out.setdefault(m.group(1), set()).add(frozenset((int(m.group(2), 16), int(m.group(3), 16))))
    return out


_LEDGE_DIR = {"SPRITE_FACING_DOWN": "south", "SPRITE_FACING_LEFT": "west", "SPRITE_FACING_RIGHT": "east",
              "SPRITE_FACING_UP": "north"}


def parse_ledges() -> list[tuple[str, int, int]]:
    """``data/tilesets/ledge_tiles.asm`` -> [(direction, standing tile, ledge tile)] (OVERWORLD only)."""
    out = []
    text = (POKERED / "data" / "tilesets" / "ledge_tiles.asm").read_text()
    for m in re.finditer(r"db\s+(SPRITE_FACING_\w+)\s*,\s*\$([0-9A-Fa-f]+)\s*,\s*\$([0-9A-Fa-f]+)", text):
        out.append((_LEDGE_DIR[m.group(1)], int(m.group(2), 16), int(m.group(3), 16)))
    return out


# --------------------------------------------------------------------------- #
# 4. Portal graph builder
# --------------------------------------------------------------------------- #
class MapData:
    def __init__(self, name: str, const: str, map_id: int, w_blocks: int, h_blocks: int,
                 tileset: str, connections: list[dict], warps: list[dict],
                 walkable: set, comp: dict, tiles: dict | None = None):
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
        self.tiles = tiles or {}

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
    tiles = decode_tiles(blk, blockset, w_blocks, h_blocks)
    pairs = parse_pair_collisions().get(tileset_const, set())
    blocked = (lambda a, b: frozenset((tiles[a], tiles[b])) in pairs) if pairs else None
    comp = connected_components(walkable, blocked)
    warps = parse_warps(map_name)
    md = MapData(map_name, const, map_id, w_blocks, h_blocks, tileset_const,
                 header["connections"], warps, walkable, comp, tiles)
    # K7: the elevation edges themselves (two walkable cells the game won't let you step between) —
    # the runtime path-finders must not plan across them (RAM walkability can't see them)
    md.cuts = sorted((c[0], c[1], d) for c in walkable for d, n in (("E", (c[0] + 1, c[1])), ("S", (c[0], c[1] + 1)))
                     if n in walkable and blocked is not None and blocked(c, n))
    return md


# Name<->const<->id helpers built once from the header set.
def build_name_index() -> tuple[dict[str, str], dict[str, str]]:
    """Return (const->map_name, map_name->const) for every header file."""
    const_to_name: dict[str, str] = {}
    name_to_const: dict[str, str] = {}
    hdr_dir = POKERED / "data" / "maps" / "headers"
    for f in sorted(hdr_dir.glob("*.asm")):
        if f.stem.endswith("Copy"):          # K4: duplicate headers re-declare a real map's const
            continue
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

    def warp_direction(md: MapData, x: int, y: int) -> str:
        """Compass bearing of a warp EXTRACTED from its tile position on the map (top edge -> north
        exit, bottom -> south, etc.). 'interior' = a door not on an edge (e.g. a mid-map building
        entrance), which isn't a compass exit."""
        fx = x / max(md.width - 1, 1)
        fy = y / max(md.height - 1, 1)
        if fy <= 0.20:
            return "north"
        if fy >= 0.80:
            return "south"
        if fx <= 0.20:
            return "west"
        if fx >= 0.80:
            return "east"
        return "interior"

    # ---- warp portals ------------------------------------------------------
    for md in maps.values():
        for wp in md.warps:
            x, y, slot = wp["x"], wp["y"], wp["slot"]
            appr = approach_cell(md, x, y)
            comp_id = md.comp.get(appr) if appr is not None else None
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
                "approach": list(appr) if appr is not None else None,
                "kind": "elevator" if "Elevator" in md.name else "warp",
                "dest_map": dest_id,
                "dest_warp": dest_slot,
                "dest_portal": dest_portal,
                "component": comp_id,
                "direction": warp_direction(md, x, y),
                "label": f"{md.name} warp {slot} -> {dest_const}",
                "note": note,
            }

    # ---- edge portals: OFFSET-AWARE cell-level pairing (K0) -----------------
    # pokered's `connection` macro shifts the crossing coordinate by -2*offset on the seam axis;
    # a crossing exists only where the shifted target cell is walkable. One portal per
    # (source component, target component), represented by a cell that really crosses.
    for md in maps.values():
        for conn in md.connections:
            direction = conn["dir"]
            dest_id = consts[conn["target_const"]][0]
            dst = maps.get(dest_id)
            groups: dict[tuple[int, int | None], list[tuple[int, int]]] = {}
            for c in border_walkable(md, direction):
                sc = md.comp[c]
                if dst is None:
                    groups.setdefault((sc, None), []).append(c)
                    continue
                t = cross_cell(md, direction, conn["offset"], dst, c)
                if t in dst.comp:
                    groups.setdefault((sc, dst.comp[t]), []).append(c)
            per_src: dict[int, int] = {}
            for sc, _tc in groups:
                per_src[sc] = per_src.get(sc, 0) + 1
            for (sc, tc), cells in sorted(groups.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
                pid = pid_edge(md, direction, sc) + ("" if per_src[sc] == 1 else f"_to{tc}")
                rep_cell = representative(cells, direction)
                portals[pid] = {
                    "id": pid,
                    "map": md.id,
                    "coord": [rep_cell[0], rep_cell[1]],
                    "kind": "edge",
                    "dest_map": dest_id,
                    "dest_warp": None,
                    "dest_portal": None,   # paired below
                    "dest_component": tc,
                    "component": sc,
                    "label": f"{md.name} {direction} edge -> {conn['target_name']}",
                    "direction": direction,
                }

    # ---- pair each edge to the neighbour's reverse edge landing in the same component ----
    for p in list(portals.values()):
        if p["kind"] != "edge":
            continue
        opp = _OPPOSITE[p["direction"]]
        cands = [q for q in portals.values()
                 if q["kind"] == "edge" and q["map"] == p["dest_map"] and q["dest_map"] == p["map"]
                 and q.get("direction") == opp and q["component"] == p.get("dest_component")]
        if cands:
            p["dest_portal"] = cands[0]["id"]

    # ---- directed LEDGE portals (K2', OVERWORLD tileset only) ----------------
    ledges = parse_ledges()
    step = {"south": (0, 1), "north": (0, -1), "west": (-1, 0), "east": (1, 0)}
    for md in maps.values():
        if md.tileset != "OVERWORLD":
            continue
        groups: dict[tuple, list[tuple[int, int]]] = {}
        for a in md.walkable:
            ta = md.tiles.get(a)
            for d, stand, ledge in ledges:
                if ta != stand:
                    continue
                dx, dy = step[d]
                if md.tiles.get((a[0] + dx, a[1] + dy)) != ledge:
                    continue
                b = (a[0] + 2 * dx, a[1] + 2 * dy)
                if 0 <= b[0] < md.width and 0 <= b[1] < md.height:
                    if b not in md.comp:
                        continue
                    key = (md.comp[a], md.id, md.comp[b], d)
                else:   # the landing is across a map edge (e.g. Route 4 -> Route 3)
                    conn = next((c for c in md.connections if c["dir"] == d), None)
                    dst = maps.get(consts[conn["target_const"]][0]) if conn else None
                    if dst is None:
                        continue
                    t = beyond_cell(md, d, conn["offset"], dst, b)
                    if t not in dst.comp:
                        continue
                    key = (md.comp[a], dst.id, dst.comp[t], d)
                if key[1] == md.id and key[0] == key[2]:
                    continue          # a hop inside one component adds nothing
                groups.setdefault(key, []).append(a)
        for (ca, dm, cb, d), cells in sorted(groups.items()):
            rep_cell = representative(cells, d)
            pid = f"{md.name.lower()}:ledge_{d}_c{ca}_to{dm}c{cb}"
            portals[pid] = {
                "id": pid, "map": md.id, "coord": [rep_cell[0], rep_cell[1]], "kind": "ledge",
                "hop": d, "dest_map": dm, "dest_warp": None, "dest_portal": None, "dest_component": cb,
                "component": ca, "direction": d,
                "label": f"{md.name} ledge hop {d} -> {maps[dm].name if dm in maps else dm}",
            }

    # ---- LAST_MAP like the game (K6): wLastMap is only set when leaving an OUTSIDE map ----
    fix_last_map(maps, portals, consts, const_to_name, id_to_const)

    # ---- gated portals (K5): story gates the static geometry can't see ----
    for p in portals.values():
        src = maps[p["map"]].name
        dst = maps[p["dest_map"]].name if p.get("dest_map") in maps else None
        why = GATED.get(src) or (GATED.get(dst) if p["kind"] in ("warp", "elevator") else None)
        if why:
            p["gated"] = why

    return {"maps": maps, "portals": portals}


OUTSIDE_TILESETS = {"OVERWORLD", "PLATEAU"}

# K5 — story gates the static geometry can't see (extend per incident). Warps on/into these maps are
# marked `gated` and skipped by routing by default.
GATED = {
    "Route5Gate": "Saffron guard wants a drink",
    "Route6Gate": "Saffron guard wants a drink",
    "Route7Gate": "Saffron guard wants a drink",
    "Route8Gate": "Saffron guard wants a drink",
    "CeruleanTrashedHouse": "a police officer blocks the back door early on",
    "Route22Gate": "badge check (Boulder Badge) for Route 23",
    "Route16Gate1F": "Cycling Road: bicycle required",
    "Route18Gate1F": "Cycling Road: bicycle required",
}


def cross_cell(md: MapData, direction: str, offset: int, dst: MapData, cell: tuple[int, int]) -> tuple[int, int]:
    """The destination cell reached by stepping off ``md``'s ``direction`` border at ``cell`` (pokered
    connection macro: the seam coordinate shifts by -2*offset)."""
    x, y = cell
    if direction == "north":
        return (x - 2 * offset, dst.height - 1)
    if direction == "south":
        return (x - 2 * offset, 0)
    if direction == "west":
        return (dst.width - 1, y - 2 * offset)
    return (0, y - 2 * offset)


def beyond_cell(md: MapData, direction: str, offset: int, dst: MapData, b: tuple[int, int]) -> tuple[int, int]:
    """A cell ``b`` lying just beyond ``md``'s border, expressed in the neighbour's coordinates."""
    x, y = b
    if direction == "south":
        return (x - 2 * offset, y - md.height)
    if direction == "north":
        return (x - 2 * offset, dst.height + y)
    if direction == "east":
        return (x - md.width, y - 2 * offset)
    return (dst.width + x, y - 2 * offset)


def approach_cell(md: MapData, x: int, y: int) -> tuple[int, int] | None:
    """The walkable cell you step from to use a warp (K3'): the warp cell itself when walkable, else the
    inward neighbour (doors on the bottom rows are entered from above, others from below), then sides."""
    if (x, y) in md.comp:
        return (x, y)
    order = ((0, -1), (0, 1), (-1, 0), (1, 0)) if y >= md.height - 2 else ((0, 1), (0, -1), (-1, 0), (1, 0))
    for dx, dy in order:
        c = (x + dx, y + dy)
        if c in md.comp:
            return c
    return None


def fix_last_map(maps: dict, portals: dict, consts: dict, const_to_name: dict, id_to_const: dict) -> None:
    """Re-resolve LAST_MAP warps the way the game does: wLastMap is only updated when warping out of an
    OUTSIDE map, so an indoor LAST_MAP warp returns to the outside map that reaches it through a chain of
    explicit warps. Applied where the slot-matching heuristic left the warp unresolved, dangling, or
    pointing at a non-outside map (e.g. Victory Road 2F, Rock Tunnel 1F)."""
    preds: dict[int, list[tuple[int, int]]] = {}
    for md in maps.values():
        for wp in md.warps:
            if wp["dest_const"] in ("LAST_MAP",) or wp["dest_const"] not in consts:
                continue
            preds.setdefault(consts[wp["dest_const"]][0], []).append((md.id, wp["slot"]))

    def outside_entries(target: int) -> set[tuple[int, int]]:
        found: set[tuple[int, int]] = set()
        seen = {target}
        q = deque([target])
        while q:
            t = q.popleft()
            for src, slot in preds.get(t, []):
                if src not in maps:
                    continue
                if maps[src].tileset in OUTSIDE_TILESETS:
                    found.add((src, slot))
                elif src not in seen:
                    seen.add(src)
                    q.append(src)
        return found

    for p in portals.values():
        if p["kind"] != "warp" or "LAST_MAP" not in p["label"]:
            continue
        dm = p["dest_map"]
        ok = (dm in maps and maps[dm].tileset in OUTSIDE_TILESETS
              and (p["dest_portal"] is None or p["dest_portal"] in portals))
        if ok:
            continue
        entries = outside_entries(p["map"])
        omaps = {m for m, _ in entries}
        if len(omaps) == 1:
            om, oslot = sorted(entries)[0]
            p["dest_map"] = om
            p["dest_warp"] = oslot
            p["dest_portal"] = f"{maps[om].name.lower()}:warp{oslot}"
            p["note"] = f"LAST_MAP -> outside map {om} (game rule: last OUTSIDE map)"
        elif omaps:
            p["note"] = (p.get("note") or "") + f"; outside candidates {sorted(omaps)}"


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
    usable = {k: v for k, v in portals.items() if not v.get("gated") and v["kind"] != "elevator"}
    portals = usable
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
        if p.get("dest_component") is not None:
            land_map, land_comp = p["dest_map"], p["dest_component"]
        elif dest is not None and dest in portals:
            land_map, land_comp = portals[dest]["map"], portals[dest]["component"]
        else:
            continue
        # arrived on dest map, in the landing component -> can walk to same-component portals
        arrivals = [q2["id"] for q2 in portals.values()
                    if q2["map"] == land_map and q2["component"] == land_comp]
        for a in arrivals:
            if a not in seen:
                seen.add(a)
                q.append((a, path + [a]))
    return None


# --------------------------------------------------------------------------- #
# 6. Validation checks
# --------------------------------------------------------------------------- #
def find_states(per_map: int = 6) -> dict[int, list[Path]]:
    """Newest ``runs/*/states/map<ID>_*.state`` files per map id (a few candidates each: the recorder
    names a state by the map at step START, so a map-crossing step's file holds the NEXT map)."""
    cands: dict[int, list[tuple[float, Path]]] = {}
    for p in (REPO / "runs").glob("*/states/map*_*.state"):
        m = re.match(r"map(\d+)_", p.name)
        if not m:
            continue
        cands.setdefault(int(m.group(1)), []).append((p.stat().st_mtime, p))
    return {k: [p for _, p in sorted(v, reverse=True)[:per_map]] for k, v in cands.items()}


def validate_collision(graph: dict) -> dict[int, dict]:
    """For each validate-id we have a save state for, load it, decode RAM walkable,
    and diff against the static walkable. Returns per-map result dict."""
    sys.path.insert(0, str(REPO / "src"))
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.games.pokemon_red.map_reader import read_collision_map

    states = find_states()
    maps = graph["maps"]
    results: dict[int, dict] = {}
    for mid in sorted(m for m in states if m in maps):
        if mid not in states:
            results[mid] = {"status": "NO_STATE"}
            continue
        ram = None
        for path in states[mid]:
            emu = PyBoyEmulator(str(ROM), window="null")
            emu.load_state(path)
            emu.tick(4)
            r = read_collision_map(emu)
            emu.close() if hasattr(emu, "close") else None
            if r is not None and r["map_id"] == mid:
                ram = r
                break
        if ram is None:
            results[mid] = {"status": "NO_MATCHING_STATE"}
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
SHIPPED = REPO / "src" / "pokemon_agent" / "games" / "pokemon_red" / "portal_graph.json"


def all_map_names() -> list[str]:
    """Every map header except the duplicate ``*Copy`` headers (K4)."""
    return [f.stem for f in sorted((POKERED / "data" / "maps" / "headers").glob("*.asm"))
            if not f.stem.endswith("Copy")]


def encode_grid(md: MapData) -> str:
    """Per-map static component grid, run-length encoded per row ('.' = not walkable):
    rows joined by '|', each row 'c*n,c*n'. Lets the runtime locate the player's component exactly
    (K7: tile-pair cuts that the live flood fill can't see)."""
    rows = []
    for y in range(md.height):
        runs, cur, n = [], None, 0
        for x in range(md.width):
            c = md.comp.get((x, y))
            v = "." if c is None else str(c)
            if v == cur:
                n += 1
            else:
                if cur is not None:
                    runs.append(f"{cur}*{n}")
                cur, n = v, 1
        runs.append(f"{cur}*{n}")
        rows.append(",".join(runs))
    return "|".join(rows)


def graph_json(graph: dict, version: str) -> dict:
    maps = graph["maps"]
    return {
        "version": version,
        "maps": {
            str(mid): {
                "id": mid, "name": md.name, "const": md.const,
                "width": md.width, "height": md.height, "tileset": md.tileset,
                "n_components": len(set(md.comp.values())) if md.comp else 0,
                "walkable_count": len(md.walkable),
                "grid": encode_grid(md),
                "cuts": ";".join(f"{x},{y},{d}" for x, y, d in getattr(md, "cuts", [])),
            } for mid, md in maps.items()
        },
        "portals": {pid: {k: v for k, v in p.items()} for pid, p in graph["portals"].items()},
    }


def version_string() -> str:
    import hashlib
    import subprocess
    try:
        commit = subprocess.check_output(["git", "-C", str(POKERED), "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    rip = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:10]
    return f"pokered@{commit}+rip@{rip}"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="all", help="'all' (default) or 'corridor'")
    ap.add_argument("--out", default=str(SHIPPED), help="where to write the shipped graph JSON")
    ap.add_argument("--no-validate", action="store_true", help="skip the RAM collision check")
    a = ap.parse_args()

    names = CORRIDOR_NAMES if a.maps == "corridor" else all_map_names()
    print("=" * 70)
    print(f"STATIC PORTAL-GRAPH RIP — {a.maps} ({len(names)} headers) from {POKERED}")
    print("=" * 70)
    graph = build_portal_graph(names)
    portals, maps = graph["portals"], graph["maps"]
    kinds: dict[str, int] = {}
    for p in portals.values():
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(f"\nRipped {len(maps)} maps, {len(portals)} portals {kinds}; "
          f"gated {sum(1 for p in portals.values() if p.get('gated'))}")
    for nm, note in LOAD_NOTES.items():
        print(f"  load note: {nm}: {note}")

    ok = True
    dangling = [p["id"] for p in portals.values() if p.get("dest_portal") and p["dest_portal"] not in portals]
    print(f"\nCHECK 0 — no dangling dest_portal: {'PASS' if not dangling else 'FAIL ' + str(dangling)}")
    ok &= not dangling

    out = graph_json(graph, version_string())
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {a.out} ({Path(a.out).stat().st_size // 1024} KB, version {out['version']})")

    if not a.no_validate:
        print("\nCHECK 1 — collision: static decode == RAM read_collision_map (every map with a save state)")
        print("         (cannot detect edge offsets, ledges or tile-pair cuts — the goldens cover those)")
        try:
            vres = validate_collision(graph)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR running validation: {e!r}")
            vres, ok = {}, False
        for mid, r in sorted(vres.items()):
            name = maps[mid].name
            if r["status"] == "MATCH":
                print(f"  map {mid:3d} {name:28s} PASS ({r['static_cells']} cells)")
            else:
                ok = False
                print(f"  map {mid:3d} {name:28s} {r['status']} {r.get('diff_count', '')}")

    print("\nCHECK 2 — golden routes")
    ids = {md.name: mid for mid, md in maps.items()}

    def comp_of_warp(map_name: str, label_part: str) -> int:
        return next(p["component"] for p in portals.values()
                    if p["map"] == ids[map_name] and label_part in p["label"])

    def first_route(a_name: str, b_name: str, comp: int | None = None):
        a_id, b_id = ids[a_name], ids[b_name]
        comps = [comp] if comp is not None else sorted({p["component"] for p in portals.values()
                                                        if p["map"] == a_id and p["component"] is not None})
        for c in comps:
            r = route(portals, a_id, c, b_id)
            if r:
                return r
        return None

    goldens = [
        ("Forest -> Pewter via the north gate", first_route("ViridianForest", "PewterCity"),
         lambda r: r and any("northgate" in x for x in r) and not any("southgate" in x for x in r)),
        ("Pewter -> Mt. Moon 1F", first_route("PewterCity", "MtMoon1F"), lambda r: bool(r)),
        ("Pewter -> Cerulean via Mt. Moon + the Route 4 ledge", first_route("PewterCity", "CeruleanCity"),
         lambda r: r and any(x.startswith("mtmoonb2f:") for x in r) and any(":ledge_" in x for x in r)),
        ("Cerulean (main) -> Pewter is unreachable",
         first_route("CeruleanCity", "PewterCity", comp_of_warp("CeruleanCity", "POKECENTER")),
         lambda r: r is None),
        # from Cerulean's SOUTH side (its main area reaches the south only through the trashed house,
        # which is gated early on — so main -> Vermilion is correctly unreachable at first)
        ("Cerulean (south side) -> Vermilion via the Underground Path, not Saffron",
         first_route("CeruleanCity", "VermilionCity", comp_of_warp("CeruleanCity", "south edge")),
         lambda r: r and any("undergroundpath" in x for x in r) and not any("saffron" in x for x in r)),
    ]
    for label, r, test in goldens:
        passed = bool(test(r))
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
        if r:
            print("        " + " | ".join(portals[x]["label"] for x in r))
    north = [p["id"] for p in portals.values() if p["kind"] == "ledge" and p["hop"] == "north"]
    print(f"  {'PASS' if not north else 'FAIL'}  no ledge hops north {north or ''}")
    ok &= not north

    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
