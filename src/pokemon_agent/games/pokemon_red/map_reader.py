"""Full current-map collision from RAM (the CPP/PokePilot-style world model).

Instead of only the on-screen 9x10 window, this decodes the ENTIRE loaded map's
walkability from ``wOverworldMap`` — so the navigator can route across a whole town
up front, around buildings, without having walked every tile first.

Decode (validated 100% against PyBoy's on-screen collision on Pallet Town):
  * ``wCurMapWidth`` (0xD369) / ``wCurMapHeight`` (0xD368) — map size in 4x4-tile BLOCKS.
  * ``wOverworldMap`` (0xC6E8) — the loaded map's block ids, padded with a 3-block
    connection BORDER; stride = width + 6, interior origin at (3, 3).
  * each block id -> its 4x4 tile ids at ``wTilesetBlocksPtr`` (0xD52C) in bank
    ``wTilesetBank`` (0xD52B).
  * a tile is walkable iff its id is in the tileset's collision list at
    ``wTilesetCollisionPtr`` (0xD530) (fixed bank), terminated by 0xFF.
  * the player walks on a 2x2-tile (16px) grid, so each block = 2x2 player cells; the
    collision-relevant tile of cell (cr, cc) is tile index (cr*2+1)*4 + (cc*2).

Coordinates returned are MAP-LOCAL player cells (same frame as wXCoord/wYCoord), so a
grid of ``width = Wb*2`` by ``height = Hb*2``.
"""
from __future__ import annotations

WOVERWORLDMAP = 0xC6E8
WCURMAP = 0xD35E
WCURMAPHEIGHT = 0xD368  # blocks
WCURMAPWIDTH = 0xD369   # blocks
WTILESETBANK = 0xD52B
WTILESETBLOCKSPTR = 0xD52C
WTILESETCOLLISIONPTR = 0xD530
WTILESETTALKINGOVERTILES = 0xD532  # up to 3 "talk-over" (counter) tile ids, 0xFF-terminated
WGRASSTILE = 0xD535     # this tileset's wild-encounter grass tile id (0xFF = none)
WCURMAPTILESET = 0xD367  # 0 = OVERWORLD
BORDER = 3  # wOverworldMap's connection border, in blocks, on every side

# OVERWORLD tileset (id 0) semantic tile ids — from the pokered disassembly
# (data/tilesets/ledge_tiles.asm, door_tile_ids.asm, home/overworld.asm). These are the tiles
# whose MEANING isn't in the walkable collision list: one-way ledges (+ hop direction), water,
# and door/warp tiles. Grass + counters come from RAM (per-tileset) and aren't hardcoded.
OVERWORLD = 0
_LEDGE_SOUTH = {0x36, 0x37}   # hop DOWN
_LEDGE_WEST = {0x27}          # hop LEFT
_LEDGE_EAST = {0x0D, 0x1D}    # hop RIGHT
_WATER = {0x14}
_OW_DOOR = {0x1B, 0x58}       # walkable tiles that trigger a warp when a warp event sits on them
GYM = 7
# a small tree the field move CUT removes (engine/overworld/cut.asm: wTileInFrontOfPlayer $3d in the
# OVERWORLD tileset, $50 in GYM). Not walkable until cut; it grows back when the map reloads.
CUT_TREE_TILES = {OVERWORLD: 0x3D, GYM: 0x50}


def _classify(t: int, tileset: int, walkable_ids: set[int], grass_tile: int,
              counter_ids: set[int]) -> str:
    """Semantic class of a tile id, for the map the agent reasons on. Ledge/water/door semantics
    apply to the OVERWORLD tileset (their ids are overworld-specific); grass/counter are read from
    RAM so they work in every tileset (e.g. FOREST grass=0x20)."""
    if t in counter_ids:
        return "counter"
    if CUT_TREE_TILES.get(tileset) == t:
        return "cut_tree"
    if grass_tile != 0xFF and t == grass_tile:
        return "grass"
    if tileset == OVERWORLD:
        if t in _LEDGE_SOUTH:
            return "ledge_s"
        if t in _LEDGE_WEST:
            return "ledge_w"
        if t in _LEDGE_EAST:
            return "ledge_e"
        if t in _WATER:
            return "water"
        if t in _OW_DOOR:
            return "door"
    return "floor" if t in walkable_ids else "wall"


def read_collision_map(emu) -> dict | None:
    """Decode the full current-map walkability. Returns
    ``{map_id, width, height, walkable: set[(x, y)]}`` in map-local player cells, or None
    if the map buffers aren't readable/sane (e.g. mid-load before the state settles)."""
    try:
        m = emu.read_memory
        wb, hb = m(WCURMAPWIDTH), m(WCURMAPHEIGHT)
        if not (0 < wb <= 64 and 0 < hb <= 64):
            return None
        bank = m(WTILESETBANK)
        blocks = m(WTILESETBLOCKSPTR) | (m(WTILESETBLOCKSPTR + 1) << 8)
        coll = m(WTILESETCOLLISIONPTR) | (m(WTILESETCOLLISIONPTR + 1) << 8)

        collset: set[int] = set()
        for i in range(0x100):
            v = m(coll + i)
            if v == 0xFF:
                break
            collset.add(v)
        if not collset:
            return None
        # counter / "talk-over" tiles for this tileset (you can talk to an NPC across one)
        counter_ids: set[int] = set()
        for i in range(3):
            v = m(WTILESETTALKINGOVERTILES + i)
            if v == 0xFF:
                break
            counter_ids.add(v)

        grass_tile = m(WGRASSTILE)
        tileset = m(WCURMAPTILESET)
        stride = wb + 2 * BORDER
        walkable: set[tuple[int, int]] = set()
        counters: set[tuple[int, int]] = set()
        grass: set[tuple[int, int]] = set()
        terrain: dict[tuple[int, int], str] = {}  # (x,y) -> semantic class (floor/wall/grass/water/ledge_*/door/counter/cut_tree)
        block_tiles: dict[int, list[int]] = {}
        for by in range(hb):
            for bx in range(wb):
                bid = m(WOVERWORLDMAP + (by + BORDER) * stride + (bx + BORDER))
                tiles = block_tiles.get(bid)
                if tiles is None:
                    tiles = [m(blocks + bid * 16 + i, bank) for i in range(16)]
                    block_tiles[bid] = tiles
                for cr in (0, 1):
                    for cc in (0, 1):
                        t = tiles[(cr * 2 + 1) * 4 + (cc * 2)]
                        cell = (bx * 2 + cc, by * 2 + cr)
                        cls = _classify(t, tileset, collset, grass_tile, counter_ids)
                        terrain[cell] = cls
                        if t in collset:
                            walkable.add(cell)
                        if cls == "counter":
                            counters.add(cell)
                        elif cls == "grass":
                            grass.add(cell)
        return {"map_id": m(WCURMAP), "width": wb * 2, "height": hb * 2,
                "walkable": walkable, "counters": counters, "grass": grass, "terrain": terrain}
    except Exception:
        return None
