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
BORDER = 3  # wOverworldMap's connection border, in blocks, on every side


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

        stride = wb + 2 * BORDER
        walkable: set[tuple[int, int]] = set()
        counters: set[tuple[int, int]] = set()
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
                        if t in collset:
                            walkable.add(cell)
                        if t in counter_ids:
                            counters.add(cell)
        return {"map_id": m(WCURMAP), "width": wb * 2, "height": hb * 2,
                "walkable": walkable, "counters": counters}
    except Exception:
        return None
