"""Full-map collision reader: pure ingest + a fixture-guarded live decode check."""
import os
from pathlib import Path

import numpy as np
import pytest

from pokemon_agent.agent.world_map import FLOOR, WALL, WorldMap

ROM = Path("roms/pokemon_red.gb")
STATE = Path("states/pallet_ready.state")
_live = pytest.mark.skipif(not (ROM.exists() and STATE.exists()),
                           reason="ROM / pallet_ready fixture not present")


def test_ingest_collision_marks_floor_and_wall():
    w = WorldMap()
    walkable = {(0, 0), (1, 0)}
    w.ingest_collision(map_id=7, width=2, height=2, walkable=walkable)
    m = w.tiles[7]
    assert m[(0, 0)] == FLOOR and m[(1, 0)] == FLOOR
    assert m[(0, 1)] == WALL and m[(1, 1)] == WALL  # not in walkable -> wall


@_live
def test_full_collision_matches_onscreen_ground_truth():
    """Decoded full-map walkability must match PyBoy's on-screen collision window exactly
    (this is the 100%-match validation used to derive the decode)."""
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.games.pokemon_red.map_reader import read_collision_map

    emu = PyBoyEmulator(str(ROM), window="null")
    try:
        emu.load_state(STATE)  # PyBoyEmulator.load_state settles ~60 frames internally
        cm = read_collision_map(emu)
        assert cm is not None and cm["map_id"] == 0  # Pallet Town, overworld
        px, py = emu.read_memory(0xD362), emu.read_memory(0xD361)
        gc = np.asarray(emu._pyboy.game_area_collision())
        onscreen = gc[::2, ::2] > 0  # 9x10 per-cell walkability, player at (row4,col4)
        walk = cm["walkable"]
        checked = 0
        for r in range(9):
            for c in range(10):
                mx, my = px - 4 + c, py - 4 + r
                if 0 <= mx < cm["width"] and 0 <= my < cm["height"]:
                    assert ((mx, my) in walk) == bool(onscreen[r, c]), f"mismatch at ({mx},{my})"
                    checked += 1
        assert checked > 40  # actually validated a real window
    finally:
        emu.close()
