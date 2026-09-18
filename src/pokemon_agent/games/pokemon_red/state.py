"""Pokémon Red WRAM offsets and state extraction, all confined to this module.

Addresses are the well-documented pokered WRAM symbols. Extraction failures
degrade to UNKNOWN / None rather than crashing the loop.
"""
from __future__ import annotations

from ...core.models import GameMode, PlayerState
from ...emulator.interface import Emulator
from .maps import map_name

# pokered WRAM (unbanked) --------------------------------------------------
WXCOORD = 0xD362      # player X (map-local)
WYCOORD = 0xD361      # player Y (map-local)
WCURMAP = 0xD35E      # current map id
WISINBATTLE = 0xD057  # 0 = not in battle, nonzero = battle
WPARTYCOUNT = 0xD163  # number of pokemon in party
WTEXTBOXID = 0xD125   # dialog/textbox indicator (heuristic)
WNUMWARPS = 0xD3AE    # number of warp/exit tiles on the current map
WWARPENTRIES = 0xD3AF  # 4 bytes each: y, x, destWarpID, destMap
WCURMAPHEIGHT = 0xD368  # current map height in 2x2-tile blocks
WCURMAPWIDTH = 0xD369   # current map width in 2x2-tile blocks
WCURMAPTILESET = 0xD367  # 0 = OVERWORLD (outdoor); nonzero = an interior tileset
WPLAYERFACING = 0xC109   # player sprite facing: 0 down, 4 up, 8 left, 0xC right
_FACING = {0: "south", 4: "north", 8: "west", 0xC: "east"}


def read_map_dims(emu: Emulator) -> tuple[int, int] | None:
    """(width, height) of the current map in TILES, or None if unreadable."""
    try:
        w = emu.read_memory(WCURMAPWIDTH) * 2
        h = emu.read_memory(WCURMAPHEIGHT) * 2
        if 0 < w <= 200 and 0 < h <= 200:
            return w, h
        return None
    except Exception:
        return None


def read_player(emu: Emulator) -> PlayerState | None:
    try:
        map_id = emu.read_memory(WCURMAP)
        return PlayerState(
            x=emu.read_memory(WXCOORD),
            y=emu.read_memory(WYCOORD),
            map_id=map_id,
            facing=_FACING.get(emu.read_memory(WPLAYERFACING)),
            is_outdoor=(emu.read_memory(WCURMAPTILESET) == 0),
            map_name=map_name(map_id),
        )
    except Exception:
        return None


def read_exits(emu: Emulator) -> list[dict]:
    """Exit/warp tiles on the current map, from RAM (exact coords, ground truth).

    Returns a list of {x, y, dest_map}. These are the door/stairs/mat tiles the
    player steps onto to leave — precise perception the vision model can't provide.
    """
    try:
        n = emu.read_memory(WNUMWARPS)
        exits: list[dict] = []
        for i in range(min(n, 32)):
            y = emu.read_memory(WWARPENTRIES + i * 4 + 0)
            x = emu.read_memory(WWARPENTRIES + i * 4 + 1)
            dest = emu.read_memory(WWARPENTRIES + i * 4 + 3)
            exits.append({"x": x, "y": y, "dest_map": dest, "dest_name": map_name(dest)})
        return exits
    except Exception:
        return []


def detect_mode(emu: Emulator) -> GameMode:
    """Coarse mode detection. Intentionally minimal for the walking slice."""
    try:
        if emu.read_memory(WISINBATTLE) != 0:
            return GameMode.BATTLE
        return GameMode.OVERWORLD
    except Exception:
        return GameMode.UNKNOWN
