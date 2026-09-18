"""Map-id -> human name, from the pokered disassembly (see constants.py).

Names are for human/log readability and semantic grounding; behavior keys off
map_id and the tileset-derived is_outdoor flag, not the name.
"""
from __future__ import annotations

from .constants import MAP_NAMES_RAW


def map_name(map_id: int) -> str:
    return MAP_NAMES_RAW.get(map_id, f"Map {map_id}")
