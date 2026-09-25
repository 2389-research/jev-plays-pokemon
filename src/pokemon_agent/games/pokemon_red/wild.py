"""Where wild Pokémon can appear, per map (ripped by scripts/gen_wild_rates.py from pokered).

A step rolls an encounter on a GRASS tile, or on ANY tile of an indoor map with wild data unless it
uses the FOREST tileset (caves, Pokémon Tower, Seafoam: every step; Viridian Forest: grass only)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _data() -> tuple[dict[int, int], frozenset]:
    try:
        raw = json.loads((Path(__file__).with_name("wild_rates.json")).read_text())
    except Exception:
        return {}, frozenset()
    return {int(k): int(v) for k, v in (raw.get("grass_rate") or {}).items()}, frozenset(raw.get("anywhere") or [])


def grass_rate(map_id) -> int | None:
    """Encounter rate (out of 256 per eligible step), None if the map has no wild Pokémon."""
    return _data()[0].get(map_id)


def encounters_anywhere(map_id) -> bool:
    """Every walkable step rolls an encounter here (not only grass)."""
    return map_id in _data()[1]
