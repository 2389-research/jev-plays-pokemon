#!/usr/bin/env python3
"""Rip each map's wild GRASS encounter rate from pokered (data/wild/maps/*.asm: def_grass_wildmons N).

Every step on grass rolls an encounter with probability N/256, so a 60-step "no encounter" limit
gave up on Viridian Forest (N=8, ~32 steps per encounter) about 15% of the time while grinding.
Where encounters can happen (engine/battle/wild_encounters.asm TryDoWildEncounter): on a GRASS tile;
or on ANY tile of an "indoor" map (id >= FIRST_INDOOR_MAP = $25) that has wild data, unless its tileset
is FOREST (Viridian Forest / Safari Zone stay grass-only). So caves, Pokémon Tower, Seafoam... roll on
every step.
Output: src/pokemon_agent/games/pokemon_red/wild_rates.json
    {"grass_rate": {map_id: N}, "anywhere": [map_id, ...]}
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rip_portals import POKERED, REPO, build_name_index, parse_header, parse_map_constants  # noqa: E402

FIRST_INDOOR_MAP = 0x25
OUT = REPO / "src" / "pokemon_agent" / "games" / "pokemon_red" / "wild_rates.json"


def main() -> None:
    consts = parse_map_constants()
    _c2n, name_to_const = build_name_index()
    # wild data files are named after the WildMons label, which is the map name (Route3WildMons)
    headers = {}
    for f in (POKERED / "data" / "maps" / "headers").glob("*.asm"):
        if not f.stem.endswith("Copy"):
            headers[f.stem] = f.stem
    rates: dict[int, int] = {}
    anywhere: list[int] = []
    for f in sorted((POKERED / "data" / "wild" / "maps").glob("*.asm")):
        m = re.search(r"def_grass_wildmons\s+(\d+)", f.read_text())
        mid = consts.get(name_to_const.get(f.stem, ""), (None,))[0]
        if m and mid is not None and int(m.group(1)) > 0:
            rates[mid] = int(m.group(1))
            try:
                tileset = parse_header(f.stem)["tileset"]
            except Exception:
                tileset = None
            if mid >= FIRST_INDOOR_MAP and tileset != "FOREST":
                anywhere.append(mid)
    OUT.write_text(json.dumps({"grass_rate": {str(k): v for k, v in sorted(rates.items())},
                               "anywhere": sorted(anywhere)}))
    print(f"wrote {OUT.relative_to(REPO)}: {len(rates)} maps, {len(anywhere)} encounter-anywhere; "
          f"forest(51)={rates.get(51)} in_anywhere={51 in anywhere}; mt moon 1F(59) in_anywhere={59 in anywhere}")


if __name__ == "__main__":
    main()
