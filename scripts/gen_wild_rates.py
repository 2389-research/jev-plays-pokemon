#!/usr/bin/env python3
"""Rip each map's wild GRASS encounter rate from pokered (data/wild/maps/*.asm: def_grass_wildmons N).

Every step on grass rolls an encounter with probability N/256, so a 60-step "no encounter" limit
gave up on Viridian Forest (N=8, ~32 steps per encounter) about 15% of the time while grinding.
Output: src/pokemon_agent/games/pokemon_red/wild_rates.json  {map_id: N}  (maps with N > 0 only).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rip_portals import POKERED, REPO, build_name_index, parse_map_constants  # noqa: E402

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
    for f in sorted((POKERED / "data" / "wild" / "maps").glob("*.asm")):
        m = re.search(r"def_grass_wildmons\s+(\d+)", f.read_text())
        mid = consts.get(name_to_const.get(f.stem, ""), (None,))[0]
        if m and mid is not None and int(m.group(1)) > 0:
            rates[mid] = int(m.group(1))
    OUT.write_text(json.dumps({str(k): v for k, v in sorted(rates.items())}))
    print(f"wrote {OUT.relative_to(REPO)}: {len(rates)} maps; forest(51)={rates.get(51)} route3(14)={rates.get(14)}")


if __name__ == "__main__":
    main()
