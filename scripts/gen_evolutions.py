#!/usr/bin/env python3
"""Rip each species' NEXT evolution(s) from pokered (data/pokemon/evos_moves.asm) for the party view.

A player knows what their Pokémon become (runs/sleeves-mtmoon: L1 benched a Magikarp as "dead weight"
without ever looking up Gyarados). Output: src/pokemon_agent/games/pokemon_red/evolutions.json
    {"Magikarp": ["Gyarados at L20"], "Eevee": ["Flareon with a Fire Stone", ...], "Kadabra": ["Alakazam by trade"]}
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rip_portals import POKERED, REPO  # noqa: E402

OUT = REPO / "src" / "pokemon_agent" / "games" / "pokemon_red" / "evolutions.json"


def main() -> None:
    from pokemon_agent.games.pokemon_red.constants import SPECIES

    def key(n):
        return re.sub(r"[^A-Z0-9]", "", str(n).upper().replace("♀", "F").replace("♂", "M"))
    disp = {key(n): n for n in SPECIES.values() if n}

    def name(const):
        return disp.get(re.sub(r"[^A-Z0-9]", "", const), const.title().replace("_", " "))
    text = (POKERED / "data/pokemon/evos_moves.asm").read_text()
    out: dict[str, list[str]] = {}
    for m in re.finditer(r"^(\w+)EvosMoves:\n(.*?)\n\s*db 0", text, re.M | re.S):
        if re.match(r"(?i)(Fossil|MonGhost|MissingNo)", m.group(1)):
            continue
        src = name(m.group(1).upper())
        for line in m.group(2).splitlines():
            ev = re.match(r"\s*db EVOLVE_LEVEL, (\d+), (\w+)", line)
            it = re.match(r"\s*db EVOLVE_ITEM, (\w+), \d+, (\w+)", line)
            tr = re.match(r"\s*db EVOLVE_TRADE, \d+, (\w+)", line)
            if ev:
                out.setdefault(src, []).append(f"{name(ev.group(2))} at L{ev.group(1)}")
            elif it:
                out.setdefault(src, []).append(f"{name(it.group(2))} with a {it.group(1).replace('_', ' ').title()}")
            elif tr:
                out.setdefault(src, []).append(f"{name(tr.group(1))} by trade")
    OUT.write_text(json.dumps(out, sort_keys=True))
    print(f"wrote {OUT.relative_to(REPO)}: {len(out)} species; Magikarp={out.get('Magikarp')} Weedle={out.get('Weedle')}")


if __name__ == "__main__":
    main()
