#!/usr/bin/env python3
"""Rip TM/HM machines and every species' TM/HM learnset from pokered.

Which Pokémon can learn Cut is player knowledge the agent needs before it can choose to use it
(runs/sleeves-misty: Wartortle, Weedle and Magikarp — none can). Sources: constants/item_constants.asm
(add_hm / add_tm order = HM01.. / TM01..) and data/pokemon/base_stats/*.asm (the ``tmhm`` macro).
Output: src/pokemon_agent/games/pokemon_red/tmhm.json
    {"machines": {"HM01": "Cut", "TM01": "Mega Punch", ...},
     "learnsets": {"Oddish": ["Swords Dance", ..., "Cut"], ...}}
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rip_portals import POKERED, REPO  # noqa: E402

OUT = REPO / "src" / "pokemon_agent" / "games" / "pokemon_red" / "tmhm.json"


def move_name(const: str) -> str:
    special = {"SOLARBEAM": "Solar Beam", "DOUBLESLAP": "Double Slap", "THUNDERBOLT": "Thunderbolt",
               "PSYCHIC_M": "Psychic", "BUBBLEBEAM": "Bubble Beam", "SELFDESTRUCT": "Self-Destruct",
               "DOUBLE_EDGE": "Double-Edge", "SOFTBOILED": "Soft-Boiled", "HI_JUMP_KICK": "Hi Jump Kick"}
    return special.get(const, const.replace("_", " ").title())


def main() -> None:
    from pokemon_agent.games.pokemon_red.constants import SPECIES

    def key(n):
        return re.sub(r"[^A-Z0-9]", "", str(n).upper().replace("♀", "F").replace("♂", "M"))
    disp = {key(n): n for n in SPECIES.values() if n}

    consts = (POKERED / "constants/item_constants.asm").read_text()
    machines: dict[str, str] = {}
    for i, m in enumerate(re.findall(r"^\s*add_hm (\w+)", consts, re.M), 1):
        machines[f"HM{i:02d}"] = move_name(m)
    for i, m in enumerate(re.findall(r"^\s*add_tm (\w+)", consts, re.M), 1):
        machines[f"TM{i:02d}"] = move_name(m)

    learnsets: dict[str, list[str]] = {}
    for f in sorted((POKERED / "data/pokemon/base_stats").glob("*.asm")):
        text = f.read_text()
        dex = re.search(r"db DEX_(\w+)", text)
        tm = re.search(r"tmhm (.*?)\n\s*; end", text, re.S)
        if not dex or not tm:
            continue
        species = disp.get(key(dex.group(1)), dex.group(1).title())
        moves = [move_name(x) for x in re.findall(r"[A-Z_]+", tm.group(1).replace("\\", " "))]
        learnsets[species] = moves
    OUT.write_text(json.dumps({"machines": machines, "learnsets": learnsets}, sort_keys=True))
    cut = sorted(s for s, ms in learnsets.items() if "Cut" in ms)
    print(f"wrote {OUT.relative_to(REPO)}: {len(machines)} machines, {len(learnsets)} species; "
          f"Cut learners ({len(cut)}): {', '.join(cut)}")


if __name__ == "__main__":
    main()
