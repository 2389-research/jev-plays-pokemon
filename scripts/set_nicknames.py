"""Set party Pokémon nicknames in a save state (writes wPartyMonNicks, like renaming in-game).

  uv run python scripts/set_nicknames.py runs/<run>/latest.state Wartortle=Dylan Spearow=Sugi ...
  uv run python scripts/set_nicknames.py runs/<run>/latest.state 1=Dylan 2=Sugi     # by party slot

Species keys match in party order (a second Zubat takes the next Zubat=... entry). A backup of the state
is written next to it (<state>.bak) before saving.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")

from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator  # noqa: E402
from pokemon_agent.games.pokemon_red.game_state import WPARTYNICKS, read_party  # noqa: E402

TERMINATOR = 0x50


def encode(name: str) -> list[int]:
    out = []
    for ch in name[:10]:
        if "A" <= ch <= "Z":
            out.append(0x80 + ord(ch) - ord("A"))
        elif "a" <= ch <= "z":
            out.append(0xA0 + ord(ch) - ord("a"))
        elif ch == " ":
            out.append(0x7F)
        else:
            raise SystemExit(f"unsupported character {ch!r} in {name!r}")
    return out + [TERMINATOR] * (11 - len(out))


def main() -> int:
    state = Path(sys.argv[1])
    pairs = [a.split("=", 1) for a in sys.argv[2:]]
    emu = PyBoyEmulator("roms/pokemon_red.gb", window="null")
    emu.load_state(state)
    emu.tick(2)
    party = read_party(emu)
    used: set[int] = set()
    plan: dict[int, str] = {}
    for key, name in pairs:
        if key.isdigit():
            slot = int(key) - 1
        else:
            slot = next((i for i, p in enumerate(party) if p["species"].lower() == key.lower() and i not in used), None)
            if slot is None:
                raise SystemExit(f"no (remaining) {key} in the party: {[p['species'] for p in party]}")
        used.add(slot)
        plan[slot] = name
    for slot, name in plan.items():
        for i, b in enumerate(encode(name)):
            emu.write_memory(WPARTYNICKS + slot * 11 + i, b)
    shutil.copy(state, state.with_suffix(state.suffix + ".bak"))
    emu.save_state(state)
    print([(p["species"], p["nickname"]) for p in read_party(emu)])
    emu.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
