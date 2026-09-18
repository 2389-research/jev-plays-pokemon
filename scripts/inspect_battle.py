# /// script
# requires-python = ">=3.12"
# ///
"""Dump + validate battle RAM against a real battle save state.

  uv run python scripts/inspect_battle.py --state battle

Loads states/<NAME>.state and prints: the context read, the (best-effort) battle
struct, the raw menu-cursor values, and the decoded on-screen tilemap — so we can
eyeball whether the FIGHT/PKMN/ITEM/RUN menu, the move list, and the HP/species
reads line up with what's actually on screen, then fix any offsets from ground truth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
from pokemon_agent.games.pokemon_red.game_state import (
    WTILEMAP,
    _decode_byte,
    read_battle,
    read_context,
    read_party,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="roms/pokemon_red.gb")
    ap.add_argument("--state", default="battle", help="states/<NAME>.state")
    args = ap.parse_args()

    path = Path("states") / f"{args.state}.state"
    if not path.exists():
        raise SystemExit(f"no fixture at {path} — capture one with scripts/play.py --state {args.state}")

    emu = PyBoyEmulator(args.rom, window="null", speed=0)
    emu.load_state(path)
    emu.tick(3)
    M = emu.read_memory

    print(f"=== {path} ===")
    print("context :", read_context(emu))
    print("party   :", read_party(emu))
    print("battle  :", read_battle(emu))
    print("menu_raw: cursor_item(CC26)=%d last(CC28)=%d cursor_yx=(%d,%d) watched_keys(CC29)=%s"
          % (M(0xCC26), M(0xCC28), M(0xCC24), M(0xCC25), hex(M(0xCC29))))

    print("\n--- on-screen tilemap (what the player sees) ---")
    for r in range(18):
        s = "".join(_decode_byte(M(WTILEMAP + r * 20 + c)) for c in range(20))
        if s.strip():
            print(f"{r:2d}|{s}")

    print("\n--- cursor-arrow tiles (0xEC/0xED candidates) ---")
    for cand in (0xED, 0xEC, 0x76, 0x77):
        cells = [(i % 20, i // 20) for i in range(20 * 18) if M(WTILEMAP + i) == cand]
        if cells:
            print(f"  tile {hex(cand)} at (x,y): {cells}")
    emu.close()


if __name__ == "__main__":
    main()
