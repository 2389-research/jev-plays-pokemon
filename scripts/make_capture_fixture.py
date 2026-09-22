# /// script
# requires-python = ">=3.12"
# ///
"""Create the CAPTURE end-to-end fixture for Phase 4 (design §6.4).

Phase 2's `states/wild_battle.state` deliberately catches nothing — its Kakuna at full HP
breaks free of a Poké Ball (RNG restored by the save state), which is right for the macro
test but useless for verifying a *catch*. This makes a fixture where the throw DETERMINISTICALLY
succeeds, by stacking the catch odds so high the restored RNG catches on the first ball:

  * same mid-wild-battle save (Viridian Forest, `in_battle==1`, FIGHT menu up, wild Kakuna);
  * drop the wild Kakuna to **1 HP** (0xCFE6 — maximises the Gen-1 catch chance);
  * inject a bag led by an **Ultra Ball** (a high catch-rate ball) plus Poké Balls + Potions.

At 1 HP an Ultra Ball catches Kakuna under this save state's RNG (verified across a range of
pre-throw frame offsets, so the loop's turn timing doesn't matter). The battle layer's
`choose_action` throws the FIRST ball in bag order, so the Ultra Ball leads. Writes
`states/capture_wild.state` (gitignored).

  uv run python scripts/make_capture_fixture.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
from pokemon_agent.games.pokemon_red import battle
from pokemon_agent.games.pokemon_red.game_state import read_battle, read_items, read_party

ROM = "roms/pokemon_red.gb"
SOURCE = Path("runs/portal-cross/states/map51_step0.state")
DEST = Path("states/capture_wild.state")

ENEMY_HP = 0xCFE6           # enemy current HP, big-endian 2 bytes
WNUMBAGITEMS = 0xD31D
WBAGITEMS = 0xD31E
ULTRA_BALL, POKE_BALL, POTION = 2, 4, 20


def inject_bag(emu) -> None:
    """[Ultra Ball ×5, Poké Ball ×5, Potion ×3], 0xFF-terminated (matches read_items order)."""
    emu.write_memory(WNUMBAGITEMS, 3)
    for off, val in [(0, ULTRA_BALL), (1, 5), (2, POKE_BALL), (3, 5),
                     (4, POTION), (5, 3), (6, 0xFF)]:
        emu.write_memory(WBAGITEMS + off, val)


def weaken_enemy(emu, hp: int = 1) -> None:
    emu.write_memory(ENEMY_HP, (hp >> 8) & 0xFF)
    emu.write_memory(ENEMY_HP + 1, hp & 0xFF)


def main() -> None:
    if not Path(ROM).exists() or not SOURCE.exists():
        raise SystemExit(f"need {ROM} and {SOURCE}")
    emu = PyBoyEmulator(ROM, window="null")
    emu.load_state(SOURCE)
    emu.tick(6)

    assert emu.read_memory(0xD057) == 1, "source must be a WILD battle (in_battle==1)"
    assert battle.fight_menu_showing(emu), "source must show the FIGHT menu"
    weaken_enemy(emu, 1)
    inject_bag(emu)

    DEST.parent.mkdir(parents=True, exist_ok=True)
    emu.save_state(DEST)
    print(f"wrote {DEST}")
    print("  battle:", read_battle(emu))
    print("  party :", [(m["species"], m["level"], f"{m['hp']}/{m['max_hp']}") for m in read_party(emu)])
    print("  bag   :", [(i["item"], i["qty"]) for i in read_items(emu)])
    emu.close()


if __name__ == "__main__":
    main()
