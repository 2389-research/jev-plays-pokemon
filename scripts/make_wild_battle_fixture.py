# /// script
# requires-python = ">=3.12"
# ///
"""Create the wild-battle fixture for the P2 battle-action macro tests (design §5).

A wild encounter is RNG, so rather than grind for one live we start from an existing
mid-wild-battle save (`runs/portal-cross/states/map51_step0.state` — Viridian Forest,
`in_battle==1`, the FIGHT menu up, a wild Kakuna, our Squirtle at 8/27 HP), then inject a
small bag so the ITEM path has something to throw/use:

    5 × Poké Ball   (throw_ball)      3 × Potion   (use_item, active mon is already low HP)

The bag is plain WRAM; injecting it is a legitimate test PRECONDITION (the player normally
carries balls) and keeps the fixture deterministic. Save states restore RNG, so the throw /
run outcomes are reproducible. Writes `states/wild_battle.state` (gitignored).

  uv run python scripts/make_wild_battle_fixture.py
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
DEST = Path("states/wild_battle.state")

WNUMBAGITEMS = 0xD31D
WBAGITEMS = 0xD31E
POKE_BALL, POTION = 4, 20


def inject_bag(emu) -> None:
    """[Poké Ball ×5, Potion ×3], 0xFF-terminated — matches read_items order."""
    emu.write_memory(WNUMBAGITEMS, 2)
    for off, val in [(0, POKE_BALL), (1, 5), (2, POTION), (3, 3), (4, 0xFF)]:
        emu.write_memory(WBAGITEMS + off, val)


def main() -> None:
    if not Path(ROM).exists() or not SOURCE.exists():
        raise SystemExit(f"need {ROM} and {SOURCE}")
    emu = PyBoyEmulator(ROM, window="null")
    emu.load_state(SOURCE)
    emu.tick(6)

    assert emu.read_memory(0xD057) == 1, "source must be a WILD battle (in_battle==1)"
    assert battle.fight_menu_showing(emu), "source must show the FIGHT menu"
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
