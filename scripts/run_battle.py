# /// script
# requires-python = ">=3.12"
# ///
"""Play a Pokémon Red battle to completion with the battle puppeteer.

  uv run python scripts/run_battle.py --state battle --move 0

Loads states/<NAME>.state, advances any intro text to the FIGHT menu, then takes
turns with the chosen move slot until the battle ends, printing the play-by-play.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
from pokemon_agent.games.pokemon_red import battle
from pokemon_agent.games.pokemon_red.game_state import read_battle
from pokemon_agent.core.models import GameButton


def _both_hp(emu):
    b = read_battle(emu) or {}
    e = b.get("enemy", {})
    a = b.get("active", {})
    return (f"{a.get('species','?')} {a.get('hp','?')}/{a.get('max_hp','?')}",
            f"{e.get('species','?')} {e.get('hp','?')}/{e.get('max_hp','?')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="roms/pokemon_red.gb")
    ap.add_argument("--state", default="battle")
    ap.add_argument("--move", type=int, default=0, help="fixed move slot each turn (ignored if --bot)")
    ap.add_argument("--bot", action="store_true", help="let a TypeSafe model choose the move each turn")
    ap.add_argument("--window", action="store_true", help="show the emulator window (watch it play)")
    ap.add_argument("--speed", type=int, default=None, help="emulation speed (default: 1 windowed, unbounded headless)")
    ap.add_argument("--max-turns", type=int, default=30)
    args = ap.parse_args()

    path = Path("states") / f"{args.state}.state"
    if not path.exists():
        raise SystemExit(f"no fixture at {path}")

    window = "SDL2" if args.window else "null"
    speed = args.speed if args.speed is not None else (1 if args.window else 0)
    emu = PyBoyEmulator(args.rom, window=window, speed=speed)
    emu.load_state(path)
    emu.tick(3)

    # advance any battle-intro text until the FIGHT menu appears
    for _ in range(40):
        if battle.fight_menu_showing(emu):
            break
        emu.press(GameButton.A)
        emu.tick(38)

    if not battle.in_battle(emu):
        raise SystemExit("this state isn't in a battle")

    moves = battle.active_moves(emu)
    me, you = _both_hp(emu)
    print(f"BATTLE START — you: {me} | enemy: {you}")
    ts_client = None
    if args.bot:
        from typesafe_sdk import TypeSafeClient
        from pokemon_agent.games.pokemon_red import battle_agent
        ts_client = TypeSafeClient()
        print(f"moves available: {moves} -> BOT chooses each turn\n")
    else:
        print(f"moves available: {moves} -> fixed slot {min(args.move, len(moves)-1)}\n")

    for turn in range(1, args.max_turns + 1):
        if not battle.fight_menu_showing(emu):
            # a message or sub-menu is up (e.g. faint / switch prompt) — nudge it
            emu.press(GameButton.A)
            emu.tick(38)
            if not battle.in_battle(emu):
                break
            continue
        if ts_client is not None:
            slot, conf = battle_agent.choose_move(ts_client, emu)
            pick = f"BOT->{moves[slot]} ({conf:.2f})"
        else:
            slot = max(0, min(args.move, len(moves) - 1))
            pick = ""
        r = battle.use_move(emu, slot)
        me, you = _both_hp(emu)
        print(f"turn {turn:>2}: {r['move']:<10} dealt {r['damage_dealt']:>2} | you: {me} | enemy: {you}  {pick}")
        if r["battle_over"]:
            break

    # let the end-of-battle text play out
    for _ in range(20):
        if not battle.in_battle(emu):
            break
        emu.press(GameButton.A)
        emu.tick(38)

    print()
    if battle.in_battle(emu):
        print("=> battle still going after the turn cap")
    else:
        me, _ = _both_hp(emu)
        print(f"=> battle over. your mon: {me}")
    emu.close()


if __name__ == "__main__":
    main()
