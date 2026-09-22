"""Deterministic, RAM-checked in-battle macros (design §5, §6.2): from a wild-battle
fixture, `throw_ball` / `use_item` / `run` must move the right RAM — no LLM, no network.

Fixture `states/wild_battle.state` is Viridian Forest (map 51) mid wild battle: `in_battle==1`,
the FIGHT/PKMN/ITEM/RUN menu up, a wild Kakuna, our Squirtle at 8/27 HP, and a bag of
5 Poké Balls + 3 Potions. Build it with `scripts/make_wild_battle_fixture.py`. states/*.state
are gitignored, so these skip when the fixture/ROM is absent — like the other live tests.

Save states restore the RNG, so a throw's catch/break-free and a run's success/fail are
reproducible from the fixture; the tests assert the deterministic outcome (ball −1, item −1
+ HP up) and treat catch/escape as the observed-but-either-way branch.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROM_PATH = ROOT / "roms" / "pokemon_red.gb"
STATE_PATH = ROOT / "states" / "wild_battle.state"


def _emu():
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator

    emu = PyBoyEmulator(str(ROM_PATH), window="null")
    emu.load_state(STATE_PATH)
    emu.tick(6)
    return emu


def _guard():
    if not ROM_PATH.exists() or not STATE_PATH.exists():
        pytest.skip("ROM/wild-battle fixture not present")


def _qty(items, name):
    return next((it["qty"] for it in items if it["item"].lower() == name.lower()), 0)


def test_fixture_is_a_wild_battle_at_the_fight_menu():
    _guard()
    from pokemon_agent.games.pokemon_red import battle, menus

    emu = _emu()
    try:
        assert emu.read_memory(0xD057) == 1, "fixture must be a WILD battle (in_battle==1)"
        assert battle.fight_menu_showing(emu) and menus.menu_open(emu)
    finally:
        emu.close()


def test_battle_menu_2x2_navigation_is_pinned():
    """Structural indices are constant (§3): pin the 2×2 cursor moves from FIGHT."""
    _guard()
    from pokemon_agent.games.pokemon_red import battle_actions as ba
    from pokemon_agent.games.pokemon_red import menus

    for option, expected in [("FIGHT", (9, 14)), ("ITEM", (9, 16)),
                             ("PKMN", (15, 14)), ("RUN", (15, 16))]:
        emu = _emu()
        try:
            ba._goto_battle_option(emu, option)
            assert menus.cursor_pos(emu) == expected, f"{option} landed at {menus.cursor_pos(emu)}"
        finally:
            emu.close()


def test_throw_ball_consumes_a_ball():
    _guard()
    from pokemon_agent.games.pokemon_red import battle_actions as ba
    from pokemon_agent.games.pokemon_red.game_state import read_items, read_party

    emu = _emu()
    try:
        balls_before = _qty(read_items(emu), "Poke Ball")
        party_before = len(read_party(emu))
        assert balls_before > 0

        res = ba.throw_ball(emu, "Poke Ball")
        assert res["ok"], res

        balls_after = _qty(read_items(emu), "Poke Ball")
        party_after = len(read_party(emu))
        # The ball was thrown (RAM-checkable, deterministic): one fewer ball.
        assert balls_after == balls_before - 1, res
        # Either it caught (party +1, battle over) or it broke free (party unchanged, the
        # turn resolved) — both are valid; the macro reports which.
        if res["caught"]:
            assert party_after == party_before + 1
            assert res["battle_over"]
        else:
            assert party_after == party_before
    finally:
        emu.close()


def test_use_item_potion_heals_and_is_consumed():
    _guard()
    from pokemon_agent.games.pokemon_red import battle_actions as ba
    from pokemon_agent.games.pokemon_red.game_state import read_items, read_party

    emu = _emu()
    try:
        hp_before = read_party(emu)[0]["hp"]
        pot_before = _qty(read_items(emu), "Potion")
        assert pot_before > 0 and hp_before < read_party(emu)[0]["max_hp"]

        res = ba.use_item(emu, "Potion")
        assert res["ok"], res

        assert _qty(read_items(emu), "Potion") == pot_before - 1, res
        assert read_party(emu)[0]["hp"] > hp_before, res
    finally:
        emu.close()


def test_run_leaves_the_battle_or_costs_the_turn():
    _guard()
    from pokemon_agent.games.pokemon_red import battle, battle_actions as ba

    emu = _emu()
    try:
        assert battle.in_battle(emu)
        res = ba.run(emu)
        assert res["ok"], res
        # escaped == in_battle now false; a failed run leaves us still in battle (turn spent).
        assert res["escaped"] == (not battle.in_battle(emu))
    finally:
        emu.close()
