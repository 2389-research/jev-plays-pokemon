"""Deterministic, RAM-checked BUY macro (design §5, §6.1): from an open Mart counter,
`shop_buy` must add the item to the bag and debit the price — no LLM, no network.

Fixture `states/mart_counter.state` is the Viridian Mart (map 42) with the BUY/SELL/QUIT
menu open (created live by walking to the clerk and talking over the counter). states/*.state
are gitignored, so this skips when the fixture/ROM is absent — like the other live tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROM_PATH = ROOT / "roms" / "pokemon_red.gb"
STATE_PATH = ROOT / "states" / "mart_counter.state"


def _emu():
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator

    emu = PyBoyEmulator(str(ROM_PATH), window="null")
    emu.load_state(STATE_PATH)
    emu.tick(6)
    return emu


def test_buy_adds_items_and_debits_money():
    if not ROM_PATH.exists() or not STATE_PATH.exists():
        pytest.skip("ROM/mart-counter fixture not present")
    from pokemon_agent.games.pokemon_red import shop
    from pokemon_agent.games.pokemon_red.game_state import read_items, read_money

    emu = _emu()
    try:
        from pokemon_agent.games.pokemon_red.shop import at_shop_menu

        assert at_shop_menu(emu), "fixture must sit at the BUY/SELL/QUIT counter menu"

        def qty_of(items, name):
            return next((it["qty"] for it in items if it["item"].lower() == name.lower()), 0)

        money_before = read_money(emu)
        antidote_before = qty_of(read_items(emu), "Antidote")

        res = shop.shop_buy(emu, "Antidote", 3)
        assert res["ok"], res

        items_after = read_items(emu)
        assert qty_of(items_after, "Antidote") == antidote_before + 3
        # Antidote is 100 each in the Viridian Mart -> money drops by 3 * price
        spent = money_before - read_money(emu)
        assert spent == 3 * 100, f"expected -300, spent {spent}"

        # and the macro left the shop (back in the overworld, no menu, not in battle)
        from pokemon_agent.games.pokemon_red.menus import menu_open

        assert not menu_open(emu)
        assert emu.read_memory(0xD057) == 0  # not in battle
    finally:
        emu.close()


def test_buy_item_not_sold_is_a_clean_noop():
    if not ROM_PATH.exists() or not STATE_PATH.exists():
        pytest.skip("ROM/mart-counter fixture not present")
    from pokemon_agent.games.pokemon_red import shop
    from pokemon_agent.games.pokemon_red.game_state import read_money

    emu = _emu()
    try:
        money_before = read_money(emu)
        res = shop.shop_buy(emu, "Master Ball", 1)  # not on the Viridian shelf
        assert res["ok"] is False
        assert read_money(emu) == money_before  # nothing spent
    finally:
        emu.close()


def test_read_shop_list_matches_the_viridian_shelf():
    if not ROM_PATH.exists() or not STATE_PATH.exists():
        pytest.skip("ROM/mart-counter fixture not present")
    from pokemon_agent.games.pokemon_red import menus, shop

    emu = _emu()
    try:
        # enter the BUY list, then the live shop list is readable from wListPointer RAM
        menus.select_option(emu, 0)  # BUY
        shop._wait_menu(emu)
        names = [s["item"] for s in shop.read_shop_list(emu)]
        assert names == ["Poke Ball", "Antidote", "Parlyz Heal", "Burn Heal"]
        assert shop.resolve_shop_index(shop.read_shop_list(emu), "Parlyz Heal") == 2
    finally:
        emu.close()
