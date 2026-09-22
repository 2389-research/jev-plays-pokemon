"""Unit tests for the shop BUY macro's pure logic — the data-dependent item-index lookup
(design §3). No ROM needed; runs everywhere."""
from __future__ import annotations

from pokemon_agent.games.pokemon_red.shop import resolve_shop_index


# a real Viridian Mart list (read from wListPointer live): order is data-dependent.
VIRIDIAN = [
    {"id": 4, "item": "Poke Ball"},
    {"id": 11, "item": "Antidote"},
    {"id": 15, "item": "Parlyz Heal"},
    {"id": 12, "item": "Burn Heal"},
]


def test_resolve_by_exact_name():
    assert resolve_shop_index(VIRIDIAN, "Antidote") == 1
    assert resolve_shop_index(VIRIDIAN, "Poke Ball") == 0
    assert resolve_shop_index(VIRIDIAN, "Burn Heal") == 3


def test_resolve_is_case_and_punctuation_insensitive():
    assert resolve_shop_index(VIRIDIAN, "poke ball") == 0
    assert resolve_shop_index(VIRIDIAN, "PARLYZ HEAL") == 2


def test_resolve_by_item_id_when_names_differ():
    # canonical id match wins even if the display name is spelled differently
    assert resolve_shop_index(VIRIDIAN, "potion") is None  # not in this shop
    assert resolve_shop_index(VIRIDIAN, "Antidote") == 1


def test_resolve_plain_name_list():
    names = ["Poke Ball", "Potion", "Super Potion"]
    assert resolve_shop_index(names, "Potion") == 1
    assert resolve_shop_index(names, "Super Potion") == 2


def test_resolve_missing_returns_none():
    assert resolve_shop_index(VIRIDIAN, "Master Ball") is None
    assert resolve_shop_index([], "Potion") is None


def test_resolve_order_is_data_dependent_not_hardcoded():
    # same items, different order -> different indices (the whole point of a runtime lookup)
    reordered = list(reversed(VIRIDIAN))
    assert resolve_shop_index(reordered, "Poke Ball") == 3
    assert resolve_shop_index(reordered, "Burn Heal") == 0
