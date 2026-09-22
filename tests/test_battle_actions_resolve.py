"""Unit tests (no ROM) for the runtime bag item-index resolution (design §3).

`resolve_item_index` computes a *specific* item's CURRENT index from the live bag every
call — bag order is data-dependent, so it must never be hardcoded. When a name matches
several entries it surfaces all candidates (a later phase lets Jev pick) and defaults to
the first; no accent-folding heuristics.
"""
from __future__ import annotations

from pokemon_agent.games.pokemon_red.battle_actions import resolve_item_index

BAG = [
    {"item": "Antidote", "qty": 2},
    {"item": "Poke Ball", "qty": 5},
    {"item": "Potion", "qty": 3},
    {"item": "Great Ball", "qty": 2},
]


def test_resolves_exact_id_regardless_of_position():
    # Potion sits at index 2 in this bag; the index is looked up, not assumed.
    res = resolve_item_index(BAG, "Potion")
    assert res["index"] == 2
    assert res["item"] == "Potion"


def test_ball_names_are_distinct_not_conflated():
    # "Poke Ball" and "Great Ball" are different items; the id tier keeps them apart.
    assert resolve_item_index(BAG, "Poke Ball")["index"] == 1
    assert resolve_item_index(BAG, "Great Ball")["index"] == 3


def test_missing_item_returns_none():
    assert resolve_item_index(BAG, "Master Ball") is None


def test_index_tracks_bag_order_not_a_constant():
    reordered = list(reversed(BAG))  # Potion now at index 1
    assert resolve_item_index(reordered, "Potion")["index"] == 1


def test_ambiguous_name_exposes_candidates_and_picks_first():
    # A fuzzy intent like "ball" matches both balls; the macro surfaces both and defaults
    # to the first so Jev (a later phase) can disambiguate — no heuristic accent-folding.
    res = resolve_item_index(BAG, "Ball")
    idxs = [c["index"] for c in res["candidates"]]
    assert set(idxs) == {1, 3}
    assert res["index"] == idxs[0]  # first candidate is the default


def test_accepts_plain_name_strings():
    res = resolve_item_index(["Antidote", "Poke Ball", "Potion"], "Poke Ball")
    assert res["index"] == 1
    assert res["item"] == "Poke Ball"
