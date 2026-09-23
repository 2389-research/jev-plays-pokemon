"""select_npc (spec F3.2): choosing WHICH sprite an approach targets.

The incident: an `approach_npc Oak` target was proposed on a torn warp frame in Pallet Town and
cached `picked=[8,5]` (Pallet's hidden intro Oak). In the lab, locality tracking ran before the
name check and chose the sprite nearest (8,5) — an invisible, already-taken Poké Ball at (8,3) —
which the agent walked to and pressed A on 9 times.

`LAB_NPCS` is the REAL `read_npcs` output captured from `states/oak_tap_alias.state` before the
hidden-sprite filter (F2) landed, so it still contains the phantoms: Blue (4,3), the taken balls
(7,3)/(8,3), and the second Oak (5,10).
"""
from __future__ import annotations

from types import SimpleNamespace

from pokemon_agent.agent.targets import select_npc

LAB_NPCS = [
    {"x": 4, "y": 3, "sprite": "Blue", "kind": "person"},
    {"x": 6, "y": 3, "sprite": "Poke Ball", "kind": "item"},
    {"x": 7, "y": 3, "sprite": "Poke Ball", "kind": "item"},
    {"x": 8, "y": 3, "sprite": "Poke Ball", "kind": "item"},
    {"x": 5, "y": 2, "sprite": "Oak", "kind": "person"},
    {"x": 2, "y": 1, "sprite": "Pokedex", "kind": "person"},
    {"x": 3, "y": 1, "sprite": "Pokedex", "kind": "person"},
    {"x": 5, "y": 10, "sprite": "Oak", "kind": "person"},
    {"x": 1, "y": 10, "sprite": "Girl", "kind": "person"},
    {"x": 2, "y": 10, "sprite": "Scientist", "kind": "person"},
    {"x": 8, "y": 10, "sprite": "Scientist", "kind": "person"},
]
HIDDEN = {(4, 3), (7, 3), (8, 3), (5, 10)}   # what the missable flags hide in this state
LAB_VISIBLE = [n for n in LAB_NPCS if (n["x"], n["y"]) not in HIDDEN]
LAB = 40
PALLET = 0


def _p(x, y, m=LAB):
    return SimpleNamespace(x=x, y=y, map_id=m)


def test_stale_pick_from_another_map_never_selects_the_poke_ball():
    """The incident: a Pallet-coordinate pick must not drag the lab choice onto the ball at (8,3)."""
    npc = select_npc(LAB_NPCS, sprite="Oak", picked=[8, 5, PALLET], player=_p(5, 11), want_kind="person")
    assert npc["sprite"] == "Oak"


def test_with_hidden_sprites_filtered_the_real_oak_is_chosen():
    npc = select_npc(LAB_VISIBLE, sprite="Oak", picked=[8, 5, PALLET], player=_p(5, 11), want_kind="person")
    assert (npc["x"], npc["y"]) == (5, 2)


def test_legacy_two_element_pick_is_ignored():
    npc = select_npc(LAB_VISIBLE, sprite="Oak", picked=[8, 5], player=_p(5, 11), want_kind="person")
    assert (npc["x"], npc["y"]) == (5, 2)


def test_same_map_pick_tracks_a_moving_npc_within_the_name_pool():
    npcs = [n if n["sprite"] != "Oak" else {**n, "x": 6} for n in LAB_VISIBLE]   # Oak stepped to (6,2)
    npc = select_npc(npcs, sprite="Oak", picked=[5, 2, LAB], player=_p(5, 8), want_kind="person")
    assert (npc["sprite"], npc["x"], npc["y"]) == ("Oak", 6, 2)


def test_unnamed_talk_never_returns_an_item_when_a_person_exists():
    # the remaining ball (6,3) is the nearest sprite to (6,4), but a person was asked for
    npc = select_npc(LAB_VISIBLE, sprite=None, picked=None, player=_p(6, 4), want_kind="person")
    assert npc["kind"] != "item"


def test_unnamed_grab_item_returns_the_item_ball():
    npc = select_npc(LAB_VISIBLE, sprite="Potion", picked=None, player=_p(5, 8), want_kind="item")
    assert (npc["sprite"], npc["x"], npc["y"]) == ("Poke Ball", 6, 3)


def test_chooser_only_sees_the_candidate_pool():
    seen = {}

    def chooser(pool):
        seen["pool"] = [(n["sprite"], n["x"], n["y"]) for n in pool]
        return None   # not confident -> fall through to nearest

    select_npc(LAB_VISIBLE, sprite=None, picked=None, player=_p(5, 8), want_kind="person", chooser=chooser)
    assert all(s != "Poke Ball" for s, _, _ in seen["pool"])


def test_sprites_without_kind_count_as_people():
    npcs = [{"x": 3, "y": 2, "sprite": "Rival"}, {"x": 3, "y": 6, "sprite": "Oak"}]
    npc = select_npc(npcs, sprite=None, picked=None, player=_p(3, 5, 0), want_kind="person")
    assert npc["sprite"] == "Oak"   # nearest person


def test_no_npcs_returns_none():
    assert select_npc([], sprite="Oak", picked=None, player=_p(0, 0), want_kind="person") is None
