"""Grind in place (spec 2026-09-23-grind-talk-shop-fixes-design F1): pace the grass on the grind map.

runs/brock-goals2-20260923: a grind step on Route 2 compiled to "travel to map 13"; already on map 13
the executor had no target, stood at (8,0) for 120 steps (0 battles) and wedged ~25x.
"""
from __future__ import annotations

from types import SimpleNamespace

from pokemon_agent.agent.routing import grind_step
from pokemon_agent.core.models import Direction

MAP = 13


def _world(rows: list[str]):
    """'G' grass, '.' floor, '#' wall."""
    from pokemon_agent.agent.world_map import WALL
    tiles, terr = {}, {}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            tiles[(x, y)] = WALL if ch == "#" else "floor"
            if ch == "G":
                terr[(x, y)] = "grass"
    return SimpleNamespace(tiles={MAP: tiles}, terrain={MAP: terr}, bounds={MAP: (len(rows[0]), len(rows))})


W = _world(["#######",
            "#GGG..#",
            "#GGG..#",
            "#.....#",
            "#######"])


def test_on_grass_keeps_going_straight():
    assert grind_step(W, MAP, (1, 1), Direction.EAST) == Direction.EAST


def test_on_grass_turns_at_the_edge_without_reversing():
    d = grind_step(W, MAP, (3, 1), Direction.EAST)     # (4,1) is floor -> turn, not back west
    assert d == Direction.SOUTH


def test_off_grass_heads_to_the_nearest_grass():
    assert grind_step(W, MAP, (5, 3), None) in (Direction.WEST, Direction.NORTH)


def test_no_grass_on_the_map_returns_none():
    bare = _world(["####", "#..#", "####"])
    assert grind_step(bare, MAP, (1, 1), None) is None


def test_unreachable_grass_returns_none():
    walled = _world(["#####", "#.#G#", "#####"])
    assert grind_step(walled, MAP, (1, 1), None) is None


def test_avoid_cells_are_not_grass_targets():
    one = _world(["#####", "#.G.#", "#####"])
    assert grind_step(one, MAP, (1, 1), None, avoid={(2, 1)}) is None


def test_a_long_pace_visits_more_than_two_tiles():
    pos, last, seen = (1, 1), None, set()
    from pokemon_agent.agent.world_map import DELTA
    for _ in range(12):
        d = grind_step(W, MAP, pos, last)
        pos = (pos[0] + DELTA[d][0], pos[1] + DELTA[d][1])
        last = d
        seen.add(pos)
        assert W.terrain[MAP].get(pos) == "grass"
    assert len(seen) >= 4       # sweeps the patch (no 2-tile ping-pong)


def test_the_no_encounter_window_follows_the_map_encounter_rate():
    """runs/sleeves-explore: a flat 60 gave up grinding in Viridian Forest (rate 8/256, ~32 steps per
    encounter) while sweeping its grass. The window is ~4.6x the expected steps for the map's rate."""
    from pokemon_agent.agent.reason_loop import GRIND_ENCOUNTER_WINDOW, grind_grass_window
    assert grind_grass_window(51) >= 140                        # Viridian Forest (8/256)
    assert grind_grass_window(13) == GRIND_ENCOUNTER_WINDOW     # Route 2 (25/256): the floor
    assert grind_grass_window(99999) == GRIND_ENCOUNTER_WINDOW  # unknown map


def test_caves_roll_encounters_on_every_tile_so_grinding_paces_anywhere():
    """Gen 1 (TryDoWildEncounter): indoor maps with wild data roll on every step unless they use the
    FOREST tileset. Mt. Moon has no grass tiles, so a grass-only grind would wedge 'no reachable grass'."""
    from pokemon_agent.agent.routing import grind_step
    from pokemon_agent.agent.world_map import WorldMap
    from pokemon_agent.games.pokemon_red.wild import encounters_anywhere, grass_rate
    assert encounters_anywhere(59) and grass_rate(59)                 # Mt. Moon 1F
    assert not encounters_anywhere(51) and grass_rate(51) == 8        # Viridian Forest: grass only
    assert not encounters_anywhere(12)                                # Route 1 (outdoor)
    w = WorldMap()
    w.ingest_collision(59, 6, 3, {(x, 1) for x in range(6)}, None, {})   # a cave corridor, no grass
    assert grind_step(w, 59, (2, 1), None) is None                        # grass-only: nothing to do
    assert grind_step(w, 59, (2, 1), None, anywhere=True) is not None     # cave: pace the corridor
