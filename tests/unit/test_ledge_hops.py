"""Ledge hopping as a planned one-way shortcut (live on Route 1: south to Pallet 36 moves with 4 hops vs
55 walking around; north is unchanged — a ledge can't be climbed)."""
from __future__ import annotations

from pokemon_agent.agent.navigator import Navigator, ledge_hops
from pokemon_agent.agent.world_map import WorldMap
from pokemon_agent.core.models import Direction


def _world():
    """A 5-wide strip: a south ledge row at y=3 (x 0..3), a gap at x=4 to walk around."""
    w = WorldMap()
    walk = {(x, y) for x in range(5) for y in range(7)} - {(x, 3) for x in range(4)}
    terrain = {(x, 3): "ledge_s" for x in range(4)}
    w.ingest_collision(1, 5, 7, walk, None, terrain)
    return w


def test_ledge_hops_map_takeoff_to_landing():
    hops = ledge_hops({(2, 3): "ledge_s", (5, 5): "ledge_e", (7, 1): "ledge_w", (0, 0): "grass"})
    assert hops[(2, 2)] == [(Direction.SOUTH, (2, 4))]
    assert hops[(4, 5)] == [(Direction.EAST, (6, 5))]
    assert hops[(8, 1)] == [(Direction.WEST, (6, 1))]
    assert (0, 0) not in hops and (-1, 0) not in hops


def test_going_down_hops_the_ledge_going_up_walks_around():
    nav = Navigator(_world())
    assert nav._bfs_first_step(1, (0, 2), {(0, 6)}) == Direction.SOUTH and nav.first_is_hop    # hop down
    assert nav._bfs_first_step(1, (0, 4), {(0, 0)}) != Direction.NORTH and not nav.first_is_hop  # never climb
