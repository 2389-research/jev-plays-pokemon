from pokemon_agent.agent.navigator import Navigator
from pokemon_agent.agent.world_map import FLOOR, WALL, WorldMap
from pokemon_agent.core.models import Direction, InteractAction, MoveAction, PlayerState

MAP = 37


def world(floors=(), walls=()):
    w = WorldMap()
    for xy in floors:
        w.tiles[MAP][xy] = FLOOR
    for xy in walls:
        w.tiles[MAP][xy] = WALL
    return w


def player(x, y, facing="south"):
    return PlayerState(x=x, y=y, map_id=MAP, facing=facing)


def test_adjacent_not_facing_turns_first():
    nav = Navigator(world(floors=[(5, 5)]))
    act, arrived = nav.step_toward(player(5, 5, "south"), {"x": 5, "y": 4, "interact": True})
    assert isinstance(act, MoveAction) and act.direction == Direction.NORTH and arrived is False


def test_adjacent_and_facing_interacts():
    nav = Navigator(world(floors=[(5, 5)]))
    act, arrived = nav.step_toward(player(5, 5, "north"), {"x": 5, "y": 4, "interact": True})
    assert isinstance(act, InteractAction) and arrived is True


def test_walks_toward_far_object():
    nav = Navigator(world(floors=[(5, 5), (5, 4), (5, 3), (5, 2)]))
    act, arrived = nav.step_toward(player(5, 5), {"x": 5, "y": 1, "interact": True})
    assert isinstance(act, MoveAction) and act.direction == Direction.NORTH and not arrived


def test_routes_around_a_wall():
    # direct north to the object is walled, and east/south are walled off too,
    # so the only route to an approach tile is the west detour.
    nav = Navigator(world(
        floors=[(5, 5), (4, 5), (4, 4), (4, 3)],
        walls=[(5, 4), (6, 5), (5, 6)],
    ))
    act, _ = nav.step_toward(player(5, 5), {"x": 5, "y": 3, "interact": True})
    assert isinstance(act, MoveAction) and act.direction == Direction.WEST


def test_exit_stand_on_tile():
    nav = Navigator(world(floors=[(5, 5), (5, 6), (5, 7)]))
    on = nav.step_toward(player(5, 7), {"x": 5, "y": 7, "interact": False})
    assert on == (None, True)
    off, arrived = nav.step_toward(player(5, 5), {"x": 5, "y": 7, "interact": False})
    assert isinstance(off, MoveAction) and off.direction == Direction.SOUTH and not arrived


def test_approaches_object_from_a_standable_side_not_an_adjacent_object():
    # target ball (11,7) sits between two other balls at (10,7) and (12,7). The nav
    # must not try to walk ONTO an adjacent ball to reach it — it routes to a free side.
    nav = Navigator(world(floors=[(10, 8), (11, 8), (11, 6)]))
    occupied = {(10, 7), (12, 7)}  # the neighbouring Poké Balls
    act, _ = nav.step_toward(player(10, 8), {"x": 11, "y": 7, "interact": True}, occupied)
    assert isinstance(act, MoveAction) and act.direction == Direction.EAST  # not NORTH onto (10,7)


def test_no_route_when_target_walled_off():
    walls = [(5, 0), (5, 2), (4, 1), (6, 1)]  # every approach tile blocked
    nav = Navigator(world(floors=[(5, 5)], walls=walls))
    assert nav.step_toward(player(5, 5), {"x": 5, "y": 1, "interact": True}) == (None, False)


def test_counter_talk_routes_across_counter():
    """An NPC behind a real counter tile is talked to from 2 tiles away in a straight line."""
    from pokemon_agent.agent.world_map import WorldMap, WALL, FLOOR
    from pokemon_agent.agent.navigator import Navigator
    from pokemon_agent.core.models import Direction, InteractAction, MoveAction, PlayerState

    w = WorldMap()
    # clerk at (0,5); counter WALL at (1,5); walkable floor at (2,5),(3,5)
    w.tiles[42].update({(0, 5): FLOOR, (1, 5): WALL, (2, 5): FLOOR, (3, 5): FLOOR})
    w.bounds[42] = (8, 8)
    w.counters[42] = {(1, 5)}
    nav = Navigator(w)
    # standing at (2,5) facing away -> should turn WEST to face the clerk across the counter
    act, arrived = nav.step_toward(PlayerState(x=2, y=5, map_id=42, facing="south"),
                                   {"x": 0, "y": 5, "interact": True})
    assert isinstance(act, MoveAction) and act.direction == Direction.WEST and not arrived
    # now facing west across the counter -> interact
    act, arrived = nav.step_toward(PlayerState(x=2, y=5, map_id=42, facing="west"),
                                   {"x": 0, "y": 5, "interact": True})
    assert isinstance(act, InteractAction) and arrived
    # a generic WALL (not a counter) must NOT be treated as talk-over
    w.counters[42] = set()
    act, arrived = nav.step_toward(PlayerState(x=2, y=5, map_id=42, facing="west"),
                                   {"x": 0, "y": 5, "interact": True})
    assert not (isinstance(act, InteractAction) and arrived)
