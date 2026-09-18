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
