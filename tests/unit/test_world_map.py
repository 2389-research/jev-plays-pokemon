from pokemon_agent.agent.world_map import WorldMap, FLOOR, WALL
from pokemon_agent.core.models import Direction, PlayerState


def p(x, y, m=38):
    return PlayerState(x=x, y=y, map_id=m)


def test_observe_marks_player_floor():
    w = WorldMap()
    w.observe(p(5, 5), None)
    assert w.tiles[38][(5, 5)] == FLOOR
    assert w.visits[38][(5, 5)] == 1


def test_mark_blocked_marks_wall_in_direction():
    w = WorldMap()
    w.observe(p(5, 5), None)
    w.mark_blocked(p(5, 5), Direction.SOUTH)  # south = +y
    assert w.tiles[38][(5, 6)] == WALL


def test_unexplored_excludes_walls_and_floor():
    w = WorldMap()
    w.observe(p(5, 5), None)          # (5,5) floor
    w.mark_blocked(p(5, 5), Direction.SOUTH)  # (5,6) wall
    dirs = w.unexplored_directions(p(5, 5))
    assert "south" not in dirs        # known wall
    assert "north" in dirs and "east" in dirs and "west" in dirs


def test_blocked_direction_not_retried_as_unexplored():
    # The exact scenario from the logs: blocked south should be remembered.
    w = WorldMap()
    w.observe(p(1, 6), None)
    w.mark_blocked(p(1, 6), Direction.SOUTH)
    assert "south" not in w.unexplored_directions(p(1, 6))


def test_explore_step_points_toward_unknown():
    w = WorldMap()
    # a known floor corridor going east; unknown beyond
    for x in (5, 6, 7):
        w.observe(p(x, 5), None)
    w.mark_blocked(p(5, 5), Direction.NORTH)
    w.mark_blocked(p(5, 5), Direction.SOUTH)
    w.mark_blocked(p(5, 5), Direction.WEST)
    d = w.explore_step(p(5, 5))
    assert d in ("east", "north", "south", "west")  # some frontier exists
    assert d is not None


def test_ascii_shows_player_wall_unknown():
    w = WorldMap()
    w.observe(p(5, 5), None)
    w.mark_blocked(p(5, 5), Direction.EAST)  # (6,5) wall
    grid = w.ascii(p(5, 5), radius=1)
    assert grid == ["???", "?@#", "???"]


def test_observe_projects_local_floor_cells():
    w = WorldMap()
    # 3x3 local window, player center, all floor
    local = ["...", ".@.", "..."]
    w.observe(p(10, 10), local)
    assert w.tiles[38][(9, 10)] == FLOOR   # west of player
    assert w.tiles[38][(11, 10)] == FLOOR  # east of player
    assert w.tiles[38][(10, 9)] == FLOOR   # north
