from pokemon_agent.actions.controller import ActionController
from pokemon_agent.core.models import Direction, MoveAction, PressAction, GameButton
from pokemon_agent.emulator.fake_emulator import FakeEmulator


def test_move_completed_reports_moved():
    emu = FakeEmulator(start=(2, 1))
    ctrl = ActionController(emu)
    res = ctrl.execute(MoveAction(direction=Direction.EAST, tiles=1))
    assert res.result == "completed"
    assert res.success is True
    assert res.player_moved is True


def test_move_into_wall_is_blocked_not_success():
    emu = FakeEmulator(start=(1, 1))
    ctrl = ActionController(emu)
    res = ctrl.execute(MoveAction(direction=Direction.WEST, tiles=1))
    assert res.result == "blocked"
    assert res.success is False
    assert res.player_moved is False
    assert any("movement_blocked" in e for e in res.events)


def test_multi_tile_stops_at_wall():
    # From (1,1) moving east across "#......#": can go to x=6 then hits wall at x=7
    emu = FakeEmulator(start=(1, 1))
    ctrl = ActionController(emu)
    res = ctrl.execute(MoveAction(direction=Direction.EAST, tiles=10))
    assert res.success is True
    assert emu.x == 6  # stopped against the right wall
    assert any("movement_blocked" in e for e in res.events)


def test_press_returns_completed():
    emu = FakeEmulator()
    ctrl = ActionController(emu)
    res = ctrl.execute(PressAction(button=GameButton.A))
    assert res.result == "completed"
    assert res.events == ["pressed:a"]
