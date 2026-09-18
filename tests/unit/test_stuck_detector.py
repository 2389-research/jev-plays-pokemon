from pokemon_agent.actions.stuck_detector import StuckDetector
from pokemon_agent.core.models import ActionResult, Direction, GameMode, MoveAction, PlayerState


def _blocked():
    return ActionResult(
        success=False, result="blocked", mode_before=GameMode.OVERWORLD,
        mode_after=GameMode.OVERWORLD, player_moved=False,
    )


def _ok():
    return ActionResult(
        success=True, result="completed", mode_before=GameMode.OVERWORLD,
        mode_after=GameMode.OVERWORLD, player_moved=True,
    )


def test_stuck_after_threshold():
    det = StuckDetector(threshold=3)
    action = MoveAction(direction=Direction.NORTH)
    player = PlayerState(x=1, y=1, map_id=40)
    s1 = det.update(action, _blocked(), player, None)
    s2 = det.update(action, _blocked(), player, None)
    s3 = det.update(action, _blocked(), player, None)
    assert not s1.stuck and not s2.stuck
    assert s3.stuck and s3.repeat_count == 3


def test_progress_resets():
    det = StuckDetector(threshold=3)
    action = MoveAction(direction=Direction.NORTH)
    player = PlayerState(x=1, y=1, map_id=40)
    det.update(action, _blocked(), player, None)
    det.update(action, _blocked(), player, None)
    s = det.update(action, _ok(), PlayerState(x=1, y=0, map_id=40), None)
    assert not s.stuck and s.repeat_count == 0
