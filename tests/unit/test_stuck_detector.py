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


# ---- F4 (interaction-reliability spec): the objective budget is per plan step -------------------
def _wedge_to_threshold(det, *, dist):
    """Drive non-improving, non-oscillating updates (a new tile each step) until wedged."""
    action = MoveAction(direction=Direction.EAST)
    prog = {"badges": 0, "max_party_level": 8, "map_id": 40, "tiles_known": 50}
    s = None
    for i in range(det.wedge_threshold + 1):
        s = det.update(action, _ok(), PlayerState(x=i, y=1, map_id=40),
                       progress=prog, objective_distance=dist)
    return s, action, prog


def test_objective_wedge_persists_into_the_next_step_without_a_reset():
    """CHARACTERIZATION (documents the detector's behavior when nobody calls reset_objective; it is
    not a regression test for the fix, which is covered by the reset test below and the loop-level
    test that _commit_directive calls reset_objective). The incident: a later step whose target is farther can never count as 'getting closer'
    against the all-time best distance, so it is wedged from its very first update."""
    det = StuckDetector(wedge_threshold=5)
    s, action, prog = _wedge_to_threshold(det, dist=0)
    assert s.stuck and s.kind == "no_objective_progress"
    s2 = det.update(action, _ok(), PlayerState(x=20, y=1, map_id=40), progress=prog, objective_distance=3)
    assert s2.stuck   # today's behavior this fix removes (via reset_objective at step commit)


def test_reset_objective_gives_a_new_step_a_fresh_budget_and_keeps_exploration_progress():
    det = StuckDetector(wedge_threshold=5)
    s, action, prog = _wedge_to_threshold(det, dist=0)
    assert s.stuck
    best_sig = det._best_sig
    det.reset_objective()
    s2 = det.update(action, _ok(), PlayerState(x=20, y=1, map_id=40), progress=prog, objective_distance=3)
    assert not s2.stuck
    assert det._best_sig == best_sig   # cumulative exploration progress is not forgotten
