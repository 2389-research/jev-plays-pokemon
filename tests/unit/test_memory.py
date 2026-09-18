"""AgentMemory checkpoint round-trip + the new stuck-detector signals."""
from pokemon_agent.actions.stuck_detector import StuckDetector
from pokemon_agent.agent.memory import AgentMemory
from pokemon_agent.agent.plan import AgentPlan
from pokemon_agent.agent.world_map import FLOOR, WALL
from pokemon_agent.core.models import (
    ActionResult,
    Direction,
    GameMode,
    MoveAction,
    PlayerState,
)


def _blocked():
    return ActionResult(success=False, result="blocked", mode_before=GameMode.OVERWORLD,
                        mode_after=GameMode.OVERWORLD, player_moved=False)


def test_memory_checkpoint_round_trip(tmp_path):
    m = AgentMemory()
    m.world.tiles[40][(5, 5)] = FLOOR
    m.world.tiles[40][(5, 4)] = WALL
    m.interactions.talked.add((40, 5, 4))
    m.observe_map(0); m.observe_map(40)
    m.mark_tried_failed("goto ball from top-left")
    m.note("player has Squirtle", source="observed", step=3)
    m.plan = AgentPlan(milestone="get starter", next_objective="grab a ball", mode_hint="overworld")

    path = tmp_path / "mem.json"
    m.save(path)
    back = AgentMemory.load(path)

    assert back.world.tiles[40][(5, 5)] == FLOOR
    assert back.world.tiles[40][(5, 4)] == WALL
    assert (40, 5, 4) in back.interactions.talked
    assert back.map_history == [0, 40]
    assert back.tried_failed == ["goto ball from top-left"]
    assert back.notes[-1]["source"] == "observed"
    assert back.plan.objective == "grab a ball" and back.plan.mode_hint == "overworld"


def test_memory_has_seeded_route_pallet_to_pewter():
    m = AgentMemory()
    route = m.graph.route(0, 2)  # Pallet Town -> Pewter City
    assert route and route[0] == 0 and route[-1] == 2


def _det_step(det, progress=None, forced=False, dist=None, moved=False, pos=(1, 1)):
    action = MoveAction(direction=Direction.NORTH)
    res = ActionResult(success=moved, result="completed" if moved else "blocked",
                       mode_before=GameMode.OVERWORLD, mode_after=GameMode.OVERWORLD, player_moved=moved)
    return det.update(action, res, PlayerState(x=pos[0], y=pos[1], map_id=40), None,
                      progress=progress, forced_movement=forced, objective_distance=dist)


def test_forced_movement_suppresses_stuck():
    det = StuckDetector(threshold=2)
    for _ in range(5):
        s = _det_step(det, forced=True)
        assert not s.stuck and s.kind == "forced_movement_suppressed"


def test_objective_distance_wedge_and_reset():
    det = StuckDetector(wedge_threshold=5)
    # distance never improves (stuck at 4) -> wedge fires. Vary positions each step so the
    # position-stall detector doesn't trip first (we're isolating the objective-wedge here).
    s = None
    for k in range(7):
        s = _det_step(det, progress={"money": 100}, dist=4, moved=True, pos=(k, 0))
    assert s.stuck and s.kind == "no_objective_progress"
    # a decreasing distance resets the wedge
    det2 = StuckDetector(wedge_threshold=3)
    for k, d in enumerate((4, 3, 2, 1, 0)):
        s = _det_step(det2, progress={"money": 100}, dist=d, moved=True, pos=(k, 0))
    assert not s.stuck


def test_money_drop_flags_setback():
    det = StuckDetector()
    _det_step(det, progress={"money": 500}, moved=True)
    s = _det_step(det, progress={"money": 250}, moved=True)  # whiteout halves money
    assert s.setback is True


def test_blocked_edge_ledger_round_trips(tmp_path):
    m = AgentMemory()
    m.mark_blocked_edge(1, 19, 9, "north")
    assert m.is_blocked_edge(1, 19, 9, "north")
    assert not m.is_blocked_edge(1, 19, 9, "south")
    p = tmp_path / "m.json"
    m.save(p)
    assert AgentMemory.load(p).is_blocked_edge(1, 19, 9, "north")
