from types import SimpleNamespace

from pokemon_agent.agent.planner_llm import Planner
from pokemon_agent.agent.plan import Directive, Intent
from pokemon_agent.core.models import Direction, InteractAction, MoveAction


class FakeProvider:
    def __init__(self, content): self.content = content
    def chat_json(self, system, state, image=None): return self.content, 0, {}


def _ctx(**kw):
    base = {"player": {"x": 3, "y": 3, "map_id": 1}, "reachable": {(3, 3), (3, 4), (4, 3)},
            "exit_tile": (4, 3), "map_view": ["..."], "objective": "go", "destination": "map 0",
            "goal_dir": "south", "npcs": [], "recent_trail": [], "recent_targets": [],
            "default": {"kind": "exit"}, "stuck": False, "why": "pick"}
    base.update(kw); return base


def test_propose_target_no_provider_returns_deterministic_default():
    # OFFLINE / tests: running with no LLM is intentional, not an error -> deterministic default.
    p = Planner(goal_map=0, provider=None)
    assert p.propose_target(emu=None, context=_ctx()) == {"kind": "exit"}


def test_propose_target_parses_tile():
    p = Planner(goal_map=0, provider=FakeProvider('{"kind":"tile","x":3,"y":4,"note":"south"}'))
    t = p.propose_target(emu=None, context=_ctx())
    assert t["kind"] == "tile" and (t["x"], t["y"]) == (3, 4) and t["note"] == "south"


def test_propose_target_configured_but_unreachable_tile_returns_unresolved():
    # a provider IS wired (live run) but it named an unreachable tile -> this is a model failure;
    # BREAK LOUDLY, do NOT silently substitute the deterministic default.
    p = Planner(goal_map=0, provider=FakeProvider('{"kind":"tile","x":9,"y":9,"note":"bad"}'))
    t = p.propose_target(emu=None, context=_ctx())
    assert t["kind"] == "unresolved" and "note" in t


def test_propose_target_passes_through_exit_and_approach_and_enter():
    for content, kind in [('{"kind":"exit","note":"leave"}', "exit"),
                          ('{"kind":"approach_npc","sprite":"Oak","note":"talk"}', "approach_npc"),
                          ('{"kind":"enter","map":1,"note":"door"}', "enter")]:
        p = Planner(goal_map=0, provider=FakeProvider(content))
        assert p.propose_target(emu=None, context=_ctx())["kind"] == kind


def test_propose_target_configured_but_bad_json_returns_unresolved():
    # provider wired but returns garbage -> model failure -> unresolved (flagged), not a guess.
    p = Planner(goal_map=0, provider=FakeProvider("not json"))
    assert p.propose_target(emu=None, context=_ctx())["kind"] == "unresolved"


def test_propose_target_approach_npc_without_sprite_returns_unresolved():
    p = Planner(goal_map=0, provider=FakeProvider('{"kind":"approach_npc","note":"talk"}'))
    assert p.propose_target(emu=None, context=_ctx())["kind"] == "unresolved"


def test_propose_target_enter_without_map_returns_unresolved():
    p = Planner(goal_map=0, provider=FakeProvider('{"kind":"enter","note":"door"}'))
    assert p.propose_target(emu=None, context=_ctx())["kind"] == "unresolved"


def test_propose_target_unresolved_payload_has_reason_and_note():
    p = Planner(goal_map=0, provider=FakeProvider("not json"))
    t = p.propose_target(emu=None, context=_ctx())
    assert t["kind"] == "unresolved" and t.get("reason") and t.get("note")


def _nav_loop(map_id=42):
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import ReasonStep, ReflectionPlan
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import GoalState, WaitAction
    from pokemon_agent.emulator.fake_emulator import FakeEmulator
    from pokemon_agent.observations.builder import ObservationBuilder

    class Stub:
        def reflect(self, **k): return ReflectionPlan(), 0, {}
        def step(self, **k): return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}
    emu = FakeEmulator(map_id=map_id)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=Stub(), session=Session(GoalState(primary="p", current="p")),
                         vision=False, reflect_every=100, goal_map=99)
    return loop, emu


def _d(intent=Intent.TRAVEL):
    return Directive(intent=intent, target={"kind": "map", "map": 40}, success={"on_map": 40})


def test_resolve_enter_steps_through_door_when_on_it():
    loop, _ = _nav_loop(42)
    player = SimpleNamespace(x=3, y=7, map_id=42, facing="east")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    move = loop._resolve_target({"kind": "enter", "map": 1}, _d(), obs, set(), set())
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH


def test_resolve_exit_routes_to_nearest_exit_door():
    loop, _ = _nav_loop(42)
    cells = {(x, y) for x in range(4) for y in range(8)}
    loop.world.ingest_collision(42, 4, 8, cells, None, None)
    player = SimpleNamespace(x=1, y=1, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    move = loop._resolve_target({"kind": "exit"}, _d(), obs, set(), set())
    assert isinstance(move, MoveAction) and move.direction in (Direction.EAST, Direction.SOUTH)


def test_resolve_approach_npc_interacts_when_adjacent_and_facing():
    loop, _ = _nav_loop(0)
    player = SimpleNamespace(x=3, y=3, map_id=0, facing="north")
    obs = SimpleNamespace(player=player, map_dims=(6, 6),
                          game_state={"npcs": [{"x": 3, "y": 2, "sprite": "Oak"}]}, exits=[])
    move = loop._resolve_target({"kind": "approach_npc", "sprite": "Oak"}, _d(Intent.TALK_TO),
                                obs, set(), set())
    assert isinstance(move, InteractAction)


def test_resolve_tile_interacts_on_arrival_when_flagged():
    loop, _ = _nav_loop(0)
    player = SimpleNamespace(x=2, y=2, map_id=0, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(6, 6), game_state={}, exits=[])
    move = loop._resolve_target({"kind": "tile", "x": 2, "y": 2, "interact": True}, _d(Intent.TALK_TO),
                                obs, set(), set())
    assert isinstance(move, InteractAction)


def test_resolve_tile_that_is_an_exit_door_steps_through():
    loop, _ = _nav_loop(42)
    player = SimpleNamespace(x=3, y=7, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    move = loop._resolve_target({"kind": "tile", "x": 3, "y": 7}, _d(), obs, set(), set())
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH


def test_resolve_target_unknown_or_unresolved_kind_returns_none():
    # the anti-wander guard: an unknown/unresolved kind must NOT route to an exit even when a door
    # is present -> returns None (break loudly / stall, never silently wander).
    loop, _ = _nav_loop(42)
    player = SimpleNamespace(x=1, y=1, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    assert loop._resolve_target({"kind": "unresolved"}, _d(), obs, set(), set()) is None
    assert loop._resolve_target({"kind": "bogus"}, _d(), obs, set(), set()) is None


def test_resolve_edge_steps_off_boundary_toward_goal_dir():
    loop, _ = _nav_loop(1)
    loop.world.ingest_collision(1, 4, 8, {(x, y) for x in range(4) for y in range(8)}, None, None)
    player = SimpleNamespace(x=2, y=7, map_id=1, facing="south")   # on the south boundary (h-1 == 7)
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={}, exits=[])
    move = loop._resolve_target({"kind": "edge", "dir": "south", "next_map": 0}, _d(), obs, set(), set())
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH


def test_default_target_cross_map_no_hop_is_exit():
    loop, _ = _nav_loop(42)
    loop.memory.graph.next_hop = lambda a, b: None
    player = SimpleNamespace(x=1, y=1, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={}, exits=[])
    assert loop._default_target(_d(), obs)["kind"] == "exit"


def test_default_target_cross_map_with_door_is_enter():
    loop, _ = _nav_loop(42)
    loop.memory.graph.next_hop = lambda a, b: (1, (3, 7))
    player = SimpleNamespace(x=1, y=1, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    t = loop._default_target(_d(), obs)
    assert t["kind"] == "enter" and t["map"] == 1


def test_navigate_leg_never_freezes_without_hop_uses_exit():
    # the lab-freeze case: no known route out (0xFF door unresolved) -> must MOVE, not return None
    loop, _ = _nav_loop(42)
    loop.memory.graph.next_hop = lambda a, b: None
    cells = {(x, y) for x in range(4) for y in range(8)}
    loop.world.ingest_collision(42, 4, 8, cells, None, None)
    player = SimpleNamespace(x=1, y=1, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    move = loop._navigate_leg(_d(), obs, set())
    assert isinstance(move, MoveAction)  # routed toward the exit door, not frozen


def test_navigate_leg_door_step_through_still_works():
    loop, _ = _nav_loop(42)
    loop.memory.graph.next_hop = lambda a, b: (1, (3, 7))
    player = SimpleNamespace(x=3, y=7, map_id=42, facing="east")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    move = loop._navigate_leg(_d(), obs, set())
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH


class FailProvider:
    def chat_json(self, system, state, image=None):
        raise RuntimeError("boom")


def test_navigate_leg_configured_failure_ungrounded_stalls_and_flags():
    loop, _ = _nav_loop(42)
    loop.planner.provider = FailProvider()
    loop.memory.graph.next_hop = lambda a, b: None
    cells = {(x, y) for x in range(4) for y in range(8)}
    loop.world.ingest_collision(42, 4, 8, cells, None, None)
    events = []
    loop.on_event = lambda k, p: events.append((k, p))
    player = SimpleNamespace(x=1, y=1, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={}, map_view=["x"],
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    move = loop._navigate_leg(_d(), obs, set())
    assert move is None
    assert any(k == "proposer_failed" for k, _ in events)


def test_navigate_leg_configured_failure_grounded_proceeds_and_flags():
    loop, _ = _nav_loop(42)
    loop.planner.provider = FailProvider()
    loop.memory.graph.next_hop = lambda a, b: (1, (3, 7))
    player = SimpleNamespace(x=3, y=7, map_id=42, facing="east")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={}, map_view=["x"],
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    events = []
    loop.on_event = lambda k, p: events.append((k, p))
    move = loop._navigate_leg(_d(), obs, set())
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH
    assert any(k == "proposer_failed" for k, _ in events)


# --- THE REGRESSION TESTS: a reflect/proposer model's stated target is actually navigated to ---
import json as _json

_DELTA = {Direction.NORTH: (0, -1), Direction.SOUTH: (0, 1), Direction.EAST: (1, 0), Direction.WEST: (-1, 0)}


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _applied(player, move):
    dx, dy = _DELTA[move.direction]
    return (player.x + dx, player.y + dy)


class ProposerProvider:
    """A provider whose chat_json emits the exact target a reflect/proposer model would name."""
    def __init__(self, obj): self._obj = obj
    def chat_json(self, system, state, image=None): return _json.dumps(self._obj), 0, {}


def _lab_like_obs(px, py):
    player = SimpleNamespace(x=px, y=py, map_id=42, facing="south")
    return SimpleNamespace(player=player, map_dims=(4, 8), game_state={}, map_view=["room"],
                           exits=[{"x": 3, "y": 7, "dest_map": 1}])


def test_reflect_style_exit_proposal_is_navigated_toward_the_door():
    loop, _ = _nav_loop(42)
    loop.planner.provider = ProposerProvider({"kind": "exit", "note": "head south to the exit and leave"})
    loop.memory.graph.next_hop = lambda a, b: None
    loop.world.ingest_collision(42, 4, 8, {(x, y) for x in range(4) for y in range(8)}, None, None)
    obs = _lab_like_obs(1, 1)
    move = loop._navigate_leg(_d(), obs, set())
    assert isinstance(move, MoveAction)
    assert _manhattan(_applied(obs.player, move), (3, 7)) < _manhattan((1, 1), (3, 7))


def test_reflect_style_named_tile_proposal_is_navigated_toward():
    loop, _ = _nav_loop(42)
    loop.planner.provider = ProposerProvider({"kind": "tile", "x": 3, "y": 7, "note": "go to the exit at (3,7)"})
    loop.memory.graph.next_hop = lambda a, b: None
    loop.world.ingest_collision(42, 4, 8, {(x, y) for x in range(4) for y in range(8)}, None, None)
    obs = _lab_like_obs(1, 1)
    move = loop._navigate_leg(_d(), obs, set())
    assert isinstance(move, MoveAction)
    assert _manhattan(_applied(obs.player, move), (3, 7)) < _manhattan((1, 1), (3, 7))


def test_reflect_style_proposal_emits_target_event_and_folds_note():
    from pokemon_agent.agent.reasoner import ReflectionPlan
    loop, _ = _nav_loop(42)
    loop._plan = ReflectionPlan()  # ensure the note is assignable in the fake setup
    loop.planner.provider = ProposerProvider({"kind": "exit", "note": "leave the lab, gym is north"})
    loop.memory.graph.next_hop = lambda a, b: None
    loop.world.ingest_collision(42, 4, 8, {(x, y) for x in range(4) for y in range(8)}, None, None)
    events = []
    loop.on_event = lambda k, p: events.append((k, p))
    loop._navigate_leg(_d(), _lab_like_obs(1, 1), set())
    assert any(k == "target" for k, _ in events)
    assert loop._plan is not None and "leave the lab" in (loop._plan.next_objective or "")


def test_dispatch_servo_talk_without_tile_uses_navigate_leg():
    loop, _ = _nav_loop(0)
    calls = []
    loop._navigate_leg = lambda d, o, b: calls.append("nav")
    loop._servo_step = lambda d, o, b: calls.append("servo")
    d_notile = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 0}, success={"talked_on_map": 0})
    d_tile = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 0, "x": 2, "y": 2},
                       success={"talked_on_map": 0})
    loop._dispatch_servo(d_notile, None, set())
    loop._dispatch_servo(d_tile, None, set())
    assert calls == ["nav", "servo"]  # no-tile talk -> approach_npc via nav; tile talk -> servo
