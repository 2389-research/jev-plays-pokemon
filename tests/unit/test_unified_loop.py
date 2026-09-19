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
