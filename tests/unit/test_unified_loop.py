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


def test_candidate_exits_lists_doors_and_reachable_edge_openings():
    # L2 should be handed every way OFF the map as a coordinate: warp doors (with dest) AND reachable
    # map-boundary openings (edge tiles), so it can SELECT one instead of us guessing the nearest door.
    loop, _ = _nav_loop(0)
    obs = SimpleNamespace(map_dims=(6, 8),
                          exits=[{"x": 2, "y": 7, "dest_map": 40, "dest_name": "Oaks Lab"}])
    reachable = {(2, 7), (3, 0), (0, 4), (2, 3)}  # door tile, north-edge, west-edge, interior tile
    cands = loop._candidate_exits(obs, reachable)
    doors = [c for c in cands if c["kind"] == "door"]
    edges = {(c["x"], c["y"], c["dir"]) for c in cands if c["kind"] == "edge"}
    assert doors == [{"x": 2, "y": 7, "kind": "door", "dest_map": 40, "dest": "Oaks Lab"}]
    assert edges == {(3, 0, "N"), (0, 4, "W")}          # boundary openings, with direction
    assert (2, 3) not in {(c["x"], c["y"]) for c in cands}  # interior tile is not a way off


def test_propose_target_accepts_bare_coordinate_with_why():
    # L2's primary output is just a coordinate + why — no "kind" needed.
    p = Planner(goal_map=0, provider=FakeProvider('{"x":3,"y":4,"why":"head south toward the north edge exit"}'))
    t = p.propose_target(emu=None, context=_ctx())
    assert t["kind"] == "tile" and (t["x"], t["y"]) == (3, 4)
    assert "south" in t["note"]


def test_resolve_tile_at_map_edge_steps_off_to_cross():
    # routing to a boundary opening tile: on arrival the ROUTER steps off the edge to cross maps.
    loop, _ = _nav_loop(0)
    player = SimpleNamespace(x=3, y=0, map_id=0, facing="north")   # already ON the north boundary
    obs = SimpleNamespace(player=player, map_dims=(6, 8), exits=[], game_state={})
    move = loop._resolve_target({"kind": "tile", "x": 3, "y": 0}, _d(Intent.TRAVEL), obs, set(), set())
    assert isinstance(move, MoveAction) and move.direction == Direction.NORTH


def test_approach_counter_npc_bumps_then_talks_across_the_counter():
    # A counter NPC (nurse at 3,1) sits behind a COUNTER cell (3,2): you can't stand adjacent, you
    # talk from 2 tiles away (3,3) — but ONLY after bumping the counter (a blocked step into it).
    loop, _ = _nav_loop(41)
    walk = {(x, y) for x in range(6) for y in range(8)}
    loop.world.ingest_collision(41, 6, 8, walk, {(3, 2)}, None)   # (3,2) is a counter cell
    player = SimpleNamespace(x=3, y=3, map_id=41, facing="north")  # at the across-counter tile, facing it
    obs = SimpleNamespace(player=player, map_dims=(6, 8),
                          game_state={"npcs": [{"x": 3, "y": 1, "sprite": "Nurse"}]}, exits=[])
    tgt = {"kind": "approach_npc", "sprite": "Nurse"}
    loop._counter_bumped = False
    bump = loop._approach_npc(tgt, _d(Intent.TALK_TO), obs, set(), set())
    assert isinstance(bump, MoveAction) and bump.direction == Direction.NORTH   # bump the counter first
    assert loop._counter_bumped is True
    talk = loop._approach_npc(tgt, _d(Intent.TALK_TO), obs, set(), set())
    assert isinstance(talk, InteractAction)                                     # then talk across it


def test_approach_non_counter_npc_two_away_routes_closer_not_talks():
    # WITHOUT a counter between them, an NPC 2 tiles away is not talkable over the gap: the agent
    # must route to the 1-adjacent tile, not interact across open ground.
    loop, _ = _nav_loop(41)
    walk = {(x, y) for x in range(6) for y in range(8)}
    loop.world.ingest_collision(41, 6, 8, walk, set(), None)      # no counters
    player = SimpleNamespace(x=3, y=3, map_id=41, facing="north")
    obs = SimpleNamespace(player=player, map_dims=(6, 8),
                          game_state={"npcs": [{"x": 3, "y": 1, "sprite": "Nurse"}]}, exits=[])
    move = loop._approach_npc({"kind": "approach_npc", "sprite": "Nurse"},
                              _d(Intent.TALK_TO), obs, set(), set())
    assert isinstance(move, MoveAction)          # routes toward the adjacent tile (3,2), no talk-over


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
    loop._navigate_leg = lambda d, o, b: "nav-result"
    loop._servo_step = lambda d, o, b: "servo-result"
    d_notile = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 0}, success={"talked_on_map": 0})
    d_tile = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 0, "x": 2, "y": 2},
                       success={"talked_on_map": 0})
    d_travel = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 5}, success={"on_map": 5})
    # no-tile talk -> nav (approach_npc); tile talk -> servo; travel -> nav. And the servo's RETURN
    # value is passed through (step_once branches on it).
    assert loop._dispatch_servo(d_notile, None, set()) == "nav-result"
    assert loop._dispatch_servo(d_tile, None, set()) == "servo-result"
    assert loop._dispatch_servo(d_travel, None, set()) == "nav-result"


def test_strategize_talk_step_carries_sprite_name():
    import json as _json
    plan = {"plan": "deliver", "steps": [{"map": 40, "talk": True, "who": "Oak",
                                          "done_when": "no_item:Oak's Parcel", "why": "hand it over"}]}
    class P:
        def chat_json(self, system, state, image=None): return _json.dumps(plan), 0, {}
    from pokemon_agent.agent.planner_llm import Planner

    class FakeEmu:
        def read_memory(self, a): return 0
    pl = Planner(goal_map=40, strategist=P())
    import pokemon_agent.agent.planner_llm as m
    pl_party, pl_items, pl_badges = m.read_party, m.read_items, m.read_badges
    m.read_party = lambda e: []; m.read_items = lambda e: []; m.read_badges = lambda e: {"count": 0}
    try:
        quest = pl.strategize(FakeEmu(), memory=None, why="deliver the parcel")
    finally:
        m.read_party, m.read_items, m.read_badges = pl_party, pl_items, pl_badges
    talk = [d for d in quest if d.intent.name == "TALK_TO"]
    assert talk and (talk[0].target or {}).get("sprite") == "Oak"


def test_approach_npc_substring_matches_named_sprite():
    # a free-form who ("the Mart clerk") must reach the RAM sprite label "Clerk" via substring match
    loop, _ = _nav_loop(0)
    player = SimpleNamespace(x=3, y=3, map_id=0, facing="north")
    obs = SimpleNamespace(player=player, map_dims=(6, 6),
                          game_state={"npcs": [{"x": 3, "y": 2, "sprite": "Clerk"}]}, exits=[])
    move = loop._resolve_target({"kind": "approach_npc", "sprite": "the Mart clerk"},
                                _d(Intent.TALK_TO), obs, set(), set())
    assert isinstance(move, InteractAction)  # matched Clerk; adjacent (3,3) + facing north


def test_strategize_talk_step_without_who_has_no_sprite():
    import json as _json
    plan = {"plan": "x", "steps": [{"map": 40, "talk": True, "who": "   ",
                                    "done_when": "talked", "why": "y"}]}
    class P:
        def chat_json(self, system, state, image=None): return _json.dumps(plan), 0, {}
    from pokemon_agent.agent.planner_llm import Planner
    import pokemon_agent.agent.planner_llm as m

    class FakeEmu:
        def read_memory(self, a): return 0
    pl = Planner(goal_map=40, strategist=P())
    pl_party, pl_items, pl_badges = m.read_party, m.read_items, m.read_badges
    m.read_party = lambda e: []; m.read_items = lambda e: []; m.read_badges = lambda e: {"count": 0}
    try:
        quest = pl.strategize(FakeEmu(), memory=None, why="z")
    finally:
        m.read_party, m.read_items, m.read_badges = pl_party, pl_items, pl_badges
    talk = [d for d in quest if d.intent.name == "TALK_TO"]
    assert talk and "sprite" not in (talk[0].target or {})   # whitespace who -> no sprite key


class NpcPickClient:
    """Fake TypeSafe client: system_one returns a Choice answer selecting a fixed index."""
    def __init__(self, index, conf=0.9): self._i, self._c = index, conf
    def system_one(self, *, state, questions):
        from types import SimpleNamespace as NS
        return NS(answers={"npc": NS(choice=str(self._i), confidence=self._c, probabilities=None)},
                  usage=None)


def test_choose_npc_picks_index_against_objective():
    from pokemon_agent.agent.typesafe_reasoner import TypeSafeReasoner
    r = TypeSafeReasoner(client=NpcPickClient(1))
    cands = [{"sprite": "Rival", "x": 5, "y": 7, "talked_to": False},
             {"sprite": "Oak", "x": 3, "y": 2, "talked_to": False}]
    idx, conf = r.choose_npc(objective="deliver Oak's Parcel to Professor Oak", candidates=cands)
    assert idx == 1 and conf == 0.9


def test_approach_npc_uses_jev_pick_over_nearest():
    # Jev must be able to pick the NON-nearest NPC. Rival is adjacent-north (the nearest); Oak is
    # 3 tiles SOUTH. If the code used the nearest fallback it would interact with the Rival (player
    # faces north, adjacent) -> InteractAction. Using Jev's pick it routes SOUTH toward Oak. The
    # SOUTH MoveAction is only producible by the Jev path, so it distinguishes the two.
    loop, _ = _nav_loop(0)
    loop.world.ingest_collision(0, 6, 8, {(x, y) for x in range(6) for y in range(8)}, None, None)

    class R:
        def choose_npc(self, *, objective, candidates):
            return 1, 0.9  # index 1 == Oak, the FARTHER npc (not the nearest)
    loop.reasoner = R()
    player = SimpleNamespace(x=3, y=3, map_id=0, facing="north")
    obs = SimpleNamespace(player=player, map_dims=(6, 8),
                          game_state={"npcs": [{"x": 3, "y": 2, "sprite": "Rival"},
                                               {"x": 3, "y": 6, "sprite": "Oak"}]}, exits=[])
    d = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 0}, success={"talked_on_map": 0})
    move = loop._resolve_target({"kind": "approach_npc", "sprite": None}, d, obs, set(), set())
    # SOUTH = toward Jev's pick (Oak); the nearest-fallback would have interacted with the Rival instead
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH


def test_approach_npc_single_candidate_skips_jev():
    loop, _ = _nav_loop(0)
    player = SimpleNamespace(x=3, y=3, map_id=0, facing="north")
    obs = SimpleNamespace(player=player, map_dims=(6, 6),
                          game_state={"npcs": [{"x": 3, "y": 2, "sprite": "Oak"}]}, exits=[])
    d = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 0}, success={"talked_on_map": 0})
    move = loop._resolve_target({"kind": "approach_npc", "sprite": None}, d, obs, set(), set())
    assert isinstance(move, InteractAction)


def test_approach_npc_caches_jev_pick_across_frames():
    loop, _ = _nav_loop(0)
    loop.world.ingest_collision(0, 6, 8, {(x, y) for x in range(6) for y in range(8)}, None, None)
    calls = {"n": 0}

    class R:
        def choose_npc(self, *, objective, candidates):
            calls["n"] += 1
            return 1, 0.9  # Oak
    loop.reasoner = R()
    target = {"kind": "approach_npc", "sprite": None}
    d = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 0}, success={"talked_on_map": 0})

    def obs_at(px, py):
        player = SimpleNamespace(x=px, y=py, map_id=0, facing="north")
        return SimpleNamespace(player=player, map_dims=(6, 8),
                               game_state={"npcs": [{"x": 3, "y": 2, "sprite": "Rival"},
                                                    {"x": 3, "y": 6, "sprite": "Oak"}]}, exits=[])
    loop._resolve_target(target, d, obs_at(3, 3), set(), set())  # frame 1: Jev picks
    loop._resolve_target(target, d, obs_at(3, 4), set(), set())  # frame 2: uses the position cache, NOT Jev
    assert calls["n"] == 1 and target.get("picked") is not None


def test_collision_is_reread_every_step_not_cached(monkeypatch):
    # regression: a STALE/truncated collision read on map-entry must not be cached forever. We
    # re-read the RAM collision every step, so a later (settled) read corrects a bad earlier one.
    import pokemon_agent.games.pokemon_red.map_reader as mr
    calls = {"n": 0}

    def fake_read(emu):
        calls["n"] += 1
        if calls["n"] == 1:  # first read: truncated 10-wide (stale map-entry decode)
            return {"map_id": 0, "width": 10, "height": 18,
                    "walkable": {(x, y) for x in range(10) for y in range(18)},
                    "counters": set(), "terrain": {}}
        return {"map_id": 0, "width": 20, "height": 18,   # settled: full 20-wide
                "walkable": {(x, y) for x in range(20) for y in range(18)},
                "counters": set(), "terrain": {}}
    monkeypatch.setattr(mr, "read_collision_map", fake_read)

    loop, _ = _nav_loop(0)
    loop.step_once()
    assert loop.world.bounds[0] == (10, 18)          # ingested the first read
    loop.step_once()
    assert loop.world.bounds[0] == (20, 18)           # re-read corrected it (not cached once)
    assert calls["n"] >= 2                              # collision is read on EVERY step


# --- flow router (_route_flow): Jev routes navigate-vs-dialogue; menu is deterministic RAM ---
class _FlowStub:
    def __init__(self, ans): self.ans = ans
    def choose_flow(self, **kw): return self.ans


def test_route_flow_jev_routes_confident_dialogue():
    loop, _ = _nav_loop(0)
    loop.reasoner = _FlowStub({"dialogue": ("yes", 0.9), "menu": ("no", 0.9)})
    ctx = {"screen_text_raw": "This is private property!", "menu": {"open": False}, "kind": "overworld"}
    assert loop._route_flow(None, ctx) == "dialogue"


def test_route_flow_no_text_skips_jev_and_navigates():
    loop, _ = _nav_loop(0)
    loop.reasoner = _FlowStub({"dialogue": ("yes", 0.99), "menu": ("no", 0.99)})  # must NOT be consulted
    assert loop._route_flow(None, {"screen_text_raw": "", "menu": {"open": False}}) == "navigate"


def test_route_flow_menu_open_is_deterministic_and_wins():
    # A on a menu SELECTS -> the RAM cursor signal must route to menu even when Jev reads the text as
    # dialogue (the nurse YES/NO decodes as prose, so Jev leans dialogue — RAM must win).
    loop, _ = _nav_loop(0)
    loop.reasoner = _FlowStub({"dialogue": ("yes", 0.99), "menu": ("no", 0.3)})
    ctx = {"screen_text_raw": "Shall we heal your POKMON?", "menu": {"open": True}}
    assert loop._route_flow(None, ctx) == "menu"


def test_route_flow_low_confidence_falls_to_dialogue_when_text_present():
    loop, _ = _nav_loop(0)
    loop.reasoner = _FlowStub({"dialogue": ("yes", 0.3), "menu": ("no", 0.3)})   # both hedged
    ctx = {"screen_text_raw": "some ambiguous text", "menu": {"open": False}}
    assert loop._route_flow(None, ctx) == "dialogue"      # safe = close the box


def test_route_flow_deterministic_fallback_without_jev():
    loop, _ = _nav_loop(0)
    loop.reasoner = object()                              # no choose_flow -> ctx_kind dispatch
    assert loop._route_flow(None, {"screen_text_raw": "x", "menu": {"open": False},
                                   "kind": "dialog"}) == "dialogue"
    assert loop._route_flow(None, {"screen_text_raw": "", "menu": {"open": False},
                                   "kind": "overworld"}) == "navigate"


# --- farm-exp grinding drifts toward the goal through grass (never paces in place) ---
def test_farm_step_advances_through_grass_toward_goal():
    loop, _ = _nav_loop(0)
    loop.world.terrain[0] = {(2, 1): "grass", (1, 2): "grass", (3, 2): "grass"}  # N=forward, W/E=side
    loop.world.tiles[0] = {}
    player = SimpleNamespace(x=2, y=2, map_id=0, facing="north")
    obs = SimpleNamespace(player=player, map_dims=(6, 6), game_state={}, exits=[])
    loop._farm_age = 1                                   # not a weave step
    mv = loop._farm_step(obs, (2, 0), set())             # goal is due north
    assert isinstance(mv, MoveAction) and mv.direction == Direction.NORTH   # forward, on grass


def test_farm_step_weaves_sideways_every_third_step_never_backward():
    loop, _ = _nav_loop(0)
    loop.world.terrain[0] = {(2, 1): "grass", (1, 2): "grass", (3, 2): "grass"}
    loop.world.tiles[0] = {}
    player = SimpleNamespace(x=2, y=2, map_id=0, facing="north")
    obs = SimpleNamespace(player=player, map_dims=(6, 6), game_state={}, exits=[])
    loop._farm_age = 2                                   # -> 3 inside -> weave step
    mv = loop._farm_step(obs, (2, 0), set())
    assert mv.direction in (Direction.WEST, Direction.EAST)   # perpendicular weave, never south (backward)


def _run_farm(loop, start, goal, steps):
    """Simulate N farm-exp steps in an all-grass field, returning the path of (x,y) and the moves taken."""
    x, y = start
    path, moves = [(x, y)], []
    d2v = {Direction.NORTH: (0, -1), Direction.SOUTH: (0, 1), Direction.EAST: (1, 0), Direction.WEST: (-1, 0)}
    for _ in range(steps):
        player = SimpleNamespace(x=x, y=y, map_id=0, facing="north")
        obs = SimpleNamespace(player=player, map_dims=(20, 20), game_state={}, exits=[])
        mv = loop._farm_step(obs, goal, set())
        assert mv is not None
        vx, vy = d2v[mv.direction]
        x, y = x + vx, y + vy
        path.append((x, y)); moves.append(mv.direction)
    return path, moves


def test_farm_step_weaves_FORWARD_never_paces_in_one_row():
    # The core guarantee: over a run the agent nets progress toward the goal AND weaves across lanes —
    # it does NOT ping-pong left/right in the same row waiting for a battle.
    loop, _ = _nav_loop(0)
    loop.world.tiles[0] = {}
    loop.world.terrain[0] = {(x, y): "grass" for x in range(20) for y in range(20)}  # open grass field
    start, goal = (5, 12), (5, 0)                        # goal is due north
    path, moves = _run_farm(loop, start, goal, 12)

    # (a) net FORWARD: ended well north of the start (y decreases going north), not stuck on the row
    assert path[-1][1] <= start[1] - 6, f"expected strong northward progress, got {path}"
    # (b) it actually weaved: visited more than one column
    assert len({p[0] for p in path}) > 1, "never weaved sideways"
    # (c) never two lateral steps in a row (that is what caused row ping-pong)
    lat = {Direction.EAST, Direction.WEST}
    assert not any(a in lat and b in lat for a, b in zip(moves, moves[1:])), f"two laterals in a row: {moves}"


def test_farm_step_advances_through_open_lane_when_forward_isnt_grass():
    # A vertical NON-grass lane straight ahead with grass on both sides: the agent must still climb the
    # lane (net north) rather than oscillate east/west across it forever.
    loop, _ = _nav_loop(0)
    loop.world.tiles[0] = {}
    terr = {}
    for y in range(20):
        terr[(4, y)] = "floor"; terr[(6, y)] = "floor"   # grass columns flanking the bare lane at x=5
    loop.world.terrain[0] = terr
    start, goal = (5, 12), (5, 0)
    path, _ = _run_farm(loop, start, goal, 10)
    assert path[-1][1] < start[1], f"did not advance north up the lane: {path}"
