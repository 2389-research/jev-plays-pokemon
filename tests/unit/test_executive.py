"""Executive dispatch: deterministic servo, success-replan, stack suspension (spec §9)."""
from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.plan import Directive, Intent, ReflectionPlan
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep
from pokemon_agent.agent.session import Session
from pokemon_agent.core.models import GoalState, WaitAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.observations.builder import ObservationBuilder


class StubReasoner:
    def reflect(self, **kw):
        return ReflectionPlan(next_objective="go"), 0, {}

    def step(self, **kw):
        self.last_directive = kw.get("directive")
        return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}


def _loop(goal_map=99, map_id=0, events=None):
    emu = FakeEmulator(map_id=map_id)
    session = Session(GoalState(primary="reach pewter", current="reach pewter"))
    sink = (lambda k, p: events.append((k, p))) if events is not None else None
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=StubReasoner(), session=session, vision=False,
                         reflect_every=100, goal_map=goal_map, on_event=sink)
    return loop, emu


def test_warp_exit_dir_steps_off_the_map_edge():
    from pokemon_agent.core.models import Direction
    f = ReasoningLoop._warp_exit_dir
    assert f((3, 7), (4, 8)) == Direction.SOUTH   # bottom-row door -> step south (buildings)
    assert f((3, 0), (4, 8)) == Direction.NORTH
    assert f((0, 3), (4, 8)) == Direction.WEST
    assert f((3, 3), (4, 8)) == Direction.EAST    # right-edge door
    assert f((2, 2), None) == Direction.SOUTH     # unknown dims -> default downward


def test_servo_steps_through_door_when_standing_on_it():
    # the Viridian Mart wedge: LunaRoute says "go to map 40" (needs to exit the building via the
    # door), the player is already standing ON the exit warp -> the servo must step THROUGH it
    # (off the bottom edge) to fire the warp, not report "arrived" and wait forever.
    from types import SimpleNamespace
    from pokemon_agent.core.models import Direction, MoveAction
    loop, _ = _loop(map_id=42)
    loop.memory.graph.next_hop = lambda a, b: (1, (3, 7))  # next hop toward 40 is map 1 via the door
    player = SimpleNamespace(x=3, y=7, map_id=42, facing="east")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}, {"x": 4, "y": 7, "dest_map": 1}])
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 40}, success={"on_map": 40})
    move = loop._servo_step(d, obs, set())
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH


def test_navigate_leg_steps_through_door_on_arrival():
    # L2/BFS delivered us onto the Mart exit warp -> _navigate_leg must step THROUGH it (south,
    # off the bottom edge) to fire the warp, not report arrived and wait (the Mart wedge).
    from types import SimpleNamespace
    from pokemon_agent.core.models import Direction, MoveAction
    loop, _ = _loop(map_id=42)
    loop.memory.graph.next_hop = lambda a, b: (1, (3, 7))
    player = SimpleNamespace(x=3, y=7, map_id=42, facing="east")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 40}, success={"on_map": 40})
    move = loop._navigate_leg(d, obs, set())
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH


def test_navigate_leg_falls_back_to_exit_tile_without_provider():
    # no LunaRoute provider (offline/tests) -> L2 falls back to routing toward the exit tile, so
    # navigation still works deterministically instead of stalling.
    from types import SimpleNamespace
    from pokemon_agent.core.models import MoveAction
    loop, _ = _loop(map_id=42)
    assert loop.planner.provider is None            # StubReasoner exposes no provider
    loop.memory.graph.next_hop = lambda a, b: (1, (3, 7))
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 40}, success={"on_map": 40})
    wp = loop._pick_waypoint(SimpleNamespace(player=SimpleNamespace(x=1, y=1, map_id=42), exits=[]),
                             d, goal_dir="south", next_map=1, exit_tile=(3, 7), occupied=set())
    assert wp == (3, 7)                              # fell back to the exit tile


def test_servo_walks_toward_same_map_tile_no_model_call():
    # plan-driven: the compiled quest queue holds the active tile directive; the servo routes to it
    # deterministically (no arbiter, no planner.plan, no model call).
    from collections import deque as _deque
    from pokemon_agent.agent.quest_reconciler import QuestStep
    loop, emu = _loop(map_id=0)
    # target is open floor south of the start (2,2) -> servo should step south, no executor
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 0, "x": 2, "y": 4},
                  success={"on_map": 99}, quest_id="q1")
    loop._plan_steps = [QuestStep(id="q1", map=0, done_when="on_map", status="active")]
    loop._quest = _deque([d])            # already compiled; _manage_directive pops it as the active one
    start = (emu.x, emu.y)
    loop.step_once()
    assert (emu.x, emu.y) != start and emu.y > start[1]  # moved south toward the target
    assert loop._directive is d  # committed to the directive (no replan)


def test_success_predicate_triggers_replan():
    # success on the active directive marks its step done and advances to the next compiled directive.
    from collections import deque as _deque
    from pokemon_agent.agent.quest_reconciler import QuestStep
    events = []
    loop, emu = _loop(map_id=5, events=events)
    done = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 5}, success={"on_map": 5},
                     quest_id="q1")
    nxt = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 9, "x": 2, "y": 4},
                    success={"on_map": 9}, quest_id="q2")
    loop._plan_steps = [QuestStep(id="q1", map=5, done_when="on_map", status="active"),
                        QuestStep(id="q2", map=9, done_when="on_map", status="pending")]
    loop._quest = _deque([nxt])
    loop._directive = done  # already-satisfied directive (emu is on map 5)
    out = loop._manage_directive(loop.builder.build(capture_screenshot=False)[0])
    kinds = [k for k, _ in events]
    assert "directive_done" in kinds
    assert out is nxt and loop._directive is nxt        # advanced to the next compiled directive
    steps = {s.id: s.status for s in loop._plan_steps}
    assert steps["q1"] == "done" and steps["q2"] == "active"   # step statuses advanced in order


def test_emergency_heal_preempts_then_restores_plan():
    # the near-faint emergency reflex is the ONLY preemption: it injects a HEAL directive ahead of
    # the plan; once HP is safe again the plan resumes exactly where it was.
    from collections import deque as _deque
    from pokemon_agent.agent.quest_reconciler import QuestStep
    import pokemon_agent.agent.reason_loop as rl
    loop, emu = _loop(map_id=0)
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 9}, success={"on_map": 9},
                  quest_id="q1")
    loop._plan_steps = [QuestStep(id="q1", map=9, done_when="on_map", status="active")]
    loop._quest = _deque([d])
    state = {"party": [{"hp": 0, "max_hp": 20}, {"hp": 3, "max_hp": 26}]}  # a fainted member
    orig = rl.game_signals
    rl.game_signals = lambda emu: {"party": state["party"], "hp_frac": 0.0, "min_level": 5,
                                   "badges": 0, "items": []}
    try:
        obs = loop.builder.build(capture_screenshot=False)[0]
        out = loop._manage_directive(obs)
        assert out.intent == Intent.HEAL and loop._directive.intent == Intent.HEAL  # preempted
        state["party"] = [{"hp": 26, "max_hp": 26}]                                 # HP safe again
        out2 = loop._manage_directive(obs)
        assert out2 is d and loop._directive is d                                   # plan resumed
    finally:
        rl.game_signals = orig


def test_wedged_step_marked_and_replaced_by_l1_next_gate():
    # a wedged step is marked `wedged` and L1's next review replaces just that step; the rest of the
    # plan survives (no quest.clear(), no escalation-score path).
    from collections import deque as _deque
    from pokemon_agent.agent.quest_reconciler import QuestStep
    from pokemon_agent.agent.reason_loop import SERVO_FAIL_LIMIT
    loop, emu = _loop(map_id=1, goal_map=2)
    loop._plan_steps = [QuestStep(id="q1", map=5, done_when="on_map", status="active"),
                        QuestStep(id="q2", map=9, done_when="on_map", status="pending")]
    d_active = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 5}, success={"on_map": 5},
                         quest_id="q1")
    d_pending = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 9}, success={"on_map": 9},
                          quest_id="q2")
    loop._quest = _deque([d_pending])
    loop._directive = d_active
    loop._qid = 10                                    # so L1's fresh step ids don't reuse "q1"/"q2"
    loop._servo_fail = SERVO_FAIL_LIMIT               # the active step is wedged (no route)

    obs = loop.builder.build(capture_screenshot=False)[0]
    loop._manage_directive(obs)
    assert any(s.id == "q1" and s.status == "wedged" for s in loop._plan_steps)  # marked, not cleared
    assert loop._l1_event is True                                                # L1 armed for next gate
    assert any(s.id == "q2" for s in loop._plan_steps)                           # plan NOT cleared

    # next gate: L1 replaces the wedged step (remove q1, add a fresh retry step); q2 survives.
    loop.planner.revise_quests = lambda emu, ctx: {
        "change": True,
        "add": [{"map": 5, "talk": False, "done_when": "on_map", "why": "retry via another route"}],
        "remove": ["q1"]}
    loop._manage_directive(obs)
    ids = [s.id for s in loop._plan_steps]
    assert "q1" not in ids                                          # the wedged step was replaced
    assert "q2" in ids                                              # the rest of the plan survived
    assert all(s.status != "wedged" for s in loop._plan_steps)     # nothing left wedged
    # a fresh retry step for map 5 replaced the wedged one; the plan advanced onto it (now active)
    assert any(s.map == 5 and s.status in ("pending", "active") for s in loop._plan_steps)


# --- L1 strategic planner (Task 6a) ---
from collections import deque
from pokemon_agent.agent.plan import Intent


def test_run_l1_inserts_and_recompiles():
    loop, _ = _loop(map_id=1, goal_map=2)
    loop.planner.revise_quests = lambda emu, ctx: {
        "change": True, "mission": "reach Pewter", "milestone": "deliver parcel",
        "add": [{"map": 42, "talk": True, "who": "clerk", "done_when": "has_item:Oak's Parcel", "why": "get parcel"}],
        "remove": []}
    obs, _ = loop.builder.build(capture_screenshot=False)
    loop._run_l1(obs)
    assert any(s.map == 42 and s.talk for s in loop._plan_steps)          # reconciled into the plan
    assert isinstance(loop._quest, deque) and any(d.intent == Intent.TALK_TO for d in loop._quest)  # recompiled
    assert loop._plan is not None and loop._plan.milestone == "deliver parcel"   # durable memory updated


def test_run_l1_no_change_keeps_plan():
    loop, _ = _loop(map_id=1, goal_map=2)
    from pokemon_agent.agent.quest_reconciler import QuestStep
    loop._plan_steps = [QuestStep(id="q1", map=2, done_when="on_map", status="active")]
    loop.planner.revise_quests = lambda emu, ctx: {"change": False, "add": [], "remove": []}
    obs, _ = loop.builder.build(capture_screenshot=False)
    loop._run_l1(obs)
    assert [s.id for s in loop._plan_steps] == ["q1"]                     # unchanged on no-change


def test_l1_due_is_a_pure_predicate():
    loop, _ = _loop(goal_map=2)
    loop.l1_every = 5
    loop._legs_since_l1, loop._blocked_for_n, loop._l1_event = 0, 0, False
    assert loop._l1_due() is False
    loop._legs_since_l1 = 5
    assert loop._l1_due() is True                # cadence
    loop._legs_since_l1 = 0; loop._blocked_for_n = 6
    assert loop._l1_due() is True                # blocked (>= BLOCK_TRIGGER)
    loop._blocked_for_n = 0; loop._l1_event = True
    assert loop._l1_due() is True                # event flag
