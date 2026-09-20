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


def test_emergency_heal_pings_l1_not_freeze():
    # a near-faint party is NOT a target-less HEAL directive (that froze the agent on WaitAction) —
    # it forces an L1 review that INSERTS a routed heal quest. The guard stops it re-forcing L1 once
    # a heal errand is already in the plan.
    from collections import deque as _deque
    from pokemon_agent.agent.quest_reconciler import QuestStep
    import pokemon_agent.agent.reason_loop as rl
    loop, emu = _loop(map_id=1, goal_map=2)
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 2}, success={"on_map": 2},
                  quest_id="q1")
    loop._plan_steps = [QuestStep(id="q1", map=2, done_when="on_map", status="active")]
    loop._quest = _deque([])
    loop._directive = d
    calls = {"n": 0}

    def fake_pipeline(emu, ctx, planner, *, hard_event, on_trace=None):
        calls["n"] += 1
        assert ctx["signals"].get("emergency_heal") is True     # L1 is told it's an emergency
        assert hard_event is True                                # emergency implies hard_event
        return {"mission": "", "milestone": "heal",
                "add": [{"map": 3, "talk": True, "who": "nurse", "done_when": "hp_frac>=0.95",
                         "why": "heal at the center"}], "remove": []}
    orig_pipeline = rl.run_l1_pipeline
    rl.run_l1_pipeline = fake_pipeline
    faint = [{"hp": 0, "max_hp": 20}]                            # a fainted member
    orig = rl.game_signals
    rl.game_signals = lambda emu: {"party": faint, "hp_frac": 0.0, "min_level": 5,
                                   "badges": 0, "items": []}
    try:
        obs = loop.builder.build(capture_screenshot=False)[0]
        out = loop._manage_directive(obs)
        assert calls["n"] == 1                                    # L1 fired for the emergency
        assert any((s.done_when or "").startswith("hp_frac") for s in loop._plan_steps)  # heal step landed
        assert not (out is not None and out.intent == Intent.HEAL and out.target is None)  # NO freeze
        # guard: a heal step is now in the plan -> a second near-faint call does NOT re-force L1
        loop._manage_directive(obs)
        assert calls["n"] == 1
    finally:
        rl.game_signals = orig
        rl.run_l1_pipeline = orig_pipeline


def test_wedged_step_replaced_by_l1():
    # a wedged step is marked `wedged`; wedge runs BEFORE the L1 gate so L1's reconcile sees it and
    # replaces just that step in the SAME gate. The rest of the plan survives (no quest.clear()).
    from collections import deque as _deque
    from pokemon_agent.agent.quest_reconciler import QuestStep
    from pokemon_agent.agent.reason_loop import SERVO_FAIL_LIMIT
    import pokemon_agent.agent.reason_loop as rl
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
    orig_pipeline = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: {
        "add": [{"map": 5, "talk": False, "done_when": "on_map", "why": "retry via another route"}],
        "remove": ["q1"]}
    try:
        loop._manage_directive(loop.builder.build(capture_screenshot=False)[0])
        ids = [s.id for s in loop._plan_steps]
        assert "q1" not in ids                                          # the wedged step was replaced
        assert "q2" in ids                                              # the rest of the plan survived
        assert all(s.status != "wedged" for s in loop._plan_steps)     # nothing left wedged
        # a fresh retry step for map 5 replaced the wedged one; the plan advanced onto it (now active)
        assert any(s.map == 5 and s.status in ("pending", "active") for s in loop._plan_steps)
    finally:
        rl.run_l1_pipeline = orig_pipeline


def test_wedge_resets_block_and_fires_l1_once():
    # a hard block marks the step wedged, RESETS the block counter (so L1 doesn't re-fire every
    # step), and fires L1 exactly once for the wedge.
    from collections import deque as _deque
    from pokemon_agent.agent.quest_reconciler import QuestStep
    from pokemon_agent.agent.reason_loop import BLOCK_TRIGGER
    import pokemon_agent.agent.reason_loop as rl
    loop, emu = _loop(map_id=1, goal_map=2)
    loop._plan_steps = [QuestStep(id="q1", map=5, done_when="on_map", status="active"),
                        QuestStep(id="q2", map=9, done_when="on_map", status="pending")]
    d_active = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 5}, success={"on_map": 5},
                         quest_id="q1")
    d_pending = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 9}, success={"on_map": 9},
                          quest_id="q2")
    loop._quest = _deque([d_pending])
    loop._directive = d_active
    loop._blocked_for_n = BLOCK_TRIGGER               # navigation deadlock over budget
    calls = {"n": 0}

    def no_change(emu, ctx, planner, *, hard_event, on_trace=None):
        calls["n"] += 1
        return None
    orig_pipeline = rl.run_l1_pipeline
    rl.run_l1_pipeline = no_change
    try:
        obs = loop.builder.build(capture_screenshot=False)[0]
        loop._manage_directive(obs)
        assert any(s.id == "q1" and s.status == "wedged" for s in loop._plan_steps)  # marked wedged
        assert loop._blocked_for_n == 0                                              # counter reset
        assert calls["n"] == 1                                                       # L1 fired for the wedge
        # a second call must NOT re-fire L1 (the block counter was reset; the plan advanced)
        loop._manage_directive(obs)
        assert calls["n"] == 1
    finally:
        rl.run_l1_pipeline = orig_pipeline


def test_recompile_preserves_active_talk_leftover():
    # a talk step's TRAVEL was already popped (step active) with its TALK_TO still queued; a recompile
    # must keep that TALK_TO or the talk is skipped (the step falsely completing on the travel on_map).
    from collections import deque as _deque
    from pokemon_agent.agent.quest_reconciler import QuestStep
    loop, _ = _loop(map_id=40, goal_map=2)
    loop._plan_steps = [QuestStep(id="q1", map=40, talk=True, who="Oak",
                                  done_when="no_item:Oak's Parcel", status="active"),
                        QuestStep(id="q2", map=9, done_when="on_map", status="pending", kind="travel")]
    talk = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 40, "sprite": "Oak"},
                     success={"no_item": "Oak's Parcel"}, quest_id="q1")
    loop._quest = _deque([talk])
    loop._recompile_quest()
    pairs = [(d.intent, d.quest_id) for d in loop._quest]
    assert (Intent.TALK_TO, "q1") in pairs             # active step's leftover TALK_TO survived
    assert any(d.quest_id == "q2" for d in loop._quest)  # the pending step compiled after it


def test_run_l1_resets_counters_on_exception():
    # a raising run_l1_pipeline must not leave the gate counters set (else L1 retries + re-raises
    # every step). The resets live in a finally.
    import pokemon_agent.agent.reason_loop as rl
    loop, _ = _loop(map_id=1, goal_map=2)
    loop._legs_since_l1 = 5
    loop._l1_event = True

    def boom(emu, ctx, planner, *, hard_event, on_trace=None):
        raise RuntimeError("kaboom")
    orig_pipeline = rl.run_l1_pipeline
    rl.run_l1_pipeline = boom
    try:
        loop._run_l1(loop.builder.build(capture_screenshot=False)[0])
        assert loop._legs_since_l1 == 0 and loop._l1_event is False
    finally:
        rl.run_l1_pipeline = orig_pipeline


# --- L1 strategic planner (Task 6a) ---
from collections import deque
from pokemon_agent.agent.plan import Intent


def test_run_l1_inserts_and_recompiles():
    import pokemon_agent.agent.reason_loop as rl
    loop, _ = _loop(map_id=1, goal_map=2)
    orig_pipeline = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: {
        "mission": "reach Pewter", "milestone": "deliver parcel",
        "add": [{"map": 42, "talk": True, "who": "clerk", "done_when": "has_item:Oak's Parcel", "why": "get parcel"}],
        "remove": []}
    try:
        obs, _ = loop.builder.build(capture_screenshot=False)
        loop._run_l1(obs)
        assert any(s.map == 42 and s.talk for s in loop._plan_steps)          # reconciled into the plan
        assert isinstance(loop._quest, deque) and any(d.intent == Intent.TALK_TO for d in loop._quest)  # recompiled
        assert loop._plan is not None and loop._plan.milestone == "deliver parcel"   # durable memory updated
    finally:
        rl.run_l1_pipeline = orig_pipeline


def test_run_l1_no_change_keeps_plan():
    import pokemon_agent.agent.reason_loop as rl
    loop, _ = _loop(map_id=1, goal_map=2)
    from pokemon_agent.agent.quest_reconciler import QuestStep
    loop._plan_steps = [QuestStep(id="q1", map=2, done_when="on_map", status="active")]
    orig_pipeline = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: None
    try:
        obs, _ = loop.builder.build(capture_screenshot=False)
        loop._run_l1(obs)
        assert [s.id for s in loop._plan_steps] == ["q1"]                     # unchanged on no-change
        assert loop._l1_last["change"] is False
    finally:
        rl.run_l1_pipeline = orig_pipeline


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


def test_recorder_extra_has_l1_fields():
    import pokemon_agent.agent.reason_loop as rl
    loop, _ = _loop(map_id=1, goal_map=2)
    orig_pipeline = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: {
        "mission": "reach Pewter", "milestone": "deliver parcel",
        "add": [{"map": 42, "talk": False, "done_when": "on_map", "kind": "travel", "why": "mart"}], "remove": []}
    try:
        obs, _ = loop.builder.build(capture_screenshot=False)
        loop._run_l1(obs)
        assert loop._l1_last is not None and loop._l1_last["change"] is True
        assert loop._plan.milestone == "deliver parcel"
        assert "assessment" in loop._l1_last
    finally:
        rl.run_l1_pipeline = orig_pipeline


def test_l1_plans_parcel_errand_end_to_end():
    import pokemon_agent.agent.reason_loop as rl
    from pokemon_agent.agent.plan import Intent
    loop, _ = _loop(map_id=1, goal_map=2)   # in Viridian, goal Pewter
    loop._qid = 50
    # stub L1: when blocked at Viridian, insert the parcel errand (Mart -> get parcel -> Lab -> deliver)
    orig_pipeline = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: {
        "mission": "reach Pewter", "milestone": "deliver Oak's Parcel",
        "add": [
            {"map": 42, "talk": True, "who": "clerk", "done_when": "has_item:Oak's Parcel", "why": "get parcel"},
            {"map": 40, "talk": True, "who": "Oak", "done_when": "no_item:Oak's Parcel", "why": "deliver"},
        ], "remove": []}
    try:
        obs, _ = loop.builder.build(capture_screenshot=False)
        loop._run_l1(obs)
        maps = [s.map for s in loop._plan_steps]
        assert 42 in maps and 40 in maps                       # Mart + Lab planned
        # the compiled queue contains a TALK_TO to Oak whose success checks the parcel is gone
        talk_oak = [d for d in loop._quest if d.intent == Intent.TALK_TO and (d.target or {}).get("sprite") == "Oak"]
        assert talk_oak and "no_item" in talk_oak[0].success   # deliver step is machine-checkable
    finally:
        rl.run_l1_pipeline = orig_pipeline


def test_l1_plan_supersedes_bootstrap_default():
    # regression: a synthesized bootstrap "travel to goal_map" step must NOT stay the active
    # directive once L1 supplies a real plan. In the live bug it stayed travel->Route 2 (a
    # story-gated hop) for 300+ steps while L1's parcel errand sat behind it (planned-not-executed).
    import pokemon_agent.agent.reason_loop as rl
    from pokemon_agent.agent.plan import Intent
    from pokemon_agent.agent.quest_reconciler import QuestStep
    loop, _ = _loop(map_id=1, goal_map=2)
    loop.planner.strategist = object()      # a provider is available (so bootstrap defers to L1)
    obs = loop.builder.build(capture_screenshot=False)[0]
    orig_pipeline = rl.run_l1_pipeline
    try:
        # 1) bug precondition: L1 declines on the empty plan -> a PROVISIONAL default is committed.
        rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: None
        d0 = loop._manage_directive(obs)
        assert d0 is not None and d0.target_map == 2                  # committed the goal-travel default
        prov = [s for s in loop._plan_steps if s.provisional]
        assert len(prov) == 1 and prov[0].status == "active"         # the default is provisional + active

        # 2) L1 now supplies the real parcel errand -> the provisional default must be superseded.
        loop._l1_event = True                                         # force the L1 gate this call
        rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: {
            "mission": "reach Pewter", "milestone": "deliver Oak's Parcel",
            "add": [
                {"map": 42, "talk": True, "who": "clerk", "done_when": "has_item:Oak's Parcel", "why": "get parcel"},
                {"map": 40, "talk": True, "who": "Oak", "done_when": "no_item:Oak's Parcel", "why": "deliver"},
            ], "remove": []}
        out = loop._manage_directive(obs)
        assert not any(s.provisional for s in loop._plan_steps)      # provisional default dropped
        assert all(s.map != 2 for s in loop._plan_steps)            # no lingering goal-travel default
        assert out is not None and out.target_map == 42             # active directive now leads to the Mart
        assert loop._directive is out and out.quest_id is not None  # it's L1's first real step
    finally:
        rl.run_l1_pipeline = orig_pipeline


# --- Task 8: _run_l1 wired to run_l1_pipeline ---

def test_run_l1_uses_pipeline_heal_proposal_and_compiles():
    # run_l1_pipeline's proposal shape (add/remove/mission/milestone/assessment, no "change" key) —
    # a dict return always means "apply it": a heal action step must land in the plan and compile
    # into the quest queue without raising.
    import pokemon_agent.agent.reason_loop as rl
    loop, _ = _loop(map_id=1, goal_map=2)
    orig_pipeline = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: {
        "add": [{"kind": "action", "map": 41, "done_when": "hp_frac>=1.0", "talk": True,
                 "who": "Nurse", "why": "heal"}],
        "remove": [], "mission": "", "milestone": "", "assessment": "heal"}
    try:
        obs, _ = loop.builder.build(capture_screenshot=False)
        loop._run_l1(obs)   # must not raise
        assert any((s.done_when or "").startswith("hp_frac") for s in loop._plan_steps)
        assert isinstance(loop._quest, deque) and len(loop._quest) > 0
        assert loop._l1_last["change"] is True
    finally:
        rl.run_l1_pipeline = orig_pipeline


def test_run_l1_uses_pipeline_none_leaves_plan_untouched():
    # a None return (triage said no-change, or decide was unsalvageable) -> NO reconcile at all;
    # the standing plan is kept exactly as-is and _l1_last reflects "no change".
    import pokemon_agent.agent.reason_loop as rl
    from pokemon_agent.agent.quest_reconciler import QuestStep
    loop, _ = _loop(map_id=1, goal_map=2)
    loop._plan_steps = [QuestStep(id="q1", map=2, done_when="on_map", status="active")]
    before = list(loop._plan_steps)
    orig_pipeline = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: None
    try:
        obs, _ = loop.builder.build(capture_screenshot=False)
        loop._run_l1(obs)
        assert loop._plan_steps == before                # untouched
        assert loop._l1_last["change"] is False
    finally:
        rl.run_l1_pipeline = orig_pipeline


def test_run_l1_passes_hard_event_flag_through():
    # the periodic cadence call is NOT a hard event; a wedge/event-driven call IS.
    import pokemon_agent.agent.reason_loop as rl
    loop, _ = _loop(map_id=1, goal_map=2)
    seen = []
    orig_pipeline = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: (
        seen.append(hard_event) or None)
    try:
        obs, _ = loop.builder.build(capture_screenshot=False)
        loop._run_l1(obs, hard_event=False)
        loop._run_l1(obs, hard_event=True)
        loop._run_l1(obs, emergency=True)   # emergency implies hard_event even if caller forgets
        assert seen == [False, True, True]
    finally:
        rl.run_l1_pipeline = orig_pipeline


# --- Task 9: completion-provenance events + full quest serialization + L1 trace ---

def test_recorder_plan_steps_include_done_when_and_kind():
    # the recorder's plan_steps must carry the acceptance criterion (done_when/kind/why/talk/who),
    # not just {id, map, status} — otherwise diagnosing a wedged run needs RAM forensics.
    from collections import deque as _deque
    from pokemon_agent.agent.quest_reconciler import QuestStep
    loop, _ = _loop(map_id=0)
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 0, "x": 2, "y": 4},
                  success={"on_map": 99}, quest_id="q1")
    loop._plan_steps = [QuestStep(id="q1", map=0, talk=True, who="Oak", done_when="on_map",
                                  why="scripted stop", status="active", kind="travel")]
    loop._quest = _deque([d])

    class StubRecorder:
        def __init__(self):
            self.calls = []

        def on_event(self, kind, payload):
            pass

        def record(self, **kw):
            self.calls.append(kw)

    stub = StubRecorder()
    loop.recorder = stub
    loop.step_once()
    assert stub.calls, "recorder.record was never called"
    steps = stub.calls[0]["extra"]["plan_steps"]
    assert len(steps) == 1
    s = steps[0]
    assert s["id"] == "q1" and s["map"] == 0 and s["status"] == "active"
    assert s["done_when"] == "on_map" and s["kind"] == "travel"
    assert s["why"] == "scripted stop" and s["talk"] is True and s["who"] == "Oak"


def test_mark_step_done_emits_provenance_event():
    from pokemon_agent.agent.quest_reconciler import QuestStep
    events = []
    loop, _ = _loop(events=events)
    loop._plan_steps = [QuestStep(id="q1", map=0, done_when="on_map", status="active", kind="travel")]
    loop._mark_step("q1", "done")
    matches = [p for k, p in events if k == "step_done"]
    assert len(matches) == 1
    assert matches[0]["id"] == "q1"
    assert matches[0]["done_when"] == "on_map"
    # "step_kind" (not "kind"): the recorder stamps the event type under "kind", so the step's own
    # kind is carried under a distinct key to avoid clobbering the "step_done" marker in the log.
    assert matches[0]["step_kind"] == "travel"


def test_mark_step_wedged_emits_provenance_event_with_reason():
    from pokemon_agent.agent.quest_reconciler import QuestStep
    events = []
    loop, _ = _loop(events=events)
    loop._plan_steps = [QuestStep(id="q1", map=5, done_when="has_item:Potion", status="active",
                                  kind="action")]
    loop._mark_step("q1", "wedged", reason="no route / blocked")
    matches = [p for k, p in events if k == "step_wedged"]
    assert len(matches) == 1
    assert matches[0]["id"] == "q1"
    assert matches[0]["done_when"] == "has_item:Potion"
    assert matches[0]["reason"]   # non-empty, provenance for the wedge


def test_wedge_via_manage_directive_emits_step_wedged():
    # end-to-end: the real wedge path (_manage_directive) must ALSO surface step_wedged, not just
    # a direct _mark_step call.
    from collections import deque as _deque
    from pokemon_agent.agent.quest_reconciler import QuestStep
    from pokemon_agent.agent.reason_loop import BLOCK_TRIGGER
    import pokemon_agent.agent.reason_loop as rl
    events = []
    loop, _ = _loop(map_id=1, goal_map=2, events=events)
    loop._plan_steps = [QuestStep(id="q1", map=5, done_when="on_map", status="active"),
                        QuestStep(id="q2", map=9, done_when="on_map", status="pending")]
    d_active = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 5}, success={"on_map": 5},
                         quest_id="q1", reason="heading to route 5")
    d_pending = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 9}, success={"on_map": 9},
                          quest_id="q2")
    loop._quest = _deque([d_pending])
    loop._directive = d_active
    loop._blocked_for_n = BLOCK_TRIGGER
    orig_pipeline = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: None
    try:
        obs = loop.builder.build(capture_screenshot=False)[0]
        loop._manage_directive(obs)
        matches = [p for k, p in events if k == "step_wedged"]
        assert len(matches) == 1
        assert matches[0]["id"] == "q1" and matches[0]["done_when"] == "on_map"
        assert matches[0]["reason"]
    finally:
        rl.run_l1_pipeline = orig_pipeline


def test_l1_trace_recorded_into_l1_last():
    # once Task 8's run_l1_pipeline is patched to invoke on_trace with stage events, _run_l1 must
    # thread those events into self._l1_last so the recorder captures WHY, not just the outcome.
    import json
    import pokemon_agent.agent.reason_loop as rl
    loop, _ = _loop(map_id=1, goal_map=2)
    orig_pipeline = rl.run_l1_pipeline

    def fake_pipeline(emu, ctx, planner, *, hard_event, on_trace=None):
        if on_trace is not None:
            on_trace({"stage": "triage", "change": True, "why": "blocked"})
            on_trace({"stage": "decide", "add": 1, "remove": []})
        return {"mission": "reach Pewter", "milestone": "m", "assessment": "ok",
                "add": [{"map": 42, "talk": False, "done_when": "on_map", "kind": "travel", "why": "x"}],
                "remove": []}
    rl.run_l1_pipeline = fake_pipeline
    try:
        obs, _ = loop.builder.build(capture_screenshot=False)
        loop._run_l1(obs)
        assert loop._l1_last is not None and loop._l1_last["change"] is True
        trace = loop._l1_last.get("trace")
        assert isinstance(trace, list) and len(trace) == 2
        assert trace[0] == {"stage": "triage", "change": True, "why": "blocked"}
        assert trace[1] == {"stage": "decide", "add": 1, "remove": []}
        json.dumps(loop._l1_last)   # must stay JSON-serializable for the recorder
    finally:
        rl.run_l1_pipeline = orig_pipeline


def test_l1_trace_resets_between_reviews():
    # a stale trace from a previous review must not bleed into the next one.
    import pokemon_agent.agent.reason_loop as rl
    loop, _ = _loop(map_id=1, goal_map=2)
    orig_pipeline = rl.run_l1_pipeline

    def first(emu, ctx, planner, *, hard_event, on_trace=None):
        if on_trace is not None:
            on_trace({"stage": "triage", "change": False, "why": "first"})
        return None

    def second(emu, ctx, planner, *, hard_event, on_trace=None):
        if on_trace is not None:
            on_trace({"stage": "triage", "change": False, "why": "second"})
        return None
    try:
        obs, _ = loop.builder.build(capture_screenshot=False)
        rl.run_l1_pipeline = first
        loop._run_l1(obs)
        rl.run_l1_pipeline = second
        loop._run_l1(obs)
        assert loop._l1_last["trace"] == [{"stage": "triage", "change": False, "why": "second"}]
    finally:
        rl.run_l1_pipeline = orig_pipeline
