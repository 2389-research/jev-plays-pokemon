"""Tiered goals wiring: planner state/passthrough, pipeline, loop application, goal-met ping
(spec docs/superpowers/specs/2026-09-23-l1-tiered-goals-design.md §3)."""
from __future__ import annotations

import json

import pokemon_agent.agent.reason_loop as rl
from pokemon_agent.agent.l1_pipeline import run_l1_pipeline
from pokemon_agent.agent.plan import AgentPlan, Goal, Goals
from pokemon_agent.agent.planner_llm import Planner
from pokemon_agent.agent.quest_reconciler import QuestStep

from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.plan import ReflectionPlan
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep
from pokemon_agent.agent.session import Session
from pokemon_agent.core.models import GoalState, WaitAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.observations.builder import ObservationBuilder

B = 0xD16B


class StubReasoner:
    def reflect(self, **kw):
        return ReflectionPlan(next_objective="go"), 0, {}

    def step(self, **kw):
        return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}


def _loop(goal_map=99, map_id=0, events=None):
    emu = FakeEmulator(map_id=map_id)
    session = Session(GoalState(primary="reach pewter", current="reach pewter"))
    sink = (lambda k, p: events.append((k, p))) if events is not None else None
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=StubReasoner(), session=session, vision=False,
                         reflect_every=100, goal_map=goal_map, on_event=sink)
    return loop, emu


class StubPlanner:
    def __init__(self, *, brainstorm=None, decide=None):
        self._brainstorm, self._decide = brainstorm, decide

    def l1_brainstorm(self, emu, context):
        return self._brainstorm

    def l1_decide(self, context, brainstorm):
        return self._decide


class Capture:
    """Fake provider: returns ``obj`` and remembers the state it was shown."""
    def __init__(self, obj):
        self.obj, self.seen = obj, None

    def chat_json(self, system, state, image=None):
        self.seen = state
        return json.dumps(self.obj), 0, {}


LEGACY_CTX = {"current_map": {"id": 1, "name": "Viridian City"}, "party": [], "items": [], "badges": 0,
              "plan": [], "signals": {}, "mission": "Beat Brock", "milestone": "Deliver the parcel"}


# ---- planner: state builders + DECIDE passthrough -----------------------------------------------
def test_legacy_context_gets_derived_goals_and_no_mission_milestone():
    for method in ("l1_triage", "l1_decide"):
        prov = Capture({"change": False, "add": [], "remove": []})
        p = Planner(provider=prov, strategist=prov)
        getattr(p, method)(LEGACY_CTX, *([{"assessment": "x"}] if method == "l1_decide" else []))
        assert prov.seen["goals"]["primary"]["text"] == "Beat Brock"
        assert prov.seen["goals"]["secondary"]["text"] == "Deliver the parcel"
        assert "mission" not in prov.seen and "milestone" not in prov.seen


def test_triage_sees_goals_but_not_the_notepad():
    prov = Capture({"change": False})
    ctx = {**LEGACY_CTX, "goals": {"primary": {"text": "Beat Brock", "done_when": None}},
           "goal_status": {"primary": "none"}, "interrupted": None, "notepad": "private"}
    Planner(provider=prov).l1_triage(ctx)
    assert "goals" in prov.seen and "goal_status" in prov.seen and "notepad" not in prov.seen


def test_decide_passes_goals_notepad_interrupted_and_does_not_backfill():
    obj = {"assessment": "divert", "add": [], "remove": [],
           "goals": {"tertiary": {"text": "Heal", "done_when": "hp_frac>=1.0"}},
           "notepad": "remember the forest", "interrupted": "", "catch": "clear"}
    out = Planner(strategist=Capture(obj)).l1_decide(LEGACY_CTX, {"assessment": "x"})
    assert out["goals"] == obj["goals"] and out["notepad"] == "remember the forest"
    assert out["interrupted"] == "" and out["catch"] == "clear"
    assert "mission" not in out and "milestone" not in out          # no back-fill from context


def test_decide_without_interrupted_key_omits_it():
    out = Planner(strategist=Capture({"add": [], "remove": []})).l1_decide(LEGACY_CTX, {"assessment": "x"})
    assert "interrupted" not in out and out["catch"] is None


# ---- pipeline ------------------------------------------------------------------------------------
def test_pipeline_returns_a_goals_only_decide():
    planner = StubPlanner(brainstorm={"assessment": "a"},
                          decide={"add": [], "remove": [], "assessment": "a",
                                  "goals": {"tertiary": {"text": "Heal"}}})
    prop = run_l1_pipeline(None, {}, planner, hard_event=True)
    assert prop is not None and prop["add"] == [] and prop["goals"] == {"tertiary": {"text": "Heal"}}


def test_pipeline_still_short_circuits_legacy_echo_with_empty_catch():
    planner = StubPlanner(brainstorm={"assessment": "a"},
                          decide={"add": [], "remove": [], "assessment": "a", "mission": "m",
                                  "milestone": "reworded", "catch": []})
    assert run_l1_pipeline(None, {}, planner, hard_event=True) is None


# ---- loop application ----------------------------------------------------------------------------
def _with_prop(loop, prop):
    orig = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: prop
    try:
        obs, _ = loop.builder.build(capture_screenshot=False)
        loop._run_l1(obs)
    finally:
        rl.run_l1_pipeline = orig


def test_goals_only_edit_applied_without_reconcile():
    events = []
    loop, _ = _loop(map_id=1, goal_map=2, events=events)
    loop._plan = AgentPlan(goals=Goals(primary=Goal(text="Beat Brock")))
    loop._plan_steps = [QuestStep(id="q1", map=2, done_when="on_map", status="active")]
    called = []
    orig = rl.reconcile_quests
    rl.reconcile_quests = lambda *a, **k: called.append(1) or a[0]
    try:
        _with_prop(loop, {"add": [], "remove": [], "assessment": "divert",
                          "goals": {"tertiary": {"text": "Heal at the Pokémon Center", "done_when": "hp_frac>=1.0"}}})
    finally:
        rl.reconcile_quests = orig
    assert called == [] and [s.id for s in loop._plan_steps] == ["q1"]
    assert loop._plan.goals.tertiary.text == "Heal at the Pokémon Center"
    review = [p for k, p in events if k == "l1_review"][-1]
    assert review["change"] == "goals" and loop._l1_last["change"] == "goals"
    assert any(k == "goals_changed" for k, _ in events) and not any(k == "quest" for k, _ in events)


def test_goals_only_echo_is_no_change():
    events = []
    loop, _ = _loop(map_id=1, goal_map=2, events=events)
    loop._plan = AgentPlan(goals=Goals(primary=Goal(text="Beat Brock")))
    _with_prop(loop, {"add": [], "remove": [], "goals": {"primary": {"text": "beat  brock"}}})
    assert [p for k, p in events if k == "l1_review"][-1]["change"] is False


def test_goals_tier_beats_legacy_milestone_alongside_a_step_edit():
    loop, _ = _loop(map_id=1, goal_map=2)
    loop._plan = AgentPlan(goals=Goals(secondary=Goal(text="Deliver the parcel")))
    _with_prop(loop, {"add": [{"kind": "travel", "map": 13, "done_when": "on_map", "why": "north"}],
                      "remove": [], "milestone": "Deliver the parcel",
                      "goals": {"secondary": {"text": "Reach Pewter City"}}})
    assert loop._plan.goals.secondary.text == "Reach Pewter City"
    assert loop._plan.milestone == "Reach Pewter City"


def test_empty_catch_with_step_edit_keeps_the_catch_goal_and_clear_clears_it():
    loop, _ = _loop(map_id=1, goal_map=2)
    loop._plan = AgentPlan(battle_goals={"catch": ["Pidgey"]})
    step = {"kind": "travel", "map": 13, "done_when": "on_map", "why": "north"}
    _with_prop(loop, {"add": [step], "remove": [], "catch": []})
    assert loop._plan.battle_goals["catch"] == ["Pidgey"]
    _with_prop(loop, {"add": [], "remove": [], "catch": "clear"})
    assert loop._plan.battle_goals["catch"] == []


def test_removed_steps_no_longer_feed_tried_failed():
    loop, _ = _loop(map_id=1, goal_map=2)
    loop._plan = AgentPlan()
    loop._plan_steps = [QuestStep(id="q1", map=2, done_when="on_map", status="pending", why="old idea")]
    _with_prop(loop, {"add": [], "remove": ["q1"]})
    assert loop._plan.tried_failed == []


def test_l1_context_carries_goals_and_notepad():
    loop, _ = _loop(map_id=1, goal_map=2)
    loop._plan = AgentPlan(goals=Goals(primary=Goal(text="Beat Brock", done_when="badges>=1")), notepad="n")
    seen = {}
    orig = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: seen.update(ctx)
    try:
        obs, _ = loop.builder.build(capture_screenshot=False)
        loop._run_l1(obs)
    finally:
        rl.run_l1_pipeline = orig
    assert seen["goals"]["primary"]["text"] == "Beat Brock" and seen["goal_status"]["primary"] == "unmet"
    assert seen["notepad"] == "n" and "mission" not in seen


# ---- goal-met ping --------------------------------------------------------------------------------
def _set_level(emu, level, party=1):
    emu.write_memory(0xD163, party)
    emu.write_memory(B + 0x21, level)
    emu.write_memory(B + 0x22, 0)
    emu.write_memory(B + 0x23, 20)
    emu.write_memory(B + 2, 20)


def test_ping_fires_once_on_unmet_to_met():
    events = []
    loop, emu = _loop(map_id=1, goal_map=2, events=events)
    loop._plan = AgentPlan(goals=Goals(tertiary=Goal(text="Grind", done_when="level>=10")))
    _set_level(emu, 8)
    loop._check_goal_ping()
    assert loop._l1_event is False
    _set_level(emu, 10)
    loop._check_goal_ping()
    assert loop._l1_event is True and [k for k, _ in events].count("goal_met") == 1
    loop._l1_event = False
    _set_level(emu, 8); loop._check_goal_ping()
    _set_level(emu, 10); loop._check_goal_ping()      # flips back: latched, no second ping
    assert loop._l1_event is False


def test_no_ping_on_resume_when_already_met():
    loop, emu = _loop(map_id=1, goal_map=2)
    loop._plan = AgentPlan(goals=Goals(secondary=Goal(text="Grind", done_when="level>=5")))
    _set_level(emu, 9)
    loop._check_goal_ping(); loop._check_goal_ping()
    assert loop._l1_event is False


def test_goal_written_already_met_never_pings():
    loop, emu = _loop(map_id=1, goal_map=2)
    loop._plan = AgentPlan(goals=Goals(tertiary=Goal(text="Grind", done_when="level>=20")))
    _set_level(emu, 9)
    loop._check_goal_ping()
    _with_prop(loop, {"add": [], "remove": [], "goals": {"tertiary": {"text": "Grind a bit", "done_when": "level>=5"}}})
    loop._l1_event = False
    loop._check_goal_ping()
    assert loop._l1_event is False


# ---- Jev menu prompt --------------------------------------------------------------------------------
def test_jev_menu_prompt_plan_hides_the_notepad():
    from pokemon_agent.agent.reasoner import Reasoner
    prov = Capture({"location": "", "objective": "", "reasoning": "", "action": {"type": "wait", "frames": 1}})
    plan = AgentPlan(goals=Goals(primary=Goal(text="Beat Brock")), notepad="n")
    try:
        Reasoner(prov).step(primary_goal="g", image=None, local_map=None, map_view=None, player_desc="",
                            exits=[], game_state=None, social_memory=None, map_history=[], recent=[],
                            previous=None, plan=plan)
    except Exception:
        pass   # only the prompt matters
    cp = prov.seen["current_plan"]
    assert "notepad" not in cp and cp["goals"]["primary"]["text"] == "Beat Brock"
