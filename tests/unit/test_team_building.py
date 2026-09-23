"""Team building: L1 is told why a team matters and whether its catch goal can actually fire.

brock-goals5 (after the Boulder Badge): L1 set catch ["any"] with 0 Poké Balls in the bag, so the battle
layer silently kept grinding (CAPTURE needs a ball) and nothing told L1. Before Brock it never
considered catching; every L1 prompt still framed the run as "working toward the first gym (Brock)".
"""
from __future__ import annotations

from pokemon_agent.agent import planner_llm as P
from pokemon_agent.agent.signals import catch_status


def test_no_l1_prompt_hardcodes_the_first_gym():
    for name in ("STRATEGIST_SYSTEM", "L1_SYSTEM", "BRAINSTORM_SYSTEM", "DECIDE_SYSTEM", "TRIAGE_SYSTEM"):
        assert "first gym (Brock" not in getattr(P, name), name


def test_brainstorm_and_decide_carry_team_guidance_and_the_catch_signal():
    for name in ("BRAINSTORM_SYSTEM", "DECIDE_SYSTEM"):
        text = getattr(P, name)
        assert "team" in text.lower() and "Poké Balls" in text and "SIGNALS.catch" in text, name


def test_catch_status_without_balls_is_not_ready_and_says_why():
    st = catch_status({"catch": ["any"]}, [{"item": "Antidote", "qty": 1}], [{"species": "Squirtle"}])
    assert st == {"goal": ["any"], "ready": False, "why": "no Poké Balls in the bag",
                  "balls": 0, "party_size": 1, "party_max": 6}


def test_catch_status_ready_with_a_ball_and_a_free_slot():
    st = catch_status({"catch": ["Pidgey"]}, [{"item": "Poke Ball", "qty": 5}], [{"species": "Squirtle"}])
    assert st["ready"] is True and st["balls"] == 5 and st["why"] == ""


def test_catch_status_full_party_and_no_goal():
    six = [{"species": "X"}] * 6
    assert catch_status({"catch": ["any"]}, [{"item": "Poke Ball", "qty": 2}], six)["why"] == "party is full (6/6)"
    st = catch_status({}, [{"item": "Poke Ball", "qty": 2}], [{"species": "Squirtle"}])
    assert st["goal"] == [] and st["ready"] is False and st["why"] == "no catch goal set"


def test_l1_context_signals_include_the_catch_status():
    from types import SimpleNamespace  # noqa: F401
    import pokemon_agent.agent.reason_loop as rl
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.plan import AgentPlan, ReflectionPlan
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import ReasonStep
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import GoalState, WaitAction
    from pokemon_agent.emulator.fake_emulator import FakeEmulator
    from pokemon_agent.observations.builder import ObservationBuilder

    class Stub:
        def reflect(self, **kw):
            return ReflectionPlan(next_objective="go"), 0, {}

        def step(self, **kw):
            return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}
    emu = FakeEmulator(map_id=14)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu), reasoner=Stub(),
                         session=Session(GoalState(primary="g", current="g")), vision=False, reflect_every=100,
                         goal_map=2)
    loop._plan = AgentPlan(battle_goals={"catch": ["any"]})
    seen = {}
    orig = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda emu, ctx, planner, *, hard_event, on_trace=None: seen.update(ctx)
    try:
        obs, _ = loop.builder.build(capture_screenshot=False)
        loop._run_l1(obs)
    finally:
        rl.run_l1_pipeline = orig
    assert seen["signals"]["catch"]["goal"] == ["any"] and seen["signals"]["catch"]["ready"] is False
