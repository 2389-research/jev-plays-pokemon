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


# ---- quantities: "buy 5 Poké Balls" (brock-goals5 steps 1595-1663: verify:"at least 4 Potions" looped) ----
def test_has_item_with_a_count_parses_and_evaluates():
    from pokemon_agent.agent.planner_llm import Planner
    from pokemon_agent.games.pokemon_red import predicates
    from pokemon_agent.emulator.fake_emulator import FakeEmulator
    assert Planner._parse_done_when("has_item:Poke Ball>=5", 56) == {"item_count": [4, 5]}
    emu = FakeEmulator(map_id=56)
    emu.write_memory(0xD31D, 1); emu.write_memory(0xD31E, 4); emu.write_memory(0xD31F, 3); emu.write_memory(0xD320, 0xFF)
    assert not predicates.evaluate({"item_count": [4, 5]}, emu)
    emu.write_memory(0xD31F, 5)
    assert predicates.evaluate({"item_count": [4, 5]}, emu)


def test_count_criteria_are_valid_steps_and_goals():
    from pokemon_agent.agent import goals as G
    from pokemon_agent.agent.l1_pipeline import validate_step
    assert validate_step({"kind": "action", "map": 56, "talk": True, "done_when": "has_item:Poke Ball>=5"})[0]
    assert G.clean_criterion("has_item:Poke Ball>=5") == "has_item:Poke Ball>=5"


def _mart_loop(success):
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.plan import Directive, Intent, ReflectionPlan
    from pokemon_agent.agent.quest_reconciler import QuestStep
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
    emu = FakeEmulator(map_id=56)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu), reasoner=Stub(),
                         session=Session(GoalState(primary="g", current="g")), vision=False, reflect_every=100,
                         goal_map=2)
    loop._plan_steps = [QuestStep(id="q18", map=56, talk=True, done_when="x", status="active")]
    loop._directive = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 56, "sprite": "the Mart clerk"},
                                success=success, quest_id="q18")
    return loop, emu


def test_a_count_buy_purchases_only_what_is_missing(monkeypatch):
    from pokemon_agent.games.pokemon_red import shop as shop_macro
    loop, emu = _mart_loop({"item_count": [4, 5]})
    emu.write_memory(0xD31D, 1); emu.write_memory(0xD31E, 4); emu.write_memory(0xD31F, 2); emu.write_memory(0xD320, 0xFF)
    bought = {}

    def buy(e, item, qty):
        bought.update(item=item, qty=qty)
        e.write_memory(0xD31F, 2 + qty)            # the balls arrive
        return {"ok": True, "item": item, "qty": qty}
    monkeypatch.setattr(shop_macro, "at_shop_menu", lambda e: True)
    monkeypatch.setattr(shop_macro, "shop_buy", buy)
    monkeypatch.setattr(shop_macro, "close_shop", lambda e, tries=8: None)
    obs, shot = loop.builder.build(capture_screenshot=False)
    loop._maybe_shop(obs, shot)
    assert bought == {"item": "Poke Ball", "qty": 3} and loop._plan_steps[0].status == "active"


def test_a_talk_at_the_counter_that_names_no_item_wedges_with_a_hint(monkeypatch):
    from pokemon_agent.games.pokemon_red import shop as shop_macro
    loop, emu = _mart_loop({"verify": "does the bag contain at least 4 Potions?"})
    monkeypatch.setattr(shop_macro, "at_shop_menu", lambda e: True)
    monkeypatch.setattr(shop_macro, "close_shop", lambda e, tries=8: None)
    obs, shot = loop.builder.build(capture_screenshot=False)
    loop._maybe_shop(obs, shot)
    step = loop._plan_steps[0]
    assert step.status == "wedged" and "has_item:<item>>=N" in step.wedge_reason and loop._l1_event
