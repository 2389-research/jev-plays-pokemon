"""Shop failure must hand control back to L1 (runs/brock-goals-20260923, steps ~420-1500).

Incident: L1 queued "buy Potions at the Viridian Mart", which doesn't stock them. The buy macro failed
("not sold here"), backed out and wedged the step — but the clerk's "anything else?" dialogue reopened
the counter menu, the menu path re-fired the SAME buy (the directive was still set), and because the
menu/dialogue paths return before _manage_directive, the wedged step never advanced and L1 was never
armed (_force_reflect is the legacy reflect flag). ~1100 steps of re-buying. Also L1 only ever saw
status "wedged", never why, so it couldn't learn the shelf.
"""
from __future__ import annotations

from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.plan import Directive, Intent, ReflectionPlan
from pokemon_agent.agent.quest_reconciler import QuestStep
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep
from pokemon_agent.agent.session import Session
from pokemon_agent.core.models import GoalState, WaitAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.games.pokemon_red import shop as shop_macro
from pokemon_agent.observations.builder import ObservationBuilder
import pokemon_agent.agent.reason_loop as rl


class StubReasoner:
    def reflect(self, **kw):
        return ReflectionPlan(next_objective="go"), 0, {}

    def step(self, **kw):
        return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}


def _loop(events):
    emu = FakeEmulator(map_id=42)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=StubReasoner(), session=Session(GoalState(primary="g", current="g")),
                         vision=False, reflect_every=100, goal_map=2,
                         on_event=lambda k, p: events.append((k, p)))
    loop._plan_steps = [QuestStep(id="q22", map=42, talk=True, done_when="has_item:Potion", status="active")]
    loop._directive = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 42, "item": "Potion"},
                                success={"has_item": 20}, quest_id="q22")
    return loop, emu


def _patch(monkeypatch, *, at_menu=True, result=None):
    calls = {"buy": 0, "close": 0}

    def buy(emu, item, qty):
        calls["buy"] += 1
        return result or {"ok": False, "reason": f"{item!r} not sold here",
                          "shop": ["Poke Ball", "Antidote", "Parlyz Heal", "Burn Heal"]}

    def close(emu, tries=8):
        calls["close"] += 1
    monkeypatch.setattr(shop_macro, "at_shop_menu", lambda emu: at_menu)
    monkeypatch.setattr(shop_macro, "shop_buy", buy)
    monkeypatch.setattr(shop_macro, "close_shop", close)
    return calls


def test_failed_buy_arms_l1_and_records_why(monkeypatch):
    events = []
    loop, _ = _loop(events)
    calls = _patch(monkeypatch)
    obs, shot = loop.builder.build(capture_screenshot=False)
    assert loop._maybe_shop(obs, shot) is not None and calls["buy"] == 1
    step = loop._plan_steps[0]
    assert step.status == "wedged" and loop._l1_event is True
    assert "not sold here" in step.wedge_reason and "Antidote" in step.wedge_reason   # the shelf, for L1


def test_wedged_buy_is_not_retried_and_the_counter_is_closed(monkeypatch):
    events = []
    loop, _ = _loop(events)
    calls = _patch(monkeypatch)
    obs, shot = loop.builder.build(capture_screenshot=False)
    loop._maybe_shop(obs, shot)                       # fails -> wedged
    out = loop._maybe_shop(obs, shot)                 # the clerk reopened the counter menu
    assert calls["buy"] == 1                          # never re-fires the same failed buy
    assert out is not None and calls["close"] == 1    # backs out of the counter instead of looping


def test_counter_menu_without_a_buy_directive_is_closed(monkeypatch):
    events = []
    loop, _ = _loop(events)
    loop._directive = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 13}, success={"on_map": 13})
    calls = _patch(monkeypatch)
    obs, shot = loop.builder.build(capture_screenshot=False)
    out = loop._maybe_shop(obs, shot)
    assert calls["buy"] == 0 and calls["close"] == 1 and out is not None


def test_l1_plan_rows_carry_the_wedge_reason(monkeypatch):
    events = []
    loop, _ = _loop(events)
    loop._plan_steps[0].status = "wedged"
    loop._plan_steps[0].wedge_reason = "can't buy Potion here: 'Potion' not sold here (shelf: Poke Ball, Antidote)"
    seen = {}
    monkeypatch.setattr(rl, "run_l1_pipeline",
                        lambda emu, ctx, planner, *, hard_event, on_trace=None: seen.update(ctx))
    obs, _ = loop.builder.build(capture_screenshot=False)
    loop._run_l1(obs)
    row = seen["plan"][0]
    assert row["status"] == "wedged" and "not sold here" in row["why_wedged"]


def test_resume_inside_a_building_resolves_the_return_warp():
    """runs/shopfix-verify-20260923: resumed inside the Viridian Mart, _prev_map was None, so the door's
    0xFF 'return' warp never resolved to Viridian City and the agent wandered the Mart for ~150 steps."""
    from pokemon_agent.agent.memory import AgentMemory
    mem = AgentMemory()
    mem.map_history = [1, 41, 1, 42]
    emu = FakeEmulator(map_id=42)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=StubReasoner(), session=Session(GoalState(primary="g", current="g")),
                         vision=False, reflect_every=100, goal_map=2, memory=mem)
    assert loop._prev_map == 1
    assert loop._resolve_exits([{"x": 3, "y": 7, "dest_map": 255}], 42)[0]["dest_map"] == 1
