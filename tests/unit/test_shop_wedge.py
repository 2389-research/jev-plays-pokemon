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
    assert loop._prev_map == 1                        # seeded from history (fallback)
    emu.write_memory(0xD365, 1)                       # the game's wLastMap inside the Viridian Mart
    assert loop._resolve_exits([{"x": 3, "y": 7, "dest_map": 255}], 42)[0]["dest_map"] == 1


# ---- spec 2026-09-23-grind-talk-shop-fixes F3 ------------------------------------------------------
def _bag(emu, pairs):
    emu.write_memory(0xD31D, len(pairs))
    for i, (iid, q) in enumerate(pairs):
        emu.write_memory(0xD31E + 2 * i, iid)
        emu.write_memory(0xD31F + 2 * i, q)
    emu.write_memory(0xD31E + 2 * len(pairs), 0xFF)


def test_satisfied_buy_closes_the_counter_instead_of_rebuying(monkeypatch):
    """brock-goals: the Antidote buy succeeded 16x ($1,600) — the counter reopened and the menu path
    re-bought though has_item was already true."""
    events = []
    loop, emu = _loop(events)
    _bag(emu, [(20, 1)])                                   # a Potion is already in the bag
    calls = _patch(monkeypatch, result={"ok": True})
    obs, shot = loop.builder.build(capture_screenshot=False)
    out = loop._maybe_shop(obs, shot)
    assert calls["buy"] == 0 and calls["close"] == 1 and out is not None


def test_ok_buy_that_did_not_reach_the_bag_is_a_failure_with_money(monkeypatch):
    """brock-goals2: 9x shop_buy ok:true at Pewter Mart with $177 (Potion $300), 0 Potions."""
    events = []
    loop, emu = _loop(events)
    emu.write_memory(0xD347, 0x00); emu.write_memory(0xD348, 0x01); emu.write_memory(0xD349, 0x77)   # $177 BCD
    calls = _patch(monkeypatch, result={"ok": True, "item": "Potion", "qty": 1})
    obs, shot = loop.builder.build(capture_screenshot=False)
    loop._maybe_shop(obs, shot)
    step = loop._plan_steps[0]
    assert calls["buy"] == 1 and step.status == "wedged" and loop._l1_event is True
    assert "did not go through" in step.wedge_reason and "$177" in step.wedge_reason
    buy = [p for k, p in events if k == "shop_buy"][-1]
    assert buy["verified"] is False


def test_game_signals_include_money():
    from pokemon_agent.agent.signals import game_signals
    emu = FakeEmulator(map_id=42)
    emu.write_memory(0xD347, 0x00); emu.write_memory(0xD348, 0x30); emu.write_memory(0xD349, 0x00)   # $3000
    assert game_signals(emu)["money"] == 3000


def test_return_warp_uses_the_games_wlastmap_not_the_previous_map():
    """brock-goals2 steps 1071-1239: at Viridian Forest North Gate (47) the north LAST_MAP doors resolved
    to the forest (_prev_map=51) instead of Route 2 (wLastMap=13) -> the off-route veto blocked the exit."""
    events = []
    loop, emu = _loop(events)
    loop._prev_map = 51
    emu.write_memory(0xD365, 13)
    out = loop._resolve_exits([{"x": 5, "y": 0, "dest_map": 255}, {"x": 5, "y": 7, "dest_map": 51}], 47)
    assert [e["dest_map"] for e in out] == [13, 51]
    emu.write_memory(0xD365, 47)                      # nonsense (== current map) -> fall back to _prev_map
    assert loop._resolve_exits([{"x": 5, "y": 0, "dest_map": 255}], 47)[0]["dest_map"] == 51


def test_l1_sees_item_quantities():
    from pokemon_agent.agent.signals import game_signals
    emu = FakeEmulator(map_id=42)
    _bag(emu, [(0x0B, 15), (20, 1)])                  # 15 Antidotes, 1 Potion
    assert game_signals(emu)["items"] == ["Antidote x15", "Potion"]


def test_a_purchase_is_not_a_whiteout_setback():
    from pokemon_agent.actions.stuck_detector import StuckDetector
    sd = StuckDetector()
    sd._last_money = 3000
    sd.note_spend(2700)
    assert sd._last_money == 2700
