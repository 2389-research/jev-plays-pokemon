"""Training the bench (runs/cerulean-team4: only Wartortle ever gained EXP — the lead always starts the
battle and nothing else fights — so L1's only option against Misty was 'heal and retry').

L1 decides: it can set a LEAD (moved to the front of the party with the in-game START > POKEMON > SWITCH
menu, so it starts battles and earns EXP) and grind a single member with done_when "level:<name>>=N".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pokemon_agent.agent import goals as G
from pokemon_agent.agent.plan import AgentPlan
from pokemon_agent.agent.planner_llm import Planner

ROM = Path("roms/pokemon_red.gb")
SAVE = Path("runs/verify-faint-20260923/latest.state")


def test_member_level_criterion_parses_and_is_a_valid_goal():
    assert Planner._parse_done_when("level:Sophie>=16", 22) == {"member_level": ["Sophie", 16]}
    assert G.clean_criterion("level:Sophie>=16") == "level:Sophie>=16"


def test_member_level_predicate_matches_nickname_or_species(monkeypatch):
    from pokemon_agent.games.pokemon_red import predicates
    party = [{"species": "Wartortle", "nickname": "Dylan", "level": 23},
             {"species": "Paras", "nickname": "Sophie", "level": 15}]
    monkeypatch.setattr(predicates, "read_party", lambda emu: party, raising=False)
    assert not predicates.evaluate({"member_level": ["Sophie", 16]}, object())
    party[1]["level"] = 16
    assert predicates.evaluate({"member_level": ["Sophie", 16]}, object())
    assert predicates.evaluate({"member_level": ["paras", 16]}, object())


def test_a_member_level_travel_step_is_a_grind():
    from pokemon_agent.agent.plan import Directive, Intent
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 35},
                  success={"member_level": ["Sophie", 16]})
    assert ReasoningLoop._is_grind(d)


def test_lead_change_detection():
    p = AgentPlan()
    ch = G.detect_change(p, {"lead": "Sophie"}, step_edit=False)
    assert ch is not None and ch.lead == "Sophie"
    G.apply_change(p, ch, pre_status={})
    assert p.battle_goals["lead"] == "Sophie"
    assert G.detect_change(p, {"lead": "sophie"}, step_edit=False) is None      # echo
    ch = G.detect_change(p, {"lead": "clear"}, step_edit=False)
    G.apply_change(p, ch, pre_status={})
    assert not p.battle_goals.get("lead")


def test_decide_passes_lead_through():
    import json

    class FP:
        def chat_json(self, s, st, image=None):
            return json.dumps({"add": [], "remove": [], "lead": "Sophie"}), 0, {}
    out = Planner(strategist=FP()).l1_decide({"plan": []}, {"assessment": "x"})
    assert out["lead"] == "Sophie"


@pytest.mark.skipif(not (ROM.exists() and SAVE.exists()), reason="ROM / party save not present")
def test_swap_to_front_on_the_real_save():
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.games.pokemon_red import party_menu
    from pokemon_agent.games.pokemon_red.game_state import read_party
    emu = PyBoyEmulator(str(ROM), window="null")
    emu.load_state(SAVE)
    emu.tick(10)
    party_menu.clear_text(emu)
    assert party_menu.swap_to_front(emu, 4) is True
    assert [p["nickname"] for p in read_party(emu)][0] == "Sophie"
    emu.close()


def test_loop_moves_the_lead_to_the_front_once(monkeypatch):
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.plan import ReflectionPlan
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import ReasonStep
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import GoalState, WaitAction
    from pokemon_agent.emulator.fake_emulator import FakeEmulator
    from pokemon_agent.games.pokemon_red import party_menu
    import pokemon_agent.agent.reason_loop as rl
    from pokemon_agent.observations.builder import ObservationBuilder

    class Stub:
        def reflect(self, **kw):
            return ReflectionPlan(next_objective="go"), 0, {}

        def step(self, **kw):
            return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}
    events = []
    emu = FakeEmulator(map_id=35)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu), reasoner=Stub(),
                         session=Session(GoalState(primary="g", current="g")), vision=False, reflect_every=100,
                         goal_map=2, on_event=lambda k, p: events.append((k, p)))
    loop._plan = AgentPlan(battle_goals={"lead": "Sophie"})
    party = [{"species": "Wartortle", "nickname": "Dylan", "level": 23},
             {"species": "Paras", "nickname": "Sophie", "level": 8}]
    monkeypatch.setattr(rl, "read_party", lambda e: party, raising=False)
    calls = []

    def swap(e, slot):
        calls.append(slot)
        party.insert(0, party.pop(slot))
        return True
    monkeypatch.setattr(party_menu, "swap_to_front", swap)
    assert loop._maybe_apply_lead() is True and calls == [1]
    assert loop._maybe_apply_lead() is False and calls == [1]          # already in front
    assert any(k == "lead_set" for k, _ in events)


# ---- L1 learns from losing (verify-train: lost to Misty's Starmie, healed, walked straight back in) ----
def test_battle_result_classification():
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    f = ReasoningLoop._classify_battle
    assert f(alive=0, enemy_hp=12, party_grew=False) == "lost"
    assert f(alive=2, enemy_hp=0, party_grew=False) == "won"
    assert f(alive=3, enemy_hp=5, party_grew=True) == "caught"
    assert f(alive=3, enemy_hp=5, party_grew=False) == "ended"


def test_a_lost_battle_is_recorded_for_l1_and_triggers_a_review():
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.plan import ReflectionPlan
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
    events = []
    emu = FakeEmulator(map_id=65)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu), reasoner=Stub(),
                         session=Session(GoalState(primary="g", current="g")), vision=False, reflect_every=100,
                         goal_map=2, on_event=lambda k, p: events.append((k, p)))
    loop._battle_track = {"start": 10, "where": "Cerulean Gym", "trainer": True, "opponents": ["Staryu", "Starmie"],
                          "alive": 0, "enemy_hp": 30, "party_size": 5}
    loop._record_battle_end(party_size_now=5)
    rec = loop._battle_log[-1]
    assert rec["result"] == "lost" and rec["opponents"] == ["Staryu", "Starmie"] and rec["where"] == "Cerulean Gym"
    assert loop._l1_event is True and any(k == "battle_lost" for k, _ in events)
