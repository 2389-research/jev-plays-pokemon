import json
from pokemon_agent.agent.planner_llm import Planner

class FP:
    def __init__(self, obj): self.obj = obj
    def chat_json(self, system, state, image=None): return json.dumps(self.obj), 0, {}

def _ctx():
    return {"current_map": {"id": 1, "name": "Viridian City"}, "party": ["Squirtle L8 20/26"],
            "items": [], "badges": 0, "plan": [], "signals": {"hp_frac": 0.8, "blocked_for_n": 7},
            "mission": "", "milestone": ""}

def test_revise_quests_parses_add_and_change():
    obj = {"assessment": "blocked at Viridian north -> need parcel", "change": True,
           "mission": "reach Pewter", "milestone": "deliver Oak's Parcel",
           "add": [{"map": 42, "talk": True, "who": "clerk", "done_when": "has_item:Oak's Parcel", "why": "get parcel"}],
           "remove": []}
    p = Planner(goal_map=2, strategist=FP(obj))
    out = p.revise_quests(emu=None, context=_ctx())
    assert out["change"] and out["add"][0]["map"] == 42 and out["milestone"]

def test_revise_quests_no_change_passthrough():
    p = Planner(goal_map=2, strategist=FP({"assessment": "on track", "change": False, "add": [], "remove": []}))
    assert p.revise_quests(emu=None, context=_ctx())["change"] is False

def test_revise_quests_drops_invalid_done_when():
    obj = {"assessment": "x", "change": True, "add": [{"map": 3, "talk": False, "done_when": "nonsense", "why": "x"}], "remove": []}
    p = Planner(goal_map=2, strategist=FP(obj))
    out = p.revise_quests(emu=None, context=_ctx())
    assert out["add"] == []                     # invalid criterion dropped

def test_revise_quests_garbage_is_no_change():
    class Bad:
        def chat_json(self, s, st, image=None): return "not json", 0, {}
    p = Planner(goal_map=2, strategist=Bad())
    assert p.revise_quests(emu=None, context=_ctx())["change"] is False

def test_revise_quests_no_provider_is_no_change():
    p = Planner(goal_map=2)          # no strategist/provider
    assert p.revise_quests(emu=None, context=_ctx())["change"] is False
