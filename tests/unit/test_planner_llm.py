"""Planner: arbiter intent -> concrete Directive (target + machine-checkable success)."""
from types import SimpleNamespace

from pokemon_agent.agent.planner_llm import Planner, intent_for_need
from pokemon_agent.agent.plan import Intent
from pokemon_agent.agent.world_graph import full_kanto_graph
from pokemon_agent.emulator.fake_emulator import FakeEmulator


def _mem():
    return SimpleNamespace(graph=full_kanto_graph())


def test_need_to_intent_mapping():
    assert intent_for_need("survive") == Intent.HEAL
    assert intent_for_need("readiness") == Intent.GRIND
    assert intent_for_need("progress") == Intent.TRAVEL
    assert intent_for_need("battle") == Intent.BATTLE


def test_travel_directive_has_goal_map_success():
    p = Planner(goal_map=2, level_target=12)
    emu = FakeEmulator(map_id=0)  # Pallet Town
    d = p.plan(Intent.TRAVEL, emu, _mem())
    assert d.intent == Intent.TRAVEL
    assert d.success == {"on_map": 2}
    assert d.target and d.target["map"] == 2
    assert "next_map" in d.target  # the graph knows the next hop toward Pewter


def test_travel_when_already_at_goal_is_not_trivially_complete():
    p = Planner(goal_map=2)
    emu = FakeEmulator(map_id=2)  # already at Pewter
    d = p.plan(Intent.TRAVEL, emu, _mem())
    # success keys off on_map == goal; at the goal the executive will detect completion
    assert d.success.get("on_map") == 2


def test_grind_directive_targets_level_and_heads_toward_goal():
    p = Planner(goal_map=2, level_target=12)
    d = p.plan(Intent.GRIND, FakeEmulator(map_id=0), _mem())
    assert d.intent == Intent.GRIND
    assert d.success == {"level": ">=12"}          # completes on level, not arrival
    assert d.target and d.target["map"] == 2       # but heads toward the goal (grass en route)
    assert d.target_bearing                         # so the servo walks it


def test_heal_directive_targets_hp():
    p = Planner(goal_map=2, heal_hp=0.8)
    d = p.plan(Intent.HEAL, FakeEmulator(map_id=1), _mem())
    assert d.intent == Intent.HEAL
    assert d.success == {"hp_frac": ">=0.8"}


def test_battle_directive_completes_when_battle_ends():
    d = Planner().plan(Intent.BATTLE, FakeEmulator(), _mem())
    assert d.intent == Intent.BATTLE and d.success == {"in_battle": 0}


class FakeProvider:
    """Returns a scripted planner JSON and records the state it was given."""

    def __init__(self, content):
        self.content = content

    def chat_json(self, system_prompt, user, image=None):
        self.state = user
        return self.content, 5, {}


def test_llm_reroute_picks_intermediate_reachable_map():
    # provider reroutes to Viridian City (1) instead of the final goal Pewter (2)
    prov = FakeProvider('{"target_map": 1, "reason": "route 2 is blocked north, go via Viridian"}')
    p = Planner(goal_map=2, provider=prov)
    d = p.plan(Intent.TRAVEL, FakeEmulator(map_id=0), _mem(), why="stuck heading north")
    assert d.target["map"] == 1                # honored the LLM's reroute
    assert d.success == {"on_map": 1}          # success keys off the chosen subgoal
    assert "blocked" in d.reason
    assert prov.state["why_replan"] == "stuck heading north"  # Inner Monologue passed through


def test_llm_unreachable_or_bad_target_falls_back_to_goal():
    prov = FakeProvider('{"target_map": 999, "reason": "nonsense"}')  # 999 is unreachable
    p = Planner(goal_map=2, provider=prov)
    d = p.plan(Intent.TRAVEL, FakeEmulator(map_id=0), _mem())
    assert d.target["map"] == 2 and d.success == {"on_map": 2}  # fell back to the goal


def test_llm_garbage_json_falls_back_to_goal():
    p = Planner(goal_map=2, provider=FakeProvider("not json at all"))
    d = p.plan(Intent.TRAVEL, FakeEmulator(map_id=0), _mem())
    assert d.target["map"] == 2  # deterministic fallback on parse failure


def test_stuck_replan_uses_llm_waypoint():
    # when stuck, LunaRoute picks a concrete tile and the servo is bound to route there
    prov = FakeProvider('{"x": 11, "y": 3, "reason": "west then up the open corridor"}')
    p = Planner(goal_map=2, provider=prov)
    ctx = {"map_view": ["ruler"], "player": {"x": 19, "y": 9, "map_id": 1},
           "exits": [], "goal_dir": "north toward map 13"}
    d = p.plan(Intent.GRIND, FakeEmulator(map_id=1), _mem(),
               why="got stuck pursuing the grind directive; reroute", context=ctx)
    assert d.target["kind"] == "waypoint" and (d.target["x"], d.target["y"]) == (11, 3)
    assert d.success == {"at_xy": [1, 11, 3]}   # arrives-at termination on the current map
    assert d.target_bearing                      # servo will BFS to it


def test_stuck_without_context_or_provider_is_normal_directive():
    # no map-view context -> no waypoint; falls back to the normal deterministic directive
    p = Planner(goal_map=2, provider=FakeProvider('{"x":1,"y":1}'))
    d = p.plan(Intent.GRIND, FakeEmulator(map_id=1), _mem(), why="got stuck")
    assert d.target.get("kind") != "waypoint"


def test_strategize_builds_quest_from_llm_steps():
    # tier-2 strategist returns an ordered quest; talk steps expand to a travel+talk pair
    strat = FakeProvider('{"plan":"deliver Oak parcel","steps":['
                         '{"map":42,"talk":true,"why":"get parcel from mart clerk"},'
                         '{"map":0,"talk":true,"why":"deliver to Oak"},'
                         '{"map":1,"talk":false,"why":"return to Viridian"}]}')
    p = Planner(goal_map=2, strategist=strat)
    quest = p.strategize(FakeEmulator(map_id=1), _mem(), why="blocked by old man")
    intents = [(d.intent.value, d.target.get("map"), d.success) for d in quest]
    # map42 travel+talk, map0 travel+talk, map1 travel  -> 5 directives
    assert [i[0] for i in intents] == ["travel", "talk_to", "travel", "talk_to", "travel"]
    assert intents[0][2] == {"on_map": 42} and intents[1][2] == {"talked_on_map": 42}


def test_strategize_empty_without_provider():
    assert Planner(goal_map=2).strategize(FakeEmulator(map_id=1), _mem(), why="x") == []


class FakeKB:
    def __init__(self, texts):
        self.texts = texts

    def query_texts(self, q, top_k=5):
        self.last_q = q
        return list(self.texts)


class SeqProvider:
    """Returns scripted responses in order, recording each state it was handed."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.states = []

    def chat_json(self, system_prompt, user, image=None):
        self.states.append(user)
        return self.responses.pop(0), 5, {}


def test_strategist_uses_search_tool_then_plans():
    # the strategist first calls the search 'tool', gets results fed back, then plans
    prov = SeqProvider('{"search":["how to leave Viridian City"]}',
                       '{"plan":"deliver parcel","steps":[{"map":42,"talk":true,"why":"get parcel"}]}')
    kb = FakeKB(["Oak's Parcel gate: get the parcel from the Viridian Mart clerk"])
    searches = []
    p = Planner(goal_map=2, strategist=prov, knowledge=kb)
    p.on_search = lambda q, n: searches.append((q, n))
    quest = p.strategize(FakeEmulator(map_id=1), _mem(), why="blocked by old man north of Viridian")
    assert kb.last_q == "how to leave Viridian City"                  # model chose the query
    assert searches == [("how to leave Viridian City", 1)]           # tool-call surfaced
    # the second call received the search results back as KNOWLEDGE_GATHERED
    assert prov.states[1]["knowledge_gathered"][0]["results"] == \
        ["Oak's Parcel gate: get the parcel from the Viridian Mart clerk"]
    assert len(quest) == 2  # travel + talk from the finalized plan


def test_strategist_can_skip_search_and_plan_directly():
    prov = SeqProvider('{"plan":"x","steps":[{"map":42,"talk":false}]}')
    p = Planner(goal_map=2, strategist=prov, knowledge=FakeKB(["unused"]))
    quest = p.strategize(FakeEmulator(map_id=1), _mem(), why="blocked")
    assert len(quest) == 1 and len(prov.states) == 1  # no search round


def test_done_when_criteria_map_to_predicates():
    p = Planner(goal_map=2)
    assert p._parse_done_when("on_map", 42) == {"on_map": 42}
    assert p._parse_done_when("has_item:Oak's Parcel", 42) == {"has_item": 70}
    assert p._parse_done_when("no_item:Oaks Parcel", 42) == {"no_item": 70}
    assert p._parse_done_when("level>=12", 42) == {"level": ">=12"}
    assert p._parse_done_when("badges>=1", 42) == {"badges": ">=1"}
    assert p._parse_done_when("talked", 42) == {"talked_on_map": 42}
    assert p._parse_done_when("verify:did I get the pokedex?", 42) == {"verify": "did I get the pokedex?"}
    assert p._parse_done_when("nonsense", 42) is None


def test_strategize_uses_model_acceptance_criteria():
    # the clerk step's acceptance is has_item(parcel), NOT a loose "talked" — no premature done
    strat = FakeProvider('{"plan":"parcel","steps":['
                         '{"map":42,"talk":true,"done_when":"has_item:Oak\'s Parcel","why":"get parcel"},'
                         '{"map":40,"talk":true,"done_when":"no_item:Oak\'s Parcel","why":"deliver"}]}')
    p = Planner(goal_map=2, strategist=strat)
    quest = p.strategize(FakeEmulator(map_id=1), _mem(), why="blocked")
    talk_successes = [d.success for d in quest if d.intent.value == "talk_to"]
    assert talk_successes[0] == {"has_item": 70}   # get-parcel step
    assert talk_successes[1] == {"no_item": 70}    # delivered step
