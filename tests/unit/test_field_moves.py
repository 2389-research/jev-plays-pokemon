"""Cut as a choice: the knowledge (who can learn it), trees on the map, teach orders, cut steps.

runs/sleeves-misty ended in Vermilion with HM01 next on the story and a party (Wartortle, Weedle,
Magikarp) where nobody can learn Cut — the agent needs to KNOW that, and to be able to choose to
teach and use it; the router does the walking (interaction routing) and the menu work."""
from __future__ import annotations

from types import SimpleNamespace

import pokemon_agent.agent.reason_loop as RL
from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent import goals as goals_mod
from pokemon_agent.agent.plan import AgentPlan, Directive, Intent, ReflectionPlan
from pokemon_agent.agent.planner_llm import BRAINSTORM_SYSTEM, DECIDE_SYSTEM, Planner
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep
from pokemon_agent.agent.session import Session
from pokemon_agent.agent.world_map import WorldMap
from pokemon_agent.core.models import GoalState, WaitAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.games.pokemon_red.map_reader import _classify
from pokemon_agent.games.pokemon_red.tmhm import can_learn, hm_line, learners, machine_move
from pokemon_agent.observations.builder import ObservationBuilder


def test_learnsets_know_who_can_learn_cut():
    assert can_learn("Oddish", "Cut") and can_learn("Farfetchd", "Cut") and can_learn("Beedrill", "Cut")
    assert not can_learn("Wartortle", "Cut") and not can_learn("Magikarp", "Cut")
    assert "Paras" in learners("Cut")
    assert machine_move("HM01 Cut") == "Cut" and machine_move("TM28 Dig") == "Dig" and machine_move("Potion") is None


def test_the_party_view_says_what_an_evolution_adds():
    assert hm_line("Weedle") == "none; as Beedrill: Cut"
    assert hm_line("Wartortle") == "Surf, Strength"
    assert hm_line("Caterpie") is None


def test_cut_trees_are_their_own_terrain_class():
    assert _classify(0x3D, 0, {0x2C}, 0x52, set()) == "cut_tree"          # OVERWORLD
    assert _classify(0x50, 7, {0x2C}, 0xFF, set()) == "cut_tree"          # GYM
    assert _classify(0x3D, 17, {0x2C}, 0xFF, set()) == "wall"             # the id means nothing elsewhere
    assert "T = small tree CUT can remove" in WorldMap.SEMANTIC_LEGEND and WorldMap.SEMANTIC_SYMBOLS["cut_tree"] == "T"


def test_done_when_cut_parses_to_a_ram_check():
    assert Planner._parse_done_when("cut:15,18", 5) == {"tree_cut": [5, 15, 18]}
    assert Planner._parse_done_when("cut:(15, 18)", 5) == {"tree_cut": [5, 15, 18]}
    assert Planner._parse_done_when("cut:somewhere", 5) is None


def test_l1_knows_the_options():
    for prompt in (BRAINSTORM_SYSTEM, DECIDE_SYSTEM):
        assert "FIELD MOVES" in prompt and '"teach"' in prompt and "cut:15,18" in prompt


def test_a_teach_order_is_detected_and_stored_like_lead():
    plan = AgentPlan()
    ch = goals_mod.detect_change(plan, {"teach": {"move": "Cut", "who": "Sprout", "forget": ""}}, step_edit=False)
    assert ch is not None and ch.teach == {"move": "Cut", "who": "Sprout"}
    goals_mod.apply_change(plan, ch, pre_status={})
    assert plan.battle_goals["teach"] == {"move": "Cut", "who": "Sprout"}
    assert goals_mod.detect_change(plan, {"teach": {"move": "Cut"}}, step_edit=False) is None   # no "who"


class StubReasoner:
    def reflect(self, **kw):
        return ReflectionPlan(next_objective="go"), 0, {}

    def step(self, **kw):
        return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}


def _loop():
    events = []
    emu = FakeEmulator(map_id=5)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=StubReasoner(), session=Session(GoalState(primary="g", current="g")),
                         vision=False, reflect_every=100, goal_map=2, on_event=lambda k, p: events.append((k, p)))
    return loop, events


def test_trees_on_the_map_are_objects_you_can_name():
    loop, _ = _loop()
    loop.world.terrain[5] = {(15, 18): "cut_tree", (15, 17): "floor"}
    trees = [o for o in loop._objects_here(5) if o.get("kind") == "cut_tree"]
    assert trees == [{"name": "Cut tree at (15,18)", "x": 15, "y": 18, "kind": "cut_tree"}]


def test_a_teach_order_without_the_hm_is_reported_not_retried(monkeypatch):
    loop, events = _loop()
    loop._plan = AgentPlan(battle_goals={"teach": {"move": "Cut", "who": "Sprout"}})
    monkeypatch.setattr(RL, "read_party", lambda emu: [{"nickname": "SPROUT", "species": "Oddish", "moves": []}])
    import pokemon_agent.games.pokemon_red.game_state as gs
    monkeypatch.setattr(gs, "read_items", lambda emu: [{"item": "Potion", "qty": 2}])
    assert loop._maybe_teach() is True
    assert "teach" not in loop._plan.battle_goals                        # consumed: one attempt per order
    ev = next(p for k, p in events if k == "teach")
    assert ev["ok"] is False and "no TM/HM for Cut" in ev["detail"]
    assert loop._l1_event is True
    assert any("couldn't teach Cut" in (e.get("text") or "") for e in loop.episode.events)


def test_facing_a_tree_nobody_can_cut_wedges_with_the_reason(monkeypatch):
    loop, events = _loop()
    loop._plan_steps = []
    d = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 5, "sprite": "Cut tree at (15,18)"},
                  success={"tree_cut": [5, 15, 18]}, quest_id=None)
    monkeypatch.setattr(RL, "read_party", lambda emu: [{"nickname": "DYLAN", "species": "Wartortle", "moves": ["Bite"]},
                                                        {"nickname": "BUZZ", "species": "Weedle", "moves": []}])
    import pokemon_agent.games.pokemon_red.party_menu as pm
    monkeypatch.setattr(pm, "knows", lambda emu, slot, move: False)
    mv = loop._use_cut({"x": 15, "y": 18}, {"kind": "approach_npc", "sprite": "Cut tree at (15,18)"}, d)
    assert mv is None
    assert loop._approach_block == ("can't cut the tree at (15,18): nobody in the party knows Cut and nobody in "
                                    "the party can learn it")


# ---- routing follows the live map (runs/sleeves-vermilion) ------------------------------------------
def test_a_cut_tree_joins_the_components_the_rip_kept_apart():
    """After Cut the gym door's static component stayed unreachable: the route was "no known way" and L1
    recut the tree 4 times believing it had regrown. Routing now floods the LIVE walkable set."""
    from pokemon_agent.agent.portal_graph import PortalGraph
    pg = PortalGraph.load()
    walk = set(pg._grid(5))                                   # Vermilion City with the tree standing
    assert pg.route(5, pg.components_reachable(5, 15, 17, walk), 92) is None
    cut = walk | {(15, 18)}
    route = pg.route(5, pg.components_reachable(5, 15, 17, cut), 92)
    assert route and route[-1]["dest_map"] == 92

