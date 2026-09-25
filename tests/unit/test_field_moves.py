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



# ---- one interaction per step (runs/sleeves-surge: the trash-can puzzle) ---------------------------
def test_used_is_a_step_criterion_that_needs_a_single_who():
    from pokemon_agent.agent.l1_pipeline import validate_step
    assert Planner._parse_done_when("used", 92) == {"used": True}
    assert validate_step({"kind": "action", "map": 92, "talk": True, "who": "trash can at (9,9)", "done_when": "used"})[0]
    assert not validate_step({"kind": "action", "map": 92, "talk": False, "done_when": "used"})[0]


def test_used_steps_on_one_map_are_not_deduped_into_one():
    from pokemon_agent.agent.quest_reconciler import reconcile_quests
    ids = iter(f"q{i}" for i in range(1, 99))
    add = [{"kind": "action", "map": 92, "talk": True, "who": f"trash can at ({x},9)", "done_when": "used"}
           for x in (1, 3, 5)] + [{"kind": "action", "map": 92, "talk": True, "who": "trash can at (1,9)",
                                   "done_when": "used"}]
    out = reconcile_quests([], {"add": add, "remove": []}, next_id=lambda: next(ids))
    assert [s.who for s in out] == ["trash can at (1,9)", "trash can at (3,9)", "trash can at (5,9)"]


def test_repeated_objects_are_named_with_their_tile():
    loop, _ = _loop()
    cans = [o["name"] for o in loop._objects_here(92) if "trash can" in o["name"]]
    assert len(cans) == 16 and "trash can at (9,9)" in cans
    assert any(o["name"] == "gym statue at (3,14)" for o in loop._objects_here(92))


def test_used_counts_only_the_named_one_answering():
    loop, _ = _loop()
    d = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 92, "sprite": "trash can at (1,11)"},
                  success={"used": True}, quest_id="q7")
    loop._directive = d

    def press_and_open(at):
        loop._interact_with({}, at, {"x": at[0], "y": at[1], "sprite": "trash can"})
        loop.session.step += 1
        loop._check_answer(SimpleNamespace(game_state={"dialog_active": True, "facing": {
            "facing_sprite": None, "front_tile": list(at)}}))
    press_and_open((3, 11))                                   # a different can answered
    assert not loop._directive_satisfied(d)
    press_and_open((1, 11))
    assert loop._directive_satisfied(d)


def test_a_can_is_credited_not_the_person_standing_behind_it():
    """runs/sleeves-surge2: "only trash here" was credited to the Gentleman two tiles behind a can."""
    loop, _ = _loop()
    gentleman = {"x": 9, "y": 7, "sprite": "Gentleman", "slot": 3, "kind": "person"}
    for active, lines in ((True, ["Nope, there's only", "trash here."]), (False, [])):
        loop.session.step += 1
        loop._observe_heard(SimpleNamespace(player=SimpleNamespace(x=9, y=10, map_id=92, facing="north"),
                                            game_state={"dialog_active": active, "dialog_lines": lines,
                                                        "facing": {"front_tile": [9, 9], "facing_sprite": gentleman}}))
    m = loop.heard.by_map[92]["messages"][-1]
    assert (m["speaker"], m["at"]) == ("trash can", [9, 9])


# ---- scripted turn-backs are remembered (runs/sleeves-surge3: the thirsty Saffron guards) -----------
def test_a_scripted_push_back_marks_the_portal_and_survives_leaving_by_another_door(monkeypatch):
    loop, events = _loop()
    loop.portals = type("PG", (), {
        "portals": {"route5gate:warp1": {"id": "route5gate:warp1", "map": 70, "coord": [3, 5], "kind": "warp",
                                          "dest_map": 16, "label": "Route5Gate south door"},
                    "route5gate:warp3": {"id": "route5gate:warp3", "map": 70, "coord": [3, 0], "kind": "warp",
                                         "dest_map": 16, "label": "Route5Gate north door"}},
        "portals_on": lambda self, m: [p for p in self.portals.values() if p["map"] == m],
    })()
    loop._directive = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 81}, success={"on_map": 81},
                                quest_id="q1")
    monkeypatch.setattr(loop, "_portal_next", lambda p, t: loop.portals.portals["route5gate:warp1"])
    loop.heard.recent.append({"map": 70, "last_step": 10, "speaker": None,
                              "text": "I'm on guard duty. Gee, I'm thirsty, though!"})
    loop.session.step = 11
    loop._pos_prev, loop._last_action_type = (70, 4, 3), "advance_dialog"
    loop._target = {"kind": "tile", "x": 3, "y": 5, "portal": True}
    loop._observe_pushback(SimpleNamespace(player=SimpleNamespace(x=4, y=2, map_id=70)))   # moved by the game
    assert "route5gate:warp1" in loop.memory.blocked_portals
    assert loop.memory.blocked_portals["route5gate:warp1"]["kind"] == "script" and "thirsty" in \
        loop.memory.blocked_portals["route5gate:warp1"]["why"]
    assert loop._target is None                                   # re-route now, not after a wedge
    # our own step never counts
    loop.memory.blocked_portals.clear()
    loop._pos_prev, loop._last_action_type = (70, 4, 3), "move"
    loop._observe_pushback(SimpleNamespace(player=SimpleNamespace(x=4, y=2, map_id=70)))
    assert not loop.memory.blocked_portals
