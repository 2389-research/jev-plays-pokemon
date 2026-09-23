"""Rotate away from an NPC whose conversations don't achieve the step (spec grind-talk-shop F2).

runs/brock-goals2-20260923 steps 612-893: "talk to Brock" — Gen 1 labels Brock's sprite "Super Nerd",
so no name matched; Jev picked the Gym Guide (cached in target["picked"]) and the agent held ~9
advice conversations with him (~280 steps, budget frozen during dialogue), never trying anyone else.
"""
from __future__ import annotations

from types import SimpleNamespace

from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.plan import Directive, Intent, ReflectionPlan
from pokemon_agent.agent.quest_reconciler import QuestStep
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep
from pokemon_agent.agent.session import Session
from pokemon_agent.agent.targets import select_npc
from pokemon_agent.core.models import GoalState, InteractAction, WaitAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.observations.builder import ObservationBuilder

GYM = 54
GUIDE = {"x": 7, "y": 10, "sprite": "Gym Guide", "slot": 3, "kind": "person"}
TRAINER = {"x": 3, "y": 6, "sprite": "Cooltrainer M", "slot": 2, "kind": "person"}
BROCK = {"x": 4, "y": 1, "sprite": "Super Nerd", "slot": 1, "kind": "person"}


def _p(x, y, facing="north"):
    return SimpleNamespace(x=x, y=y, map_id=GYM, facing=facing)


# ---- select_npc ------------------------------------------------------------------------------------
def test_tried_npcs_are_skipped_even_when_they_are_nearest():
    npc = select_npc([GUIDE, TRAINER, BROCK], sprite="Brock", picked=None, player=_p(7, 11),
                     tried={(GYM, 3)})
    assert npc["sprite"] == "Cooltrainer M"


def test_tried_filter_survives_the_all_sprites_fallback_and_exhaustion_returns_none():
    item = {"x": 1, "y": 1, "sprite": "Poke Ball", "slot": 5, "kind": "item"}
    assert select_npc([GUIDE, item], sprite="Brock", picked=None, player=_p(7, 11),
                      tried={(GYM, 3)}) is None       # the person pool is exhausted: never the item
    assert select_npc([GUIDE], sprite=None, picked=[7, 10, GYM], player=_p(7, 11), tried={(GYM, 3)}) is None


def test_tried_is_keyed_by_slot_so_a_wandering_npc_stays_tried():
    moved = {**GUIDE, "x": 6, "y": 9}
    npc = select_npc([moved, TRAINER], sprite=None, picked=None, player=_p(6, 10), tried={(GYM, 3)})
    assert npc["sprite"] == "Cooltrainer M"


# ---- loop wiring ----------------------------------------------------------------------------------------
class StubReasoner:
    def reflect(self, **kw):
        return ReflectionPlan(next_objective="go"), 0, {}

    def step(self, **kw):
        return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}


def _setup(success=None):
    events = []
    emu = FakeEmulator(map_id=GYM)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=StubReasoner(), session=Session(GoalState(primary="g", current="g")),
                         vision=False, reflect_every=100, goal_map=2, on_event=lambda k, p: events.append((k, p)))
    loop._plan_steps = [QuestStep(id="q5", map=GYM, talk=True, who="Brock", done_when="badges>=1", status="active")]
    d = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": GYM, "sprite": "Brock"},
                  success=success or {"badges": ">=1"}, quest_id="q5")
    loop._directive = d
    return loop, d, events


def _obs(player, npcs):
    return SimpleNamespace(player=player, game_state={"npcs": npcs}, exits=[], map_dims=(10, 14))


def _talk_once(loop, d, target, player):
    """Stand facing the picked NPC -> interact, then a dialogue happens, then back to navigate."""
    act = loop._approach_npc(target, d, _obs(player, [GUIDE, TRAINER, BROCK]), set(), set())
    loop.session.step += 1
    loop._note_dialogue_step()
    loop.session.step += 1
    return act


def test_two_unproductive_conversations_rotate_to_someone_else():
    loop, d, events = _setup()
    target = {"kind": "approach_npc", "sprite": "Brock", "picked": [7, 10, GYM]}
    at_guide = _p(7, 11, "north")
    assert isinstance(_talk_once(loop, d, target, at_guide), InteractAction)
    assert isinstance(_talk_once(loop, d, target, at_guide), InteractAction)   # 1st unproductive: talk again
    loop._approach_npc(target, d, _obs(at_guide, [GUIDE, TRAINER, BROCK]), set(), set())
    assert [GYM, 3] in [list(t) for t in target["tried"]]
    assert target["picked"] is not None and target["picked"][:2] != [7, 10]      # moved on
    assert any(k == "npc_rotate" for k, _ in events)


def test_no_rotation_on_the_step_after_a_rotation():
    loop, d, events = _setup()
    target = {"kind": "approach_npc", "sprite": "Brock", "picked": [7, 10, GYM]}
    at_guide = _p(7, 11, "north")
    _talk_once(loop, d, target, at_guide)
    _talk_once(loop, d, target, at_guide)
    loop._approach_npc(target, d, _obs(at_guide, [GUIDE, TRAINER, BROCK]), set(), set())
    loop._approach_npc(target, d, _obs(at_guide, [GUIDE, TRAINER, BROCK]), set(), set())
    assert [k for k, _ in events].count("npc_rotate") == 1


def test_verify_success_never_rotates():
    loop, d, events = _setup(success={"verify": "did Brock accept the challenge?"})
    target = {"kind": "approach_npc", "sprite": "Brock", "picked": [7, 10, GYM]}
    for _ in range(3):
        _talk_once(loop, d, target, _p(7, 11, "north"))
    assert not target.get("tried")


def test_everyone_tried_wedges_the_step_with_a_reason():
    loop, d, events = _setup()
    target = {"kind": "approach_npc", "sprite": "Brock", "picked": None,
              "tried": [[GYM, 1], [GYM, 2], [GYM, 3]]}
    out = loop._approach_npc(target, d, _obs(_p(5, 12), [GUIDE, TRAINER, BROCK]), set(), set())
    step = loop._plan_steps[0]
    assert out is None and step.status == "wedged" and loop._l1_event is True
    assert "talked to everyone here" in step.wedge_reason and "Gym Guide" in step.wedge_reason


def test_a_travel_step_is_never_wedged_or_rotated_by_npc_talks():
    """restart340-smoke step 64: L2 aimed a TRAVEL step (on_map 2) at the Mart NPCs; exhaustion wedged
    the travel step as 'talked to everyone here'. Rotation is only for talk/grab steps."""
    loop, d, events = _setup(success={"on_map": 2})
    d.intent = Intent.TRAVEL
    target = {"kind": "approach_npc", "sprite": None, "picked": None, "tried": [[GYM, 1], [GYM, 2], [GYM, 3]]}
    loop._approach_npc(target, d, _obs(_p(5, 12), [GUIDE, TRAINER, BROCK]), set(), set())
    assert loop._plan_steps[0].status == "active"
    target2 = {"kind": "approach_npc", "sprite": None, "picked": [7, 10, GYM]}
    for _ in range(3):
        _talk_once(loop, d, target2, _p(7, 11, "north"))
    assert not target2.get("tried")
