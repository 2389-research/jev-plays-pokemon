"""Bill's House loop (runs/vermilion-team-20260923, ~2,900 steps): talk to Bill -> YES -> he walks into
the teleporter (his sprite is hidden) -> the room has no NPCs, so the talk step "left via the nearest
exit" -> entering Route 25 resets the whole event (Route25ToggleBillsScript) -> Bill is a Pokémon again.
L1 even said "talk to the PC", but the PC is a hidden bg event, not a sprite, so the harness re-targeted
the only person in the room. Fix: never walk out of the step's own map when nobody is left, and aim
talk steps at ripped background objects (PCs, machines, trash cans) by name."""
from __future__ import annotations

from types import SimpleNamespace

from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.plan import Directive, Intent, ReflectionPlan
from pokemon_agent.agent.quest_reconciler import QuestStep
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep
from pokemon_agent.agent.session import Session
from pokemon_agent.core.models import Direction, GoalState, InteractAction, MoveAction, WaitAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.games.pokemon_red.map_objects import match_objects, objects_on
from pokemon_agent.observations.builder import ObservationBuilder

BILLS = 88


def test_table_has_bills_pc_and_surges_trash_cans_but_no_hidden_items():
    pc = [o for o in objects_on(BILLS) if "PC" in o["name"]]
    assert pc and (pc[0]["x"], pc[0]["y"], pc[0]["face"]) == (1, 4, "north")
    assert sum(o["name"] == "trash can" for o in objects_on(92)) >= 15        # Vermilion Gym
    assert not any("item" in o["name"].lower() for m in range(250) for o in objects_on(m))


def test_matching_by_content_word_and_never_a_person_to_their_possession():
    objs = objects_on(BILLS)
    assert match_objects(objs, "the PC") and match_objects(objs, "Bill's machine")
    assert match_objects(objs, "cell separator")
    assert not match_objects(objs, "Bill")                    # the PERSON Bill is not his PC


class Stub:
    def reflect(self, **kw):
        return ReflectionPlan(next_objective="go"), 0, {}

    def step(self, **kw):
        return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}


def _setup(who, success=None):
    emu = FakeEmulator(map_id=BILLS)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu), reasoner=Stub(),
                         session=Session(GoalState(primary="g", current="g")), vision=False, reflect_every=100,
                         goal_map=2)
    loop._plan_steps = [QuestStep(id="q9", map=BILLS, talk=True, who=who, done_when="has_item:S.S. Ticket",
                                  status="active")]
    d = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": BILLS, "sprite": who},
                  success=success or {"has_item": 63}, quest_id="q9")
    loop._directive = d
    return loop, d


def _obs(x, y, facing="north", npcs=()):
    return SimpleNamespace(player=SimpleNamespace(x=x, y=y, map_id=BILLS, facing=facing),
                           game_state={"npcs": list(npcs)}, exits=[{"x": 2, "y": 7}, {"x": 3, "y": 7}],
                           map_dims=(8, 8))


def test_an_empty_room_on_the_steps_own_map_wedges_to_l1_instead_of_walking_out():
    loop, d = _setup("Bill")
    target = {"kind": "approach_npc", "sprite": "Bill"}
    assert loop._approach_npc(target, d, _obs(6, 6), set(), set()) is None
    step = loop._plan_steps[0]
    assert step.status == "wedged" and loop._l1_event
    assert "nobody to talk to" in step.wedge_reason and "Bill's PC" in step.wedge_reason


def test_a_talk_step_for_another_map_still_leaves_an_empty_room():
    loop, d = _setup("Oak")
    d.target["map"] = 0
    act = loop._approach_npc({"kind": "approach_npc", "sprite": "Oak"}, d, _obs(2, 6), set(), set())
    assert loop._plan_steps[0].status == "active" and act is not None


def test_the_pc_is_approached_from_below_facing_north_then_used():
    loop, d = _setup("the PC")
    target = {"kind": "approach_npc", "sprite": "the PC"}
    monster = {"x": 6, "y": 5, "sprite": "Monster", "kind": "person", "slot": 1}
    assert isinstance(loop._approach_npc(target, d, _obs(1, 5, "north", [monster]), set(), set()),
                      InteractAction)                                   # right tile, right facing: press A
    turn = loop._approach_npc(target, d, _obs(1, 5, "east", [monster]), set(), set())
    assert isinstance(turn, MoveAction) and turn.direction == Direction.NORTH


def test_using_the_object_twice_without_success_wedges():
    loop, d = _setup("the PC")
    target = {"kind": "approach_npc", "sprite": "the PC", "tried": [["obj", BILLS, 1, 4]]}
    assert loop._approach_npc(target, d, _obs(1, 5), set(), set()) is None
    assert loop._plan_steps[0].status == "wedged" and "used Bill's PC" in loop._plan_steps[0].wedge_reason
