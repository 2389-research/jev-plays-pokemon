"""Grind in place — executor wiring (spec grind-talk-shop F1 + review round 1)."""
from __future__ import annotations

from types import SimpleNamespace

from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.plan import Directive, Intent, ReflectionPlan
from pokemon_agent.agent.quest_reconciler import QuestStep
from pokemon_agent.agent.reason_loop import GRIND_ENCOUNTER_WINDOW, ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep
from pokemon_agent.agent.session import Session
from pokemon_agent.agent.world_map import WALL
from pokemon_agent.core.models import GoalState, MoveAction, WaitAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.observations.builder import ObservationBuilder

R2 = 13
B = 0xD16B


class StubReasoner:
    def reflect(self, **kw):
        return ReflectionPlan(next_objective="go"), 0, {}

    def step(self, **kw):
        return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}


def _setup(rows, level=11, success=None, intent=Intent.TRAVEL):
    events = []
    emu = FakeEmulator(map_id=R2)
    emu.write_memory(0xD163, 1)
    emu.write_memory(B + 0x21, level)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=StubReasoner(), session=Session(GoalState(primary="g", current="g")),
                         vision=False, reflect_every=100, goal_map=2, on_event=lambda k, p: events.append((k, p)))
    tiles, terr = {}, {}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            tiles[(x, y)] = WALL if ch == "#" else "floor"
            if ch == "G":
                terr[(x, y)] = "grass"
    loop.world.tiles[R2], loop.world.terrain[R2] = tiles, terr
    loop.world.bounds[R2] = (len(rows[0]), len(rows))
    loop._plan_steps = [QuestStep(id="q30", map=R2, done_when="level>=13", status="active")]
    d = Directive(intent=intent, target={"kind": "map", "map": R2}, success=success or {"level": ">=13"},
                  quest_id="q30")
    loop._directive = d
    return loop, d, events


GRASSY = ["#######", "#GGG..#", "#GGG..#", "#.....#", "#######"]


def _obs(x, y):
    return SimpleNamespace(player=SimpleNamespace(x=x, y=y, map_id=R2, facing="north"),
                           game_state={"npcs": []}, exits=[], map_dims=(7, 5))


def test_default_target_is_grind_on_the_grind_map():
    loop, d, _ = _setup(GRASSY)
    assert loop._default_target(d, _obs(5, 3)) == {"kind": "grind"}


def test_no_grind_when_met_mixed_or_a_talk_step():
    loop, d, _ = _setup(GRASSY, level=13)
    assert loop._default_target(d, _obs(5, 3)) is None                       # level already reached
    loop2, d2, _ = _setup(GRASSY, success={"level": ">=13", "has_item": 20})
    assert loop2._default_target(d2, _obs(5, 3)) is None                     # grinding can't satisfy it
    loop3, d3, _ = _setup(GRASSY, intent=Intent.TALK_TO)
    d3.target = {"kind": "npc", "map": R2}
    assert loop3._default_target(d3, _obs(5, 3))["kind"] == "approach_npc"


def test_the_proposer_never_overrides_a_grind_target():
    loop, d, _ = _setup(GRASSY)

    class Boom:
        def chat_json(self, *a, **k):
            raise AssertionError("proposer must not be called while grinding")
    loop.planner.provider = Boom()
    assert loop._propose_target(_obs(5, 3), d, {"kind": "grind"}, stuck=False) == {"kind": "grind"}


def test_navigate_leg_walks_into_the_grass():
    loop, d, _ = _setup(GRASSY)
    move = loop._navigate_leg(d, _obs(5, 3), set())
    assert isinstance(move, MoveAction) and move.direction.value in ("west", "north")


def test_no_reachable_grass_wedges_at_once_with_the_reason():
    loop, d, _ = _setup(["#####", "#...#", "#####"])
    assert loop._navigate_leg(d, _obs(1, 1), set()) is None
    step = loop._plan_steps[0]
    assert step.status == "wedged" and "no reachable grass" in step.wedge_reason and loop._l1_event


def test_stuck_verdicts_inside_the_encounter_window_do_not_count():
    loop, d, _ = _setup(GRASSY)
    loop._navigate_leg(d, _obs(1, 1), set())              # grinding starts now
    stuck = SimpleNamespace(stuck=True, kind="local_loop")
    for _ in range(10):
        loop._apply_stuck_to_budget(stuck, frozen=False)
        loop.session.step += 1
    assert loop._blocked_for_n == 0 and loop._plan_steps[0].status == "active"


def _grass_steps(loop, n, *, on=(1, 1)):
    loop.controller.emu.x, loop.controller.emu.y = on
    for _ in range(n):
        loop.session.step += 1
        loop._apply_stuck_to_budget(SimpleNamespace(stuck=False, kind=None), frozen=False)


def test_no_wild_encounter_for_the_window_wedges_with_the_reason():
    """Counted in GRASS steps (only those roll an encounter), window from the map's rate (Route 2:
    25/256 -> the 60 floor)."""
    loop, d, _ = _setup(GRASSY)
    loop._navigate_leg(d, _obs(1, 1), set())
    _grass_steps(loop, GRIND_ENCOUNTER_WINDOW + 2)
    step = loop._plan_steps[0]
    assert step.status == "wedged" and "no wild encounter in" in step.wedge_reason and "grass steps" in step.wedge_reason
    assert loop._l1_event


def test_steps_off_the_grass_only_count_toward_the_hard_cap():
    from pokemon_agent.agent.reason_loop import GRIND_HARD_CAP
    loop, d, _ = _setup(GRASSY)
    loop._navigate_leg(d, _obs(1, 1), set())
    _grass_steps(loop, GRIND_ENCOUNTER_WINDOW + 5, on=(4, 3))          # a floor tile: no encounter rolls
    assert loop._plan_steps[0].status == "active"
    _grass_steps(loop, GRIND_HARD_CAP, on=(4, 3))
    assert loop._plan_steps[0].status == "wedged"


def test_a_wild_battle_resets_the_window():
    loop, d, _ = _setup(GRASSY)
    loop._navigate_leg(d, _obs(1, 1), set())
    _grass_steps(loop, GRIND_ENCOUNTER_WINDOW - 5)
    loop._note_battle_end(wild=True)
    _grass_steps(loop, 20)
    assert loop._plan_steps[0].status == "active"
    loop._note_battle_end(wild=False)                     # a trainer battle does not reset it
    _grass_steps(loop, GRIND_ENCOUNTER_WINDOW)
    assert loop._plan_steps[0].status == "wedged"
