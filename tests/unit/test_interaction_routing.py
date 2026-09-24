"""Interaction routing: the router decides WHERE TO STAND (spec 2026-09-24-interaction-routing).

runs/sleeves-cerulean: a beaten Cooltrainer F stood on Misty's front tile (4,3). Every pathfinder
treated its goal as free, so the executor walked into him 15+ times while her east side (5,2) was open.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pokemon_agent.agent.reason_loop as RL
from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.interaction import plan, report, sides, standable
from pokemon_agent.agent.navigator import Navigator
from pokemon_agent.agent.plan import Directive, Intent, ReflectionPlan
from pokemon_agent.agent.planner_llm import Planner
from pokemon_agent.agent.quest_reconciler import QuestStep
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep
from pokemon_agent.agent.session import Session
from pokemon_agent.agent.world_map import WorldMap
from pokemon_agent.core.models import Direction, GoalState, InteractAction, MoveAction, WaitAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.observations.builder import ObservationBuilder

GYM = 200
MISTY = {"x": 4, "y": 2, "sprite": "Misty", "slot": 1, "kind": "person"}
TRAINER = {"x": 4, "y": 3, "sprite": "Cooltrainer F", "slot": 2, "kind": "person"}
# a room x 1..8, y 2..12; (3,2) is a wall west of Misty, row 1 is the wall north of her
ROOM = {(x, y) for x in range(1, 9) for y in range(2, 13)} - {(3, 2)}


# ---- pure geometry ---------------------------------------------------------------------------------
def test_standable_rejects_walls_water_sprites_and_doors_but_not_the_target_door():
    walk = {(1, 1), (2, 1), (3, 1)}
    assert standable((1, 1), walkable=walk, occupied=set())
    assert not standable((9, 9), walkable=walk, occupied=set())               # wall / water: off the set
    assert not standable((2, 1), walkable=walk, occupied={(2, 1)})            # someone stands there
    assert not standable((3, 1), walkable=walk, occupied=set(), warps={(3, 1)})  # a door warps you away
    assert standable((3, 1), walkable=walk, occupied=set(), warps={(3, 1)}, target_warp=True)


def test_sides_plain_person_counter_person_and_object_with_a_required_side():
    assert {s.stand for s in sides((4, 2))} == {(4, 3), (4, 1), (3, 2), (5, 2)}
    nurse = {s.stand: s.counter for s in sides((3, 1), counters={(3, 2)})}
    assert nurse[(3, 3)] is True and (3, 2) not in nurse          # talked to from across the counter
    pc = sides((5, 4), face="north")
    assert [(s.stand, s.face) for s in pc] == [((5, 5), Direction.NORTH)]


def test_plan_takes_the_open_side_when_someone_stands_on_the_front():
    p = plan((4, 9), (4, 2), walkable=ROOM, occupied={(4, 2): "Misty", (4, 3): "Cooltrainer F"})
    assert p[0].stand == (5, 2) and p[0].dist is not None
    front = next(s for s in p if s.stand == (4, 3))
    assert (front.status, front.by) == ("occupied", "Cooltrainer F")


def test_plan_respects_an_elevation_cut_between_stand_tile_and_target():
    p = plan((4, 9), (4, 2), walkable=ROOM, occupied={(4, 2): "Misty", (4, 3): "Cooltrainer F"},
             cuts={frozenset({(5, 2), (4, 2)})})
    assert next(s for s in p if s.stand == (5, 2)).status == "cut"
    assert all(s.dist is None for s in p)


def test_the_report_names_who_stands_where():
    sealed = ROOM - {(5, 3), (6, 2)}                     # (5,2) is open but walled off from the room
    p = plan((7, 9), (4, 2), walkable=sealed, occupied={(4, 2): "Misty", (4, 3): "Cooltrainer F"})
    assert all(s.dist is None for s in p)
    r = report("Misty", (4, 2), (7, 9), p)
    assert r.startswith("Misty (4,2): ")
    assert "south (4,3) occupied by Cooltrainer F" in r
    assert "east (5,2) open but unreachable from (7,9)" in r
    assert "north/west wall" in r


# ---- pathfinders: a goal is never exempt from occupancy ----------------------------------------------
def test_navigator_does_not_route_onto_an_occupied_tile():
    w = WorldMap()
    nav = Navigator(w)
    player = SimpleNamespace(x=4, y=6, map_id=GYM, facing="north")
    prim, arrived = nav.step_toward(player, {"x": 4, "y": 3, "interact": False}, {(4, 3)})
    assert (prim, arrived) == (None, False) and nav.goal_blocked
    prim, _ = nav.step_toward(player, {"x": 4, "y": 2, "interact": True}, {(4, 2)})   # talking to her is fine
    assert isinstance(prim, MoveAction)


class StubReasoner:
    def reflect(self, **kw):
        return ReflectionPlan(next_objective="go"), 0, {}

    def step(self, **kw):
        return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}


def _loop(monkeypatch, walkable=ROOM):
    events = []
    emu = FakeEmulator(map_id=GYM)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=StubReasoner(), session=Session(GoalState(primary="g", current="g")),
                         vision=False, reflect_every=100, goal_map=2, on_event=lambda k, p: events.append((k, p)))
    loop.portals = None
    monkeypatch.setattr(RL, "read_collision_map",
                        lambda emu: {"map_id": GYM, "width": 10, "height": 14, "walkable": set(walkable), "terrain": {}})
    loop._plan_steps = [QuestStep(id="q9", map=GYM, talk=True, who="Misty", done_when="badges>=2", status="active")]
    d = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": GYM, "sprite": "Misty"},
                  success={"badges": ">=2"}, quest_id="q9")
    loop._directive = d
    return loop, d, events


def _obs(player, npcs, exits=()):
    return SimpleNamespace(player=player, game_state={"npcs": list(npcs)}, exits=list(exits), map_dims=(10, 14))


def _walk(loop, d, player, npcs, n=12):
    """Follow the approach step by step (moves applied to the player) until it presses A."""
    for _ in range(n):
        act = loop._approach_npc({"kind": "approach_npc", "sprite": "Misty"}, d, _obs(player, npcs), set(),
                                 {(int(c["x"]), int(c["y"])) for c in npcs})
        if isinstance(act, InteractAction) or act is None:
            return act, player
        dx, dy = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}[act.direction.value]
        nxt = (player.x + dx, player.y + dy)
        if nxt in ROOM and nxt not in {(c["x"], c["y"]) for c in npcs}:
            player = SimpleNamespace(x=nxt[0], y=nxt[1], map_id=GYM, facing=act.direction.value)
        else:
            player = SimpleNamespace(x=player.x, y=player.y, map_id=GYM, facing=act.direction.value)
    return None, player


def test_misty_with_a_trainer_on_her_front_tile_is_reached_from_her_open_side(monkeypatch):
    loop, d, events = _loop(monkeypatch)
    act, player = _walk(loop, d, SimpleNamespace(x=4, y=6, map_id=GYM, facing="north"), [MISTY, TRAINER])
    assert isinstance(act, InteractAction)
    assert (player.x, player.y, player.facing) == (5, 2, "west")
    assert not any(k == "approach_blocked" for k, _ in events)


def test_no_reachable_side_wedges_the_step_with_the_occupant_named(monkeypatch):
    loop, d, events = _loop(monkeypatch, walkable=ROOM - {(5, 3), (6, 2)})
    player = SimpleNamespace(x=7, y=9, map_id=GYM, facing="north")
    assert loop._approach_npc({"kind": "approach_npc", "sprite": "Misty"}, d, _obs(player, [MISTY, TRAINER]),
                              set(), {(4, 2), (4, 3)}) is None
    step = loop._plan_steps[0]
    assert step.status == "wedged" and "occupied by Cooltrainer F" in step.wedge_reason
    assert "occupied by Cooltrainer F" in loop._approach_block


def test_a_wanderer_on_the_only_side_is_waited_for_before_reporting(monkeypatch):
    loop, d, events = _loop(monkeypatch, walkable=ROOM - {(5, 3), (6, 2)})
    player = SimpleNamespace(x=7, y=9, map_id=GYM, facing="north")
    loop._observe_npcs(_obs(player, [MISTY, {**TRAINER, "x": 4, "y": 4}]))
    loop.session.step += 1
    loop._observe_npcs(_obs(player, [MISTY, TRAINER]))                      # seen moving: a wanderer
    acts = [loop._approach_npc({"kind": "approach_npc", "sprite": "Misty"}, d, _obs(player, [MISTY, TRAINER]),
                               set(), {(4, 2), (4, 3)}) for _ in range(RL.APPROACH_WAIT + 1)]
    assert all(isinstance(a, WaitAction) for a in acts[:-1]) and acts[-1] is None


def test_rams_movement_flag_decides_who_wanders(monkeypatch):
    loop, d, events = _loop(monkeypatch, walkable=ROOM - {(5, 3), (6, 2)})
    player = SimpleNamespace(x=7, y=9, map_id=GYM, facing="north")
    beaten = {**TRAINER, "wanders": False}                  # a trainer never leaves his tile: report now
    assert loop._approach_npc({"kind": "approach_npc", "sprite": "Misty"}, d, _obs(player, [MISTY, beaten]),
                              set(), {(4, 2), (4, 3)}) is None
    loop2, d2, _ = _loop(monkeypatch, walkable=ROOM - {(5, 3), (6, 2)})
    walker = {**TRAINER, "wanders": True}
    assert isinstance(loop2._approach_npc({"kind": "approach_npc", "sprite": "Misty"}, d2,
                                          _obs(player, [MISTY, walker]), set(), {(4, 2), (4, 3)}), WaitAction)


# ---- who answered ----------------------------------------------------------------------------------
def _answer(loop, sprite, xy, front):
    loop.session.step += 1
    loop._check_answer(SimpleNamespace(game_state={"dialog_active": True, "facing": {
        "facing_sprite": {"sprite": sprite, "x": xy[0], "y": xy[1]}, "front_tile": list(front)}}))


def test_someone_else_answering_is_not_the_target_conversation(monkeypatch):
    loop, d, events = _loop(monkeypatch)
    target = {"kind": "approach_npc", "sprite": "Misty"}
    loop._talk_npc = (GYM, 1)
    loop._interact_with(target, (4, 2), MISTY)
    _answer(loop, "Cooltrainer F", (4, 3), (4, 3))
    assert "talking_to" not in target and target["wrong_answer"] == 1
    assert loop._plan_steps[0].status == "active"                       # retried once first
    loop._interact_with(target, (4, 2), MISTY)
    _answer(loop, "Cooltrainer F", (4, 3), (4, 3))
    assert loop._plan_steps[0].status == "wedged" and "Cooltrainer F" in loop._plan_steps[0].wedge_reason


def test_the_right_person_answering_passes_and_objects_ignore_a_sprite_beyond(monkeypatch):
    loop, d, events = _loop(monkeypatch)
    target = {"kind": "approach_npc", "sprite": "Misty"}
    loop._talk_npc = (GYM, 1)
    loop._interact_with(target, (4, 2), MISTY)
    _answer(loop, "Misty", (4, 2), (4, 2))
    assert "wrong_answer" not in target and "talking_to" in target
    pc = {"kind": "use_object", "object": "PC"}
    loop._interact_with(pc, (5, 4), {"x": 5, "y": 4, "sprite": "PC"})
    _answer(loop, "Bill", (5, 3), (5, 4))                                # 2 tiles away, not who answered
    assert "wrong_answer" not in pc


# ---- L2 names who/what; the router places the player ----------------------------------------------
def test_the_proposer_accepts_use_object():
    class One:
        def chat_json(self, system, state, image=None):
            return json.dumps({"kind": "use_object", "object": "PC", "why": "store a mon"}), 0, {}
    ctx = {"map_view": "", "player": {"x": 1, "y": 1, "map_id": 0}, "objective": "use the PC", "npcs": [],
           "objects": [{"name": "PC", "x": 5, "y": 4}], "reachable": {(1, 1)}, "default": {"kind": "exit"}}
    t = Planner(goal_map=0, provider=One()).propose_target(emu=None, context=ctx)
    assert (t["kind"], t["object"]) == ("use_object", "PC")


def test_leaving_skips_a_door_someone_is_standing_in(monkeypatch):
    loop, d, events = _loop(monkeypatch)
    player = SimpleNamespace(x=5, y=5, map_id=GYM, facing="north")
    obs = _obs(player, [], exits=[{"x": 5, "y": 7}, {"x": 5, "y": 1}])
    mv = loop._leave_via_nearest_exit(player, obs, set(), {(5, 7)})
    assert isinstance(mv, MoveAction) and mv.direction == Direction.NORTH


def test_full_collision_bfs_refuses_an_occupied_goal_but_reaches_an_off_set_door(monkeypatch):
    loop, d, events = _loop(monkeypatch)
    player = SimpleNamespace(x=4, y=6, map_id=GYM, facing="north")
    assert loop._bfs_full_collision(player, (4, 3), set(), {(4, 3)}) is None
    mv = loop._bfs_full_collision(player, (4, 13), set(), set())        # door just off the walkable set
    assert isinstance(mv, MoveAction) and mv.direction == Direction.SOUTH
