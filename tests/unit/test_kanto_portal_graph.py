"""The shipped Kanto portal graph + the runtime PortalGraph (spec 2026-09-23-kanto-portal-graph-design).

The graph is ripped from pokered (scripts/rip_portals.py); these tests pin the ground truth the agent
routes by: offset-aware edges, one-way ledges, tile-pair (elevation) components, gated story portals.
"""
from __future__ import annotations

import pytest

from pokemon_agent.agent.portal_graph import PortalGraph

PEWTER, CERULEAN, VERMILION, ROUTE4, MTMOON_1F, MTMOON_B2F, FOREST = 2, 3, 5, 15, 59, 61, 51


@pytest.fixture(scope="module")
def pg():
    return PortalGraph.load()


def _comp_of(pg, map_id, label_part):
    return next(p["component"] for p in pg.portals_on(map_id) if label_part in p["label"])


def test_it_covers_all_of_kanto(pg):
    assert len(pg.maps) >= 220 and pg.version
    assert {MTMOON_1F, 60, MTMOON_B2F, CERULEAN} <= set(pg.maps)


def test_pewter_to_cerulean_goes_through_mt_moon_then_the_route4_ledge(pg):
    r = pg.route(PEWTER, _comp_of(pg, PEWTER, "POKECENTER"), CERULEAN)
    ids = [p["id"] for p in r]
    assert any(i.startswith("mtmoonb2f:") for i in ids) and any(":ledge_" in i for i in ids)
    assert not any(i == "route4:edge_east_c0" for i in ids)


def test_cerulean_main_cannot_walk_back_to_pewter(pg):
    assert pg.route(CERULEAN, _comp_of(pg, CERULEAN, "POKECENTER"), PEWTER) is None


def test_ledges_are_one_way_and_honour_their_landing_component(pg):
    ledge = pg.portals["route4:ledge_south_c0_to15c3"]
    assert pg._dest_nodes(ledge) == [(ROUTE4, 3)]
    assert not [p for p in pg.portals.values() if p["kind"] == "ledge" and p.get("hop") == "north"]


def test_gated_and_elevator_portals_are_not_routed(pg):
    south = _comp_of(pg, CERULEAN, "south edge")
    r = pg.route(CERULEAN, south, VERMILION)
    assert r and not any("gate" in p["id"] and "saffron" in p["label"].lower() for p in r)
    assert not any(p.get("gated") for p in r)
    for p in pg.portals.values():
        if p.get("gated") or p["kind"] == "elevator":
            assert p not in pg.exits_from(p["map"], p["component"])


def test_component_lookup_uses_the_static_grid(pg):
    """Mt. Moon B2F splits by elevation (tile-pair collisions); a flood fill over RAM walkability
    can't see that, the shipped grid can."""
    for p in pg.portals_on(MTMOON_B2F):
        ax, ay = p.get("approach") or p["coord"]
        assert pg.component_at(MTMOON_B2F, ax, ay, walkable=set()) == p["component"]


def test_render_view_hides_ledges(pg):
    assert "ledge" not in pg.render_view(ROUTE4, 0, CERULEAN).lower()
    assert "Route 4" not in pg.render_view(ROUTE4, 0, CERULEAN).split("EXITS FROM HERE")[1].split("HOW AREAS")[0]


def test_forest_to_pewter_still_uses_the_north_gate(pg):
    r = pg.route(FOREST, 0, PEWTER)
    ids = [p["id"] for p in r]
    assert any("northgate" in i for i in ids) and not any("southgate" in i for i in ids)


# ---- executor: ledge hop (K8) and component-aware sibling doors (K3') ------------------------------
def test_controller_waits_out_a_ledge_hop():
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.core.models import Direction, MoveAction
    from pokemon_agent.emulator.fake_emulator import FakeEmulator

    class HopEmu(FakeEmulator):
        """Moving south sets the ledge flag (wMovementFlags bit 6) for a while, like a real hop."""
        def __init__(self):
            super().__init__(map_id=15)
            self.hop_frames = 0

        def tick(self, frames=1, render=True):
            super().tick(frames, render)
            if self.hop_frames > 0:
                self.hop_frames -= frames
                if self.hop_frames <= 0:
                    self.write_memory(0xD736, 0)

        def release(self, button):
            super().release(button)
            if self.read_memory(0xD736) == 0 and self.hop_frames == 0:
                self.write_memory(0xD736, 0x40)
                self.hop_frames = 20

    emu = HopEmu()
    res = ActionController(emu).execute(MoveAction(direction=Direction.SOUTH))
    assert emu.read_memory(0xD736) & 0x40 == 0 and "ledge_hop" in res.events


def test_executor_hops_at_the_ledge_takeoff_and_drops_the_target():
    from types import SimpleNamespace
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.plan import Directive, Intent, ReflectionPlan
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import ReasonStep
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import Direction, GoalState, MoveAction, WaitAction
    from pokemon_agent.emulator.fake_emulator import FakeEmulator
    from pokemon_agent.observations.builder import ObservationBuilder

    class Stub:
        def reflect(self, **kw):
            return ReflectionPlan(next_objective="go"), 0, {}

        def step(self, **kw):
            return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}
    emu = FakeEmulator(map_id=ROUTE4)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu), reasoner=Stub(),
                         session=Session(GoalState(primary="g", current="g")), vision=False, reflect_every=100,
                         goal_map=CERULEAN)
    target = {"kind": "tile", "x": 70, "y": 8, "portal": True, "hop": "south"}
    loop._target = target
    obs = SimpleNamespace(player=SimpleNamespace(x=70, y=8, map_id=ROUTE4, facing="south"),
                          exits=[], map_dims=(90, 18), game_state={"npcs": []})
    assert loop._target_reached(target, obs) is False           # standing on A is not "arrived"
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": CERULEAN}, success={"on_map": CERULEAN})
    move = loop._resolve_target(target, d, obs, {"south"}, set())   # the ledge veto doesn't block the hop
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH and loop._target is None


# ---- cerulean-team incident: the Mt. Moon door (steps 150-240) ------------------------------------------
def _loop_at(map_id):
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
    emu = FakeEmulator(map_id=map_id)
    return ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu), reasoner=Stub(),
                         session=Session(GoalState(primary="g", current="g")), vision=False, reflect_every=100,
                         goal_map=CERULEAN)


def test_door_veto_uses_the_portal_graphs_next_hop():
    """At (18,6) below the Mt. Moon door the old WorldGraph said 'next hop = Cerulean' (Route 4 edge),
    so the door (-> Mt Moon 1F) was vetoed as off-route and the agent waited 90 steps."""
    from types import SimpleNamespace
    from pokemon_agent.agent.plan import Directive, Intent
    loop = _loop_at(ROUTE4)
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": CERULEAN}, success={"on_map": CERULEAN})
    obs = SimpleNamespace(player=SimpleNamespace(x=18, y=6, map_id=ROUTE4, facing="north"))
    assert loop._next_hop_map(obs, d) == MTMOON_1F


def test_an_intermediate_travel_step_on_the_way_is_merged_into_the_next_one():
    """Inside Mt. Moon, 'go to Route 4' (either half satisfies it) walked straight back out the entrance;
    with 'go to Cerulean' next, the Route 4 step is on the way and is skipped."""
    from collections import deque
    from types import SimpleNamespace
    from pokemon_agent.agent.plan import Directive, Intent
    from pokemon_agent.agent.portal_graph import PortalGraph
    pg = PortalGraph.load()
    appr = next(p for p in pg.portals_on(MTMOON_1F) if "MT_MOON_B1F" in p["label"])
    ax, ay = appr.get("approach") or appr["coord"]
    loop = _loop_at(MTMOON_1F)
    loop._directive = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": ROUTE4},
                                success={"on_map": ROUTE4}, quest_id="q4")
    loop._quest = deque([Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": CERULEAN},
                                   success={"on_map": CERULEAN}, quest_id="q5")])
    obs = SimpleNamespace(player=SimpleNamespace(x=ax, y=ay, map_id=MTMOON_1F, facing="north"))
    assert loop._travel_on_the_way(obs) is True
    # not on the way: heading for B2F, then Route 3 (the route to Route 3 leaves by the entrance)
    loop._directive = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": MTMOON_B2F},
                                success={"on_map": MTMOON_B2F}, quest_id="q7")
    loop._quest = deque([Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 14},
                                   success={"on_map": 14}, quest_id="q8")])
    assert loop._travel_on_the_way(obs) is False


# ---- verify-mtmoon3: 273x 'blocked moving west' at Mt Moon 1F (10,22) — a tile-pair (elevation) cut ----
def test_shipped_graph_carries_elevation_cut_edges(pg):
    cuts = pg.cut_edges(MTMOON_1F)
    assert frozenset({(10, 22), (9, 22)}) in cuts and len(cuts) >= 100
    assert pg.cut_edges(PEWTER) == set()


def test_navigator_never_plans_across_a_cut_edge():
    from pokemon_agent.agent.navigator import Navigator
    from pokemon_agent.agent.world_map import WorldMap
    wm = WorldMap()
    for x in range(5):
        for y in range(3):
            wm.tiles[1][(x, y)] = "floor"
    wm.bounds[1] = (5, 3)
    # a wall of cut edges between x=1 and x=2 except on row 2
    wm.cut_edges = {1: {frozenset({(1, 0), (2, 0)}), frozenset({(1, 1), (2, 1)})}}
    d = Navigator(wm)._bfs_first_step(1, (1, 0), {(3, 0)})
    assert d.value == "south"          # detours via row 2 instead of stepping east across the cut


def test_loop_loads_cut_edges_onto_the_world():
    loop = _loop_at(MTMOON_1F)
    assert frozenset({(10, 22), (9, 22)}) in loop.world.cut_edges.get(MTMOON_1F, set())


_STUCK = __import__("pathlib").Path("runs/verify-mtmoon3-20260923")


@pytest.mark.skipif(not (_STUCK / "latest.state").exists() or not __import__("pathlib").Path("roms/pokemon_red.gb").exists(),
                    reason="stuck Mt Moon state / ROM not present")
def test_portal_tile_route_detours_around_the_cut_at_mt_moon_1f():
    """verify-mtmoon4: the portal-tile path (_bfs_full_collision over RAM walkability) still stepped
    west across the (10,22)-(9,22) elevation edge; it must detour (the correct first step is east)."""
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.memory import AgentMemory
    from pokemon_agent.agent.plan import Directive, Intent, ReflectionPlan
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import ReasonStep
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import GoalState, WaitAction
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.observations.builder import ObservationBuilder

    class Stub:
        def reflect(self, **kw):
            return ReflectionPlan(next_objective="go"), 0, {}

        def step(self, **kw):
            return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}
    emu = PyBoyEmulator("roms/pokemon_red.gb", window="null")
    emu.load_state(_STUCK / "latest.state")
    emu.tick(4)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu), reasoner=Stub(),
                         session=Session(GoalState(primary="g", current="g")), vision=False, reflect_every=100,
                         goal_map=2, memory=AgentMemory.load(_STUCK / "latest.mem.json"))
    obs, _ = loop.builder.build(capture_screenshot=False)
    assert (obs.player.map_id, obs.player.x, obs.player.y) == (MTMOON_1F, 10, 22)
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": CERULEAN}, success={"on_map": CERULEAN})
    move = loop._resolve_target({"kind": "tile", "x": 5, "y": 5, "portal": True}, d, obs, set(), set())
    assert move is not None and move.direction.value != "west"
    emu.close()
