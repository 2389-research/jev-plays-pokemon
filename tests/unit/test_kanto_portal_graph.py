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
