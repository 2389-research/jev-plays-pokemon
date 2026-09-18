"""Warp-aware navigation layer: 0xFF resolution + off-route door avoidance."""
from types import SimpleNamespace

from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.session import Session
from pokemon_agent.agent.reasoner import ReasonStep, ReflectionPlan
from pokemon_agent.core.models import GoalState, WaitAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.observations.builder import ObservationBuilder


class StubReasoner:
    def reflect(self, **kw):
        return ReflectionPlan(), 0, {}

    def step(self, **kw):
        return ReasonStep(action=WaitAction(frames=1)), 0, {}


def _loop():
    emu = FakeEmulator(map_id=0)
    return ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=StubReasoner(), session=Session(GoalState(primary="g", current="g")),
                         vision=False, reflect_every=100, goal_map=2)


def test_resolve_exits_maps_0xff_to_prev_map():
    loop = _loop()
    loop._prev_map = 0  # we came from Pallet
    out = loop._resolve_exits([{"x": 4, "y": 11, "dest_map": 255, "dest_name": "Map 255"}], cur_map=39)
    assert out[0]["dest_map"] == 0  # building door now routes back to Pallet, not "map 255"


def test_resolve_exits_leaves_real_dest_untouched():
    loop = _loop()
    loop._prev_map = 0
    out = loop._resolve_exits([{"x": 1, "y": 1, "dest_map": 12}], cur_map=0)
    assert out[0]["dest_map"] == 12


def test_can_step_avoids_off_route_door():
    loop = _loop()
    # player aligned with the FakeEmulator's actual position so _confirmed_wall agrees;
    # north and south of (2,2) are open floor in the fake grid.
    player = SimpleNamespace(x=2, y=2, map_id=0, facing="north")
    exits = [{"x": 2, "y": 1, "dest_map": 39}]  # a door directly NORTH leading to Blue's House (39)
    # heading toward next hop map 12: the north door leads OFF route (39) -> must not step onto it
    assert not loop._can_step(player, "north", 12, exits, set())
    # but if the next hop IS that map, entering the door is correct
    assert loop._can_step(player, "north", 39, exits, set())
    # a plain tile with no door ahead is fine (south of (2,2) is open floor)
    assert loop._can_step(player, "south", 12, exits, set())


def test_can_step_respects_blocked_dirs():
    loop = _loop()
    player = SimpleNamespace(x=2, y=2, map_id=0, facing="north")
    assert not loop._can_step(player, "north", 12, [], {"north"})
