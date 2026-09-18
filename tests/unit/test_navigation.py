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


def test_blocked_dirs_masks_off_route_door():
    # off-route-door avoidance now lives in _blocked_dirs (the servo's BFS routes around the
    # masked directions) — the old greedy _can_step compass fallback is gone.
    loop = _loop()
    player = SimpleNamespace(x=2, y=2, map_id=0, facing="north")
    exits = [{"x": 2, "y": 1, "dest_map": 39}]  # a door directly NORTH leading to Blue's House (39)
    obs = SimpleNamespace(player=player, exits=exits, game_state={})
    # heading toward next hop map 12: the north door leads OFF route (39) -> masked out
    assert "north" in loop._blocked_dirs(obs, allowed_next=12)
    # but if the next hop IS that map, the door is on-route -> not masked
    assert "north" not in loop._blocked_dirs(obs, allowed_next=39)
