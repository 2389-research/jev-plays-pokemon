"""Regression: the agent must step THROUGH the forest gate door, not freeze at a non-walkable one.

Reproduces a real wedge captured live (states/gate_door_wedge.state): the agent stood at (4,1) on
the Viridian Forest South Gate one tile below warp door (4,0), which is NON-walkable — so the move
north failed forever. The south gate has a second, WALKABLE warp to the forest at (5,0); portal
routing must prefer it. Deterministic (portal graph + servo), no network — only needs the ROM.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROM_PATH = ROOT / "roms" / "pokemon_red.gb"
STATE_PATH = ROOT / "states" / "gate_door_wedge.state"


def _loop(emu):
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import ReasonStep, ReflectionPlan
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import GoalState, WaitAction
    from pokemon_agent.observations.builder import ObservationBuilder

    class Stub:
        def reflect(self, **k): return ReflectionPlan(), 0, {}
        def step(self, **k): return ReasonStep(location="", objective="", reasoning="",
                                               action=WaitAction(frames=1)), 0, {}
    return ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=Stub(), session=Session(GoalState(primary="p", current="p")),
                         vision=False, reflect_every=100, goal_map=2)


def test_agent_steps_through_forest_gate_door():
    if not ROM_PATH.exists() or not STATE_PATH.exists():
        pytest.skip("ROM/state fixture not present")
    from types import SimpleNamespace

    from pokemon_agent.agent.plan import Directive, Intent
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator

    emu = PyBoyEmulator(str(ROM_PATH), window="null")
    try:
        emu.load_state(STATE_PATH)
        emu.tick(4)
        cur_map = lambda: emu.read_memory(0xD35E)
        px, py = emu.read_memory(0xD362), emu.read_memory(0xD361)
        assert cur_map() == 50 and (px, py) == (4, 1)   # the captured wedge

        loop = _loop(emu)
        # portal routing must pick a WALKABLE door tile toward the forest (5,0), not (4,0)
        portal = loop._portal_next(SimpleNamespace(map_id=50, x=px, y=py), 2)
        assert portal is not None
        coll = __import__("pokemon_agent.games.pokemon_red.map_reader", fromlist=["read_collision_map"]).read_collision_map(emu)
        assert tuple(portal["coord"]) in coll["walkable"], "chosen door tile must be walkable"

        # and it actually crosses into the forest within a few steps (was frozen forever before)
        directive = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 2}, success={"on_map": 2})
        for _ in range(8):
            obs, _ = loop.builder.build(capture_screenshot=False)
            mv = loop._navigate_leg(directive, obs, set())
            assert mv is not None, "agent produced no move (frozen)"
            loop.controller.execute(mv)
            if cur_map() != 50:
                break
        assert cur_map() == 51, f"expected to cross into Viridian Forest (51), still on {cur_map()}"
    finally:
        emu.close()
