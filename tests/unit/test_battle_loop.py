"""The loop's mode dispatch routes into the battle sub-policy and wins (no network)."""
from pathlib import Path

import pytest

from pokemon_agent.agent.reasoner import ReasonStep, ReflectionPlan
from pokemon_agent.core.models import Direction, MoveAction

ROM = Path("roms/pokemon_red.gb")
BATTLE = Path("states/battle.state")


class Stub:
    """No .client -> the battle policy falls back to move slot 0 (Tackle), deterministic."""

    def reflect(self, **k):
        return ReflectionPlan(), 0, {}

    def step(self, **k):
        return ReasonStep(location="", objective="", reasoning="",
                          action=MoveAction(direction=Direction.SOUTH)), 0, {}


@pytest.mark.skipif(not (ROM.exists() and BATTLE.exists()),
                    reason="needs ROM + battle.state (rival battle) fixture")
def test_loop_dispatches_into_battle_and_wins():
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import GoalState
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.games.pokemon_red import battle
    from pokemon_agent.games.pokemon_red.progress import read_progress
    from pokemon_agent.observations.builder import ObservationBuilder

    emu = PyBoyEmulator(str(ROM), window="null", speed=0)
    emu.load_state(BATTLE); emu.tick(3)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=Stub(), session=Session(GoalState(primary="win")),
                         vision=False, reflect_every=99)
    assert battle.in_battle(emu)
    won = False
    for _ in range(90):
        loop.step_once()
        if not battle.in_battle(emu):
            won = True
            break
    assert won                                   # the battle policy fought it to a finish
    assert read_progress(emu).party_levels[0] >= 6   # Squirtle leveled up from the win
    emu.close()
