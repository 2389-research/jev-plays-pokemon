"""The go-to navigation macro drives the player to a target over the emulator."""
from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep, ReflectionPlan
from pokemon_agent.agent.session import Session
from pokemon_agent.agent.world_map import WALL
from pokemon_agent.core.models import GoToAction, GoalState, WaitAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.observations.builder import ObservationBuilder


class GotoThenWait:
    """Emits a GoToAction on the first step, then waits."""

    def __init__(self, action):
        self.first = action
        self.calls = 0

    def reflect(self, **kwargs):
        return ReflectionPlan(next_objective="go"), 0, {}

    def step(self, **kwargs):
        self.calls += 1
        act = self.first if self.calls == 1 else WaitAction(frames=1)
        return ReasonStep(location="", objective="", reasoning="", action=act), 0, {}


def test_goto_macro_walks_player_to_target():
    # (2,4) is open floor straight south of the start; reachable under either
    # 1-tile or the fake's 2-tile stepping, so the assertion is fidelity-robust.
    emu = FakeEmulator(start=(2, 2), map_id=40)
    session = Session(GoalState(primary="g", current="g"))
    reasoner = GotoThenWait(GoToAction(x=2, y=4, interact=False, label="spot (2,4)"))
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=reasoner, session=session, vision=False, reflect_every=100)
    result = loop.step_once()
    assert (emu.x, emu.y) == (2, 4)  # the macro pathed onto the target tile
    assert result.result == "completed"


def test_goto_explores_when_no_route_known():
    # target is an object whose every neighbor is a known wall -> no direct route.
    # the macro should EXPLORE (map the area) and move, not give up in place.
    emu = FakeEmulator(start=(2, 2), map_id=40)
    session = Session(GoalState(primary="g", current="g"))
    reasoner = GotoThenWait(GoToAction(x=4, y=4, interact=True, label="thing (4,4)"))
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=reasoner, session=session, vision=False, reflect_every=100)
    for xy in [(3, 4), (5, 4), (4, 3), (4, 5)]:  # wall off every approach tile
        loop.world.tiles[40][xy] = WALL
    result = loop.step_once()
    assert (emu.x, emu.y) != (2, 2)          # it moved (explored) instead of stalling
    assert "goto_explore" in (result.events or [])
    assert result.player_moved is True
