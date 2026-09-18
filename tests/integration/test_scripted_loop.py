"""Full agent loop over FakeEmulator with a scripted MockProvider — no network, no ROM."""
from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.loop import AgentLoop
from pokemon_agent.agent.session import Session
from pokemon_agent.core.models import AgentDecision, Direction, GoalState, MoveAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.observations.builder import ObservationBuilder
from pokemon_agent.providers.mock import MockProvider


def test_scripted_navigation_reaches_doorway():
    # Default maze: start (2,2); the opening ("doorway") is at bottom row cols 3-4.
    emu = FakeEmulator()
    # Script: south, south, south, south -> should walk down the left corridor then blocked,
    # so route east first then south through the opening at x=3.
    script = [
        AgentDecision(action=MoveAction(direction=Direction.SOUTH, tiles=1), decision_note="s"),
        AgentDecision(action=MoveAction(direction=Direction.EAST, tiles=1), decision_note="e"),
        AgentDecision(action=MoveAction(direction=Direction.SOUTH, tiles=3), decision_note="s"),
    ]
    provider = MockProvider(script)
    session = Session(GoalState(current="reach doorway"))
    loop = AgentLoop(
        builder=ObservationBuilder(emu), provider=provider,
        controller=ActionController(emu), session=session,
    )
    loop.run(max_steps=len(script))
    # Player should have descended into the bottom opening (y grew).
    assert emu.y > 2
    assert session.step == 3


def test_observation_exposes_walkability_and_actions():
    emu = FakeEmulator()
    obs, shot = ObservationBuilder(emu).build()
    assert obs.walkability is not None
    assert "@" in "".join(obs.walkability)
    assert "move_north" in obs.available_actions
    assert shot is None  # no screenshot unless requested
