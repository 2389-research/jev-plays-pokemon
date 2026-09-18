"""The loop re-plans (off-cycle) when the decider reports low confidence."""
from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep, ReflectionPlan
from pokemon_agent.agent.session import Session
from pokemon_agent.core.models import Direction, GoalState, MoveAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.observations.builder import ObservationBuilder


class StubReasoner:
    """Always returns a move with a fixed confidence; counts reflect() calls."""

    def __init__(self, confidence: float):
        self.confidence = confidence
        self.reflect_calls = 0

    def reflect(self, **kwargs):
        self.reflect_calls += 1
        return ReflectionPlan(next_objective="go"), 1, {}

    def step(self, **kwargs):
        rstep = ReasonStep(location="l", objective="o", reasoning="r",
                           action=MoveAction(direction=Direction.SOUTH))
        return rstep, 1, {"confidence": self.confidence}


def _loop(conf, **kw):
    emu = FakeEmulator()
    session = Session(GoalState(primary="leave", current="leave"))
    reasoner = StubReasoner(conf)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=reasoner, session=session, vision=False,
                         reflect_every=100, reflect_cooldown=1, **kw)
    return loop, reasoner


def test_low_confidence_forces_extra_reflection():
    loop, reasoner = _loop(0.10, low_conf_reflect=0.5)
    for _ in range(3):
        loop.step_once()
    # step0: periodic reflect. step0-end low conf -> force. step... -> at least one extra reflect.
    assert reasoner.reflect_calls >= 2


def test_high_confidence_does_not_force_reflection():
    loop, reasoner = _loop(0.95, low_conf_reflect=0.5)
    for _ in range(3):
        loop.step_once()
    assert reasoner.reflect_calls == 1  # only the step-0 periodic one


def test_disabled_by_default_no_forced_reflection():
    loop, reasoner = _loop(0.01)  # low_conf_reflect defaults to None
    for _ in range(3):
        loop.step_once()
    assert reasoner.reflect_calls == 1
