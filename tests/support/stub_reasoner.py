"""A deterministic, network-free reasoner for capture/replay integration tests: always steps
SOUTH and exposes a `provider` stub so the loop builds its Planner (whose overworld legs then
emit `l2_propose_target` capture records)."""
from __future__ import annotations

from pokemon_agent.agent.reasoner import ReasonStep, ReflectionPlan
from pokemon_agent.core.models import Direction, MoveAction


class StubProvider:
    model = "stub"

    def chat_json(self, system, user, image=None):
        return ('{"kind":"exit"}', 7, {"total_tokens": 5})


class StubReasoner:
    def __init__(self):
        self.provider = StubProvider()

    def reflect(self, **kwargs):
        return ReflectionPlan(next_objective="go south"), 1, {}

    def step(self, **kwargs):
        rstep = ReasonStep(location="l", objective="o", reasoning="r",
                           action=MoveAction(direction=Direction.SOUTH))
        return rstep, 1, {"confidence": 0.9}
