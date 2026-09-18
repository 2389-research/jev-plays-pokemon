"""Deterministic providers for offline agent-loop testing."""
from __future__ import annotations

from ..core.models import AgentDecision
from .interface import AgentRequest, AgentResponse, ProviderError


class MockProvider:
    """Replays a fixed list of decisions; cycles or raises when exhausted."""

    def __init__(self, decisions: list[AgentDecision], *, cycle: bool = False):
        if not decisions:
            raise ValueError("MockProvider needs at least one decision")
        self._decisions = decisions
        self._i = 0
        self._cycle = cycle

    def complete(self, request: AgentRequest) -> AgentResponse:
        if self._i >= len(self._decisions):
            if not self._cycle:
                raise ProviderError("MockProvider exhausted")
            self._i = 0
        d = self._decisions[self._i]
        self._i += 1
        return AgentResponse(decision=d, raw_content=d.model_dump_json(), model="mock")
