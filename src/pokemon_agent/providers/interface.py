"""Provider abstraction. AgentRequest carries semantic content only —
no OpenAI message objects, no image wire-format. Providers translate."""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from ..core.models import AgentDecision, GoalState, Observation
from ..emulator.interface import ImageObservation


class AgentRequest(BaseModel):
    system_prompt: str
    observation: Observation
    goal: GoalState
    session_summary: str | None = None
    plan: dict | None = None  # {strategy, next_checkpoint, scene} from the vision planner
    image: ImageObservation | None = None

    model_config = {"arbitrary_types_allowed": True}


class AgentResponse(BaseModel):
    decision: AgentDecision
    raw_content: str
    model: str
    latency_ms: int = 0
    usage: dict = {}


class ProviderError(RuntimeError):
    """Raised when a provider cannot return a usable decision."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class ModelProvider(Protocol):
    def complete(self, request: AgentRequest) -> AgentResponse: ...
