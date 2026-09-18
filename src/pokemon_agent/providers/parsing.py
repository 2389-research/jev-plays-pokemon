"""Tolerant parsing of model output into an AgentDecision."""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ..core.models import AgentDecision

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_DANGLING_THINK = re.compile(r"^.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_fences(text: str) -> str:
    t = text.strip()
    # Some models (e.g. glm vision) inline chain-of-thought in <think>...</think>.
    t = _THINK.sub("", t)
    if "</think>" in t.lower():  # unbalanced closing tag: drop everything up to it
        t = _DANGLING_THINK.sub("", t)
    t = t.strip()
    t = _FENCE.sub("", t).strip()
    # If there is extra prose around a JSON object, grab the outermost braces.
    if not t.startswith("{"):
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start : end + 1]
    return t


def parse_decision(content: str) -> AgentDecision:
    """Raise ValueError with a helpful message on failure (caller may retry)."""
    if not content or not content.strip():
        raise ValueError("empty model content")
    cleaned = strip_fences(content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}")
    try:
        return AgentDecision.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"JSON did not match AgentDecision schema: {e}") from e
