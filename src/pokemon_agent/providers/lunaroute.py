"""LunaRoute provider (OpenAI-compatible gateway).

Env:
  LUMAROUTE_API_KEY   (note: LUMA spelling on this machine) or LUNAROUTE_API_KEY
  LUNAROUTE_BASE_URL  default https://gw.lunaroute.com/v1
  LUNAROUTE_MODEL     e.g. glm-5.3-flash (text) or glm-5.3-vision (vision)
  LUNAROUTE_REASONING_EFFORT  default "none"
"""
from __future__ import annotations

import base64
import os
import time

from ..core.models import GoalState, Observation
from ..emulator.interface import ImageObservation
from .interface import AgentRequest, AgentResponse, ProviderError
from .parsing import parse_decision

DEFAULT_BASE_URL = "https://gw.lunaroute.com/v1"

# Backoff schedule (seconds) for retryable statuses.
_BACKOFF = [0.5, 1.5, 4.0]


def _api_key() -> str:
    key = os.environ.get("LUMAROUTE_API_KEY") or os.environ.get("LUNAROUTE_API_KEY")
    if not key:
        raise ProviderError("LUMAROUTE_API_KEY / LUNAROUTE_API_KEY not set", status=401)
    return key


def _user_content(obs: Observation, goal: GoalState, image: ImageObservation | None, plan: dict | None = None):
    payload = {"goal": goal.model_dump(), "observation": obs.to_model_json()}
    if plan:
        payload["current_plan"] = plan
    text = "Decide the next action for this game state:\n" + _json(payload)
    if image is None:
        return text
    b64 = base64.b64encode(image.data).decode("ascii")
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:{image.mime_type};base64,{b64}"}},
    ]


def _json(obj) -> str:
    import json

    return json.dumps(obj, separators=(",", ":"))


class LunaRouteProvider:
    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int = 512,
        timeout: float = 60.0,
    ):
        from openai import OpenAI

        self.model = model or os.environ.get("LUNAROUTE_MODEL") or "glm-5.3-flash"
        self.reasoning_effort = reasoning_effort or os.environ.get("LUNAROUTE_REASONING_EFFORT", "none")
        self.max_tokens = max_tokens
        self._client = OpenAI(
            api_key=_api_key(),
            base_url=base_url or os.environ.get("LUNAROUTE_BASE_URL", DEFAULT_BASE_URL),
            timeout=timeout,
        )

    # --- discovery --------------------------------------------------------
    def list_models(self) -> list[str]:
        return [m.id for m in self._client.models.list().data]

    # --- generic structured call (used by the planner/reflector) ----------
    def chat_json(self, system_prompt: str, user, image: ImageObservation | None = None) -> tuple[str, int, dict]:
        """One JSON-object call. `user` may be a str or a dict (serialized).
        Returns (raw_content, latency_ms, usage). Caller parses the JSON."""
        if not isinstance(user, str):
            user = _json(user)
        content_block = user
        if image is not None:
            b64 = base64.b64encode(image.data).decode("ascii")
            content_block = [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": f"data:{image.mime_type};base64,{b64}"}},
            ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_block},
        ]
        return self._call(messages)

    # --- main -------------------------------------------------------------
    def complete(self, request: AgentRequest) -> AgentResponse:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": _user_content(request.observation, request.goal, request.image, request.plan)},
        ]
        content, latency_ms, usage = self._call(messages)
        try:
            decision = parse_decision(content)
        except ValueError as first_err:
            # one tolerant retry with an explicit correction message
            messages.append({"role": "assistant", "content": content or ""})
            messages.append({
                "role": "user",
                "content": f"That was not valid. Error: {first_err}. "
                           "Return ONLY the required JSON object, nothing else.",
            })
            content, latency_ms, usage = self._call(messages)
            try:
                decision = parse_decision(content)
            except ValueError as second_err:
                raise ProviderError(f"model output invalid after retry: {second_err}") from second_err
        return AgentResponse(
            decision=decision, raw_content=content, model=self.model,
            latency_ms=latency_ms, usage=usage,
        )

    # --- transport with error handling ------------------------------------
    def _call(self, messages) -> tuple[str, int, dict]:
        from openai import (
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            NotFoundError,
            RateLimitError,
        )

        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            # STREAM: the LunaRoute gateway's non-streaming (buffered) completions path
            # hangs — the request returns 0 bytes and blocks until the client timeout,
            # which looks like a freeze. The streaming path works, so we always stream and
            # accumulate. include_usage puts the usage block in the final chunk.
            stream=True,
            stream_options={"include_usage": True},
        )
        if self.reasoning_effort and self.reasoning_effort != "provider_default":
            kwargs["reasoning_effort"] = self.reasoning_effort

        last_exc: Exception | None = None
        for attempt in range(len(_BACKOFF) + 1):
            try:
                t = time.time()
                content, finish_reason, usage = self._stream(kwargs)
                latency_ms = int((time.time() - t) * 1000)
                if finish_reason == "length" and not content.strip():
                    raise ProviderError(
                        "empty content with finish_reason=length (reasoning likely ate the budget; "
                        "lower reasoning_effort or raise max_tokens)"
                    )
                return content, latency_ms, usage
            except AuthenticationError as e:  # 401
                raise ProviderError("authentication failed (check LUMAROUTE_API_KEY)", status=401) from e
            except NotFoundError as e:  # 404
                raise ProviderError(
                    f"model '{self.model}' not found — run list-models to see the current roster",
                    status=404,
                ) from e
            except RateLimitError as e:  # 429 -> backoff
                last_exc = e
                self._sleep(attempt)
            except APIStatusError as e:
                status = getattr(e, "status_code", None)
                if status == 503:  # backoff
                    last_exc = e
                    self._sleep(attempt)
                elif status == 504:  # do NOT retry-storm; upstream may still be computing
                    raise ProviderError("gateway timeout (504); surfaced without retry", status=504, retryable=False) from e
                else:
                    raise ProviderError(f"API error {status}: {e}", status=status) from e
            except APITimeoutError as e:
                raise ProviderError("request timed out", status=504) from e
        raise ProviderError(f"exhausted retries: {last_exc}", retryable=True)

    def _stream(self, kwargs) -> tuple[str, str | None, dict]:
        """Run a streaming completion and accumulate it into (content, finish_reason, usage).

        The final chunk (with include_usage) carries usage and an empty choices list."""
        parts: list[str] = []
        finish_reason: str | None = None
        usage: dict = {}
        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage.model_dump()
            for choice in (chunk.choices or []):
                delta = getattr(choice, "delta", None)
                if delta is not None and getattr(delta, "content", None):
                    parts.append(delta.content)
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
        return "".join(parts), finish_reason, usage

    @staticmethod
    def _sleep(attempt: int) -> None:
        if attempt < len(_BACKOFF):
            time.sleep(_BACKOFF[attempt])
