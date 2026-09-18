"""One JSONL record per decision step, enough to reconstruct what the agent saw.

Screenshot logging is configurable because it uses disk:
  none         -> never save pixels
  errors_only  -> save the frame only when the action did not complete
  every_step   -> save the pre-action frame every step

Frames are written next to the log as <episode>_shots/step_00042.png and the
record stores the relative path. Never logs API keys.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ..core.models import ActionResult, Observation
from ..emulator.interface import ImageObservation
from ..providers.interface import AgentResponse

SCREENSHOT_MODES = ("none", "errors_only", "every_step")


class EpisodeLogger:
    def __init__(self, path: str | Path, *, provider: str, model: str, screenshot_mode: str = "none"):
        if screenshot_mode not in SCREENSHOT_MODES:
            raise ValueError(f"screenshot_mode must be one of {SCREENSHOT_MODES}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_id = uuid.uuid4().hex[:8]
        self.provider = provider
        self.model = model
        self.screenshot_mode = screenshot_mode
        self.shots_dir = self.path.parent / f"{self.episode_id}_shots"
        self._fh = self.path.open("a", encoding="utf-8")

    @property
    def wants_screenshot(self) -> bool:
        """Whether the loop should bother capturing a frame for the log."""
        return self.screenshot_mode != "none"

    def _maybe_write_shot(self, step: int, shot: ImageObservation | None, result: ActionResult) -> str | None:
        if shot is None or self.screenshot_mode == "none":
            return None
        if self.screenshot_mode == "errors_only" and result.result == "completed":
            return None
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        fname = f"step_{step:05d}.png"
        (self.shots_dir / fname).write_bytes(shot.data)
        return str((self.shots_dir / fname).relative_to(self.path.parent))

    def record(
        self,
        *,
        step: int,
        observation: Observation,
        response: AgentResponse,
        result: ActionResult,
        plan=None,
        screenshot: ImageObservation | None = None,
    ) -> None:
        shot_path = self._maybe_write_shot(step, screenshot, result)
        rec = {
            "episode_id": self.episode_id,
            "step": step,
            "timestamp": time.time(),
            "provider": self.provider,
            "model": response.model or self.model,
            "observation_id": observation.observation_id,
            "observation": observation.to_model_json(),
            "plan": plan.model_dump() if plan is not None else None,
            "decision": response.decision.model_dump(),
            "decision_note": response.decision.decision_note,
            "agent_response_raw": response.raw_content,
            "action_result": result.model_dump(),
            "screenshot": shot_path,
            "latency_ms": response.latency_ms,
            "usage": response.usage,
        }
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
