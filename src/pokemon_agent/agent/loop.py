"""The agent loop. Contains NO PyBoy-specific and NO provider-specific code.

Two-tier control:
  * a periodic vision PLANNER (every `reflect_every` steps, and at step 0) looks at
    the screen and sets a plan + next_checkpoint;
  * the per-step ACTOR navigates toward that checkpoint each step.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from ..actions.controller import ActionController
from ..actions.stuck_detector import StuckDetector
from ..core.models import ActionResult, MoveAction
from ..observations.builder import ObservationBuilder
from ..providers.interface import AgentRequest, ModelProvider, ProviderError
from .planner import Planner
from .prompts import SYSTEM_PROMPT
from .session import Session
from .world_map import WorldMap


class AgentLoop:
    def __init__(
        self,
        *,
        builder: ObservationBuilder,
        provider: ModelProvider,
        controller: ActionController,
        session: Session,
        vision: bool = False,
        planner: Optional[Planner] = None,
        reflect_every: int = 10,
        logger=None,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ):
        self.builder = builder
        self.provider = provider
        self.controller = controller
        self.session = session
        self.vision = vision
        self.planner = planner
        self.reflect_every = max(1, reflect_every)
        self.logger = logger
        self.on_event = on_event or (lambda kind, payload: None)
        self.stuck = StuckDetector()
        self._last_stuck = None
        self._last_reflect_step = -1
        self.world = WorldMap()

    def _reflect_reason(self) -> str | None:
        if self.planner is None:
            return None
        if self.session.step % self.reflect_every == 0:
            return "scheduled"
        # stall-triggered, but not right after a reflection (avoid thrash)
        if self._last_stuck and self._last_stuck.stuck and self.session.step - self._last_reflect_step >= 3:
            return f"stalled: {self._last_stuck.reason} (x{self._last_stuck.repeat_count})"
        return None

    def step_once(self) -> ActionResult:
        reflect_reason = self._reflect_reason()
        log_shot = bool(self.logger and getattr(self.logger, "wants_screenshot", False))
        obs, shot = self.builder.build(
            recent_events=self.session.recent_events,
            stuck=self._last_stuck,
            capture_screenshot=self.vision or reflect_reason is not None or log_shot,
        )
        self.session.record_position(obs.player)
        # accumulate + inject the persistent exploration map
        self.world.observe(obs.player, obs.walkability)
        obs.explored_map = self.world.ascii(obs.player)
        obs.unexplored_directions = self.world.unexplored_directions(obs.player)
        obs.suggested_explore = self.world.explore_step(obs.player)
        self.on_event("observation", {"observation": obs.to_model_json()})

        # --- vision reflection -> plan (scheduled or stall-recovery) ---
        if reflect_reason is not None:
            player_desc = obs.player.model_dump() if obs.player else "unknown"
            plan, p_latency, _ = self.planner.reflect(
                primary_goal=self.session.goal.primary,
                trajectory_summary=self.session.trajectory_summary(),
                previous_plan=self.session.plan,
                image=shot,
                player_desc=str(player_desc),
                local_map=obs.walkability,
                explored_map=obs.explored_map,
                stall_reason=None if reflect_reason == "scheduled" else reflect_reason,
            )
            self.session.plan = plan
            self._last_reflect_step = self.session.step
            self.on_event("plan", {"plan": plan.model_dump(), "latency_ms": p_latency, "reason": reflect_reason})

        request = AgentRequest(
            system_prompt=SYSTEM_PROMPT,
            observation=obs,
            goal=self.session.goal,
            session_summary=self.session.summary,
            plan=self.session.plan.as_actor_context() if self.session.plan else None,
            image=shot if self.vision else None,
        )
        self.on_event("thinking", {"step": self.session.step})
        response = self.provider.complete(request)  # may raise ProviderError
        self.on_event("decision", {
            "action": response.decision.action.model_dump(),
            "decision_note": response.decision.decision_note,
            "latency_ms": response.latency_ms,
        })

        result = self.controller.execute(response.decision.action)
        self.session.record_result(result)
        # learn walls from real blocked moves
        if result.result == "blocked" and isinstance(response.decision.action, MoveAction):
            self.world.mark_blocked(obs.player, response.decision.action.direction)

        player = obs.player
        self._last_stuck = self.stuck.update(
            response.decision.action, result, player, shot.data if shot else None,
        )
        self.on_event("result", {"result": result.model_dump(), "stuck": self._last_stuck.model_dump()})

        if self.logger:
            self.logger.record(step=self.session.step, observation=obs, response=response,
                               result=result, plan=self.session.plan, screenshot=shot)

        self.session.step += 1
        return result

    def run(self, max_steps: int = 50) -> None:
        for _ in range(max_steps):
            if not self.session.running:
                break
            try:
                self.step_once()
            except ProviderError as e:
                self.session.running = False
                self.on_event("error", {"message": str(e), "status": e.status})
                raise
