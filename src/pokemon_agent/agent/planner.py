"""Periodic vision-based reflection: look at the screen, keep a subgoal checklist,
notate landmarks, and set the next concrete checkpoint.

Runs infrequently (every N steps) and on demand when the actor stalls. Between
reflections the fast text actor navigates toward `next_checkpoint` using the
deterministic local walkability map.
"""
from __future__ import annotations

import json
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from ..emulator.interface import ImageObservation
from ..providers.parsing import strip_fences


class Subgoal(BaseModel):
    text: str
    status: Literal["active", "done"] = "active"


class Plan(BaseModel):
    scene: str = Field(default="", description="what the planner sees in the screenshot")
    strategy: str = Field(default="", description="short overall approach to the primary goal")
    subgoals: list[Subgoal] = Field(default_factory=list, description="ordered checklist")
    landmarks: list[str] = Field(default_factory=list, description="notated points of interest w/ locations")
    next_checkpoint: str = Field(default="", description="the concrete immediate target to navigate to")
    memory: str = Field(default="", description="running natural-language notes of what's been learned/done")
    progress: Literal["on_track", "no_progress", "achieved", "unknown"] = "unknown"

    def active_subgoal(self) -> str:
        for sg in self.subgoals:
            if sg.status == "active":
                return sg.text
        return self.next_checkpoint or self.strategy

    def as_actor_context(self) -> dict:
        return {
            "active_subgoal": self.active_subgoal(),
            "next_checkpoint": self.next_checkpoint,
            "scene": self.scene,
            "landmarks": self.landmarks,
        }


class ReflectiveProvider(Protocol):
    def chat_json(self, system_prompt: str, user, image: ImageObservation | None = None) -> tuple[str, int, dict]: ...


REFLECT_SYSTEM = """You are the strategist for an AI playing Pokémon Red. You look at a
screenshot of the current Game Boy screen plus a text walkability map and decide the plan.

You are given: the PRIMARY GOAL, your previous SUBGOALS (a checklist), previous
LANDMARKS, your MEMORY, a summary of recent movement, and a local walkability map
(`@` = the player, `.` = walkable floor, `#` = wall/blocked; screen-relative, so up=north).
If `stall_reason` is set, the actor got stuck — figure out why and change the plan to recover.

Do all of this:
1. Look at the screenshot. Describe the area and where key features are (stairs, doors,
   exits, ledges, signs, NPCs) with rough screen locations.
2. Update the SUBGOAL checklist: mark any completed subgoal "done", keep the current one
   "active", and ADD a new subgoal if the next step of the plan isn't represented. Keep the
   list short (2-5 items). Exactly one should be "active".
3. Update LANDMARKS: carry forward useful ones and add newly-seen points of interest, each
   as a short string with a location (e.g. "stairs: bottom-left of this room",
   "exit door: north wall", "the big lab: south end of town").
4. Set NEXT_CHECKPOINT: one concrete, reachable sub-target to walk to now.
5. Update MEMORY: carry forward prior facts, add what you just learned/accomplished, drop
   nothing important. A few short sentences.
6. Judge progress toward the primary goal.

Return ONLY a JSON object:
{
  "scene": "what you see and where key features are",
  "strategy": "one sentence overall approach",
  "subgoals": [{"text": "...", "status": "active|done"}, ...],
  "landmarks": ["label: location", ...],
  "next_checkpoint": "one concrete sub-target to navigate to now",
  "memory": "durable notes, updated from the previous memory",
  "progress": "on_track | no_progress | achieved | unknown"
}"""


class Planner:
    def __init__(self, provider: ReflectiveProvider):
        self.provider = provider

    def reflect(
        self,
        *,
        primary_goal: str,
        trajectory_summary: str,
        previous_plan: Plan | None,
        image: ImageObservation | None,
        player_desc: str,
        local_map: list[str] | None = None,
        explored_map: list[str] | None = None,
        stall_reason: str | None = None,
    ) -> tuple[Plan, int, dict]:
        prev = previous_plan.model_dump() if previous_plan else None
        user = {
            "primary_goal": primary_goal,
            "player": player_desc,
            "walkability_map": local_map,
            "explored_map": explored_map,
            "recent_movement": trajectory_summary,
            "previous_subgoals": prev["subgoals"] if prev else None,
            "previous_landmarks": prev["landmarks"] if prev else None,
            "memory": prev["memory"] if prev else None,
            "stall_reason": stall_reason,
        }
        content, latency_ms, usage = self.provider.chat_json(REFLECT_SYSTEM, user, image=image)
        plan = self._parse(content, fallback=previous_plan)
        if plan is None:  # empty/blank (vision cold-starts do this) — retry once
            content2, l2, _ = self.provider.chat_json(REFLECT_SYSTEM, user, image=image)
            latency_ms += l2
            plan = self._parse(content2, fallback=previous_plan) or Plan(
                strategy="(reflection unavailable; keep exploring toward the goal)",
                next_checkpoint="explore to find an exit", progress="unknown",
            )
        return plan, latency_ms, usage

    @staticmethod
    def _parse(content: str, fallback: Plan | None) -> Plan | None:
        try:
            data = json.loads(strip_fences(content))
            plan = Plan.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError):
            return fallback
        if not (plan.scene.strip() or plan.strategy.strip() or plan.next_checkpoint.strip() or plan.subgoals):
            return fallback
        return plan
