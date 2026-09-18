"""Integrated reason-and-act loop.

Every decision is a single vision call that RE-GROUNDS on the current screen and
everything learned so far, instead of blindly following a stale checkpoint:

  where am I?  ->  what do I need to do?  ->  what have I already tried?  ->  next action

The model receives the screenshot, the persistent explored map, a log of the
recent (action -> result) outcomes, and its own notes from the previous step, and
returns one action plus updated notes. Result + new state feed the next call.
"""
from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from ..core.models import AgentAction, AgentDecision
from ..emulator.interface import ImageObservation
from ..providers.parsing import strip_fences
from .plan import AgentPlan, ReflectionPlan  # ReflectionPlan is an alias of AgentPlan (the unified contract)


class ReasonStep(BaseModel):
    location: str = Field(default="", description="where the player is right now")
    objective: str = Field(default="", description="what to do next to progress toward the goal")
    tried: str = Field(default="", description="what has been attempted that did/didn't work")
    reasoning: str = Field(default="", description="why this action")
    action: AgentAction

    def to_decision(self) -> AgentDecision:
        return AgentDecision(action=self.action, decision_note=self.reasoning or self.objective)


class VisionProvider(Protocol):
    def chat_json(self, system_prompt: str, user, image: ImageObservation | None = None) -> tuple[str, int, dict]: ...


REASON_SYSTEM = """You are playing Pokémon Red, deciding ONE button action at a time by
reasoning over the current screen and everything you've learned so far.

Each turn you receive:
- PRIMARY_GOAL and CURRENT_PLAN (a strategy + next_objective set by a periodic
  reflection that reviewed your whole explored map and history). Pursue the plan's
  next_objective; if its explore_note names an unexplored area, head toward that
  '?' region. The plan is your higher-level guide between reflections.
- your previous notes (location/objective/tried),
- a SCREENSHOT of the current screen,
- PLAYER: your exact tile {x, y}, map_id (a location index), and is_outdoor
  (true = you're outside; a building's interior is a different map_id). Trust these
  over the screenshot. If is_outdoor is true you have already left the building — do
  NOT go back into a building unless the goal needs it.
- MAP_HISTORY: the ordered list of map_ids you've been in. If it alternates between
  two ids (e.g. [38,37,38,37]) you are OSCILLATING between two areas — stop and take
  a different exit than the one that keeps sending you back.
- EXITS: every exit tile on this map as {x, y, dest_map, dest_name}. GROUND TRUTH —
  trust it over the screenshot. dest_name tells you where each exit LEADS: to leave
  a building choose the exit whose dest_name is outdoor / makes progress, and do NOT
  take the exit that leads back where you just came from (check RECENT's map numbers).
  To use an exit, walk until your PLAYER x,y equals that EXIT's x,y.
- LOCAL_MAP: walkability right around you ('@'=you, '.'=floor, '#'=wall; screen-relative, up=north),
- MAP_VIEW: the whole current map in map coordinates with a coordinate ruler
  ('@'=you, 'E'=exit tile, '.'=floor, '#'=wall, '?'=unexplored). Read your position
  and the exits' positions straight off this grid.
- RECENT: your last steps as "(x,y)mMAP action -> result" — your own trajectory,
  including which MAP you were on each step.
- GAME_STATE: rich game data — party (each Pokémon's level/HP/status/moves), any
  active battle, badges, money, items, NPCS (people on screen as {x,y,facing}),
  DIALOG_ACTIVE (true = a text box / menu is open), and SCREEN_TEXT (the decoded
  on-screen text). Use it:
  * If DIALOG_ACTIVE is true, a conversation/menu is open: read SCREEN_TEXT and
    respond — use advance_dialog to continue text, or press A/B or move the cursor
    to choose a menu option. Do NOT walk away mid-dialog.
  * To TALK TO A PERSON: pick an NPC from NPCS, walk to a tile ADJACENT to its
    (x,y), face toward it, then use "interact" (press A). You must be next to them
    and facing them. NPCS coordinates are in the same map tiles as your PLAYER x,y.
    Each NPC has "talked_to" (you already had a real conversation with them) and
    "interact_did_nothing" (you pressed A facing them and NOTHING happened — likely a
    wall/counter between you, so move to a different adjacent tile, don't keep pressing).
    Only believe you talked to someone if talked_to is true or recent_dialog shows it.
- SOCIAL_MEMORY: talked_to_count, tiles_where_A_did_nothing (don't retry these blindly),
  recent_interactions (each A-press result: "talked" or "nothing"), and recent_dialog
  (what people/signs actually said). Do NOT assume an interaction worked unless a dialog
  appeared. If pressing A did nothing, reposition (a different adjacent tile / facing).
  * In battle, choose moves using party/enemy HP. If party is empty you still need
    a starter Pokémon — talk to Prof. Oak to get one.

Coordinate rule: x increases EAST, y increases SOUTH. To reach an exit tile:
if exit.x > your x, go EAST; if exit.x < your x, go WEST; if exit.y > your y, go
SOUTH; if exit.y < your y, go NORTH. Alternate axes as needed and route around '#' walls.

Think it through, RE-GROUNDING on the screenshot each time (do not trust a stale plan):
1. WHERE AM I? Read the screenshot: what room/area, and where are exits (doors, stairs,
   mats, ledges), NPCs, and objects — relative to yourself.
2. WHAT DO I NEED TO DO to progress toward the goal from here?
3. REVIEW YOUR RECENT TRAJECTORY. Look at RECENT carefully:
   - Are you revisiting the same (x,y) tiles? Then you are circling — stop and go a NEW way.
   - Did your MAP number bounce back and forth (e.g. 38 -> 37 -> 38)? Then you keep taking an
     exit that leads back where you came from. Identify which exit sends you back, and take a
     DIFFERENT exit instead (compare exits' dest_map to the map you just came from).
   - Do NOT repeat a move that was just blocked.
4. WHAT HAVE I TRIED / WHAT SHOULD I CHECK? If you're stuck or looping, decide what to check
   (an unexplored '?' area, a different exit, or interacting with something) and do that.
5. Pick ONE concrete next action toward an exit/objective.

To LEAVE a room you must step onto its exit tile (a door, stairs, or mat) — usually at an
edge. If you don't see the exit, EXPLORE unmapped ('?') areas until you find it.

Return ONLY a JSON object:
{
  "location": "where you are, from the screenshot",
  "objective": "what to do next to progress",
  "tried": "what you've attempted and what's blocked (carry this forward)",
  "reasoning": "one sentence: why this action",
  "action": {"type":"move","direction":"north|south|east|west","tiles":1}
}
Valid actions: move (direction+tiles 1-10), press (button), interact, advance_dialog, wait (frames)."""


REFLECT_SYSTEM = """You are the strategist for an AI playing Pokémon Red. Periodically you
STEP BACK and think about the bigger picture, then set a plan the moment-to-moment
actor will follow.

You are given the PRIMARY_GOAL, your current PLAYER position, MAP_VIEW (the whole
current map you've mapped: '@'=you, 'E'=exit, '.'=walked/floor, '#'=wall,
'?'=UNEXPLORED), MAP_HISTORY (areas visited), SOCIAL_MEMORY (dialog you've read and
who/what you've interacted with), GAME_STATE (party/npcs/etc), and RECENT steps.

Reflect honestly:
- What have you accomplished, and are you actually making progress toward the goal?
- Are you circling the same tiles without progress? If so, you are NOT exploring —
  the answer is usually in the '?' (unexplored) parts of MAP_VIEW. Name a specific
  unexplored area to go check.
- If an NPC you need (e.g. Prof. Oak) is visible but you haven't talked to them, the
  plan is to walk adjacent and interact.
- Use SOCIAL_MEMORY: follow instructions the game already gave you; don't redo done things.

Return ONLY a JSON object:
{
  "progress": "what you've done / found and whether you're progressing",
  "plan": "the strategy to reach the goal from here",
  "next_objective": "one concrete objective for the next several steps",
  "explore_note": "which unexplored '?' area to check next, if you are searching"
}"""


class Reasoner:
    def __init__(self, provider: VisionProvider):
        self.provider = provider

    def reflect(
        self,
        *,
        primary_goal: str,
        player_desc: str,
        map_view: list[str] | None,
        map_history: list[int],
        social_memory: dict | None,
        game_state: dict | None,
        recent: list[str],
        previous: "ReflectionPlan | None",
    ) -> tuple["ReflectionPlan", int, dict]:
        user = {
            "primary_goal": primary_goal,
            "player": player_desc,
            "map_view": map_view,
            "map_history": map_history,
            "social_memory": social_memory,
            "game_state": game_state,
            "recent": recent,
            "previous_plan": previous.model_dump() if previous else None,
        }
        content, latency, usage = self.provider.chat_json(REFLECT_SYSTEM, user, image=None)
        try:
            plan = ReflectionPlan.model_validate(json.loads(strip_fences(content)))
        except Exception:
            plan = previous or ReflectionPlan(next_objective="explore unmapped areas toward the goal")
        return plan, latency, usage

    def step(
        self,
        *,
        primary_goal: str,
        image: ImageObservation | None,
        local_map: list[str] | None,
        map_view: list[str] | None,
        player_desc: str,
        exits: list[dict],
        game_state: dict | None,
        social_memory: dict | None,
        map_history: list[int],
        recent: list[str],
        previous: ReasonStep | None,
        plan: "ReflectionPlan | None" = None,
        targets: list[dict] | None = None,
        route_hint: str | None = None,
        blocked_dirs: set[str] | None = None,
    ) -> tuple[ReasonStep, int, dict]:
        user = {
            "primary_goal": primary_goal,
            "current_plan": plan.model_dump() if plan else None,
            "player": player_desc,
            "available_targets": targets,
            "route_hint": route_hint,
            "blocked_directions": sorted(blocked_dirs or []),
            "game_state": game_state,
            "social_memory": social_memory,
            "map_history": map_history,
            "exits": exits,
            "local_map": local_map,
            "map_view": map_view,
            "recent": recent,
            "previous_notes": (
                {"location": previous.location, "objective": previous.objective, "tried": previous.tried}
                if previous else None
            ),
        }
        content, latency, usage = self.provider.chat_json(REASON_SYSTEM, user, image=image)
        step = self._parse(content, previous)
        if step is None:  # blank/cold-start — retry once, then a safe explore fallback
            content2, l2, _ = self.provider.chat_json(REASON_SYSTEM, user, image=image)
            latency += l2
            step = self._parse(content2, previous)
        if step is None:
            from ..core.models import MoveAction, Direction
            step = ReasonStep(
                location="(unclear)", objective="explore to find an exit",
                reasoning="reasoning unavailable; explore",
                action=MoveAction(direction=Direction.NORTH),
            )
        return step, latency, usage

    @staticmethod
    def _parse(content: str, fallback: ReasonStep | None) -> ReasonStep | None:
        try:
            data = json.loads(strip_fences(content))
            return ReasonStep.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError):
            return None
