"""The Planner: turns the arbiter's intent into a concrete `Directive` (spec §4).

Division of labor (spec §2): the NeedsArbiter owns the coarse INTENT (deterministic,
RAM-driven, stable); this planner owns the concrete TARGET and the replan reasoning.
For a ``travel`` intent that means *choosing which map to head toward* — normally the next
hop on the graph route, but on a stuck/impossible replan the LLM can reroute to a different
reachable subgoal instead of re-issuing the identical failing directive (the loop the old
deterministic planner fell into). The base-level geometry is handed to BFS (LLM+P), and
``success`` predicates are computed deterministically here so they stay machine-checkable.

The LLM (LunaRoute) is consulted only for the travel target and only on a replan trigger —
rare and cheap (bold commitment). If no provider is wired, or the call fails / returns an
unreachable map, the planner falls back to the deterministic graph route. So the agent
still navigates with the LLM offline; the LLM makes it *smarter about rerouting*, not
load-bearing for the happy path.
"""
from __future__ import annotations

import json

from ..games.pokemon_red import needs
from ..games.pokemon_red.maps import map_name
from ..games.pokemon_red.needs import WCURMAP
from ..providers.parsing import strip_fences
from .plan import Directive, Intent

# arbiter need name -> executor intent
_NEED_TO_INTENT = {
    "survive": Intent.HEAL,
    "battle": Intent.BATTLE,
    "readiness": Intent.GRIND,
    "progress": Intent.TRAVEL,
    "idle": Intent.TRAVEL,
}


def intent_for_need(need_name: str) -> Intent:
    return _NEED_TO_INTENT.get(need_name, Intent.TRAVEL)


PLANNER_SYSTEM = """You are the STRATEGIC PLANNER for an agent playing Pokémon Red. You are
called only when a new plan is needed (a directive finished, failed, got stuck, or the
situation changed) — so think, don't rubber-stamp.

Your ONE job right now: pick the next MAP to travel toward, given the mission and the world
graph. Usually that is the next hop on the route to the goal. BUT if WHY_REPLAN says the
last attempt got stuck or was impossible, do NOT pick the same map again — reroute: choose a
different reachable map that makes progress (an intermediate area, or backtrack to try
another edge), using TRIED_FAILED and the ROUTE/NEIGHBORS to reason about it.

You are given: MISSION, GOAL_MAP, CURRENT_MAP, ROUTE (map ids from here to the goal),
NEIGHBORS (adjacent maps you can walk to now, with direction), MAP_HISTORY (recent maps —
if it alternates between two ids you are oscillating), TRIED_FAILED, and WHY_REPLAN.

Return ONLY a JSON object:
{"target_map": <int map id to head toward next>, "reason": "one sentence why"}
target_map MUST be a real map id from ROUTE or NEIGHBORS (a reachable map), never the
current map."""


class Planner:
    def __init__(self, *, goal_map: int | None = None, level_target: int = 0,
                 heal_hp: float = 0.80, reflector=None, provider=None):
        self.goal_map = goal_map
        self.level_target = level_target
        self.heal_hp = heal_hp
        self.reflector = reflector      # optional generative strategist (maintains AgentPlan)
        self.provider = provider        # optional chat_json provider (LunaRoute) for target selection

    def plan(self, intent: Intent, emu, memory=None, *, why: str = "") -> Directive:
        """Compute the concrete directive for ``intent``. ``why`` explains the replan
        (fed to the LLM as Inner Monologue) so it never re-issues a failed directive."""
        if intent == Intent.TRAVEL:
            return self._travel(emu, memory, why)
        if intent == Intent.GRIND:
            # grinding needs a destination: grass is en route to the goal, so head that way
            # and level up along the corridor. Success is the level, not arrival.
            cur = emu.read_memory(WCURMAP)
            graph = getattr(memory, "graph", None)
            target = self._goal_travel_target(cur, self.goal_map, graph)
            return Directive(
                intent=Intent.GRIND, target=target,
                success={"level": f">={self.level_target}"} if self.level_target else {"level": ">=999"},
                reason=f"grind party to level {self.level_target} (heading toward grass en route to the goal)",
            )
        if intent == Intent.HEAL:
            return Directive(intent=Intent.HEAL, target=None, success={"hp_frac": f">={self.heal_hp}"},
                             reason="restore party HP")
        if intent == Intent.BATTLE:
            return Directive(intent=Intent.BATTLE, target=None, success={"in_battle": 0},
                             reason="win/flee the current battle")
        return self._travel(emu, memory, why)

    # --- travel: LLM picks the target map; graph does the geometry + fallback --
    def _travel(self, emu, memory, why: str) -> Directive:
        cur = emu.read_memory(WCURMAP)
        goal = self.goal_map
        if goal is None or cur == goal:
            return Directive(intent=Intent.TRAVEL, target=None,
                             success={"on_map": goal if goal is not None else -1},
                             reason="hold position / explore (no distinct goal map)")
        graph = getattr(memory, "graph", None)
        target_map, reason = self._choose_target_map(cur, goal, graph, memory, why)
        return self._travel_directive(cur, target_map, goal, graph, reason)

    def _choose_target_map(self, cur, goal, graph, memory, why) -> tuple[int, str]:
        """The default target is the FINAL goal (bold commitment — one persistent travel
        directive across the corridor; the servo walks the route each step). The LLM is asked
        to reroute to an INTERMEDIATE reachable map only when a replan needs it; any problem
        falls back to the goal."""
        default = goal
        default_reason = f"travel toward map {goal} ({map_name(goal)})"
        if self.provider is None or graph is None:
            return default, default_reason
        try:
            route = graph.route(cur, goal) or []
            neighbors = [{"map": m, "name": map_name(m), "dir": graph.direction_between(cur, m)}
                         for m in graph.neighbors(cur)]
            state = {
                "mission": f"reach map {goal} ({map_name(goal)}) — the next gym town",
                "goal_map": goal,
                "current_map": {"id": cur, "name": map_name(cur)},
                "route": [{"map": m, "name": map_name(m)} for m in route],
                "neighbors": neighbors,
                "map_history": list(getattr(memory, "map_history", []) or [])[-10:],
                "tried_failed": list(getattr(memory, "tried_failed", []) or []),
                "why_replan": why or "starting a new travel directive",
            }
            content, _, _ = self.provider.chat_json(PLANNER_SYSTEM, state)
            data = json.loads(strip_fences(content))
            tm = int(data.get("target_map"))
            reason = str(data.get("reason") or "").strip() or default_reason
            # validate: must be reachable and not the current map
            if tm != cur and graph.route(cur, tm):
                return tm, reason
        except Exception:
            pass
        return default, default_reason

    def _goal_travel_target(self, cur, target_map, graph) -> dict | None:
        """A target dict toward ``target_map``, with the exact exit tile on the CURRENT map
        attached when the graph knows it (so the servo BFS walks straight to it)."""
        if target_map is None:
            return None
        target = {"kind": "map", "map": target_map}
        if graph is not None:
            hop = graph.next_hop(cur, target_map)
            if hop is not None:
                next_map, tile = hop
                target = {"kind": "warp", "map": target_map, "next_map": next_map}
                if tile is not None:
                    target["x"], target["y"] = int(tile[0]), int(tile[1])
        return target

    def _travel_directive(self, cur, target_map, goal, graph, reason) -> Directive:
        """Build the travel directive toward ``target_map``."""
        target = self._goal_travel_target(cur, target_map, graph)
        # success keys off the FINAL goal when heading straight there, else the chosen subgoal
        success_map = goal if target_map == goal else target_map
        return Directive(intent=Intent.TRAVEL, target=target, success={"on_map": success_map},
                         reason=reason)
