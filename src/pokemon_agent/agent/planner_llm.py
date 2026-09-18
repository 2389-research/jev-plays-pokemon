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
from ..games.pokemon_red.constants import MAP_NAMES_RAW
from ..games.pokemon_red.game_state import read_badges, read_items, read_party
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

TOOL — knowledge base: you may look things up in a Pokémon Red guide before choosing. To
search, reply with ONLY {"search": ["query1", "query2"]} (1-3 queries); results come back in
KNOWLEDGE_GATHERED and you can search again or finalize. SEARCH_ROUNDS_LEFT limits searches.

Return ONLY a JSON object (when ready):
{"target_map": <int map id to head toward next>, "reason": "one sentence why"}
target_map MUST be a real map id from ROUTE or NEIGHBORS (a reachable map), never the
current map."""


WAYPOINT_SYSTEM = """You are the navigation module for a Pokémon Red agent. The deterministic
pathfinder is STUCK. Pick ONE distant waypoint tile to commit to that breaks the deadlock.

COORDINATE SYSTEM (read carefully): the grid uses (x, y). x = COLUMN, read off the TWO header
rows — the first is the tens digit, the second the units digit — 0 at the left, increasing to
the RIGHT. y = ROW, labeled 'y<n>' at the start of each line, increasing DOWNWARD (south);
y=0 is the NORTH edge. Symbols: '@'=you, '.'=walkable floor, '#'=wall, 'N'=an NPC (NEVER
target it — walking into one only talks to it), 'D'=a door/building exit, '?'=unknown.

GOAL_DIR says which way the next area is. WHY_STUCK says what went wrong.

Pick a WALKABLE '.' tile that is: (a) reachable from '@' WITHOUT crossing '#' or 'N';
(b) AT LEAST 4 tiles away (Manhattan); (c) as far along a clear path toward GOAL_DIR as you
can. TRACE the path tile-by-tile in your head first and make sure every step is '.'.

Return ONLY JSON: {"path": "(x,y)->(x,y)->...", "x": <int>, "y": <int>, "reason": "..."}"""


STRATEGIST_SYSTEM = """You are the STRATEGIC planner (tier 2) for an agent playing Pokémon Red,
working toward the first gym (Brock, in Pewter City, north). The fast navigator is BLOCKED and
cannot proceed on its own — usually a STORY GATE (an NPC who won't move, a locked path, a
required item/errand). Work out the SEQUENCE of steps that unblocks progress.

TOOL — knowledge base: you can look things up in a Pokémon Red guide/knowledge base before you
commit. To search, reply with ONLY {"search": ["query1", "query2"]} (1-3 queries); you'll get
the results back in KNOWLEDGE_GATHERED and can search again or finalize. SEARCH FIRST to ground
your plan in the guides rather than guessing, especially for story gates. SEARCH_ROUNDS_LEFT
tells you how many more searches you may do; when it hits 0 you must output the final plan.

You are given: WHY_BLOCKED, CURRENT_MAP (id + name), PARTY, ITEMS, BADGES, MAPS (an id→name
table — use these exact ids), and KNOWLEDGE_GATHERED (results of your searches so far — TRUST
these over your own memory when they conflict).

Reason about what the game requires here (e.g. fetch an item from a shop and deliver it, talk to
a specific person, enter a building), then output an ORDERED list of steps. Each step is a MAP to
go to, optionally talking to an NPC once there. The LAST step should continue toward the gym once
unblocked.

Return ONLY JSON:
{"plan": "one-line summary", "steps": [{"map": <int map id>, "talk": <true|false>, "why": "<short>"}]}"""


class Planner:
    def __init__(self, *, goal_map: int | None = None, level_target: int = 0,
                 heal_hp: float = 0.80, reflector=None, provider=None, strategist=None, knowledge=None):
        self.goal_map = goal_map
        self.level_target = level_target
        self.heal_hp = heal_hp
        self.reflector = reflector      # optional generative strategist (maintains AgentPlan)
        self.provider = provider        # tier-1 chat_json provider (LunaRoute-fast) for targets
        self.strategist = strategist    # tier-2 chat_json provider (LunaRoute-strong) for quests
        self.knowledge = knowledge      # optional Orrery KnowledgeBase for retrieval-grounded quests
        self.on_search = None           # optional callback(query, n_results) for logging KB tool-calls

    def _llm_with_search(self, provider, system: str, state: dict, *, final_key: str,
                         max_rounds: int = 3) -> dict:
        """Run an LLM call where the model MAY use the knowledge base as a tool: each round it
        returns either {"search": [...]} (we run the queries, feed results back as
        KNOWLEDGE_GATHERED) or its final JSON object (which contains ``final_key``). The model
        decides when and what to look up. Returns the final parsed dict."""
        gathered: list[dict] = []
        data: dict = {}
        for round_i in range(max_rounds + 1):
            s = {**state, "knowledge_gathered": gathered, "search_rounds_left": max_rounds - round_i}
            try:
                content, _, _ = provider.chat_json(system, s)
                data = json.loads(strip_fences(content))
            except Exception:
                return data
            queries = data.get("search")
            is_search = (isinstance(queries, list) and queries and self.knowledge is not None
                         and final_key not in data and round_i < max_rounds)
            if not is_search:
                return data
            for q in [str(x) for x in queries][:3]:
                res = self.knowledge.query_texts(q, top_k=4)
                gathered.append({"query": q, "results": res})
                if self.on_search:
                    self.on_search(q, len(res))
        return data

    def strategize(self, emu, memory=None, *, why: str = "") -> list[Directive]:
        """Tier-2 problem solving: given a blocked situation, return an ORDERED quest of
        directives that unblocks progress (travel/enter to a map, optionally talk there). Empty
        list if no strategist is wired or the plan can't be parsed."""
        prov = self.strategist or self.provider
        if prov is None:
            return []
        cur = emu.read_memory(WCURMAP)
        try:
            state = {
                "why_blocked": why,
                "current_map": {"id": cur, "name": map_name(cur)},
                "goal": f"reach map {self.goal_map} ({map_name(self.goal_map)})" if self.goal_map is not None else "progress",
                "party": [f"{p['species']} L{p['level']}" for p in read_party(emu)],
                "items": [it["item"] for it in read_items(emu)],
                "badges": read_badges(emu)["count"],
                "maps": {str(mid): name for mid, name in MAP_NAMES_RAW.items()},
            }
            # the strategist MAY search the knowledge base as a tool before finalizing the quest
            data = self._llm_with_search(prov, STRATEGIST_SYSTEM, state, final_key="steps", max_rounds=3)
            plan_note = str(data.get("plan") or "quest").strip()
            quest: list[Directive] = []
            for step in data.get("steps", []):
                mp = int(step["map"])
                why_s = str(step.get("why") or plan_note)[:80]
                quest.append(Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": mp},
                                       success={"on_map": mp}, reason=f"quest: go to {map_name(mp)} — {why_s}"))
                if step.get("talk"):
                    quest.append(Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": mp},
                                           success={"talked_on_map": mp},
                                           reason=f"quest: talk to someone in {map_name(mp)} — {why_s}"))
            return quest
        except Exception:
            return []

    def plan(self, intent: Intent, emu, memory=None, *, why: str = "", context: dict | None = None) -> Directive:
        """Compute the concrete directive for ``intent``. ``why`` explains the replan
        (fed to the LLM as Inner Monologue) so it never re-issues a failed directive.

        When the replan is because we're STUCK/impossible on a target-bearing intent, and a
        map view + provider are available, ask LunaRoute for a concrete WAYPOINT tile on the
        current map and route there (the LLM's spatial reasoning gets teeth — BFS alone can't
        un-stick itself)."""
        stuck = context is not None and ("stuck" in why or "impossible" in why)
        if stuck and intent in (Intent.TRAVEL, Intent.GRIND):
            wp = self._waypoint(emu, context)
            if wp is not None:
                cur = emu.read_memory(WCURMAP)
                x, y, reason = wp
                return Directive(intent=intent, target={"kind": "waypoint", "map": cur, "x": x, "y": y},
                                 success={"at_xy": [cur, x, y]}, reason=f"unstuck waypoint: {reason}")
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

    def _waypoint(self, emu, context: dict) -> tuple[int, int, str] | None:
        """Ask LunaRoute for a concrete walkable tile on the CURRENT map to route toward, to
        break a deterministic-pathfinder deadlock. VALIDATES the pick against the deterministic
        reachable set (rejecting walls / unreachable / too-close picks) and retries once — so a
        good presentation does the heavy lifting and a guardrail catches the occasional miss.
        Returns (x, y, reason) or None."""
        if self.provider is None:
            return None
        player = context.get("player") or {}
        reachable = context.get("reachable")  # set[(x,y)] BFS-reachable, avoiding NPCs
        state = {
            "goal_dir": context.get("goal_dir"),
            "player": player,
            "map_view": context.get("map_view"),
            "why_stuck": context.get("why", "the pathfinder is oscillating without progress"),
        }
        for _ in range(2):  # one retry if the model picks an invalid tile
            try:
                content, _, _ = self.provider.chat_json(WAYPOINT_SYSTEM, state)
                data = json.loads(strip_fences(content))
                x, y = int(data["x"]), int(data["y"])
            except Exception:
                continue
            reason = str(data.get("reason") or "").strip() or "head toward the goal corridor"
            px, py = player.get("x"), player.get("y")
            far_enough = px is None or (abs(x - px) + abs(y - py) >= 3)
            ok = far_enough and (reachable is None or (x, y) in reachable)
            if ok:
                return x, y, reason
        return None

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
            # the tier-1 planner MAY search the KB as a tool before choosing the reroute target
            data = self._llm_with_search(self.provider, PLANNER_SYSTEM, state, final_key="target_map")
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
