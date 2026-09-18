"""The executive: one coherent hierarchy driven by a single active Directive.

This is the 3T "executive" middle layer (spec docs/superpowers/specs/2026-09-17-
planner-executor-design.md). Each step:

  1. perceive + update memory
  2. mode controllers own the step when they apply (battle / passive dialog)
  3. directive management — the arbiter sets the INTENT, the planner computes the
     concrete Directive (target + machine-checkable success); RAM owns termination
     detection; replan fires only on explicit triggers (bold commitment)
  4. deterministic SERVO — BFS / ripped connection directions route toward the
     directive's target (LLM+P: the classical solver does the geometry)
  5. the calibrated EXECUTOR (Jev) only for residual decisions, its Choice masked +
     biased BY the directive (teeth)

There is ONE source of truth: the active Directive. The old rival controllers
(travel autopilot, anti-seam-oscillation, frontier-escape recovery,
effective_goal_map routing) are gone — they are subsumed by "serve the directive."

When no planner is configured (no goal_map / level_target), the loop runs the plain
executor→action path (legacy behavior for the vertical-slice tests).
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Optional

from pathlib import Path

from ..actions.controller import ActionController
from ..actions.stuck_detector import StuckDetector
from ..core.models import ActionResult, Direction, GoToAction, InteractAction, MenuSelectAction, MoveAction
from ..games.pokemon_red import predicates
from ..games.pokemon_red.maps import map_name
from ..games.pokemon_red.progress import progress_vector
from ..games.pokemon_red.routes import next_direction
from ..games.pokemon_red.state import detect_mode, read_player
from ..observations.builder import ObservationBuilder
from .memory import AgentMemory
from .navigator import Navigator
from .plan import Directive, Intent
from .reasoner import Reasoner, ReasonStep
from .session import Session
from .targets import build_targets
from .world_map import DELTA, WALL

MAX_GOTO_STEPS = 18  # safety bound on one navigation macro
SERVO_FAIL_LIMIT = 6  # consecutive servo no-route steps before the directive is deemed impossible
REPLAN_COOLDOWN = 8   # min steps between soft replans (stuck/low-conf) — enforces bold commitment
WARP_BACK = 0xFF     # a warp's dest_map of 0xFF means "return to the map you came from" (wLastMap)

# priority of each intent, for preemption/suspension (higher preempts lower).
_INTENT_PRIORITY = {
    Intent.HEAL: 100, Intent.BATTLE: 90, Intent.SHOP: 60,
    Intent.GRIND: 50, Intent.TALK_TO: 40, Intent.GRAB_ITEM: 40,
    Intent.ENTER: 30, Intent.TRAVEL: 10,
}

# screen-relative neighbor of the player in the local walkability window (up = north)
_SCREEN_DELTA = {
    Direction.NORTH: (-1, 0),
    Direction.SOUTH: (1, 0),
    Direction.EAST: (0, 1),
    Direction.WEST: (0, -1),
}


class ReasoningLoop:
    def __init__(
        self,
        *,
        builder: ObservationBuilder,
        controller: ActionController,
        reasoner: Reasoner,
        session: Session,
        logger=None,
        recent_max: int = 14,
        vision: bool = True,
        reflect_every: int = 8,
        low_conf_reflect: float | None = None,
        reflect_cooldown: int = 4,
        memory: AgentMemory | None = None,
        checkpoint_every: int = 0,
        checkpoint_dir: str | Path | None = None,
        goal_map: int | None = None,
        level_target: int = 0,
        strategist_provider=None,
        knowledge=None,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ):
        self.builder = builder
        self.controller = controller
        self.reasoner = reasoner
        self.session = session
        self.logger = logger
        self.vision = vision
        self.on_event = on_event or (lambda kind, payload: None)
        self.reflect_every = max(1, reflect_every)
        # when the decider reports confidence below this, re-plan on the NEXT step
        # (throttled by reflect_cooldown) instead of thrashing. None = off.
        self.low_conf_reflect = low_conf_reflect
        self.reflect_cooldown = max(1, reflect_cooldown)
        self._force_reflect = False
        self._last_reflect_step = -(10**9)
        # persistent memory (world map + graph + interactions + plan + ledger), checkpointed
        self.memory = memory or AgentMemory()
        self.world = self.memory.world              # alias — same object gets serialized
        self.interactions = self.memory.interactions
        self._map_history = self.memory.map_history  # alias
        self.stuck = StuckDetector()
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.goal_map = goal_map
        # needs arbiter + planner: the two-tier control. The arbiter sets the INTENT; the
        # planner computes the concrete directive. Active only when a goal/level is set —
        # otherwise the loop runs the plain executor path (legacy vertical-slice behavior).
        self.arbiter = None
        self.planner = None
        if goal_map is not None or level_target > 0:
            from .needs_arbiter import NeedsArbiter
            from .planner_llm import Planner
            # the LLM planner needs a chat_json provider for travel-target selection: the
            # generative reasoner exposes it directly, or via its reflector (TypeSafe case).
            prov = getattr(reasoner, "provider", None) or getattr(
                getattr(reasoner, "reflector", None), "provider", None)
            self.arbiter = NeedsArbiter(goal_map=goal_map, level_target=level_target)
            self.planner = Planner(goal_map=goal_map, level_target=level_target,
                                   reflector=reasoner, provider=prov,
                                   strategist=strategist_provider or prov, knowledge=knowledge)
            # surface the planner's knowledge-base tool-calls in the run log
            self.planner.on_search = lambda q, n: self.on_event(
                "kb_search", {"step": self.session.step, "query": q, "results": n})
        # executive state (the single source of truth + its suspension stack + quest queue)
        self._directive: Directive | None = None
        self._dstack: list[Directive] = []
        self._quest: deque[Directive] = deque()   # tier-2 quest steps, executed in order (FIFO)
        self._in_quest = False                      # holding a quest step (overrides arbiter intent)
        self._replan_next = False       # a trigger (stuck / low-conf) asked for a replan
        self._servo_fail = 0            # consecutive servo no-route steps (impossibility)
        self._steps_since_replan = REPLAN_COOLDOWN  # cooldown counter (bold commitment)
        self._prev_map: int | None = None  # last DISTINCT map (resolves 0xFF "return" warps)
        self._collision_seen: set[int] = set()  # maps whose full RAM collision we've ingested
        self._recent: deque[str] = deque(maxlen=recent_max)
        self._prev: ReasonStep | None = None
        self._plan = self.memory.plan  # strategic AgentPlan (reflection), separate from the directive

    # ------------------------------------------------------------------ step
    def step_once(self) -> ActionResult:
        want_shot = self.vision or bool(self.logger and getattr(self.logger, "wants_screenshot", False))
        obs, shot = self.builder.build(capture_screenshot=want_shot)
        self.session.record_position(obs.player)
        # FULL-MAP collision from RAM (wOverworldMap): load the whole current map's walkability
        # so the navigator routes across a town up front (not just the on-screen window).
        # Guarded against STALE reads: after an in-game map transition the map buffer isn't
        # settled for a few frames, so a read can decode the PREVIOUS map. We only accept a
        # decode where the player stands on a walkable cell (always true for a correct read),
        # and re-ingest if a cached grid ever marks the player's own tile as a wall.
        if obs.player is not None:
            mid = obs.player.map_id
            stale_cache = self.world.tiles.get(mid, {}).get((obs.player.x, obs.player.y)) == WALL
            if mid not in self._collision_seen or stale_cache:
                from ..games.pokemon_red.map_reader import read_collision_map
                cm = read_collision_map(self.controller.emu)
                if (cm is not None and cm["map_id"] == mid
                        and (obs.player.x, obs.player.y) in cm["walkable"]):  # reject stale reads
                    self.world.ingest_collision(cm["map_id"], cm["width"], cm["height"], cm["walkable"])
                    self._collision_seen.add(mid)
                    # re-apply learned obstacles (signs / bump-blocked tiles) so a fresh ingest
                    # doesn't wipe what we discovered by bumping — they'd read as walkable again.
                    for (mp, ex, ey, edir) in self.memory.blocked_edges:
                        if mp == mid:
                            dx, dy = DELTA[Direction(edir)]
                            self.world.tiles[mid][(ex + dx, ey + dy)] = WALL
        self.world.observe(obs.player, obs.walkability)
        obs.map_view = self.world.render(obs.player, obs.exits, obs.map_dims)
        obs.unexplored_directions = self.world.unexplored_directions(obs.player)
        if obs.player and (not self._map_history or self._map_history[-1] != obs.player.map_id):
            if self._map_history:                      # remember where we came FROM
                self._prev_map = self._map_history[-1]
            self._map_history.append(obs.player.map_id)
        # resolve 0xFF ("return to last map") warps so building doors become real graph edges
        # (otherwise a house exit reads as "map 255" and the agent can't route back out).
        if obs.player:
            obs.exits = self._resolve_exits(obs.exits, obs.player.map_id)
        if obs.player and obs.exits:  # learn real cross-map warp edges as we see them
            self.memory.graph.observe_exits(obs.player.map_id, obs.exits)
        self.interactions.record_dialog(self.session.step, obs.player, obs.game_state)
        if obs.game_state and obs.game_state.get("npcs"):
            obs.game_state["npcs"] = self.interactions.annotate_npcs(obs.player, obs.game_state["npcs"])

        player_desc = obs.player.model_dump() if obs.player else "unknown"
        ctx = (obs.game_state or {}).get("context") or {}
        ctx_kind = ctx.get("kind")

        # --- mode controllers own the step (battle / passive dialog) ---
        if ctx.get("in_battle"):
            rstep, latency, usage, result = self._battle_turn(obs)
            self._prev = rstep
            self._emit_reason(rstep, latency)
            return self._finish(obs, rstep, result, latency, usage, shot)
        if ctx_kind == "dialog":
            return self._advance_dialog(obs, shot)

        # --- directive management (only when the planner is active) ---
        directive = self._manage_directive(obs) if self.planner is not None else None
        # blocked_dirs = known walls + one-way ledges + OFF-ROUTE building doors (so neither
        # the servo nor the executor wanders into a house/lab that isn't the next hop).
        allowed_next = self._next_hop_map(obs, directive)
        blocked_dirs = self._blocked_dirs(obs, allowed_next)
        if self.planner is not None and directive is not None:
            # deterministic SERVO: route toward the directive's target with no model call
            if directive is not None and directive.target_bearing and ctx_kind == "overworld":
                servo = self._servo_step(directive, obs, blocked_dirs)
                if servo is not None:
                    self._servo_fail = 0
                    return self._act_servo(obs, directive, servo, shot)
                self._servo_fail += 1
            else:
                self._servo_fail = 0

        # --- reflection (periodic / forced): maintains the strategic AgentPlan ---
        if self.session.step % self.reflect_every == 0 or self._force_reflect:
            self._force_reflect = False
            self._last_reflect_step = self.session.step
            self._plan, rlat, _ = self.reasoner.reflect(
                primary_goal=self.session.goal.primary,
                player_desc=str(player_desc),
                map_view=obs.map_view,
                map_history=self._map_history[-12:],
                social_memory=self.interactions.summary(),
                game_state=obs.game_state,
                recent=list(self._recent),
                previous=self._plan,
            )
            self.on_event("reflect", {"step": self.session.step, "latency_ms": rlat,
                                      "plan": self._plan.model_dump()})

        # --- executor (Jev): the residual decision, its Choice masked+biased by the directive ---
        targets = build_targets(obs.game_state, obs.exits)
        # under a travel/grind directive, don't offer the executor a building exit that isn't
        # the next hop — otherwise, when the servo has no step, it wanders into a house/mart.
        if directive is not None and directive.intent in (Intent.TRAVEL, Intent.GRIND):
            allowed_next = self._next_hop_map(obs, directive)
            targets = [t for t in targets
                       if t.get("interact", True) or t.get("dest_map") == allowed_next]
        rstep, latency, usage = self.reasoner.step(
            primary_goal=self.session.goal.primary,
            image=shot if self.vision else None,
            local_map=obs.walkability,
            map_view=obs.map_view,
            player_desc=str(player_desc),
            exits=obs.exits,
            game_state=obs.game_state,
            social_memory=self.interactions.summary(),
            map_history=self._map_history[-12:],
            recent=list(self._recent),
            previous=self._prev,
            plan=self._plan,
            targets=targets,
            route_hint=None,
            blocked_dirs=blocked_dirs,
            directive=directive,
        )
        self._prev = rstep
        # confidence-driven control: unsure -> re-plan next step (reflect + directive replan)
        conf = (usage or {}).get("confidence")
        if (self.low_conf_reflect is not None and conf is not None
                and conf < self.low_conf_reflect
                and self.session.step - self._last_reflect_step >= self.reflect_cooldown):
            self._force_reflect = True
            self._replan_next = True
        self._emit_reason(rstep, latency)
        result = self._dispatch(rstep, obs, blocked_dirs, ctx_kind)
        return self._finish(obs, rstep, result, latency, usage, shot)

    # --------------------------------------------------- directive lifecycle
    def _manage_directive(self, obs) -> Directive | None:
        """Set/refresh the single active directive per the replan triggers (spec §6).

        RAM owns termination detection (the success predicate); the planner owns the next
        directive and runs only on a trigger."""
        emu = self.controller.emu
        intent = self.arbiter.intent(emu)
        self._steps_since_replan += 1

        # --- termination: did the active directive succeed? ---
        if self._directive is not None and predicates.evaluate(self._directive.success, emu, memory=self.memory):
            self.on_event("directive_done", {"step": self.session.step,
                                             "intent": self._directive.intent.value,
                                             "reason": self._directive.reason})
            self._servo_fail = 0
            if self._in_quest:
                # advance the quest to its next step (FIFO); when empty, the quest is complete.
                self._directive = self._quest.popleft() if self._quest else None
                if self._directive is None:
                    self._in_quest = False
                    self.on_event("quest_done", {"step": self.session.step})
                else:
                    self._commit_directive("next quest step")
                    return self._directive
            else:
                self._directive = self._dstack.pop() if self._dstack else None

        # --- holding a quest step: it overrides the arbiter's intent; only drop it if the step
        # is impossible (re-strategize) or a SURVIVE emergency preempts. ---
        if self._in_quest and self._directive is not None:
            if self._servo_fail >= SERVO_FAIL_LIMIT:
                self._in_quest = False
                self._quest.clear()  # this quest step can't be reached — fall through to re-plan
            elif intent == Intent.HEAL and self._directive.intent != Intent.HEAL:
                self._in_quest = False  # emergency heal preempts; re-derive the quest later
            else:
                return self._carry_plan()

        # --- replan? ---
        if self._directive is None or self._should_replan(intent):
            why = self._replan_reason(intent)
            # BLOCKED and it smells like a story gate -> tier-2 quest (Jev escalation router).
            if ("impossible" in why or "stuck" in why) and self._try_quest(obs, why):
                self._directive = self._quest.popleft()
                self._in_quest = True
                self._commit_directive("quest start")
                return self._directive
            # otherwise: normal replan (with suspension + LLM waypoint on stuck).
            new_prio = _INTENT_PRIORITY.get(intent, 0)
            cur_prio = _INTENT_PRIORITY.get(self._directive.intent, 0) if self._directive else -1
            if self._directive is not None and new_prio > cur_prio:
                self._dstack.append(self._directive)
                self.on_event("directive_suspend", {"step": self.session.step,
                                                     "intent": self._directive.intent.value})
                why = f"preempted by higher-priority {intent.value}"
            context = self._replan_context(obs, why) if ("stuck" in why or "impossible" in why) else None
            self._directive = self.planner.plan(intent, emu, self.memory, why=why, context=context)
            self._commit_directive(self._directive.reason)
        return self._carry_plan()

    def _commit_directive(self, note: str) -> None:
        self._replan_next = False
        self._servo_fail = 0
        self._steps_since_replan = 0
        self.on_event("directive", {"step": self.session.step, "intent": self._directive.intent.value,
                                    "target": self._directive.target, "success": self._directive.success,
                                    "reason": self._directive.reason})

    def _carry_plan(self) -> Directive | None:
        self.memory.plan = self._plan  # keep the checkpointed plan carrying the live directive
        if self._plan is not None:
            self._plan.directive = self._directive
            self._plan.stack = list(self._dstack)
        return self._directive

    def _replan_context(self, obs, why: str) -> dict | None:
        """The map view + reachable set handed to the tier-1 planner so LunaRoute can pick a
        concrete waypoint tile to route around a deadlock (LLM routing with teeth)."""
        if obs.player is None:
            return None
        gd = next_direction(self.memory.graph, obs.player.map_id, self.goal_map) \
            if self.goal_map is not None else None
        npcs = {(int(n["x"]), int(n["y"])) for n in ((obs.game_state or {}).get("npcs") or [])
                if "x" in n and "y" in n}
        return {
            "map_view": self.world.render_labeled(obs.player, obs.exits, npcs),
            "player": {"x": obs.player.x, "y": obs.player.y, "map_id": obs.player.map_id},
            "goal_dir": (f"{gd} toward map {self.goal_map}" if gd else None),
            "reachable": self._reachable_cells(obs.player, npcs),
            "why": why,
        }

    def _try_quest(self, obs, why: str) -> bool:
        """Jev escalation router: is this block a story gate needing a sub-quest? If so, ask the
        tier-2 strategist for an ordered quest and load it. Returns True if a quest was set."""
        strategize = getattr(self.planner, "strategize", None)
        if strategize is None or obs.player is None or not self._should_escalate(obs):
            return False
        quest = strategize(self.controller.emu, self.memory, why=why)
        if not quest:
            return False
        self._quest = deque(quest)
        self.on_event("quest", {"step": self.session.step, "len": len(quest),
                                "plan": [d.reason for d in quest]})
        return True

    def _should_escalate(self, obs) -> bool:
        """Calibrated (Jev) decision: does this stuck need tier-2 problem-solving vs a reroute?
        Falls back to escalate-on-impossibility when no calibrated router is available."""
        judge = getattr(self.reasoner, "judge", None)
        if judge is None:
            return True  # no Jev router -> the impossibility trigger already gates this
        try:
            state = {
                "situation": "the navigator is blocked and cannot reach its target after repeated tries",
                "player": {"x": obs.player.x, "y": obs.player.y, "map_id": obs.player.map_id},
                "map_view": obs.map_view,
                "npcs": [(n.get("x"), n.get("y"), n.get("sprite"))
                         for n in ((obs.game_state or {}).get("npcs") or [])],
                "recent": list(self._recent)[-8:],
            }
            score = judge(
                "Is the agent blocked by a STORY GATE or puzzle that needs a change of plan — "
                "talking to an NPC, fetching/delivering an item, or entering a building — rather "
                "than just walking a different path around an obstacle?", state)
            self.on_event("escalate", {"step": self.session.step, "score": round(float(score), 2)})
            return float(score) >= 0.5
        except Exception:
            return True

    def _replan_reason(self, intent: Intent) -> str:
        """A short human explanation of WHY we are replanning (Inner Monologue for the LLM)."""
        if self._directive is None:
            return "initial plan"
        if self._servo_fail >= SERVO_FAIL_LIMIT:
            return (f"the {self._directive.intent.value} directive was impossible — no route "
                    f"toward {self._directive.target} after repeated tries; reroute")
        if self._replan_next:
            return f"got stuck / low confidence pursuing the {self._directive.intent.value} directive; reroute"
        if intent != self._directive.intent:
            return f"the active need changed to {intent.value}"
        return "replan"

    def _should_replan(self, intent: Intent) -> bool:
        """The explicit, few replan triggers (bold commitment — everything else holds)."""
        if self._directive is None:
            return True
        if intent != self._directive.intent:      # trigger #6: arbiter intent changed/downgraded
            return True
        # triggers #3/#4 (stuck / sustained low conf) honor a cooldown so we don't re-emit the
        # SAME directive every step and thrash — bold commitment. Only replan if the condition
        # PERSISTED past the cooldown window.
        if self._replan_next and self._steps_since_replan >= REPLAN_COOLDOWN:
            return True
        if self._servo_fail >= SERVO_FAIL_LIMIT:   # trigger #2: directive impossible (no route)
            self.on_event("directive_impossible", {"step": self.session.step,
                                                    "intent": self._directive.intent.value})
            return True
        return False

    def _servo_step(self, directive: Directive, obs, blocked_dirs: set[str]):
        """The deterministic servo: one concrete step toward ``directive.target`` (LLM+P),
        now WARP-AWARE so it uses the world model instead of walking blindly.

        Priority:
          1. an explicit object/NPC tile (talk_to/grab_item) -> BFS + interact
          2. cross-map: find the next hop toward the target map. If that hop is reached by a
             WARP DOOR on this map (a building/gate) -> BFS to that door tile and step on it.
             Otherwise it's a map-EDGE connection -> walk the compass direction, but NEVER
             step onto a warp door that leads OFF-route (route around it instead).
        Returns a MoveAction / InteractAction, or None (residual -> executor)."""
        player = obs.player
        if player is None:
            return None
        exits = obs.exits or []  # already 0xFF-resolved in step_once
        occupied = {(int(n["x"]), int(n["y"])) for n in ((obs.game_state or {}).get("npcs") or [])
                    if "x" in n and "y" in n}

        # 1. an explicit tile — an object/NPC to talk to, or the exact exit/door tile the
        #    planner attached (including a building's return door) -> BFS straight to it,
        #    routing around NPCs blocking the way.
        txy = directive.target_xy
        if txy is not None:
            interact = directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM)
            avoid = set(occupied)
            # an on-map WAYPOINT is always reachable without leaving the map, so never route
            # THROUGH a building door to get to it (doors are walkable -> the agent wandered
            # into the Mart). Block every warp tile for waypoint routing.
            if (directive.target or {}).get("kind") == "waypoint":
                avoid |= {(int(e["x"]), int(e["y"])) for e in exits}
            return self._bfs_move(player, txy, interact=interact, blocked_dirs=blocked_dirs,
                                  occupied=avoid)

        # 2. no tile: a cross-map target reached by a door or a map-edge connection.
        tmap = directive.target_map
        if tmap is None or player.map_id == tmap:
            return None
        hop = self.memory.graph.next_hop(player.map_id, tmap)
        if hop is None:
            return None  # no known route -> residual; the impossible trigger will replan
        next_map, _tile = hop

        # is the next hop reached by a WARP DOOR on this map? (building/gate/return door)
        door = next((e for e in exits if e.get("dest_map") == next_map), None)
        if door is not None:
            return self._bfs_move(player, (int(door["x"]), int(door["y"])), interact=False,
                                  blocked_dirs=blocked_dirs, occupied=occupied)

        # else a map-EDGE connection (walk off the edge into next_map): BFS across the
        # accumulated collision map to the boundary edge in the connection direction, routing
        # AROUND off-route doors. This is real pathfinding, not a greedy compass walk — it's
        # what stops the drift into houses/labs when the direct bearing is blocked.
        d = next_direction(self.memory.graph, player.map_id, tmap)
        if d is None:
            return None
        edge = self._edge_goals(obs.map_dims, d)
        if edge:
            nav = Navigator(self.world)
            # route around off-route doors AND NPCs (pressing into an NPC talks to it instead
            # of moving — the infinite "advance_dialog" bump loop at Viridian's Gambler).
            off_route = {(int(e["x"]), int(e["y"])) for e in exits if e.get("dest_map") != next_map}
            off_route |= occupied
            step = nav._bfs_first_step(player.map_id, (player.x, player.y), edge, off_route)
            if step is not None and step.value not in blocked_dirs:
                return MoveAction(direction=step)
            # BFS returned None because we're ALREADY on the boundary edge -> take the final
            # step OFF the edge in the connection direction to actually cross the seam (the
            # servo used to stop here and hand a confused executor the crossing).
            if step is None and (player.x, player.y) in edge and d not in blocked_dirs:
                return MoveAction(direction=Direction(d))
        # fallback (edge unknown / no BFS path yet): greedy compass with door + NPC avoidance
        if self._can_step(player, d, next_map, exits, blocked_dirs, occupied):
            return MoveAction(direction=Direction(d))
        for alt in _lateral(d):
            if self._can_step(player, alt, next_map, exits, blocked_dirs, occupied):
                return MoveAction(direction=Direction(alt))
        return None

    @staticmethod
    def _edge_goals(map_dims, d: str) -> set[tuple[int, int]]:
        """The set of boundary tiles on the edge in direction ``d`` — walking off any of them
        crosses the map-edge connection. Empty if map dimensions are unknown."""
        if not map_dims:
            return set()
        w, h = map_dims
        if d == "north":
            return {(x, 0) for x in range(w)}
        if d == "south":
            return {(x, h - 1) for x in range(w)}
        if d == "west":
            return {(0, y) for y in range(h)}
        if d == "east":
            return {(w - 1, y) for y in range(h)}
        return set()

    def _can_step(self, player, d: str, next_map: int, exits, blocked_dirs: set[str],
                  occupied: set | None = None) -> bool:
        """True if stepping `d` is safe: not a known wall/ledge, not an NPC (bumping talks
        to it), and not a warp door that leads somewhere other than the intended next hop."""
        if d in blocked_dirs or self._confirmed_wall(Direction(d)):
            return False
        ax, ay = _tile_ahead(player, d)
        if occupied and (ax, ay) in occupied:
            return False  # an NPC stands there — stepping in talks to it, doesn't move
        for e in exits:
            if (int(e["x"]), int(e["y"])) == (ax, ay) and e.get("dest_map") != next_map:
                return False  # a door to an off-route map (e.g. Blue's House) — don't enter
        return True

    def _bfs_move(self, player, xy, *, interact: bool, blocked_dirs: set[str], occupied=None):
        """One BFS step toward tile ``xy`` over the learned map (routing around ``occupied``
        NPC tiles); InteractAction on arrival (interact) or None on arrival (a warp/exit tile
        you just step onto)."""
        if xy is None:
            return None
        nav = Navigator(self.world)
        prim, arrived = nav.step_toward(player, {"x": int(xy[0]), "y": int(xy[1]), "interact": interact},
                                        occupied)
        if arrived:
            return InteractAction() if interact else None
        if isinstance(prim, MoveAction) and prim.direction.value not in blocked_dirs:
            return prim
        return None

    def _resolve_exits(self, exits, cur_map: int) -> list[dict]:
        """Resolve each warp's dest: 0xFF ('return to last map') -> the map we came from, so
        building doors become real, routable graph edges (the world model knows where they go)."""
        out: list[dict] = []
        for e in (exits or []):
            dest = e.get("dest_map")
            if dest == WARP_BACK and self._prev_map is not None and self._prev_map != cur_map:
                dest = self._prev_map
                out.append({**e, "dest_map": dest, "dest_name": map_name(dest)})
            else:
                out.append(e)
        return out

    def _act_servo(self, obs, directive, servo, shot) -> ActionResult:
        result = self.controller.execute(servo)
        if isinstance(servo, MoveAction) and result.result == "blocked" and obs.player:
            if self._confirmed_wall(servo.direction):
                self.world.mark_blocked(obs.player, servo.direction)
            self.memory.mark_blocked_edge(obs.player.map_id, obs.player.x, obs.player.y,
                                          servo.direction.value)
        rstep = ReasonStep(location="servo", objective=directive.reason,
                           tried="", reasoning=f"servo->{directive.intent.value}: {_desc(servo)}",
                           action=servo)
        self._prev = rstep
        self._emit_reason(rstep, 0)
        return self._finish(obs, rstep, result, 0, {}, shot)

    # --------------------------------------------------- executor dispatch
    def _dispatch(self, rstep, obs, blocked_dirs, ctx_kind) -> ActionResult:
        if isinstance(rstep.action, MenuSelectAction):
            from ..games.pokemon_red.menus import select_option
            before = detect_mode(self.controller.emu)
            r = select_option(self.controller.emu, rstep.action.index)
            return ActionResult(
                success=bool(r.get("ok")), result="completed" if r.get("ok") else "invalid",
                mode_before=before, mode_after=detect_mode(self.controller.emu),
                events=[f"menu_select:{rstep.action.index}"],
                detail=f"menu select {rstep.action.index} ({rstep.action.label})",
            )
        if isinstance(rstep.action, GoToAction):
            occupied = {
                (int(n["x"]), int(n["y"]))
                for n in (obs.game_state or {}).get("npcs", [])
                if "x" in n and "y" in n
            }
            return self._navigate(rstep.action, occupied)
        result = self.controller.execute(rstep.action)
        if result.result == "blocked" and isinstance(rstep.action, MoveAction) and obs.player:
            if self._confirmed_wall(rstep.action.direction):
                self.world.mark_blocked(obs.player, rstep.action.direction)
            forced = bool(((obs.game_state or {}).get("context") or {}).get("forced_movement"))
            if not forced and ctx_kind == "overworld":
                self.memory.mark_blocked_edge(obs.player.map_id, obs.player.x, obs.player.y,
                                              rstep.action.direction.value)
        return result

    def _advance_dialog(self, obs, shot) -> ActionResult:
        from ..core.models import AdvanceDialogAction
        action = AdvanceDialogAction()
        result = self.controller.execute(action)
        rstep = ReasonStep(location="dialog", objective="advance text",
                           tried="", reasoning="advancing passive dialog", action=action)
        self._prev = rstep
        self._emit_reason(rstep, 0)
        return self._finish(obs, rstep, result, 0, {}, shot)

    def _reachable_cells(self, player, npcs: set[tuple[int, int]]) -> set[tuple[int, int]]:
        """BFS-reachable cells from the player over the ingested collision, avoiding walls,
        NPCs, and out-of-bounds — the deterministic ground truth the LLM's waypoint is validated
        against (so it can't commit to a wall or an unreachable tile)."""
        bounds = self.world.bounds.get(player.map_id)
        start = (player.x, player.y)
        if bounds is None:
            return {start}  # no ingested collision -> can't compute reachability (BFS would be unbounded)
        w, h = bounds
        m = self.world.tiles.get(player.map_id, {})
        seen = {start}
        q = deque([start])
        while q:
            x, y = q.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                n = (x + dx, y + dy)
                if n in seen or n in npcs:
                    continue
                if not (0 <= n[0] < w and 0 <= n[1] < h):
                    continue
                if m.get(n) == WALL:
                    continue
                seen.add(n)
                q.append(n)
        return seen

    def _next_hop_map(self, obs, directive) -> int | None:
        """The map id of the next hop toward the active directive's target (for door filtering).
        None when there's no target-bearing directive / no known route."""
        if obs.player is None or directive is None or directive.target_map is None:
            return None
        hop = self.memory.graph.next_hop(obs.player.map_id, directive.target_map)
        return hop[0] if hop else None

    def _blocked_dirs(self, obs, allowed_next: int | None = None) -> set[str]:
        """Directions the executor/servo must not take: known walls (dead-end ledger), one-way
        ledges (never HOP a ledge for routing), and OFF-ROUTE building doors — a warp tile
        adjacent to the player whose destination is not the intended next hop (so we never
        wander into a house/lab that isn't on the way). This is the world model giving the
        navigator a notion of what each door is."""
        blocked: set[str] = set()
        if obs.player is not None:
            m = self.world.tiles.get(obs.player.map_id, {})
            for d, (dx, dy) in DELTA.items():
                if (m.get((obs.player.x + dx, obs.player.y + dy)) == WALL
                        or self.memory.is_blocked_edge(obs.player.map_id, obs.player.x, obs.player.y, d.value)):
                    blocked.add(d.value)
            if allowed_next is not None:
                for e in (obs.exits or []):
                    for d, (dx, dy) in DELTA.items():
                        if ((int(e["x"]), int(e["y"])) == (obs.player.x + dx, obs.player.y + dy)
                                and e.get("dest_map") != allowed_next):
                            blocked.add(d.value)
        blocked |= set(self.controller.emu.ledge_dirs())
        return blocked

    def _emit_reason(self, rstep, latency) -> None:
        self.on_event("reason", {
            "step": self.session.step, "latency_ms": latency, "location": rstep.location,
            "objective": rstep.objective, "tried": rstep.tried,
            "action": rstep.action.model_dump(), "reasoning": rstep.reasoning,
        })

    # --- shared step tail: bookkeeping, stuck/setback, logging, checkpoint ----
    def _finish(self, obs, rstep, result, latency, usage, shot) -> ActionResult:
        from ..games.pokemon_red.game_state import read_screen_text
        _, caused_dialog = read_screen_text(self.builder.emu)
        self.interactions.record_action(self.session.step, obs.player, rstep.action, caused_dialog)
        # BUMP GUARD: an overworld move that didn't change position hit something the static
        # collision doesn't know — a SIGN or a stationary NPC. Mark that edge blocked so the
        # servo/BFS routes AROUND it instead of re-bumping forever (the Viridian sign loop).
        # Timing-independent (doesn't depend on the dialog box having finished opening).
        ctx_now = ((obs.game_state or {}).get("context") or {})
        if (isinstance(rstep.action, MoveAction) and obs.player is not None
                and not ctx_now.get("forced_movement") and not ctx_now.get("in_battle")):
            now = read_player(self.controller.emu)
            if now is not None and (now.x, now.y, now.map_id) == (obs.player.x, obs.player.y, obs.player.map_id):
                self.memory.mark_blocked_edge(obs.player.map_id, obs.player.x, obs.player.y,
                                              rstep.action.direction.value)
                # Mark the bumped-into TILE as a wall so the servo's BFS (which reads the
                # collision map, not the edge ledger) routes AROUND it — without this the sign
                # at (19,8) stays "walkable" and BFS keeps steering north into it forever.
                dx, dy = DELTA[rstep.action.direction]
                self.world.tiles[obs.player.map_id][(obs.player.x + dx, obs.player.y + dy)] = WALL
        pos = f"({obs.player.x},{obs.player.y})m{obs.player.map_id}" if obs.player else "(?)"
        self._recent.append(f"{pos} {_desc(rstep.action)} -> {result.result}")

        ctx = (obs.game_state or {}).get("context") or {}
        try:
            pv = progress_vector(self.builder.emu)
            if pv is not None and obs.player is not None:  # count intra-map exploration as progress
                pv = {**pv, "tiles_known": len(self.world.tiles.get(obs.player.map_id, {}))}
        except Exception:
            pv = None
        suppress = bool(ctx.get("forced_movement") or ctx.get("in_battle"))
        stuck = self.stuck.update(
            rstep.action, result, obs.player, shot if self.vision else None,
            progress=pv, forced_movement=suppress,
            objective_distance=self._objective_distance(obs.player),
        )
        if stuck.setback:
            self.memory.note(f"setback at step {self.session.step}", source="observed", step=self.session.step)
        if stuck.stuck:
            # trigger #3: hand stuck UP to the planner (replan) instead of a rival frontier heuristic.
            self._replan_next = True
        if stuck.stuck or stuck.setback:
            self.on_event("stuck", {"step": self.session.step, "kind": stuck.kind,
                                    "setback": stuck.setback, "repeat_count": stuck.repeat_count})

        if self.logger:
            from ..providers.interface import AgentResponse
            resp = AgentResponse(decision=rstep.to_decision(), raw_content=rstep.model_dump_json(),
                                 model="reasoner", latency_ms=latency, usage=usage)
            self.logger.record(step=self.session.step, observation=obs, response=resp,
                               result=result, plan=None, screenshot=shot)
        self.session.step += 1
        if self.checkpoint_every and self.checkpoint_dir and self.session.step % self.checkpoint_every == 0:
            self._checkpoint()
        return result

    # --- battle sub-policy (mode dispatch routes here when in_battle) ---------
    def _battle_turn(self, obs):
        """Advance battle intro/result text; when the FIGHT menu is up, choose a move
        (Jev if the executor is TypeSafe, else the first slot) and execute the turn."""
        from ..core.models import AdvanceDialogAction, MenuSelectAction
        from ..games.pokemon_red import battle, battle_agent
        emu = self.controller.emu
        mode_before = detect_mode(emu)
        if not battle.fight_menu_showing(emu):
            res = self.controller.execute(AdvanceDialogAction())
            rstep = ReasonStep(location="battle", objective="advance battle text",
                               reasoning="advancing battle text", action=AdvanceDialogAction())
            return rstep, 0, {}, res
        client = getattr(self.reasoner, "client", None)
        conf = 0.0
        if client is not None:
            slot, conf = battle_agent.choose_move(client, emu)
        else:
            slot = 0
        r = battle.use_move(emu, slot)
        result = ActionResult(
            success=bool(r.get("ok")), result="completed",
            mode_before=mode_before, mode_after=detect_mode(emu),
            events=[f"battle_move:{r.get('move')}", f"dmg:{r.get('damage_dealt')}"],
            detail=f"battle: {r.get('move')} dealt {r.get('damage_dealt')} (over={r.get('battle_over')})",
        )
        rstep = ReasonStep(location="battle", objective=f"use {r.get('move')}",
                           reasoning=f"battle move {slot} ({r.get('move')}) conf {conf:.2f}",
                           action=MenuSelectAction(index=slot, label=f"move:{r.get('move')}"))
        return rstep, 0, {"confidence": conf}, result

    def _objective_distance(self, player) -> int | None:
        """Graph-distance (map hops) from the player's map to the active directive's target
        map, if any — feeds the stuck detector's objective-progress signal."""
        d = self._directive
        if not (player and d and d.target_map is not None):
            return None
        route = self.memory.graph.route(player.map_id, d.target_map)
        return (len(route) - 1) if route else None

    def _checkpoint(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.memory.plan = self._plan
        self.controller.emu.save_state(self.checkpoint_dir / "latest.state")
        self.memory.save(self.checkpoint_dir / "latest.mem.json")
        self.on_event("checkpoint", {"step": self.session.step, "dir": str(self.checkpoint_dir)})

    def _confirmed_wall(self, direction) -> bool:
        """True only if the LIVE collision map says the tile ahead is blocked.

        A bare `blocked` move result is not enough: during a warp/transition the game
        ignores input, so a walkable tile reports blocked. Marking a wall from that
        corrupts the map. The collision matrix is RAM ground truth, so we consult it and
        mark a wall only when it agrees the tile is solid."""
        lm = self.controller.emu.local_map_ascii()
        if not lm:
            return False
        anchor = next(((r, row.find("@")) for r, row in enumerate(lm) if "@" in row), None)
        if anchor is None:
            return False
        dr, dc = _SCREEN_DELTA[direction]
        nr, nc = anchor[0] + dr, anchor[1] + dc
        if 0 <= nr < len(lm) and 0 <= nc < len(lm[nr]):
            return lm[nr][nc] == "#"
        return False

    def _navigate(self, action: GoToAction, occupied: set | None = None) -> ActionResult:
        """Drive to a GoToAction's target, a step at a time (used when the executor emits a
        goto). Follows the BFS path and presses A on arrival for an object/NPC; explores
        toward the frontier when no route is known yet. Bounded; learns walls."""
        nav = Navigator(self.world)
        target = {"x": action.x, "y": action.y, "interact": action.interact}
        emu = self.controller.emu
        events: list[str] = []
        last: ActionResult | None = None
        blocks = 0
        start = read_player(emu)
        for _ in range(MAX_GOTO_STEPS):
            player = read_player(emu)
            self.world.observe(player, emu.local_map_ascii())
            prim, arrived = nav.step_toward(player, target, occupied)
            if arrived:
                if action.interact:
                    last = self.controller.execute(InteractAction())
                    from ..games.pokemon_red.game_state import read_screen_text
                    _, caused = read_screen_text(emu)
                    self.interactions.record_action(self.session.step, player, InteractAction(), caused)
                    events.append(f"goto_reached:{action.label}" + ("" if caused else ":nothing"))
                else:
                    events.append("goto_arrived")
                break
            if prim is None:
                d = self.world.explore_step(player)
                if d is None:
                    events.append("goto_no_path")
                    break
                prim = MoveAction(direction=Direction(d))
                events.append("goto_explore")
            res = self.controller.execute(prim)
            last = res
            if isinstance(prim, MoveAction) and res.result == "blocked":
                if self._confirmed_wall(prim.direction):
                    self.world.mark_blocked(player, prim.direction)
                blocks += 1
                if blocks >= 5:
                    events.append("goto_stuck")
                    break
            else:
                blocks = 0
        mode_after = detect_mode(emu)
        end = read_player(emu)
        moved = bool(start and end and (start.x, start.y, start.map_id) != (end.x, end.y, end.map_id))
        detail = f"goto ({action.label}): {' '.join(events) or 'no progress'}"
        if last is not None:
            last.events = (last.events or []) + events
            last.player_moved = moved
            last.detail = detail
            return last
        return ActionResult(
            success=False, result="blocked", mode_before=mode_after, mode_after=mode_after,
            player_moved=moved, events=events or ["goto_no_path"], detail=detail,
        )

    def run(self, max_steps: int = 40) -> None:
        for _ in range(max_steps):
            if not self.session.running:
                break
            self.step_once()


def _tile_ahead(player, d: str) -> tuple[int, int]:
    dx, dy = DELTA[Direction(d)]
    return player.x + dx, player.y + dy


def _lateral(d: str) -> list[str]:
    """The two directions perpendicular to `d` (to route around an obstacle/door)."""
    if d in ("north", "south"):
        return ["east", "west"]
    return ["north", "south"]


def _desc(action) -> str:
    d = action.model_dump()
    if d.get("type") == "move":
        return f"move {d['direction']} x{d.get('tiles', 1)}"
    if d.get("type") == "goto":
        return f"goto {d.get('label') or (d.get('x'), d.get('y'))}"
    if d.get("type") == "menu_select":
        return f"menu[{d.get('index')}] {d.get('label') or ''}"
    return d.get("type", "?") + (f" {d.get('button')}" if d.get("button") else "")
