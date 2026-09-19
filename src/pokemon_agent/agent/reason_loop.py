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
QUEST_STEP_BUDGET = 60  # steps to keep working one quest step before re-strategizing (anti-churn)
JEV_OVERRIDE_CONF = 0.6  # Jev may override the BFS pathing suggestion only at/above this confidence
POLICY_BUDGET = 15       # steps a Jev-chosen routing policy runs before it's re-selected for the area
WARP_BACK = 0xFF     # a warp's dest_map of 0xFF means "return to the map you came from" (wLastMap)
TARGET_REPROPOSE_LIMIT = 2  # times the proposer may re-pick a DIFFERENT target to unstick before L1

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
        recorder=None,
        pather: str = "bfs",
        on_event: Optional[Callable[[str, dict], None]] = None,
    ):
        self.builder = builder
        self.controller = controller
        self.reasoner = reasoner
        self.session = session
        self.logger = logger
        self.recorder = recorder      # full-fidelity per-step run recorder (optional)
        self.vision = vision
        _on_event = on_event or (lambda kind, payload: None)
        if recorder is not None:  # tee events into the recorder so each step's record carries them
            def _tee(kind, payload):
                recorder.on_event(kind, payload)
                _on_event(kind, payload)
            self.on_event = _tee
        else:
            self.on_event = _on_event
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
        self.level_target = level_target
        self.knowledge = knowledge           # Orrery KB (also used for battle type lookups)
        self.pather = pather                 # "bfs" (deterministic) or "jev" (calibrated per-step direction)
        self._battle_kb: dict[str, list[str]] = {}  # cache: enemy species -> type knowledge
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
        self._heal_quest = False                     # the active quest IS the heal errand (don't self-preempt)
        self._quest_step_age = 0                     # steps spent on the current quest step (wedge budget)
        self._replan_next = False       # a trigger (stuck / low-conf) asked for a replan
        self._servo_fail = 0            # consecutive servo no-route steps (impossibility)
        # L2 (tactical navigator) state: the grid waypoint LunaRoute picked to head toward on the
        # current map. BFS routes to it; it's re-picked on arrival or when BFS can't get closer.
        self._leg_wp: tuple[int, int] | None = None
        self._leg_wp_map: int | None = None
        self._leg_wp_fail = 0           # consecutive legs where L2/BFS couldn't make progress
        self._recent_wps: deque[tuple[int, int]] = deque(maxlen=6)  # L2's recent picks (anti-oscillation)
        # unified mid-level target (the proposer's typed choice, held across frames — cadence B)
        self._target: dict | None = None
        self._target_map: int | None = None
        self._target_stuck = 0          # consecutive legs the current target made no progress
        self._recent_targets: deque = deque(maxlen=6)
        # Jev-picked routing POLICY (shortest / dodge-grass / farm-exp), re-chosen per leg/area
        self._policy: str | None = None
        self._policy_map: int | None = None
        self._policy_age = 0
        self._farm_last: Direction | None = None   # farm-exp pacing: last graze step (for back-and-forth)
        self._farm_age = 0
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
        # FULL-MAP collision from RAM (wOverworldMap): the RAM collision map is GROUND TRUTH for
        # walkability — it already encodes walls, signs, trees, and ledges as non-walkable (verified
        # against the game). We trust it and never overwrite it with bump-discovered guesses (those
        # created permanent FALSE walls that wedged navigation). The only obstacles NOT in it are
        # moving sprites (NPCs), which we read separately from sprite RAM each step.
        # Guarded against STALE reads: after a map transition the buffer isn't settled for a few
        # frames, so a read can decode the PREVIOUS map. We accept a decode only when the player
        # stands on a walkable cell (always true for a correct read).
        if obs.player is not None:
            mid = obs.player.map_id
            if mid not in self._collision_seen:
                from ..games.pokemon_red.map_reader import read_collision_map
                cm = read_collision_map(self.controller.emu)
                if (cm is not None and cm["map_id"] == mid
                        and (obs.player.x, obs.player.y) in cm["walkable"]):  # reject stale reads
                    self.world.ingest_collision(cm["map_id"], cm["width"], cm["height"],
                                                cm["walkable"], cm.get("counters"), cm.get("terrain"))
                    self._collision_seen.add(mid)
        self.world.observe(obs.player, obs.walkability)
        # the SEMANTIC map (grass/water/ledges/doors + legend) is what the agent reasons on; fall
        # back to the plain floor/wall render before any collision has been ingested.
        npc_cells = {(int(n["x"]), int(n["y"])) for n in ((obs.game_state or {}).get("npcs") or [])
                     if "x" in n and "y" in n}
        obs.map_view = (self.world.render_semantic(obs.player, obs.exits, npc_cells)
                        or self.world.render(obs.player, obs.exits, obs.map_dims))
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

        # --- menu is a mode: Jev operates it (a bounded 1-of-N calibrated choice) ---
        if ctx_kind == "menu":
            return self._jev_turn(obs, player_desc, self._blocked_dirs(obs, None), None, shot)

        # --- overworld: LunaRoute sets the target, deterministic BFS routes to it. Jev is NOT a
        # navigator — when there's no clean route the DECISION goes up to LunaRoute (re-plan /
        # strategize), never to a heuristic fallback or Jev. (No planner -> legacy Jev path.) ---
        if self.planner is None:
            self._maybe_reflect(obs, player_desc)
            return self._jev_turn(obs, player_desc, self._blocked_dirs(obs, None), None, shot)

        directive = self._manage_directive(obs)
        blocked_dirs = self._blocked_dirs(obs, self._next_hop_map(obs, directive))
        # NOTE: no separate _maybe_reflect on the planner path — reflection is now folded INTO the
        # mid-level proposer inside _navigate_leg (its note becomes self._plan.next_objective). The
        # legacy no-planner path above still calls _maybe_reflect.
        if directive is not None and directive.target_bearing:
            servo = self._dispatch_servo(directive, obs, blocked_dirs)
            if servo is not None:
                self._servo_fail = 0
                return self._act_servo(obs, directive, servo, shot)
            self._servo_fail += 1     # no clean route -> escalate to LunaRoute (see _manage_directive)
            self._replan_next = True
        # awaiting a fresh target/plan from LunaRoute -> a brief wait; the next step re-plans.
        from ..core.models import WaitAction
        rstep = ReasonStep(location="await-plan",
                           objective=(directive.reason if directive else "awaiting plan"),
                           reasoning="no clean route; deferring the decision to LunaRoute",
                           action=WaitAction(frames=6))
        self._prev = rstep
        self._emit_reason(rstep, 0)
        return self._finish(obs, rstep, self.controller.execute(rstep.action), 0, {}, shot)

    def _dispatch_servo(self, directive, obs, blocked_dirs):
        """Route a target-bearing directive to the right servo: a talk/grab with a concrete TILE ->
        the deterministic BFS+interact servo; anything else (incl. a talk/grab with NO tile yet) ->
        the unified nav leg, whose approach_npc finds and reaches the person."""
        if directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM) and directive.target_xy is not None:
            return self._servo_step(directive, obs, blocked_dirs)
        return self._navigate_leg(directive, obs, blocked_dirs)

    def _maybe_reflect(self, obs, player_desc) -> None:
        """Periodic/forced strategic reflection (maintains the AgentPlan) — LunaRoute."""
        if not (self.session.step % self.reflect_every == 0 or self._force_reflect):
            return
        self._force_reflect = False
        self._last_reflect_step = self.session.step
        self._plan, rlat, _ = self.reasoner.reflect(
            primary_goal=self.session.goal.primary, player_desc=str(player_desc),
            map_view=obs.map_view, map_history=self._map_history[-12:],
            social_memory=self.interactions.summary(), game_state=obs.game_state,
            recent=list(self._recent), previous=self._plan)
        self.on_event("reflect", {"step": self.session.step, "latency_ms": rlat,
                                  "plan": self._plan.model_dump()})

    def _jev_turn(self, obs, player_desc, blocked_dirs, directive, shot) -> ActionResult:
        """Jev drives one step: menu operation, or (no-planner legacy) overworld choice."""
        ctx_kind = ((obs.game_state or {}).get("context") or {}).get("kind")
        targets = build_targets(obs.game_state, obs.exits)
        rstep, latency, usage = self.reasoner.step(
            primary_goal=self.session.goal.primary,
            image=shot if self.vision else None, local_map=obs.walkability,
            map_view=obs.map_view, player_desc=str(player_desc), exits=obs.exits,
            game_state=obs.game_state, social_memory=self.interactions.summary(),
            map_history=self._map_history[-12:], recent=list(self._recent),
            previous=self._prev, plan=self._plan, targets=targets,
            route_hint=None, blocked_dirs=blocked_dirs, directive=directive)
        self._prev = rstep
        self._emit_reason(rstep, latency)
        # sustained low confidence from the calibrated decider -> force a strategic reflection on
        # the NEXT step (throttled by the cooldown), instead of thrashing the same choice.
        conf = (usage or {}).get("confidence")
        if (self.low_conf_reflect is not None and conf is not None
                and conf < self.low_conf_reflect
                and self.session.step - self._last_reflect_step >= self.reflect_cooldown):
            self._force_reflect = True
            self._replan_next = True
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

        # --- termination: did the active directive succeed? (VERIFY its acceptance criterion) ---
        if self._directive is not None and self._directive_satisfied(self._directive):
            self.on_event("directive_done", {"step": self.session.step,
                                             "intent": self._directive.intent.value,
                                             "reason": self._directive.reason})
            self._servo_fail = 0
            if self._in_quest:
                # advance the quest to its next step (FIFO); when empty, the quest is complete.
                self._directive = self._quest.popleft() if self._quest else None
                if self._directive is None:
                    self._in_quest = False
                    self._heal_quest = False
                    self.on_event("quest_done", {"step": self.session.step})
                else:
                    self._commit_directive("next quest step")
                    return self._directive
            else:
                self._directive = self._dstack.pop() if self._dstack else None

        # --- holding a quest step: it overrides the arbiter's intent. DON'T abandon it on a
        # transient stall (that caused re-quest churn) — keep working it (servo/waypoint/executor)
        # until its acceptance criterion is met, a SURVIVE emergency preempts, or it's been wedged
        # far too long (then re-strategize with the progress so far as feedback). ---
        if self._in_quest and self._directive is not None:
            self._quest_step_age += 1
            # wedged = worked this step too long, OR L2+BFS made no progress for SERVO_FAIL_LIMIT
            # consecutive steps (a genuine navigation dead-end the tactical navigator can't solve).
            if self._quest_step_age > QUEST_STEP_BUDGET or self._servo_fail >= SERVO_FAIL_LIMIT:
                self.on_event("quest_step_wedged", {"step": self.session.step,
                                                    "reason": self._directive.reason})
                self._in_quest = False
                self._heal_quest = False
                self._quest.clear()
                # RE-STRATEGIZE from the current state instead of abandoning the errand: the
                # strategist sees what we now hold (e.g. the parcel) + that we're blocked, and
                # re-derives the next objective (deliver it). Bypass the escalation gate — we're
                # already mid-errand, so recovery shouldn't depend on a borderline score.
                if self._try_quest(obs, "the current quest step wedged; re-plan from the current "
                                        "state (keep pursuing the objective)", force=True):
                    self._directive = self._quest.popleft()
                    self._in_quest = True
                    self._commit_directive("re-strategized quest")
                    return self._directive
            elif intent == Intent.HEAL and self._directive.intent != Intent.HEAL and not self._heal_quest:
                self._in_quest = False  # emergency heal preempts a NON-heal quest; re-derive later
            else:
                return self._carry_plan()   # (a heal quest keeps running until HP is restored)

        # --- replan? ---
        if self._directive is None or self._should_replan(intent):
            why = self._replan_reason(intent)
            blocked = "impossible" in why or "stuck" in why
            # A HEAL need has no built-in target — the agent doesn't know WHERE a Poké Center is.
            # Rather than issue a target-less directive it can't act on (the "stressed, doesn't know
            # where to go" behavior), send it to the tier-2 strategist, which SEARCHES the KB for the
            # nearest Poké Center and returns a routed heal quest (go there -> heal at the nurse).
            if intent == Intent.HEAL and self._try_quest(
                    obs, "party HP is low and I need to heal, but I don't know where the nearest "
                         "Poké Center is — find it, travel there, and heal at the nurse", force=True):
                self._directive = self._quest.popleft()
                self._in_quest = True
                self._heal_quest = True
                self._commit_directive("heal errand — find & route to a Poké Center")
                return self._directive
            # ESCALATION LADDER. On a genuine block, JEV ROUTES the recovery (the user's "use Jev
            # to route between parts of the loop"): is this a STORY GATE needing a sub-quest, or
            # just a navigation deadlock? Within-map rerouting is now L2's job (the tactical
            # navigator re-picks a tile every leg), so a block that survives L2 is either a real
            # story gate (-> strategist quest) or a plan-level reroute (-> planner picks a new
            # target map). `_try_quest` consults the calibrated router for the former.
            if blocked and self._try_quest(obs, why):
                self._directive = self._quest.popleft()
                self._in_quest = True
                self._commit_directive("re-strategized (story gate)")
                return self._directive
            # normal replan: LunaRoute (L1) sets the next target MAP; L2 handles the tiles.
            new_prio = _INTENT_PRIORITY.get(intent, 0)
            cur_prio = _INTENT_PRIORITY.get(self._directive.intent, 0) if self._directive else -1
            if self._directive is not None and new_prio > cur_prio:
                self._dstack.append(self._directive)
                self.on_event("directive_suspend", {"step": self.session.step,
                                                     "intent": self._directive.intent.value})
                why = f"preempted by higher-priority {intent.value}"
            self._directive = self.planner.plan(intent, emu, self.memory, why=why)
            self._commit_directive(self._directive.reason)
        return self._carry_plan()

    def _directive_satisfied(self, directive) -> bool:
        """VERIFY a directive's acceptance criterion. RAM-checkable criteria (has_item, on_map,
        level, ...) are read deterministically; a model-authored ``verify:`` criterion (a fact
        not in RAM) is judged by the calibrated verifier over assembled game-state evidence —
        this is the "when it thinks it's done, check the criteria" step."""
        success = directive.success or {}
        if "verify" in success:
            judge = getattr(self.reasoner, "judge", None)
            # throttle the model check (it's an LLM/Jev call): only every few steps of the step
            if judge is None or (self._quest_step_age % 4 != 0):
                return False
            try:
                from ..games.pokemon_red.game_state import read_game_state
                gs = read_game_state(self.controller.emu)
                evidence = {"party": gs.get("party"), "items": gs.get("items"),
                            "badges": gs.get("badges"), "screen_text": gs.get("screen_text"),
                            "map_id": self.controller.emu.read_memory(0xD35E)}
                return float(judge(success["verify"], evidence)) >= 0.6
            except Exception:
                return False
        return predicates.evaluate(success, self.controller.emu, memory=self.memory)

    def _commit_directive(self, note: str) -> None:
        self._replan_next = False
        self._servo_fail = 0
        self._steps_since_replan = 0
        self._quest_step_age = 0
        self._leg_wp = None          # a new directive -> the L2 navigator picks a fresh waypoint
        self._leg_wp_fail = 0
        self._recent_wps.clear()
        self._policy = None          # and re-picks the routing policy for the new objective
        self._target = None
        self._target_stuck = 0
        self._recent_targets.clear()
        self.on_event("directive", {"step": self.session.step, "intent": self._directive.intent.value,
                                    "target": self._directive.target, "success": self._directive.success,
                                    "reason": self._directive.reason})

    def _carry_plan(self) -> Directive | None:
        self.memory.plan = self._plan  # keep the checkpointed plan carrying the live directive
        if self._plan is not None:
            self._plan.directive = self._directive
            self._plan.stack = list(self._dstack)
        return self._directive

    def _try_quest(self, obs, why: str, *, force: bool = False) -> bool:
        """Jev escalation router: is this block a story gate needing a sub-quest? If so, ask the
        tier-2 strategist for an ordered quest and load it. ``force`` bypasses the escalation
        check (used to re-plan mid-errand from the current state). Returns True if set."""
        strategize = getattr(self.planner, "strategize", None)
        if strategize is None or obs.player is None or (not force and not self._should_escalate(obs)):
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

    def _default_target(self, directive, obs) -> dict | None:
        """The deterministic typed target for this leg (what the router would do with no LLM). None
        means 'arrived on the target map' (the success predicate ends the directive)."""
        player = obs.player
        txy = directive.target_xy
        tmap = directive.target_map
        if directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM) and txy is None:
            return {"kind": "approach_npc", "sprite": (directive.target or {}).get("sprite")}
        if txy is not None and (tmap is None or tmap == player.map_id):
            # NOTE: in production a talk/grab WITH a concrete tile is dispatched to _servo_step (see
            # _dispatch_servo), so this interact=True case is only exercised by the resolver's unit test;
            # kept for completeness (a tile target that should press A on arrival).
            interact = directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM)
            return {"kind": "tile", "x": txy[0], "y": txy[1], "interact": interact}
        if tmap is None or tmap == player.map_id:
            return None
        hop = self.memory.graph.next_hop(player.map_id, tmap)
        if hop is None:
            return {"kind": "exit"}          # no known route out -> leave; the graph learns the edge
        next_map, _tile = hop
        door = next((e for e in (obs.exits or []) if e.get("dest_map") == next_map), None)
        if door is not None:
            return {"kind": "enter", "map": next_map}
        goal_dir = next_direction(self.memory.graph, player.map_id, tmap)
        return {"kind": "edge", "dir": goal_dir, "next_map": next_map}

    @staticmethod
    def _target_reached(target, obs) -> bool:
        """Only a 'tile' target can be 'reached' in place (others resolve to a move each frame and
        end when the map changes / an interaction fires)."""
        if target.get("kind") != "tile":
            return False
        return (obs.player.x, obs.player.y) == (int(target["x"]), int(target["y"]))

    def _current_target(self, directive, obs) -> dict | None:
        """Hold the current typed target across frames; (re)pick via the proposer on
        reached / stuck / map-change. Returns None only when we've arrived on the target map."""
        player = obs.player
        default = self._default_target(directive, obs)
        if default is None:
            self._target = None
            return None
        held = self._target
        if (held is not None and self._target_map == player.map_id
                and not self._target_reached(held, obs)):
            return held
        proposed = self._propose_target(obs, directive, default, stuck=False)
        if proposed.get("kind") == "unresolved":
            # a configured model failed AND there's no grounded route (proposer already flagged it).
            # Don't cache the failure — retry the model next frame (a transient 429/cold-start recovers);
            # this leg stalls visibly. Persistent failures keep flagging + stalling = a visible break.
            self._target = None
            self._target_map = None
            return proposed
        self._target = proposed
        self._target_map = player.map_id
        self._target_stuck = 0
        return self._target

    def _propose_target(self, obs, directive, default, *, stuck) -> dict:
        """Ask the mid-level proposer for the typed target (folding reflection: its note becomes the
        visible short-term objective). Deterministic default when offline. Never None."""
        prov = getattr(self.planner, "provider", None) if self.planner else None
        if prov is None:
            return default
        player = obs.player
        occupied = {(int(n["x"]), int(n["y"])) for n in ((obs.game_state or {}).get("npcs") or [])
                    if "x" in n and "y" in n}
        tmap = directive.target_map
        goal_dir = next_direction(self.memory.graph, player.map_id, tmap) if tmap is not None else None
        exit_tile = None
        if tmap is not None:
            hop = self.memory.graph.next_hop(player.map_id, tmap)
            if hop is not None:
                door = next((e for e in (obs.exits or []) if e.get("dest_map") == hop[0]), None)
                if door is not None:
                    exit_tile = (int(door["x"]), int(door["y"]))
        ctx = {
            "map_view": obs.map_view,
            "player": {"x": player.x, "y": player.y, "map_id": player.map_id},
            "objective": directive.reason,
            "destination": (f"{map_name(tmap)} (map {tmap})" if tmap is not None else "the goal"),
            "goal_dir": goal_dir,
            "exit_tile": exit_tile,
            "npcs": [{"x": int(n["x"]), "y": int(n["y"]), "sprite": n.get("sprite"),
                      "talked_to": n.get("talked_to")}
                     for n in ((obs.game_state or {}).get("npcs") or []) if "x" in n and "y" in n],
            "recent_trail": list(self._recent)[-8:],
            "recent_targets": [dict(t) for t in self._recent_targets],
            "reachable": self._reachable_cells(player, occupied),
            "default": default,
            "stuck": stuck,
            "why": ("the last target was unreachable or made no progress; propose a DIFFERENT one"
                    if stuck else "pick the next target toward the goal"),
        }
        target = self.planner.propose_target(self.controller.emu, ctx)
        if not target:
            target = default
        if target.get("kind") == "unresolved":
            # a CONFIGURED model failed. ALWAYS flag it loudly (logs + viewer) so it's debuggable.
            self.on_event("proposer_failed", {"step": self.session.step, "reason": target.get("reason"),
                                              "raw": target.get("raw"), "default": default, "stuck": stuck})
            if default.get("kind") != "exit":
                target = default          # a GROUNDED route (enter/edge/tile/approach) -> proceed on it
                                          # (correct navigation, not a wander) — but the failure is flagged.
            else:
                if self._plan is not None:
                    self._plan.next_objective = f"[proposer failed] {target.get('reason')}"
                return target             # UNGROUNDED (would be an exit/explore guess) -> break visibly
        note = target.get("note")
        if note and self._plan is not None:
            self._plan.next_objective = note      # fold the reflection note (mid-level -> visible objective)
        self._recent_targets.append({k: target.get(k) for k in ("kind", "x", "y", "map", "sprite")})
        self.on_event("target", {"step": self.session.step, "target": target, "stuck": stuck})
        return target

    def _navigate_leg(self, directive: Directive, obs, blocked_dirs: set[str]):
        """UNIFIED within-map navigation: the mid-level proposer picks ONE typed target toward the
        goal (held across frames), Jev picks the routing policy for tile targets (doors and edge
        crossings route deterministically), and the router enacts it. When a target can't make
        progress, RE-PROPOSE a different one (the get-unstuck job);
        only after that also fails do we return None so L1 (quest/strategize) escalates. The old
        'return None -> wait forever' freeze is gone: the default target is always 'exit' (leave)."""
        player = obs.player
        if player is None:
            return None
        occupied = {(int(n["x"]), int(n["y"])) for n in ((obs.game_state or {}).get("npcs") or [])
                    if "x" in n and "y" in n}
        target = self._current_target(directive, obs)
        if target is None:
            return None  # arrived on the target map; the success predicate ends the directive
        if target.get("kind") == "unresolved":
            return None  # configured model failed with no grounded route -> already flagged; STALL
                         # visibly (step_once's await-plan wait) rather than wander deterministically
        move = self._resolve_target(target, directive, obs, blocked_dirs, occupied)
        if move is not None:
            self._target_stuck = 0
            return move
        # stuck: re-propose a DIFFERENT target (the proposer's whole purpose), a couple of times
        self._target_stuck += 1
        if self._target_stuck <= TARGET_REPROPOSE_LIMIT:
            self._target = self._propose_target(obs, directive, self._default_target(directive, obs) or {"kind": "exit"}, stuck=True)
            self._target_map = player.map_id
            if self._target.get("kind") == "unresolved":
                self._target = None
                return None  # flagged; stall (don't wander)
            move = self._resolve_target(self._target, directive, obs, blocked_dirs, occupied)
            if move is not None:
                return move
        return None  # genuinely wedged -> _manage_directive escalates (story gate / re-strategize / heal)

    def _policy_route(self, obs, wp, avoid):
        """POLICY pathing: Jev picks a routing objective (shortest / dodge-grass / farm-exp) for the
        area, and the deterministic weighted router executes it toward the L2 waypoint. The policy
        is re-chosen on a new map/leg or every POLICY_BUDGET steps — one calibrated call per leg,
        not per step."""
        from .routing import policy_first_step
        player = obs.player
        if (player.x, player.y) == tuple(wp):
            return None  # arrived -> L2 re-picks the next waypoint
        if (self._policy is None or self._policy_map != player.map_id
                or self._policy_age >= POLICY_BUDGET):
            self._policy = self._pick_policy(obs)
            self._policy_map = player.map_id
            self._policy_age = 0
        self._policy_age += 1
        if self._policy == "farm-exp":
            return self._farm_step(obs, wp, avoid)   # pace the grass like a player, drifting to goal
        d = policy_first_step(self.world, player.map_id, (player.x, player.y), tuple(wp),
                              self._policy, avoid)
        return MoveAction(direction=d) if d is not None else None

    def _farm_step(self, obs, wp, avoid):
        """farm-exp GRAZING: pace back-and-forth across adjacent grass (each grass step rolls a wild
        encounter) while periodically advancing toward the goal — the way a player walks in and out
        of a grass patch to grind. Every 4th step it advances toward the target so it still makes
        progress; the rest it oscillates over grass tiles."""
        from .routing import policy_first_step
        player = obs.player
        tiles = self.world.tiles.get(player.map_id, {})
        terr = self.world.terrain.get(player.map_id, {})

        def grass_walkable(d: Direction) -> bool:
            nb = (player.x + DELTA[d][0], player.y + DELTA[d][1])
            return terr.get(nb) == "grass" and tiles.get(nb) != WALL and nb not in avoid

        self._farm_age += 1
        grass_dirs = [d for d in DELTA if grass_walkable(d)]
        # every 4th step (or when there's no grass to pace) advance toward the goal
        if not grass_dirs or self._farm_age % 4 == 0:
            d = policy_first_step(self.world, player.map_id, (player.x, player.y), tuple(wp),
                                  "farm-exp", avoid)
            self._farm_last = None
            return MoveAction(direction=d) if d is not None else None
        # otherwise graze: prefer reversing the last graze step (walk back into the grass you came
        # from) to keep triggering encounters in place; else step onto any adjacent grass.
        rev = {Direction.NORTH: Direction.SOUTH, Direction.SOUTH: Direction.NORTH,
               Direction.EAST: Direction.WEST, Direction.WEST: Direction.EAST}
        d = rev[self._farm_last] if (self._farm_last and rev[self._farm_last] in grass_dirs) else grass_dirs[0]
        self._farm_last = d
        return MoveAction(direction=d)

    def _pick_policy(self, obs) -> str:
        """Jev chooses the routing objective for this leg from HP / level / objective / grass."""
        from .routing import grass_nearby
        player = obs.player
        party = (obs.game_state or {}).get("party") or []
        tot = sum(int(p.get("max_hp") or 0) for p in party)
        hp_frac = (sum(int(p.get("hp") or 0) for p in party) / tot) if tot else None
        level = min((int(p.get("level") or 0) for p in party), default=None)
        gn = grass_nearby(self.world, player.map_id, (player.x, player.y))
        pol, conf = self.reasoner.choose_policy(
            hp_frac=hp_frac, level=level, level_target=self.level_target,
            objective=(self._directive.reason if self._directive else ""),
            area=map_name(player.map_id), grass_nearby=gn)
        self.on_event("policy", {"step": self.session.step, "policy": pol, "conf": round(conf, 2),
                                 "hp_frac": round(hp_frac, 2) if hp_frac is not None else None,
                                 "level": level, "grass": gn})
        return pol

    def _jev_path(self, obs, wp, blocked_dirs, avoid):
        """L3 pathing via JEV: Jev picks the step direction toward waypoint ``wp``, given the
        semantic map + trail + a BFS shortest-path suggestion it may follow or override. Returns
        a MoveAction, or None (arrived / no legal direction)."""
        player = obs.player
        if (player.x, player.y) == tuple(wp):
            return None  # arrived at the waypoint -> re-pick next step
        # directions Jev must not take: known walls (blocked_dirs) + tiles occupied by NPCs/off-route
        blk = set(blocked_dirs)
        for d, (dx, dy) in DELTA.items():
            if (player.x + dx, player.y + dy) in avoid:
                blk.add(d.value)
        # the deterministic shortest-path hint (Jev may follow or override it)
        bfs_move = self._bfs_move(player, wp, interact=False, blocked_dirs=blocked_dirs, occupied=avoid)
        bfs_dir = bfs_move.direction.value if isinstance(bfs_move, MoveAction) else None
        # the 4 immediate neighbor tiles' semantic class (Jev's minimal local awareness) — from the
        # RAM terrain grid, with NPC/off-route tiles surfaced as blocking too.
        terr = self.world.terrain.get(player.map_id, {})
        neighbors = {}
        for dd, (dx, dy) in DELTA.items():
            nb = (player.x + dx, player.y + dy)
            neighbors[dd.value] = "npc/blocked" if nb in avoid else terr.get(nb, "wall")
        # party HP fraction — so Jev can avoid grass (wild-encounter risk) when the team is hurt
        party = (obs.game_state or {}).get("party") or []
        tot = sum(int(p.get("max_hp") or 0) for p in party)
        hp_frac = (sum(int(p.get("hp") or 0) for p in party) / tot) if tot else None
        d, conf, probs = self.reasoner.path_step(
            player={"x": player.x, "y": player.y}, target=wp, neighbors=neighbors,
            recent=list(self._recent)[-8:], blocked_dirs=blk, hp_frac=hp_frac,
            objective=(self._directive.reason if self._directive else ""), bfs_suggestion=bfs_dir)
        if d is None:
            return None
        # Jev is free to OVERRIDE the BFS shortest path — but only when it's CONFIDENT. A
        # low-confidence disagreement is almost always a flip-flop that reverses the path and
        # oscillates (Jev is calibrated, so its confidence is the right signal for "I really do
        # know better here"). On a low-confidence disagreement, defer to BFS.
        overrode = False
        if bfs_dir is not None and d.value != bfs_dir:
            if conf < JEV_OVERRIDE_CONF:
                d = Direction(bfs_dir)
            else:
                overrode = True
        self.on_event("jev_path", {"step": self.session.step, "dir": d.value, "conf": round(conf, 2),
                                   "bfs": bfs_dir, "override": overrode, "probs": probs, "target": list(wp)})
        return MoveAction(direction=d)

    def _leave_via_nearest_exit(self, player, obs, blocked_dirs, occupied):
        """Head to the nearest exit door and step through it — used when we can't route to the
        target map (e.g. inside a building whose door destination isn't known yet). Routes around
        NPCs (the rival blocking the lab door) via BFS; steps through when standing on the door."""
        exits = obs.exits or []
        tiles = [(int(e["x"]), int(e["y"])) for e in exits]
        if not tiles:
            return None
        for (ex, ey) in tiles:  # already on a door -> step off it to fire the warp
            if (player.x, player.y) == (ex, ey):
                d = self._warp_exit_dir((ex, ey), obs.map_dims)
                return MoveAction(direction=d) if d is not None else None
        tgt = min(tiles, key=lambda c: abs(c[0] - player.x) + abs(c[1] - player.y))
        return self._bfs_move(player, tgt, interact=False, blocked_dirs=blocked_dirs,
                              occupied=(occupied - {tgt}))

    def _route_to_tile(self, obs, xy, blocked_dirs, avoid):
        """Route one step toward tile ``xy`` under the active pather (policy / jev / bfs)."""
        if self.pather == "policy" and getattr(self.reasoner, "choose_policy", None) is not None:
            return self._policy_route(obs, xy, avoid)
        if self.pather == "jev" and getattr(self.reasoner, "path_step", None) is not None:
            return self._jev_path(obs, xy, blocked_dirs, avoid)
        return self._bfs_move(obs.player, xy, interact=False, blocked_dirs=blocked_dirs, occupied=avoid)

    def _enter_map(self, next_map, obs, blocked_dirs, occupied):
        """Resolve an ``enter(next_map)`` target: route to the door for that map and step through it.
        Falls back to leaving via the nearest exit when the specific door isn't in view."""
        player = obs.player
        door = next((e for e in (obs.exits or []) if e.get("dest_map") == next_map), None)
        if door is None:
            return self._leave_via_nearest_exit(player, obs, blocked_dirs, occupied)
        dxy = (int(door["x"]), int(door["y"]))
        if (player.x, player.y) == dxy:  # step THROUGH (don't gate on blocked: the game warps you)
            d = self._warp_exit_dir(dxy, obs.map_dims)
            return MoveAction(direction=d) if d is not None else None
        return self._bfs_move(player, dxy, interact=False, blocked_dirs=blocked_dirs, occupied=occupied)

    def _cross_edge(self, goal_dir, next_map, obs, blocked_dirs, occupied):
        """Resolve a map-EDGE crossing toward ``goal_dir``: BFS to the boundary edge (funnels through
        the gap), then step off it."""
        player = obs.player
        if goal_dir is not None and self._on_goal_edge(player, obs.map_dims, goal_dir):
            if goal_dir not in blocked_dirs:
                return MoveAction(direction=Direction(goal_dir))
        return self._edge_step(player, obs, goal_dir, next_map, blocked_dirs, occupied)

    def _approach_npc(self, target, directive, obs, blocked_dirs, occupied):
        """Resolve an ``approach_npc`` target: choose the person, reach them, interact when adjacent +
        facing. Choice order (cheap+deterministic first): a pick already cached for this leg; a named
        sprite (exact/substring); Jev's calibrated pick among MULTIPLE candidates against the objective;
        else the nearest not-yet-talked. The pick is cached on the target so it's stable across frames."""
        player = obs.player
        npcs = [n for n in ((obs.game_state or {}).get("npcs") or []) if "x" in n and "y" in n]
        if not npcs:
            return self._leave_via_nearest_exit(player, obs, blocked_dirs, occupied)
        sprite = target.get("sprite") if isinstance(target, dict) else target
        cached = target.get("picked") if isinstance(target, dict) else None

        def by_name(name):
            s = str(name).lower()
            hits = [n for n in npcs if (nm := str(n.get("sprite") or "").lower()) and (nm in s or s in nm)]
            return hits[0] if hits else None

        npc = None
        if cached:                              # stick with the leg's pick (match by sprite label)
            npc = by_name(cached)
        if npc is None and sprite:              # a named who -> exact/substring match
            npc = by_name(sprite)
            if npc is None:
                self.on_event("approach_npc_miss", {"step": self.session.step, "sprite": sprite,
                                                    "seen": [n.get("sprite") for n in npcs]})
        if npc is None and len(npcs) > 1 and getattr(self.reasoner, "choose_npc", None) is not None:
            cands = [{"sprite": n.get("sprite"), "x": int(n["x"]), "y": int(n["y"]),
                      "talked_to": bool(n.get("talked_to"))} for n in npcs]
            idx, conf = self.reasoner.choose_npc(
                objective=(directive.reason if directive else ""), candidates=cands)
            if idx is not None:
                npc = npcs[idx]
                self.on_event("npc_pick", {"step": self.session.step, "sprite": npc.get("sprite"),
                                           "conf": round(float(conf), 2), "n": len(npcs)})
        if npc is None:                         # fallback: nearest not-yet-talked
            fresh = [n for n in npcs if not n.get("talked_to")] or npcs
            npc = min(fresh, key=lambda n: abs(int(n["x"]) - player.x) + abs(int(n["y"]) - player.y))
        if isinstance(target, dict) and npc.get("sprite"):
            target["picked"] = npc.get("sprite")   # cache for the rest of the leg (stable, no re-pick)

        nx, ny = int(npc["x"]), int(npc["y"])
        adj = {Direction.NORTH: (nx, ny + 1), Direction.SOUTH: (nx, ny - 1),
               Direction.EAST: (nx - 1, ny), Direction.WEST: (nx + 1, ny)}  # tile you stand on to face npc
        facing_map = {"north": Direction.NORTH, "south": Direction.SOUTH,
                      "east": Direction.EAST, "west": Direction.WEST}
        for d, stand in adj.items():
            if (player.x, player.y) == stand:
                if facing_map.get(getattr(player, "facing", None)) == d:
                    return InteractAction()          # adjacent AND facing -> talk
                return MoveAction(direction=d)        # adjacent, turn to face (a blocked step turns you)
        others = occupied - {(nx, ny)}
        # try each of the 4 stand-tiles nearest-first; take the first BFS-reachable one (cheap, and
        # avoids burning a re-propose cycle when the single nearest stand-tile happens to be a wall).
        for stand in sorted(adj.values(), key=lambda c: abs(c[0] - player.x) + abs(c[1] - player.y)):
            mv = self._bfs_move(player, stand, interact=False, blocked_dirs=blocked_dirs, occupied=others)
            if mv is not None:
                return mv
        return None

    def _resolve_target(self, target, directive, obs, blocked_dirs, occupied):
        """Turn a typed target ({tile|exit|enter|approach_npc}) into ONE move. Returns a MoveAction /
        InteractAction, or None when even this target can't make progress (caller then unsticks)."""
        if not target:
            return None
        kind = target.get("kind")
        player = obs.player
        if kind == "tile":
            xy = (int(target["x"]), int(target["y"]))
            door = next((e for e in (obs.exits or []) if (int(e["x"]), int(e["y"])) == xy), None)
            if (player.x, player.y) == xy:
                # a model-named tile that IS an exit door: step THROUGH the warp (the model said "leave
                # via (4,11)" — honor it), don't just stop on the doormat.
                if door is not None:
                    d = self._warp_exit_dir(xy, obs.map_dims)
                    return MoveAction(direction=d) if d is not None else None
                return InteractAction() if target.get("interact") else None
            avoid = set(occupied)
            return self._route_to_tile(obs, xy, blocked_dirs, avoid)
        if kind == "enter":
            return self._enter_map(int(target["map"]), obs, blocked_dirs, occupied)
        if kind == "approach_npc":
            return self._approach_npc(target, directive, obs, blocked_dirs, occupied)
        if kind == "edge":
            return self._cross_edge(target.get("dir"), target.get("next_map"), obs, blocked_dirs, occupied)
        if kind != "exit":
            return None  # unknown / unresolved kind -> no move (never silently wander)
        # exit: leave via nearest door; if none, try a boundary-edge crossing toward the goal
        move = self._leave_via_nearest_exit(player, obs, blocked_dirs, occupied)
        if move is not None:
            return move
        goal_dir = target.get("dir")
        if goal_dir is None and directive is not None and directive.target_map is not None:
            goal_dir = next_direction(self.memory.graph, player.map_id, directive.target_map)
        return self._cross_edge(goal_dir, None, obs, blocked_dirs, occupied)

    def _pick_waypoint(self, obs, directive, goal_dir, next_map, exit_tile, occupied):
        """L2: ask LunaRoute for the next grid tile to head toward on this map. It is given the
        current grid PLUS memory (its recent movement trail + its own recent picks) and an
        explicit objective/destination chain, so it knows where it must go and doesn't oscillate.
        Falls back to the exit tile when no provider is wired (offline / tests)."""
        player = obs.player
        if self.planner is None or getattr(self.planner, "provider", None) is None:
            return exit_tile
        why = ("pick the next tile toward the goal" if self._leg_wp_fail == 0
               else "the previous tile was unreachable or made no progress; pick a DIFFERENT tile")
        tmap = directive.target_map
        dest = (f"{map_name(tmap)} (map {tmap})" if tmap is not None else "the goal")
        if goal_dir and next_map is not None and next_map != tmap:
            dest += f" — next hop is {map_name(next_map)} to the {goal_dir}"
        elif goal_dir:
            dest += f" — head {goal_dir}"
        ctx = {
            "map_view": (self.world.render_semantic(player, obs.exits, occupied)
                         or self.world.render_labeled(player, obs.exits, occupied)),
            "player": {"x": player.x, "y": player.y, "map_id": player.map_id},
            "objective": directive.reason or f"reach map {tmap}",
            "destination": dest,
            "goal_dir": goal_dir,
            "exit_tile": exit_tile,
            "recent_trail": list(self._recent)[-8:],           # past frames: pos -> action -> result
            "recent_waypoints": [list(w) for w in self._recent_wps],  # tiles I recently chose
            "reachable": self._reachable_cells(player, occupied),
            "why": why,
        }
        wp = self.planner.next_waypoint(self.controller.emu, ctx)
        if wp is not None:
            self._recent_wps.append((wp[0], wp[1]))
            self.on_event("waypoint", {"step": self.session.step, "x": wp[0], "y": wp[1],
                                       "reason": wp[2]})
            return (wp[0], wp[1])
        return exit_tile

    def _edge_step(self, player, obs, goal_dir, next_map, blocked_dirs, occupied):
        """Cross a map-EDGE connection: BFS to the boundary edge in ``goal_dir`` (routing around
        off-route doors + NPCs), then step off it. Pure geometry; None if there's no clean route."""
        if goal_dir is None:
            return None
        edge = self._edge_goals(obs.map_dims, goal_dir)
        if not edge:
            return None
        nav = Navigator(self.world)
        off_route = {(int(e["x"]), int(e["y"])) for e in (obs.exits or []) if e.get("dest_map") != next_map}
        off_route |= occupied
        step = nav._bfs_first_step(player.map_id, (player.x, player.y), edge, off_route)
        if step is not None and step.value not in blocked_dirs:
            return MoveAction(direction=step)
        if step is None and (player.x, player.y) in edge and goal_dir not in blocked_dirs:
            return MoveAction(direction=Direction(goal_dir))
        return None

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
        gs = obs.game_state or {}
        occupied = {(int(n["x"]), int(n["y"])) for n in (gs.get("npcs") or [])
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
            dxy = (int(door["x"]), int(door["y"]))
            if (player.x, player.y) == dxy:
                # ALREADY standing on the warp door: arriving on the tile isn't enough — a door
                # sits on a map edge and you must step OFF that edge to trigger the transition
                # (a building exits downward). Without this the servo "reaches" the door and then
                # waits forever on the doormat (the Viridian Mart exit wedge).
                d = self._warp_exit_dir(dxy, obs.map_dims)
                if d is not None and d.value not in blocked_dirs:
                    return MoveAction(direction=d)
                return None
            return self._bfs_move(player, dxy, interact=False,
                                  blocked_dirs=blocked_dirs, occupied=occupied)

        # else a map-EDGE connection (walk off the edge into next_map): BFS across the known
        # collision map to the boundary edge in the connection direction, routing around
        # off-route doors and NPCs. PURE GEOMETRY: if BFS finds a clean route we take it; if it
        # can't (blocked, gated, unknown), we return None and hand the DECISION up to the LLM —
        # no greedy compass/lateral improvising (that just oscillated at gates).
        d = next_direction(self.memory.graph, player.map_id, tmap)
        if d is None:
            return None
        edge = self._edge_goals(obs.map_dims, d)
        if not edge:
            return None
        nav = Navigator(self.world)
        off_route = {(int(e["x"]), int(e["y"])) for e in exits if e.get("dest_map") != next_map}
        off_route |= occupied
        step = nav._bfs_first_step(player.map_id, (player.x, player.y), edge, off_route)
        if step is not None and step.value not in blocked_dirs:
            return MoveAction(direction=step)
        # already ON the boundary edge -> take the final step OFF it to cross the connection.
        if step is None and (player.x, player.y) in edge and d not in blocked_dirs:
            return MoveAction(direction=Direction(d))
        return None  # no clean route -> escalate (LLM), don't improvise

    @staticmethod
    def _on_goal_edge(player, map_dims, goal_dir: str) -> bool:
        """True if the player stands on the map's boundary edge in ``goal_dir`` — the point from
        which stepping that direction crosses a map-edge connection to the next map."""
        if goal_dir == "north":
            return player.y <= 0
        if not map_dims:
            return False
        w, h = map_dims
        if goal_dir == "south":
            return player.y >= h - 1
        if goal_dir == "west":
            return player.x <= 0
        if goal_dir == "east":
            return player.x >= w - 1
        return False

    @staticmethod
    def _warp_exit_dir(xy, map_dims) -> Direction | None:
        """The direction to step to LEAVE through a warp door at ``xy``. A door sits on a map
        edge; stepping off that edge triggers the transition. Buildings almost always exit
        downward, so that's the default when dims are unknown or the tile isn't on an edge."""
        x, y = xy
        if map_dims:
            w, h = map_dims
            if y >= h - 1:
                return Direction.SOUTH
            if y <= 0:
                return Direction.NORTH
            if x <= 0:
                return Direction.WEST
            if x >= w - 1:
                return Direction.EAST
        return Direction.SOUTH

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
        # (no wall-learning on a blocked move — walkability is RAM ground truth; a blocked result
        # is a transient/turn-in-place artifact, never a reason to mark a wall.)
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
        # (no wall-learning on a blocked move — the RAM collision map is ground truth.)
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
        # NOTE: no bump-discovery of walls. Walkability is RAM ground truth (the collision map,
        # which already includes walls/signs/trees/ledges as non-walkable); NPCs come from sprite
        # RAM. A "blocked" move never marks a wall — that only ever created permanent FALSE walls
        # that wedged navigation (the Route 1 gap, the Mart doormat).
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
        if self.recorder is not None:  # full-fidelity record of everything the agent saw this step
            d = self._directive
            tgt = self._target
            extra = {
                "objective": (d.reason if d else None),
                "directive": ({"intent": d.intent.value, "target": d.target, "success": d.success}
                              if d else None),
                "in_quest": self._in_quest,
                "quest_remaining": [q.reason for q in self._quest],
                "waypoint": ([tgt["x"], tgt["y"]] if tgt and tgt.get("kind") == "tile" else None),
                "target": tgt,   # the unified mid-level typed target (kind/x/y/map/sprite)
                "goal_map": self.goal_map,
                "routing_policy": (self._policy if self.pather == "policy" else self.pather),
            }
            self.recorder.record(step=self.session.step, obs=obs, action=rstep.action,
                                 result=result, extra=extra)
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
            slot, conf = battle_agent.choose_move(client, emu,
                                                  type_knowledge=self._battle_type_knowledge(emu))
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

    def _battle_type_knowledge(self, emu) -> list[str] | None:
        """Retrieve type-effectiveness guidance for the current enemy from the KB (once per
        enemy species, cached — battles have many turns). None when no KB is configured."""
        if self.knowledge is None:
            return None
        from ..games.pokemon_red.game_state import read_battle
        enemy = ((read_battle(emu) or {}).get("enemy") or {}).get("species", "")
        if enemy not in self._battle_kb:
            from ..games.pokemon_red.battle_agent import battle_lookup_query
            kt = self.knowledge.query_texts(battle_lookup_query(emu), top_k=3)
            self._battle_kb[enemy] = kt
            if kt:
                self.on_event("kb_search", {"step": self.session.step,
                                            "query": f"battle vs {enemy}", "results": len(kt)})
        return self._battle_kb.get(enemy) or None

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


def _desc(action) -> str:
    d = action.model_dump()
    if d.get("type") == "move":
        return f"move {d['direction']} x{d.get('tiles', 1)}"
    if d.get("type") == "goto":
        return f"goto {d.get('label') or (d.get('x'), d.get('y'))}"
    if d.get("type") == "menu_select":
        return f"menu[{d.get('index')}] {d.get('label') or ''}"
    return d.get("type", "?") + (f" {d.get('button')}" if d.get("button") else "")
