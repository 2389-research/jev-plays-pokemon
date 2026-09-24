"""The executive: one coherent hierarchy driven by a single active Directive.

This is the 3T "executive" middle layer (spec docs/superpowers/specs/2026-09-17-
planner-executor-design.md). Each step:

  1. perceive + update memory
  2. mode controllers own the step when they apply (battle / passive dialog)
  3. directive management — the L1 plan (_plan_steps -> _quest) is the source of
     Directives (target + machine-checkable success); RAM owns termination detection;
     the near-faint emergency-heal reflex is the only preemption; a wedged step is
     marked and L1 replaces it at the next gate (the plan is never cleared)
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
from ..games.pokemon_red.map_reader import read_collision_map
from ..games.pokemon_red.maps import map_name
from ..games.pokemon_red.progress import progress_vector
from ..games.pokemon_red.routes import next_direction
from ..games.pokemon_red.state import detect_mode, read_player
from .portal_graph import PortalGraph
from ..observations.builder import ObservationBuilder
from . import goals as goals_mod
from .l1_pipeline import run_l1_pipeline
from .memory import AgentMemory
from .navigator import Navigator
from .plan import AgentPlan, Directive, Intent
from .quest_reconciler import QuestStep, compile_steps_to_directives, reconcile_quests
from .reasoner import Reasoner, ReasonStep
from .session import Session
from .signals import game_signals, needs_emergency_heal
from .targets import build_targets, name_matches, npc_key, select_npc
from .world_map import DELTA, WALL

MAX_GOTO_STEPS = 18  # safety bound on one navigation macro
L1_EVERY_N_LEGS_DEFAULT = 5  # cadence: run the L1 strategic review every N legs by default
BLOCK_TRIGGER = 6  # blocked-for-N legs at/above which L1 fires early (navigation deadlock)
SERVO_FAIL_LIMIT = 6  # consecutive servo no-route steps before the directive is deemed impossible
JEV_OVERRIDE_CONF = 0.6  # Jev may override the BFS pathing suggestion only at/above this confidence
POLICY_BUDGET = 15       # steps a Jev-chosen routing policy runs before it's re-selected for the area
JEV_NPC_CONF = 0.4  # trust Jev's NPC pick only at/above this confidence (else fall back to nearest)
# Conversation/script gate (interaction-reliability F5): between two text boxes the screen briefly
# shows no text and the flow router says "navigate"; during scripted sequences the game ignores input
# (wJoyIgnore != 0). In both the agent must WAIT, not plan/move/wedge.
WJOYIGNORE = 0xCD6B        # non-zero while a script owns the controls
GRIND_ENCOUNTER_WINDOW = 60  # grinding: steps without a WILD battle before the grind step wedges
WLASTMAP = 0xD365          # wLastMap: where a LAST_MAP (0xFF) warp returns to (the last OUTDOOR map)
CONVO_GRACE_STEPS = 2      # steps after a dialogue step still treated as the same conversation
FORCED_WAIT_MAX_STEPS = 40  # consecutive forced waits before giving up on the gate (never stall forever)
FLOW_MIN_CONF = 0.55  # trust Jev's flow-gate pick (dialogue vs navigate) only at/above this; else _safe_flow
WARP_BACK = 0xFF     # a warp's dest_map of 0xFF means "return to the map you came from" (wLastMap)
RESUME_EVERY = 50    # steps between always-on resumable checkpoints written into the record-dir
TARGET_REPROPOSE_LIMIT = 2  # times the proposer may re-pick a DIFFERENT target to unstick before L1

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
        l1_every: int = 0,
        capture_mode: str = "off",
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
        # Distillation capture (design §3): a side-effect-only per-decision log, mounted like on_event.
        from ..logging.capture import Capture
        self.capture = Capture(recorder.dir, mode=capture_mode) if recorder is not None else Capture(".", mode="off")
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
        # always-on resumable checkpoint: any recorded run drops latest.state + latest.mem.json in its
        # own record-dir (periodically + at run-end), so the run can be continued with --resume-from.
        self._resume_dir = recorder.dir if recorder is not None else None
        self.goal_map = goal_map
        self.level_target = level_target
        self.knowledge = knowledge           # Orrery KB (also used for battle type lookups)
        self.pather = pather                 # "bfs" (deterministic) or "jev" (calibrated per-step direction)
        # PortalGraph: ground-truth cross-map routing (warps + edges + walk-reachability, ripped from
        # game data). L1 reads it for reachability; L2/servo for the exact next portal. None if the
        # packaged data is unavailable (falls back to the coarse WorldGraph).
        try:
            self.portals: PortalGraph | None = PortalGraph.load()
            # elevation edges the RAM collision map can't see -> every path-finder honours them
            self.world.cut_edges = {mid: c for mid in self.portals.maps
                                    if (c := self.portals.cut_edges(mid))}
        except Exception:
            self.portals = None
        self._battle_kb: dict[str, list[str]] = {}  # cache: enemy species -> type knowledge
        # battle_L2 state (design §2.1): the objective is chosen ONCE on the battle-start edge
        # and cached for the fight. `_last_in_battle` detects that false->true edge; the loop
        # clears both when the battle ends. `_battle_goals` carries L1's standing battle goals
        # (e.g. {"catch": ["Pidgey"]}); empty -> GRIND-EXP.
        self._last_in_battle: bool = False
        self._battle_objective: str | None = None
        self._battle_goals: dict = {}
        # L1 planner: the strategist that owns the plan. Active only when a goal/level is set —
        # otherwise the loop runs the plain executor path (legacy vertical-slice behavior).
        # NeedsArbiter is no longer consulted here; needs flow into L1 as signals + the
        # near-faint emergency-heal reflex (signals.needs_emergency_heal).
        self.planner = None
        if goal_map is not None or level_target > 0:
            from .planner_llm import Planner
            # the LLM planner needs a chat_json provider for travel-target selection: the
            # generative reasoner exposes it directly, or via its reflector (TypeSafe case).
            prov = getattr(reasoner, "provider", None) or getattr(
                getattr(reasoner, "reflector", None), "provider", None)
            self.planner = Planner(goal_map=goal_map, level_target=level_target,
                                   reflector=reasoner, provider=prov,
                                   strategist=strategist_provider or prov, knowledge=knowledge)
            # surface the planner's knowledge-base tool-calls in the run log
            self.planner.on_search = lambda q, n: self.on_event(
                "kb_search", {"step": self.session.step, "query": q, "results": n})
        # executive state (the single source of truth + the compiled quest queue)
        self._directive: Directive | None = None
        self._quest: deque[Directive] = deque()   # compiled plan steps, executed in order (FIFO)
        self._quest_step_age = 0                     # steps spent on the current directive (verify throttle)
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
        self._counter_bumped = False    # bumped a counter this leg (talk-over-counter NPCs: nurse/clerk)
        self._edge_attempt: tuple | None = None  # (map,x,y) we last tried to step OFF to cross a map edge
        self._recent_targets: deque = deque(maxlen=6)
        # Jev-picked routing POLICY (shortest / dodge-grass / farm-exp), re-chosen per leg/area
        self._policy: str | None = None
        self._policy_map: int | None = None
        self._policy_age = 0
        self._farm_last: Direction | None = None   # farm-exp: last step (to forbid two laterals in a row)
        self._farm_age = 0
        self._farm_weave = 1   # farm-exp weave sign (+1/-1): alternates lateral lanes into a zig-zag
        self._prev_map: int | None = None  # last DISTINCT map (resolves 0xFF "return" warps)
        # on resume the saved history already ends with the current map, so the "came FROM" map is
        # the entry before it — seed it, or a building's 0xFF door never resolves (stuck inside)
        hist = self.memory.map_history
        if len(hist) >= 2 and hist[-2] != hist[-1]:
            self._prev_map = hist[-2]
        self._recent: deque[str] = deque(maxlen=recent_max)
        self._prev: ReasonStep | None = None
        self._plan = self.memory.plan  # strategic AgentPlan (reflection), separate from the directive
        # L1 strategic planner (Task 6a): the canonical ordered plan (QuestSteps) L1 revises, plus
        # the gate state that decides WHEN to run a strategic review (cadence / blocked / event).
        self.l1_every = l1_every or L1_EVERY_N_LEGS_DEFAULT
        self._plan_steps: list[QuestStep] = []   # canonical plan L1 reconciles + recompiles into _quest
        self._qid = 0                            # monotonic QuestStep id counter
        self._legs_since_l1 = 0                  # legs since the last L1 review (cadence trigger)
        self._blocked_for_n = 0                  # consecutive legs blocked (>= BLOCK_TRIGGER fires L1)
        self._l1_event = False                   # a one-shot event asked for an L1 review next gate
        self._l1_last = None                     # last L1 review's {change, assessment, trace} (for the recorder)
        self._l1_trace: list[dict] = []          # current/last review's pipeline stage events
        # tiered goals (spec §3.2): last-seen status per tier (absent = not yet observed -> a resume
        # or a freshly written goal never pings) and the goals that already pinged (one ping each)
        self._goal_prev_status: dict[str, str] = {}
        self._goal_pinged: set[tuple] = set()
        self._notepad_changed = False            # the recorder logs the full notepad only when it changed
        # grind in place (F1): which step is grinding, since when, the last WILD battle's end, last dir
        self._grind_qid: str | None = None
        self._grind_start = 0
        self._last_wild_battle_end: int | None = None
        self._grind_last = None
        self._battle_wild = False
        self._talk_npc = None                    # npc_key of the NPC we're about to talk to (F2 rotation)
        # conversation/script gate (F5): last step routed to dialogue, consecutive forced waits, and
        # whether forced waiting is armed (a timeout disarms it until the script flag clears)
        self._last_dialog_step = -(10**9)
        self._forced_waits = 0
        self._forced_wait_armed = True
        # per-step flags, set in step_once: conversation-ness is decided at STEP START (so a step
        # whose own action triggers a script still counts), and whether this step was routed to
        # dialogue / was a forced wait (for the block budget + forced-wait counter in _finish)
        self._step_convo = False
        self._step_dialogue = False
        self._step_forced_wait = False
        # attach the loop's Capture onto planner/reasoner so their layer methods can reach it
        self._mount_capture()

    def _mount_capture(self) -> None:
        """Attach the loop's Capture onto the planner + reasoner so their layer methods can
        reach it via getattr(self, "capture", None). No-op when a component is absent."""
        cap = getattr(self, "capture", None)
        if cap is None:
            return
        for comp in (getattr(self, "planner", None), getattr(self, "reasoner", None)):
            if comp is not None:
                try:
                    comp.capture = cap
                except Exception:
                    pass

    def _cap_det(self, layer: str, inp, out) -> None:
        """Deterministic-layer capture (§3, YAGNI-gated): records a pure-function decision for
        replay/attribution, but ONLY under --capture distill (cap.deterministic). Best-effort;
        never raises / never changes behavior."""
        cap = getattr(self, "capture", None)
        if cap is None or not getattr(cap, "deterministic", False):
            return
        cap.record(layer, kind="deterministic", model=None, input=inp, parsed=out)

    # ------------------------------------------------------------------ step
    def step_once(self) -> ActionResult:
        want_shot = self.vision or bool(self.logger and getattr(self.logger, "wants_screenshot", False))
        obs, shot = self.builder.build(capture_screenshot=want_shot)
        self.session.record_position(obs.player)
        self._step_convo = self._in_conversation()   # decided BEFORE this step's action (F5)
        self._step_dialogue = False
        self._step_forced_wait = False
        # open a fresh per-decision capture buffer for this step (flushed in _finish)
        self.capture.begin_step(self.session.step, anchor=f"states/*_step{self.session.step}.state")
        # FULL-MAP collision from RAM (wOverworldMap) is GROUND TRUTH for walkability — it already
        # encodes walls, signs, trees, and ledges as non-walkable (verified against the game); the
        # only obstacles NOT in it are moving sprites (NPCs), read separately from sprite RAM. We
        # RE-READ it EVERY step (a full-map decode is cheap next to the LLM/Jev calls) rather than
        # caching once per map: re-reading self-heals a STALE half-settled read on map-entry (it's
        # overwritten on the next step) and picks up mid-map changes (cut trees, smashed rocks,
        # opened doors). We reject only a decode that doesn't contain the player — a transition
        # frame still showing the PREVIOUS map — so we never ingest a mismatched grid.
        if obs.player is not None:
            mid = obs.player.map_id
            from ..games.pokemon_red.map_reader import read_collision_map
            cm = read_collision_map(self.controller.emu)
            if (cm is not None and cm["map_id"] == mid
                    and (obs.player.x, obs.player.y) in cm["walkable"]):  # reject stale/transition reads
                self.world.ingest_collision(cm["map_id"], cm["width"], cm["height"],
                                            cm["walkable"], cm.get("counters"), cm.get("terrain"))
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

        # keep the compact battle-goals (catch list + level_target) in sync with L1's plan each
        # step, so the battle layer sees the CURRENT goals on the next battle-start edge (§7.1).
        self._sync_battle_goals()

        # --- nickname prompt / name keyboard: always decline deterministically (an A-spam would type
        # "AAAAAAAAAA" as the Pokemon's name) ---
        from ..games.pokemon_red import menus as _menus
        if _menus.handle_nickname(self.controller.emu):
            from ..core.models import WaitAction
            rstep = ReasonStep(location="nickname", objective="decline the nickname",
                               reasoning="nickname prompt: answer NO (keep the species name)",
                               action=WaitAction(frames=1))
            self._prev = rstep
            self._emit_reason(rstep, 0)
            return self._finish(obs, rstep, ActionResult(success=True, result="completed",
                                                         mode_before=detect_mode(self.controller.emu),
                                                         mode_after=detect_mode(self.controller.emu),
                                                         detail="declined nickname"), 0, {}, shot)

        # --- battle owns the step (deterministic RAM bit — a fact, never a classification) ---
        if ctx.get("in_battle"):
            self._battle_wild = self.controller.emu.read_memory(0xD057) == 1   # 1 wild, 2 trainer
            rstep, latency, usage, result = self._battle_turn(obs)
            self._prev = rstep
            self._emit_reason(rstep, latency)
            return self._finish(obs, rstep, result, latency, usage, shot)
        # battle just ended -> clear the cached objective so the NEXT battle re-detects the
        # false->true edge in _battle_turn and re-runs battle_L2 (design §2.1 edge reset).
        if self._last_in_battle:
            self.on_event("battle_end", {"step": self.session.step,
                                         "objective": self._battle_objective})
            self._last_in_battle = False
            self._battle_objective = None
            self._note_battle_end(wild=self._battle_wild)

        # --- FLOW ROUTER: menu is deterministic (RAM cursor); navigate-vs-dialogue is a calibrated
        # Jev choice (falls back to the deterministic ctx_kind detector when Jev isn't wired). Replaces
        # the brittle has_upper dialog heuristic as the router. ---
        flow = self._route_flow(obs, ctx)
        if flow == "menu":
            shopped = self._maybe_shop(obs, shot)   # SHOP directive at an open Mart counter -> buy macro
            if shopped is not None:
                return shopped
            return self._jev_turn(obs, player_desc, self._blocked_dirs(obs, None), None, shot)
        if flow == "dialogue":
            self._note_dialogue_step()
            self._step_dialogue = True
            return self._advance_dialog(obs, shot)
        # flow == "navigate" -- but the gap between two text boxes of a conversation, or a script
        # that owns the controls, is not the agent's turn: wait instead of planning/moving/wedging
        # (F5). The raw text detector alone never forces a wait: the flow router just said navigate.
        if self._should_defer_to_script():
            self._step_forced_wait = True
            return self._forced_wait(obs, shot)
        # otherwise fall through to the navigation path below

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
        self._legs_since_l1 += 1       # a planner leg elapsed -> feeds the L1 cadence gate
        if directive is not None and directive.target_bearing:
            servo = self._dispatch_servo(directive, obs, blocked_dirs)
            if servo is not None:
                self._servo_fail = 0
                # NOTE: do NOT clear _blocked_for_n here — a servo move exists even while CIRCLING
                # (a route to a tile we keep revisiting). _blocked_for_n is driven by the stuck
                # detector in _finish (consecutive no-PROGRESS legs), so circling accumulates toward
                # the wedge/L1 trigger instead of oscillating 0<->1 and never firing.
                return self._act_servo(obs, directive, servo, shot)
            self._servo_fail += 1     # no clean route -> navigation deadlock (feeds the wedge + L1)
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
        result = self._dispatch(rstep, obs, blocked_dirs, ctx_kind)
        return self._finish(obs, rstep, result, latency, usage, shot)

    # ---------------------------------------------------- L1 strategic planner
    def _next_qid(self) -> str:
        self._qid += 1
        return f"q{self._qid}"

    def _l1_due(self) -> bool:
        """PURE predicate (no side effects): should L1 run now? Fires on the cadence, on a
        one-shot event flag, or when navigation has been blocked for too long. Resets live only
        in ``_run_l1`` so this can be polled freely."""
        return (self._legs_since_l1 >= self.l1_every
                or self._l1_event
                or self._blocked_for_n >= BLOCK_TRIGGER)

    def _recompile_quest(self) -> None:
        """Recompile the pending plan steps into the executable quest queue (deque of Directives),
        PRESERVING the active step's not-yet-run sub-directives. A talk/fetch step compiles to
        TRAVEL + TALK_TO sharing one quest_id; once TRAVEL has been popped (step is `active`) the
        TALK_TO still sits in `_quest`. Rebuilding from `pending` alone would drop it and the talk
        would be skipped (the step falsely completing on the travel's on_map), so we keep the
        active step's leftover directives at the head of the queue."""
        active = next((s for s in self._plan_steps if s.status == "active"), None)
        keep = [d for d in self._quest if active is not None and d.quest_id == active.id]
        pending = compile_steps_to_directives([s for s in self._plan_steps if s.status == "pending"])
        self._quest = deque(keep + pending)

    def _has_heal_step(self) -> bool:
        """True if the plan already carries a heal quest (a pending/active step whose criterion is an
        hp_frac threshold) — so a near-faint emergency doesn't re-force L1 every step once a heal
        errand is in the plan."""
        return any((s.done_when or "").startswith("hp_frac")
                   for s in self._plan_steps if s.status in ("pending", "active"))

    def _record_l1_trace(self, event: dict) -> None:
        """``on_trace`` callback for ``run_l1_pipeline``: append each pipeline stage event
        (triage/brainstorm/decide/invalid_criterion) to the current review's trace. ``_run_l1``
        resets ``_l1_trace`` at the start of each review and merges it into ``_l1_last`` so the
        recorder captures WHY L1 decided what it decided, not just the final change/assessment."""
        self._l1_trace.append(event)

    def _run_l1(self, obs, emergency: bool = False, hard_event: bool = False) -> None:
        """L1 strategic review: run the L1 pipeline (triage/brainstorm/decide/validate) to see
        whether the standing plan needs to change; if so, deterministically reconcile the proposal
        into the canonical plan (preserving progress) and recompile the quest queue. Durable
        strategy (tiered goals + notepad, mirrored to mission/milestone) lives on the AgentPlan. ``emergency``
        (near-faint) is surfaced in the context so L1 inserts a routed heal quest. ``hard_event``
        tells the pipeline this review was forced by an event (wedge/emergency/plan-exhausted)
        rather than the periodic cadence, so it skips the cheap triage gate. Never raises: any
        failure emits ``l1_failed`` and changes nothing. The gate counters are cleared in
        ``finally`` so a failing review doesn't re-fire every step."""
        self._l1_trace: list[dict] = []   # reset per review; _record_l1_trace appends pipeline stages
        try:
            if self.planner is None:
                return
            if self._plan is None:
                self._plan = AgentPlan()
            emu = self.controller.emu
            signals = game_signals(emu)
            signals["blocked_for_n"] = self._blocked_for_n
            signals["emergency_heal"] = emergency
            from ..games.pokemon_red.game_state import read_items
            from .signals import catch_status
            signals["catch"] = catch_status(self._plan.battle_goals, read_items(emu), signals["party"])
            cur_mid = obs.player.map_id if obs.player else None
            context = {
                "current_map": {"id": cur_mid,
                                "name": map_name(cur_mid) if cur_mid is not None else None},
                "party": signals["party"],
                "items": signals["items"],
                "badges": signals["badges"],
                "plan": [{"id": s.id, "map": s.map, "kind": s.kind, "talk": s.talk,
                          "done_when": s.done_when, "status": s.status,
                          **({"why_wedged": s.wedge_reason} if s.status == "wedged" and s.wedge_reason else {})}
                         for s in self._plan_steps],
                "signals": signals,
            }
            # tiered goals (spec §3.1): drop a paused focus whose own criterion is met, then show L1
            # its goals, their RAM status, the paused focus and its notepad
            if goals_mod.refresh_interrupted(self._plan, emu, self.memory):
                self.on_event("goals_changed", {"step": self.session.step, "interrupted": None,
                                                "why": "interrupted focus met"})
            context.update(goals_mod.goals_view(self._plan, emu, self.memory))
            pre_status = context["goal_status"]
            prop = run_l1_pipeline(emu, context, self.planner, hard_event=(hard_event or emergency),
                                   on_trace=self._record_l1_trace)
            step_edit = prop is not None and bool(prop.get("add") or prop.get("remove"))
            change = (goals_mod.detect_change(self._plan, prop, step_edit=step_edit)
                      if prop is not None else None)
            if step_edit:
                self._plan_steps = reconcile_quests(
                    self._plan_steps,
                    {"add": prop.get("add", []), "remove": prop.get("remove", [])},
                    next_id=self._next_qid,
                    on_event=lambda kind, payload: self.on_event(
                        kind, {"step": self.session.step, **payload}))
                # a provisional bootstrap default (generic goal-travel) is SUPERSEDED the moment L1
                # supplies a real step — otherwise reconcile keeps it first (adds go after the active
                # step) and the real plan sits behind a possibly story-gated default forever.
                real = [s for s in self._plan_steps
                        if not s.provisional and s.status in ("pending", "active")]
                if real and any(s.provisional for s in self._plan_steps):
                    dropped = {s.id for s in self._plan_steps if s.provisional}
                    if self._directive is not None and self._directive.quest_id in dropped:
                        self._directive = None   # the committed default is gone -> advance to L1's first real step
                    self._plan_steps = [s for s in self._plan_steps if not s.provisional]
                self._recompile_quest()
                if change is not None:   # after the step edit succeeded: a failed review changes nothing
                    self._apply_goals_change(change, pre_status)
                self._l1_last = {"change": True, "assessment": prop.get("assessment"),
                                 "trace": self._l1_trace}
                self.on_event("l1_review", {"step": self.session.step, "change": True,
                                            "assessment": prop.get("assessment"),
                                            "add": len(prop.get("add", [])),
                                            "anchors": [a.get("after") for a in prop.get("add", [])
                                                        if isinstance(a, dict)],
                                            "remove": prop.get("remove", [])})
                self.on_event("quest", {"step": self.session.step, "len": len(self._quest),
                                        "plan": [d.reason for d in self._quest]})
            elif change is not None:
                # goals / notepad / catch only: the step queue is untouched (no reconcile, no
                # recompile, no quest event) and the review is tagged so thrash metrics stay honest
                self._apply_goals_change(change, pre_status)
                self._l1_last = {"change": "goals", "assessment": prop.get("assessment"),
                                 "trace": self._l1_trace}
                self.on_event("l1_review", {"step": self.session.step, "change": "goals",
                                            "assessment": prop.get("assessment")})
            else:
                self._l1_last = {"change": False, "assessment": "", "trace": self._l1_trace}
                self.on_event("l1_review", {"step": self.session.step, "change": False,
                                            "assessment": ""})
        except Exception as e:  # never let a strategic review break the loop
            self._l1_last = {"change": False, "assessment": "l1_failed", "trace": self._l1_trace}
            self.on_event("l1_failed", {"step": self.session.step, "error": str(e)})
        finally:
            # clear the cadence/event gate even on failure, so a raising review doesn't re-fire
            # every step. (blocked_for_n is deliberately NOT reset here — it's cleared at the wedge
            # and on commit; resetting it here would suppress wedge detection.)
            self._legs_since_l1 = 0
            self._l1_event = False

    def _apply_goals_change(self, change, pre_status: dict) -> None:
        """Apply a detected goals/notepad/interrupted/catch change (goals.apply_change) and emit it.
        A tier L1 just rewrote is forgotten by the goal-met ping, so a goal written already-met
        never pings."""
        before = {"goals": self._plan.goals.model_dump(), "interrupted": self._plan.interrupted.model_dump()}
        goals_mod.apply_change(self._plan, change, pre_status=pre_status)
        for tier in change.goals:          # a rewritten tier is a new goal instance: fresh status + latch
            self._goal_prev_status.pop(tier, None)
        self._goal_pinged = {k for k in self._goal_pinged if k[0] not in change.goals}
        if change.goals or change.drop_interrupted or before["interrupted"] != self._plan.interrupted.model_dump():
            self.on_event("goals_changed", {"step": self.session.step, "before": before,
                                            "after": {"goals": self._plan.goals.model_dump(),
                                                      "interrupted": self._plan.interrupted.model_dump()}})
        if change.notepad is not None:
            self._notepad_changed = True
            self.on_event("notepad_changed", {"step": self.session.step, "len": len(self._plan.notepad),
                                              "truncated": self._plan.notepad_truncated})

    def _check_goal_ping(self) -> None:
        """Goal-met ping (spec §3.2), every step: when a tier's criterion goes unmet -> met, arm an L1
        review once for that goal (hp_frac flips in battle, so each goal pings at most once). The
        first observation of a tier only records it (resume / fresh goal -> no spurious ping)."""
        if self._plan is None:
            return
        emu = self.controller.emu
        for tier in goals_mod.GOAL_TIERS:
            g = getattr(self._plan.goals, tier)
            st = goals_mod.goal_status(g, emu, self.memory)
            prev = self._goal_prev_status.get(tier)
            self._goal_prev_status[tier] = st
            key = (tier, g.text, g.done_when)
            if prev == "unmet" and st == "met" and key not in self._goal_pinged:
                self._goal_pinged.add(key)
                self._l1_event = True
                self.on_event("goal_met", {"step": self.session.step, "tier": tier, "text": g.text,
                                           "done_when": g.done_when})

    # --------------------------------------------------- directive lifecycle
    def _mark_step(self, quest_id: str | None, status: str, reason: str | None = None) -> None:
        """Set the status of the plan step tagged ``quest_id`` (no-op if it's not in the plan) and
        emit a completion-provenance event naming WHY the step transitioned — the log alone should
        show a step's acceptance criterion + fate, no RAM forensics required. ``reason`` is the
        best-available explanation for a ``wedged`` transition (falls back to a static string)."""
        if quest_id is None:
            return
        for s in self._plan_steps:
            if s.id == quest_id:
                s.status = status
                if status == "done":
                    # NB: key is "step_kind", not "kind" — the recorder's on_event stamps the event
                    # type under "kind" ("step_done"), so a payload "kind" would clobber that marker.
                    self.on_event("step_done", {"step": self.session.step, "id": s.id,
                                                "done_when": s.done_when, "step_kind": s.kind})
                elif status == "wedged":
                    s.wedge_reason = (reason or "no route / blocked")[:200]
                    self.on_event("step_wedged", {"step": self.session.step, "id": s.id,
                                                  "done_when": s.done_when,
                                                  "reason": reason or "no route / blocked"})
                return

    def _step_by_qid(self, quest_id: str | None) -> QuestStep | None:
        if quest_id is None:
            return None
        return next((s for s in self._plan_steps if s.id == quest_id), None)

    def _manage_directive(self, obs) -> Directive | None:
        """Plan-driven executive: the L1 plan (``_plan_steps`` -> ``_quest``) is the source of
        directives. RAM owns termination (the success predicate); the near-faint emergency-heal
        reflex is the only preemption; a wedged step is marked ``wedged`` and L1 replaces it at the
        next gate (the plan is never cleared). There is no arbiter intent / priority stack /
        escalation-score path."""
        emu = self.controller.emu
        self._check_goal_ping()                # a goal just became met -> L1 review at the gate below
        party = game_signals(emu)["party"]     # L1 builds its own full signals in _run_l1
        emergency = needs_emergency_heal(party)  # near-faint -> force an L1 heal ping (not a directive)
        if self._directive is not None:
            self._quest_step_age += 1
        satisfied = self._directive is not None and self._directive_satisfied(self._directive)

        # --- 1. wedge (BEFORE the L1 gate, so reconcile sees the `wedged` step and can replace it):
        # the active step can't make progress -> mark it, reset the block/servo counters (the
        # replacement starts fresh; L1 fires ONCE per wedge, not every step) and arm L1. Never
        # clears the plan.
        if (self._directive is not None and not satisfied
                and (self._servo_fail >= SERVO_FAIL_LIMIT or self._blocked_for_n >= BLOCK_TRIGGER)):
            if self._directive.quest_id is not None:
                self._mark_step(self._directive.quest_id, "wedged", reason=self._directive.reason)
            self._blocked_for_n = 0
            self._servo_fail = 0
            self._l1_event = True     # L1 replaces just this step at the next gate (below)
            self.on_event("quest_step_wedged", {"step": self.session.step,
                                                "reason": self._directive.reason})

        # --- 2. L1 gate: periodic / event / blocked review, OR an unhandled near-faint emergency
        # (which makes L1 INSERT a routed heal quest — heal is an L1 ping, never a target-less
        # directive that would freeze the agent). The heal-step guard stops it re-forcing L1 once a
        # heal errand is already in the plan.
        ran_l1 = False
        if self._l1_due() or (emergency and not self._has_heal_step()):
            # event-driven (wedge / near-faint / navigation deadlock) -> hard_event, so the pipeline
            # skips its cheap triage gate; a review firing ONLY off the periodic cadence is not.
            hard_event = bool(self._l1_event) or self._blocked_for_n >= BLOCK_TRIGGER or emergency
            self._run_l1(obs, emergency=emergency, hard_event=hard_event)
            ran_l1 = True

        # --- 3. bootstrap: L1 owns the plan. On an empty plan, let L1 populate it FIRST (so we don't
        # commit a generic "travel to goal_map" step that lands us on a story-gated hop and wedges
        # forever). Only when L1 can't help (offline / declined) do we synthesize a PROVISIONAL
        # default — which _run_l1 drops the moment L1 supplies real steps.
        if not self._plan_steps:
            if not ran_l1 and self.planner is not None and (self.planner.strategist or self.planner.provider):
                # bootstrap on an empty plan is plan-exhausted, not a periodic cadence tick -> hard_event
                self._run_l1(obs, hard_event=True)
            if not self._plan_steps:
                gm = self.goal_map if self.goal_map is not None else (obs.player.map_id if obs.player else 0)
                self._plan_steps.append(QuestStep(id=self._next_qid(), map=gm, done_when="on_map",
                                                  status="pending", provisional=True, kind="travel"))
                self._recompile_quest()

        # --- 3b. the active directive's step was removed/replaced by L1 -> advance to the plan ----
        if self._directive is not None and self._directive.quest_id is not None:
            step = self._step_by_qid(self._directive.quest_id)
            if step is None or step.status in ("done", "wedged"):
                self._directive = None

        # --- 3c. an intermediate travel step that is merely ON THE WAY to the next one: skip it ------
        if self._directive is not None and not satisfied and self._travel_on_the_way(obs):
            self.on_event("travel_step_merged", {"step": self.session.step, "skipped": self._directive.reason,
                                                 "next": self._quest[0].reason})
            satisfied = True

        # --- 4. termination / advance -----------------------------------------------------------
        if self._directive is None or satisfied:
            if self._directive is not None:
                qid = self._directive.quest_id
                # A step can compile to SEVERAL directives sharing one quest_id (a talk/fetch step is
                # TRAVEL(reach map) + TALK_TO(criterion)). Mark the STEP done only when its FINAL
                # directive completes — NOT the intermediate travel half. Otherwise the step
                # false-completes on arrival (e.g. a heal marked hp_frac>=1.0 the instant you enter
                # the Poké Center, before ever reaching the nurse), which also drives re-plan thrash.
                more_for_step = bool(self._quest) and self._quest[0].quest_id == qid
                self.on_event("directive_done", {"step": self.session.step,
                                                 "intent": self._directive.intent.value,
                                                 "reason": self._directive.reason})
                if not more_for_step:
                    self._mark_step(qid, "done")
                self._servo_fail = 0
            if self._quest:
                self._directive = self._quest.popleft()
                self._mark_step(self._directive.quest_id, "active")
                self._commit_directive(self._directive.reason)
                return self._directive
            self._directive = None   # plan exhausted; the next gate / bootstrap refills
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
        self._servo_fail = 0
        self._blocked_for_n = 0      # a fresh directive starts with a clean block counter
        self.stuck.reset_objective()  # ...and a fresh objective budget measured against ITS target
        self._quest_step_age = 0
        self._leg_wp = None          # a new directive -> the L2 navigator picks a fresh waypoint
        self._leg_wp_fail = 0
        self._recent_wps.clear()
        self._policy = None          # and re-picks the routing policy for the new objective
        self._target = None
        self._target_map = None      # defensive: never pair a stale map with the (now cleared) target
        self._target_stuck = 0
        self._counter_bumped = False  # a fresh leg re-bumps a counter before talking over it
        self._edge_attempt = None
        self._recent_targets.clear()
        self.on_event("directive", {"step": self.session.step, "intent": self._directive.intent.value,
                                    "target": self._directive.target, "success": self._directive.success,
                                    "reason": self._directive.reason})

    def _carry_plan(self) -> Directive | None:
        """Checkpoint the durable plan carrying the live directive (L1 owns _plan_steps/_plan)."""
        self.memory.plan = self._plan
        if self._plan is not None:
            self._plan.directive = self._directive
        return self._directive

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
            # arrived — but a level goal isn't met by standing here: pace the grass (F1)
            if self._is_grind(directive) and not self._directive_satisfied(directive):
                return {"kind": "grind"}
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

    def _target_reached(self, target, obs) -> bool:
        """Only a 'tile' target can be 'reached' in place (others resolve to a move each frame and
        end when the map changes / an interaction fires). EXCEPTION: a map-EDGE opening is reached by
        stepping OFF it (one more move) to cross to the next map — standing on the boundary tile is
        NOT the end, so return it un-reached and let the router step off. Guard against a boundary
        tile that ISN'T a real crossing (a walkable edge with no adjacent map): once we've already
        issued the step-off from this exact tile and we're still standing on it, it's a dead end ->
        treat it as reached so we re-propose instead of stepping into the wall forever."""
        if target.get("kind") != "tile":
            return False
        xy = (int(target["x"]), int(target["y"]))
        if (obs.player.x, obs.player.y) != xy:
            return False
        if target.get("hop"):
            return False   # a ledge take-off cell: arriving is not the end, the hop is (K8)
        dims = getattr(obs, "map_dims", None)
        if dims:
            w, h = dims
            on_edge = xy[0] <= 0 or xy[1] <= 0 or xy[0] >= w - 1 or xy[1] >= h - 1
            if on_edge and self._edge_attempt != (obs.player.map_id, xy[0], xy[1]):
                return False   # first arrival at an edge opening -> step OFF to cross (not yet reached)
        return True

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
        # Ground-truth PortalGraph waypoint: for a cross-map hop the graph covers, head straight to
        # the EXACT next portal tile (deterministic) instead of asking the LLM proposer, which
        # oscillated at gates. This is what actually crosses Viridian Forest toward Pewter.
        if default is not None and default.get("kind") == "grind":
            return default   # grinding is mechanics, not a destination: never let the proposer override it
        tmap0 = directive.target_map if directive is not None else None
        portal = self._portal_next(obs.player, tmap0) if (tmap0 is not None and obs.player is not None) else None
        self._cap_det("portal_next", {"map_id": getattr(obs.player, "map_id", None), "target_map": tmap0}, portal)
        if portal is not None:
            cx, cy = int(portal["coord"][0]), int(portal["coord"][1])
            self.on_event("portal_hop", {"step": self.session.step, "from_map": obs.player.map_id,
                                         "to_map": portal["dest_map"], "coord": [cx, cy],
                                         "goal_map": int(tmap0)})
            tgt = {"kind": "tile", "x": cx, "y": cy, "portal": True,
                   "note": f"portal toward {map_name(portal['dest_map'])}"}
            if portal.get("kind") == "ledge":
                tgt["hop"] = portal.get("hop")   # reach the take-off cell, then hop (K8)
            return tgt
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
        reach = self._reachable_cells(player, occupied)
        ctx = {
            "map_view": obs.map_view,
            "player": {"x": player.x, "y": player.y, "map_id": player.map_id},
            "objective": directive.reason,
            "milestone": (self._plan.milestone if self._plan else None),
            "focus": (self._plan.goals.tertiary.text or None) if self._plan else None,
            "destination": (f"{map_name(tmap)} (map {tmap})" if tmap is not None else "the goal"),
            "goal_dir": goal_dir,
            "exit_tile": exit_tile,
            "npcs": [{"x": int(n["x"]), "y": int(n["y"]), "sprite": n.get("sprite"),
                      "talked_to": n.get("talked_to")}
                     for n in ((obs.game_state or {}).get("npcs") or []) if "x" in n and "y" in n],
            "recent_trail": list(self._recent)[-8:],
            "recent_targets": [dict(t) for t in self._recent_targets],
            "reachable": reach,
            "candidate_exits": self._candidate_exits(obs, reach),
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
        if move is None and target.get("portal") and self._report_blockers(obs, target):
            return None        # the route is blocked by objects/NPCs: L1 now knows exactly what
        if target.get("kind") == "grind":
            if move is None:   # no reachable grass on this map: hand the grind back to L1 right away
                self._wedge_active(f"grind: no reachable grass on {map_name(player.map_id)}")
            return move        # (never re-propose a grind target)
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
        """farm-exp = WEAVE FORWARD toward the waypoint through grass. The motion is a zig-zag that
        always trends to the goal: we never take two lateral steps in a row, so forward steps always
        outnumber sideways ones and net displacement is toward the destination (no pacing in one row).
          - After any lateral step, the next step ADVANCES (forward is open here — enforced below).
          - Otherwise, ~every 3rd step (or whenever the forward tile isn't grass) we shift ONE lane
            sideways onto grass, alternating left/right lanes (the zig-zag), then advance again.
          - If the forward tile is a wall, hand off to the goal-directed router to get around it.
        Every grass tile we touch rolls a wild encounter, so weaving through grass grinds EXP while we
        keep closing on the waypoint — instead of oscillating in place until the level target is hit."""
        from .routing import policy_first_step
        player = obs.player
        tiles = self.world.tiles.get(player.map_id, {})
        terr = self.world.terrain.get(player.map_id, {})
        px, py = player.x, player.y
        gx, gy = tuple(wp)
        dx, dy = gx - px, gy - py

        if abs(dy) >= abs(dx):            # goal lies mainly north/south -> advance on Y, weave on X
            fwd = Direction.SOUTH if dy > 0 else Direction.NORTH
            lat_pos, lat_neg = Direction.EAST, Direction.WEST
        else:                             # goal mainly east/west -> advance on X, weave on Y
            fwd = Direction.EAST if dx > 0 else Direction.WEST
            lat_pos, lat_neg = Direction.SOUTH, Direction.NORTH

        def nb(d):
            return (px + DELTA[d][0], py + DELTA[d][1])
        def walkable(d):
            return tiles.get(nb(d)) != WALL and nb(d) not in avoid
        def grass(d):
            return walkable(d) and terr.get(nb(d)) == "grass"

        self._farm_age += 1

        # Wall straight ahead: let the goal-directed router find the way around (it still favours grass).
        if not walkable(fwd):
            d = policy_first_step(self.world, player.map_id, (px, py), tuple(wp), "farm-exp", avoid)
            self._farm_last = None
            return MoveAction(direction=d) if d is not None else None

        # Never two laterals in a row -> forward is open here, so ADVANCE. Guarantees net forward motion.
        if self._farm_last in (lat_pos, lat_neg):
            self._farm_last = fwd
            return MoveAction(direction=fwd)

        # Weave a lane every ~3rd step, or whenever the forward tile isn't grass (so we grind on the way):
        # take ONE lateral step onto grass, preferring the current lane sign, then flip the sign (zig-zag).
        if self._farm_age % 3 == 0 or not grass(fwd):
            for lat in ((lat_pos, lat_neg) if self._farm_weave > 0 else (lat_neg, lat_pos)):
                if grass(lat):
                    self._farm_weave = -self._farm_weave
                    self._farm_last = lat
                    return MoveAction(direction=lat)

        # Advance toward the goal (through grass when it is grass; forward is open regardless).
        self._farm_last = fwd
        return MoveAction(direction=fwd)

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

    def _bfs_full_collision(self, player, xy, blocked_dirs, occupied):
        """First step of a BFS to ``xy`` over the FULL current-map collision (ground truth from RAM),
        so a portal deep in a maze is always reachable — unlike the learned-map / greedy pathers that
        stall at maze walls. None if no route or the emulator collision is unreadable."""
        try:
            coll = read_collision_map(self.controller.emu)
        except Exception:
            coll = None
        if not coll:
            return None
        goal = (int(xy[0]), int(xy[1]))
        start = (int(player.x), int(player.y))
        if start == goal:
            return None
        walk = set(coll["walkable"]) | {goal}   # the door tile may be off the walkable set
        blocked = set(occupied) - {goal}
        cuts = (getattr(self.world, "cut_edges", None) or {}).get(getattr(player, "map_id", None), set())
        prev: dict[tuple[int, int], tuple[tuple[int, int], Direction] | None] = {start: None}
        q = deque([start])
        found = False
        while q:
            cur = q.popleft()
            if cur == goal:
                found = True
                break
            cx, cy = cur
            for d, (dx, dy) in DELTA.items():
                nb = (cx + dx, cy + dy)
                if nb in walk and nb not in prev and nb not in blocked \
                        and frozenset({cur, nb}) not in cuts:     # elevation edge (tile-pair collision)
                    prev[nb] = (cur, d)
                    q.append(nb)
        if not found:
            return None
        step, first = goal, None
        while prev[step] is not None:
            first = prev[step][1]
            step = prev[step][0]
        if first is not None and first.value not in blocked_dirs:
            return MoveAction(direction=first)
        return None

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
        facing. WHICH sprite is decided by `targets.select_npc` (pure): a candidate pool (name matches,
        else the wanted kind — items for GRAB_ITEM, people otherwise), then the leg's cached pick tracked
        BY LOCALITY (only if made on this map), Jev's calibrated pick among several candidates (only for
        an unnamed/unmatched request, only when confident), else the nearest not-yet-talked. The pick is
        cached as [x, y, map_id] so Jev fires ~once per leg and we keep following a moving person —
        and a pick cached on another map (e.g. a torn warp frame) can never drag us onto the wrong sprite."""
        player = obs.player
        npcs = [n for n in ((obs.game_state or {}).get("npcs") or []) if "x" in n and "y" in n]
        if not npcs:
            return self._leave_via_nearest_exit(player, obs, blocked_dirs, occupied)
        sprite = target.get("sprite") if isinstance(target, dict) else target
        picked = target.get("picked") if isinstance(target, dict) else None
        want_kind = "item" if (directive is not None and directive.intent == Intent.GRAB_ITEM) else "person"
        named = bool(name_matches(npcs, sprite))
        if sprite and not named and picked is None:   # once per target (before a pick is cached)
            self.on_event("approach_npc_miss", {"step": self.session.step, "sprite": sprite,
                                                "seen": [n.get("sprite") for n in npcs]})

        def jev_pick(pool):
            if getattr(self.reasoner, "choose_npc", None) is None:
                return None
            cands = [{"sprite": n.get("sprite"), "x": int(n["x"]), "y": int(n["y"]),
                      "talked_to": bool(n.get("talked_to"))} for n in pool]
            idx, conf = self.reasoner.choose_npc(
                objective=(directive.reason if directive else ""), candidates=cands)
            if idx is None or conf < JEV_NPC_CONF:   # trust the calibrated pick only when confident
                return None
            self.on_event("npc_pick", {"step": self.session.step, "sprite": pool[idx].get("sprite"),
                                       "conf": round(float(conf), 2), "n": len(pool)})
            return pool[idx]

        # rotation (F2) is for steps that are ABOUT an NPC (talk / grab); a travel step that the
        # proposer happened to aim at an NPC must never be judged or wedged by conversations
        talk_step = directive is not None and directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM)
        if isinstance(target, dict) and talk_step:
            self._judge_last_talk(target, directive)
            picked = target.get("picked")
        tried = target.get("tried") if isinstance(target, dict) and talk_step else None
        npc = select_npc(npcs, sprite=sprite, picked=picked, player=player, want_kind=want_kind,
                         chooser=None if named else jev_pick, tried=tried)
        if npc is None:
            # every candidate here was talked to (twice) without achieving the step: hand it to L1
            if talk_step and directive.quest_id is not None:
                who = sorted({str(n.get("sprite")) for n in npcs if n.get("kind") != "item"})
                self._mark_step(directive.quest_id, "wedged",
                                reason=f"talked to everyone here ({', '.join(who)}); none satisfied "
                                       f"{directive.success}")
                self._l1_event = True
            return None
        if isinstance(target, dict):   # cache BY POSITION + MAP (works for spriteless/moving npcs)
            target["picked"] = [int(npc["x"]), int(npc["y"]), getattr(player, "map_id", None)]
        self._talk_npc = npc_key(npc, getattr(player, "map_id", None))

        nx, ny = int(npc["x"]), int(npc["y"])
        adj = {Direction.NORTH: (nx, ny + 1), Direction.SOUTH: (nx, ny - 1),
               Direction.EAST: (nx - 1, ny), Direction.WEST: (nx + 1, ny)}  # tile you stand on to face npc
        # COUNTER TALK: an NPC behind a real COUNTER tile (nurse, Mart clerk) can't be stood next to —
        # the adjacent tile IS the counter. You talk to them from 2 tiles away in a straight line,
        # over the counter. When the intervening cell is an actual counter (RAM's per-tileset talk-over
        # tiles, never a generic wall), the stand tile is that 2-away cell instead of the (blocked) one.
        counters = getattr(self.world, "counters", {}).get(getattr(player, "map_id", None), set())
        far = {Direction.NORTH: (nx, ny + 2), Direction.SOUTH: (nx, ny - 2),
               Direction.EAST: (nx - 2, ny), Direction.WEST: (nx + 2, ny)}
        counter_dirs = set()
        for d, mid in list(adj.items()):
            if mid in counters:
                adj[d] = far[d]        # reach the counter NPC from across the counter
                counter_dirs.add(d)
        facing_map = {"north": Direction.NORTH, "south": Direction.SOUTH,
                      "east": Direction.EAST, "west": Direction.WEST}
        for d, stand in adj.items():
            if (player.x, player.y) == stand:
                facing_ok = facing_map.get(getattr(player, "facing", None)) == d
                if d in counter_dirs:
                    # COUNTER TALK (nurse/clerk): arriving on the tile already facing the counter is
                    # NOT enough — the game only registers a talk-over-counter after you BUMP the
                    # counter (a blocked step into it). Bump once per leg, then interact; once the
                    # dialog opens, dialog-mode drives the rest.
                    if not facing_ok or not self._counter_bumped:
                        self._counter_bumped = True
                        return MoveAction(direction=d)   # blocked step into the counter -> bump/turn
                    return self._interact_with(target)
                if facing_ok:
                    return self._interact_with(target)   # adjacent AND facing -> talk
                return MoveAction(direction=d)        # adjacent, turn to face (a blocked step turns you)
        others = occupied - {(nx, ny)}
        # try each of the 4 stand-tiles nearest-first; take the first BFS-reachable one (cheap, and
        # avoids burning a re-propose cycle when the single nearest stand-tile happens to be a wall).
        for stand in sorted(adj.values(), key=lambda c: abs(c[0] - player.x) + abs(c[1] - player.y)):
            mv = self._bfs_move(player, stand, interact=False, blocked_dirs=blocked_dirs, occupied=others)
            if mv is not None:
                return mv
        return None

    def _interact_with(self, target) -> InteractAction:
        """Press A at the picked NPC, remembering who/when so the NEXT navigate step can judge whether
        that conversation achieved the step (F2 rotation)."""
        if isinstance(target, dict) and getattr(self, "_talk_npc", None) is not None:
            target["talking_to"] = list(self._talk_npc)
            target["talk_step"] = self.session.step
        return InteractAction()

    def _judge_last_talk(self, target: dict, directive) -> None:
        """F2: after a conversation with the picked NPC ends, if the step's success still doesn't hold,
        count it as unproductive; at 2 (some NPCs must be talked to twice) mark that NPC tried and drop
        the pick so select_npc moves on. Skipped for verify: criteria (the judge is throttled)."""
        tk = target.get("talking_to")
        if not tk or self._last_dialog_step <= target.get("talk_step", 10**9):
            return
        target.pop("talking_to", None)
        target.pop("talk_step", None)
        if directive is None or "verify" in (directive.success or {}) or self._directive_satisfied(directive):
            return
        talks = target.setdefault("talks", {})
        key = ",".join(str(v) for v in tk)
        talks[key] = talks.get(key, 0) + 1
        if talks[key] >= 2:
            target.setdefault("tried", []).append(list(tk))
            target["picked"] = None
            self.on_event("npc_rotate", {"step": self.session.step, "tried": list(tk), "talks": talks[key]})

    def _resolve_target(self, target, directive, obs, blocked_dirs, occupied):
        """Turn a typed target ({tile|exit|enter|approach_npc}) into ONE move. Returns a MoveAction /
        InteractAction, or None when even this target can't make progress (caller then unsticks)."""
        if not target:
            return None
        kind = target.get("kind")
        player = obs.player
        if kind == "grind":
            from .routing import grind_step
            if directive is not None and self._grind_qid != directive.quest_id:
                self._grind_qid, self._grind_start, self._grind_last = directive.quest_id, self.session.step, None
            d = grind_step(self.world, player.map_id, (player.x, player.y), self._grind_last,
                           avoid=occupied, blocked=blocked_dirs)
            self._grind_last = d
            return MoveAction(direction=d) if d is not None else None
        if kind == "tile":
            xy = (int(target["x"]), int(target["y"]))
            door = next((e for e in (obs.exits or []) if (int(e["x"]), int(e["y"])) == xy), None)
            if (player.x, player.y) == xy and target.get("hop"):
                # ledge take-off (K8): hop in the ledge direction (bypasses the routing ledge veto for
                # this one move) and drop the held target — the hop lands in another component of the
                # SAME map, so _current_target would otherwise walk us back up to the take-off cell
                self._target = None
                return MoveAction(direction=Direction(target["hop"]))
            if (player.x, player.y) == xy:
                # a model-named tile that IS an exit door: step THROUGH the warp (the model said "leave
                # via (4,11)" — honor it), don't just stop on the doormat.
                if door is not None:
                    d = self._warp_exit_dir(xy, obs.map_dims)
                    return MoveAction(direction=d) if d is not None else None
                # a map-EDGE opening L2 routed to (a boundary tile, no warp): step OFF the edge to
                # cross to the adjacent map — the router owns the crossing, L2 only named the tile.
                if obs.map_dims:
                    w, h = obs.map_dims
                    if xy[0] <= 0 or xy[1] <= 0 or xy[0] >= w - 1 or xy[1] >= h - 1:
                        d = self._warp_exit_dir(xy, obs.map_dims)
                        if d is not None:
                            self._edge_attempt = (player.map_id, xy[0], xy[1])  # so a dead-end edge can't loop
                            return MoveAction(direction=d)
                        return None
                return InteractAction() if target.get("interact") else None
            avoid = set(occupied)
            if target.get("portal"):
                # a ground-truth portal deep in a maze (e.g. the forest north gate): solve it with a
                # BFS over the FULL current-map collision, not the greedy policy pather that stalls.
                mv = self._bfs_full_collision(player, xy, blocked_dirs, avoid)
                if mv is not None:
                    return mv
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

    def _portal_route(self, player, tmap) -> list[dict] | None:
        """The full ground-truth portal route from the player to map ``tmap`` (None if off-graph)."""
        pg = self.portals
        if pg is None or player is None or player.map_id not in pg.maps or int(tmap) not in pg.maps:
            return None
        try:
            coll = read_collision_map(self.controller.emu)
        except Exception:
            coll = None
        comp = pg.component_at(player.map_id, player.x, player.y, coll["walkable"] if coll else set())
        return pg.route(player.map_id, comp, int(tmap)) if comp is not None else None

    def _travel_on_the_way(self, obs) -> bool:
        """The active TRAVEL step's map lies on the ground-truth route to the NEXT travel step's map, so
        it adds nothing and can mislead: a map id can cover separate areas (Route 4's two halves), and
        'go to Route 4' from inside Mt. Moon walks back out the entrance. Skip it; route to the next."""
        d = self._directive
        if d is None or d.intent != Intent.TRAVEL or set(d.success or {}) != {"on_map"} or not self._quest:
            return False
        nxt = self._quest[0]
        if nxt.intent != Intent.TRAVEL or nxt.quest_id == d.quest_id or set(nxt.success or {}) != {"on_map"}:
            return False
        r = self._portal_route(obs.player, nxt.success["on_map"])
        return bool(r) and any(p["dest_map"] == d.success["on_map"] for p in r[:-1])

    def _portal_next(self, player, tmap) -> dict | None:
        """The next PORTAL to head toward on the way to map ``tmap``, from the ground-truth
        PortalGraph (or None if it doesn't cover this leg). Locates the player's walkable component
        from the LIVE collision map so it stays correct even if map state changed."""
        pg = self.portals
        if pg is None or player is None or player.map_id not in pg.maps or int(tmap) not in pg.maps:
            return None
        try:
            coll = read_collision_map(self.controller.emu)
        except Exception:
            coll = None
        walk = coll["walkable"] if coll else set()
        comp = pg.component_at(player.map_id, player.x, player.y, walk)
        if comp is None:
            return None
        portal = pg.next_portal(player.map_id, comp, int(tmap))
        # A warp can sit on a NON-walkable door tile (you cannot step onto it — the move just fails).
        # When the chosen portal's tile isn't walkable, prefer a sibling warp to the same destination
        # whose tile IS walkable (e.g. the south gate has (4,0) unwalkable + (5,0) walkable -> forest).
        if portal is not None and walk and tuple(portal["coord"]) not in walk:
            sibs = [p for p in pg.portals_on(player.map_id)
                    if p["dest_map"] == portal["dest_map"] and tuple(p["coord"]) in walk
                    and p["component"] == portal["component"] and p["kind"] == portal["kind"]]
            if sibs:
                portal = sibs[0]
        return portal

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
            self._cap_det("servo_move",
                          {"player": {"x": getattr(player, "x", None), "y": getattr(player, "y", None)},
                           "target": [int(xy[0]), int(xy[1])], "interact": interact},
                          {"direction": prim.direction.value})
            return prim
        return None

    def _warp_back_map(self, cur_map: int) -> int | None:
        """Where a 0xFF ("LAST_MAP") warp leads: the game's own wLastMap (the last OUTDOOR map — so a
        gate's far doors lead on, not back where we came from), else the previous distinct map."""
        try:
            last = self.controller.emu.read_memory(WLASTMAP)
        except Exception:
            last = None
        if last is not None and last not in (WARP_BACK, cur_map) and last < 0xF8:
            return last
        if self._prev_map is not None and self._prev_map != cur_map:
            return self._prev_map
        return None

    def _resolve_exits(self, exits, cur_map: int) -> list[dict]:
        """Resolve each warp's dest: 0xFF ('return to last map') -> the map we came from, so
        building doors become real, routable graph edges (the world model knows where they go)."""
        out: list[dict] = []
        back = self._warp_back_map(cur_map)
        for e in (exits or []):
            dest = e.get("dest_map")
            if dest == WARP_BACK and back is not None:
                dest = back
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

    def _route_flow(self, obs, ctx) -> str:
        """Which control FLOW is active (battle handled upstream): 'menu' | 'dialogue' | 'navigate'.

        Menu is deterministic — the RAM cursor-arrow signal (A on a menu SELECTS, so a menu must never
        reach the dialogue/A-mash path). The genuinely fuzzy boundary — a text box is up vs the overworld
        is free — is a calibrated Jev choice fed the RAW decoded text (so it sees an all-lowercase line
        the old has_upper heuristic would have zeroed). When Jev isn't wired (e.g. --decider llm), or it
        errors, fall back to the deterministic ctx_kind detector. Low confidence -> _safe_flow."""
        menu = ctx.get("menu") or {}
        raw = (ctx.get("screen_text_raw") or "").strip()
        choose = getattr(self.reasoner, "choose_flow", None)
        if choose is None:                                   # no Jev -> deterministic fallback
            if menu.get("open"):
                return "menu"
            return "dialogue" if ctx.get("kind") == "dialog" else "navigate"
        if not raw and not menu.get("open"):                 # no text at all -> no box -> skip the Jev call
            return "navigate"
        try:
            last = getattr(getattr(self._prev, "action", None), "type", "") or ""
            ans = choose(screen_text=raw, has_text=bool(raw),
                         text_box_id=int(ctx.get("text_box_id") or 0), last_action=last)
        except Exception:                                    # Jev call failed -> deterministic fallback
            if menu.get("open"):
                return "menu"
            return "dialogue" if ctx.get("kind") == "dialog" else "navigate"
        d_ans, d_conf = ans.get("dialogue", ("no", 0.0))
        m_ans, m_conf = ans.get("menu", ("no", 0.0))
        if menu.get("open") or (m_ans == "yes" and m_conf >= FLOW_MIN_CONF):   # menu first (RAM or Jev)
            return "menu"
        if d_ans == "yes" and d_conf >= FLOW_MIN_CONF:
            return "dialogue"
        if max(d_conf, m_conf) < FLOW_MIN_CONF:              # Jev hedged -> safe default
            return self._safe_flow(ctx)
        return "navigate"

    @staticmethod
    def _safe_flow(ctx) -> str:
        """Low-confidence fallback (menu handled upstream): fail toward closing a box. Raw text present
        -> dialogue (advancing a real box clears the stall; A on the overworld is a cheap, self-
        correcting no-op — far safer than walking into an unread box forever); else navigate."""
        return "dialogue" if (ctx.get("screen_text_raw") or "").strip() else "navigate"

    @staticmethod
    def _shop_item_qty(directive: Directive) -> tuple[str | None, int]:
        """What to buy: the item NAME and quantity, from the directive's target ({item, qty}) or the
        ``has_item:<X>`` success predicate. NOTE: a parsed has_item predicate stores the item as an
        integer ID (resolve_item_id), so map it back to a NAME here — the shop macro matches names/ids
        against the live shelf and a bare numeric string matches nothing."""
        from ..games.pokemon_red.constants import ITEMS
        tgt = directive.target or {}
        raw = tgt.get("item")
        count = (directive.success or {}).get("item_count")
        if raw is None and isinstance(count, (list, tuple)) and len(count) == 2:
            raw = count[0]                      # has_item:<name>>=N — the qty is resolved against the bag
        if raw is None:
            raw = (directive.success or {}).get("has_item")
        if isinstance(raw, int):
            item = ITEMS.get(raw)
        elif isinstance(raw, str) and raw.isdigit():
            item = ITEMS.get(int(raw))
        else:
            item = str(raw) if raw else None
        try:
            qty = max(1, int(tgt.get("qty") or 1))
        except (TypeError, ValueError):
            qty = 1
        return item, qty

    @staticmethod
    def _bag_qty(items: list[dict], name: str) -> int:
        from ..games.pokemon_red.game_state import resolve_item_id
        want = resolve_item_id(name)
        return sum(int(it.get("qty") or 0) for it in items
                   if (want is not None and resolve_item_id(str(it.get("item"))) == want)
                   or str(it.get("item")).lower() == str(name).lower())

    def _maybe_shop(self, obs, shot) -> ActionResult | None:
        """When the Mart's BUY/SELL/QUIT counter menu is open AND the active directive names an item
        to acquire, run the deterministic buy macro (design §6.1) instead of letting Jev flail through
        the menu. The trigger is model-driven: L1 decides to shop by adding a step to talk to the Mart
        clerk with ``done_when has_item:<item>`` (compiled to a routed TALK_TO / SHOP) — this fires the
        buy once that talk opens the counter. Guarded by the unambiguous RAM shop-menu signal + a
        resolvable item, so it never fires on a non-shopping menu."""
        from ..games.pokemon_red import shop as shop_macro
        from ..core.models import WaitAction
        emu = self.controller.emu
        if not shop_macro.at_shop_menu(emu):
            return None
        d = self._directive
        item, qty = self._shop_item_qty(d) if d is not None else (None, 1)
        step = self._step_by_qid(d.quest_id) if d is not None else None
        satisfied = d is not None and self._directive_satisfied(d)
        if (item is None and d is not None and d.intent == Intent.TALK_TO and not satisfied
                and step is not None and step.status == "active"):
            # talking to a clerk for a step that never names WHAT to buy (e.g. verify:"at least 4
            # Potions?") would reopen the counter forever — hand it back to L1 with the fix
            self._wedge_active("at the Mart counter but the step names no item to buy — use "
                               "has_item:<item> or has_item:<item>>=N")
        if item is None or satisfied or (step is not None and step.status in ("done", "wedged")):
            # The counter is open but there is nothing (left) to buy — e.g. the clerk's "anything
            # else?" reopened it after a failed buy. Back out deterministically so the step can
            # advance and L1 can re-plan, instead of re-entering the same buy forever.
            mode_before = detect_mode(emu)
            shop_macro.close_shop(emu)
            self.on_event("shop_closed", {"step": self.session.step, "item": item, "satisfied": satisfied,
                                          "step_status": step.status if step else None})
            rstep = ReasonStep(location="mart", objective="leave the counter",
                               reasoning="SHOP: nothing to buy here; closing the counter",
                               action=WaitAction(frames=1))
            self._prev = rstep
            self._emit_reason(rstep, 0)
            return self._finish(obs, rstep, ActionResult(success=True, result="completed", mode_before=mode_before,
                                                         mode_after=detect_mode(emu), detail="shop closed"),
                                0, {}, shot)
        from ..games.pokemon_red.game_state import read_items, read_money
        mode_before = detect_mode(emu)
        held_before = self._bag_qty(read_items(emu), item)
        count = (d.success or {}).get("item_count")
        if isinstance(count, (list, tuple)) and len(count) == 2:
            qty = max(1, int(count[1]) - held_before)   # buy only what's missing
        res = shop_macro.shop_buy(emu, item, qty)
        ok = bool(res.get("ok"))
        verified = None
        if ok:
            # the macro only knows it answered YES: VERIFY the item actually reached the bag (can't
            # afford -> "not enough money", the bag is unchanged but the macro still says ok)
            verified = self._bag_qty(read_items(emu), item) > held_before
            if not verified:
                ok = False
                res = {**res, "ok": False, "reason": f"purchase did not go through (money ${read_money(emu)})"}
        self.on_event("shop_buy", {"step": self.session.step, "item": item, "qty": qty, "result": res,
                                   "verified": verified})
        self.stuck.note_spend(read_money(emu))   # money spent at a Mart is not a whiteout setback
        if not ok:
            # A definitive failure (item not on this shelf, can't afford) will NOT self-resolve — the
            # has_item goal can never be met here, so re-firing the macro every step would thrash the
            # counter. Wedge the step so L1 drops/replaces it (e.g. picks a Mart that stocks the item),
            # and force a re-plan.
            shelf = res.get("shop")
            why = f"can't buy {item} here: {res.get('reason')}" + (f" (shelf: {', '.join(shelf)})" if shelf else "")
            self._mark_step(d.quest_id, "wedged", reason=why)
            self._l1_event = True     # the L1 gate (not the legacy reflect flag) replaces the step
        rstep = ReasonStep(location="mart", objective=f"buy {qty}x {item}",
                           reasoning=f"SHOP macro: {res.get('reason', 'purchased')}",
                           action=WaitAction(frames=1))
        self._prev = rstep
        self._emit_reason(rstep, 0)
        result = ActionResult(success=ok, result="completed" if ok else "blocked",
                              mode_before=mode_before, mode_after=detect_mode(emu),
                              detail=f"shop_buy {item} x{qty}: {res.get('reason', 'ok')}")
        return self._finish(obs, rstep, result, 0, {}, shot)

    # ------------------------------------------------ conversation / script gate (F5)
    def _script_active(self) -> bool:
        """True while a game script owns the controls (wJoyIgnore != 0) — input is ignored."""
        try:
            return self.controller.emu.read_memory(WJOYIGNORE) != 0
        except Exception:
            return False

    def _in_conversation(self) -> bool:
        """A script owns the controls, or we're within CONVO_GRACE_STEPS of a step ROUTED to dialogue
        (covers the brief no-text gap between two text boxes). Deliberately NOT the raw text
        detector: when the flow router confidently routes a text read to navigate, trust the router."""
        return self._script_active() or (self.session.step - self._last_dialog_step) <= CONVO_GRACE_STEPS

    def _note_dialogue_step(self) -> None:
        """This step is routed to dialogue: the conversation is progressing (not a stall)."""
        self._last_dialog_step = self.session.step
        self._forced_waits = 0

    def _should_defer_to_script(self) -> bool:
        """Should a would-be navigation step WAIT instead (conversation gap / script in control)?
        Bounded: FORCED_WAIT_MAX_STEPS consecutive forced waits disarm the gate (event
        `script_wait_timeout`) until the script flag clears and no dialogue has been routed within the
        grace window — so a stuck flag or a false-positive text read can never stall the run."""
        script = self._script_active()
        grace = (self.session.step - self._last_dialog_step) <= CONVO_GRACE_STEPS
        if not self._forced_wait_armed:
            if not script and not grace:
                self._forced_wait_armed = True     # the condition cleared -> re-arm for next time
            self._forced_waits = 0
            return False
        if not self._in_conversation():
            self._forced_waits = 0
            return False
        if self._forced_waits >= FORCED_WAIT_MAX_STEPS:
            self.on_event("script_wait_timeout", {"step": self.session.step, "waits": self._forced_waits,
                                                  "script_active": script})
            self._forced_wait_armed = False
            self._forced_waits = 0
            return False
        self._forced_waits += 1
        return True

    def _forced_wait(self, obs, shot) -> ActionResult:
        """Let the game's script / next text box run: a short wait, no planning, no moves."""
        from ..core.models import WaitAction
        rstep = ReasonStep(location=getattr(obs.player, "map_name", "") or "",
                           objective=(self._directive.reason if self._directive else "wait"),
                           reasoning="waiting: conversation/script in progress",
                           action=WaitAction(frames=12))
        result = self.controller.execute(rstep.action)
        self._prev = rstep
        self._emit_reason(rstep, 0)
        return self._finish(obs, rstep, result, 0, {}, shot)

    def _report_blockers(self, obs, target) -> bool:
        """A portal route that exists on the map but is closed by objects/NPCs (e.g. the two fossils
        filling Mt. Moon B2F's corridor, a trainer in a doorway): wedge the step at once with the
        blockers named, so L1 can decide (pick one up / talk / battle / go around) instead of the
        executor waiting forever. False when the route isn't blocked by objects."""
        from .routing import route_blockers
        player = obs.player
        try:
            coll = read_collision_map(self.controller.emu)
        except Exception:
            coll = None
        if not coll or player is None:
            return False
        occ = {(int(n["x"]), int(n["y"])): str(n.get("sprite") or "someone")
               for n in ((obs.game_state or {}).get("npcs") or []) if "x" in n and "y" in n}
        cuts = (getattr(self.world, "cut_edges", None) or {}).get(player.map_id, set())
        blockers = route_blockers(coll["walkable"], (player.x, player.y),
                                  (int(target["x"]), int(target["y"])), occ, cuts)
        if not blockers:
            return False
        who = ", ".join(f"{name} at ({x},{y})" for name, (x, y) in blockers)
        self._wedge_active(f"the route {target.get('note') or 'to the target'} on {map_name(player.map_id)} is "
                           f"blocked by {who} — deal with them (pick up / talk / battle) or go around")
        self.on_event("route_blocked", {"step": self.session.step, "blockers": [list(c) + [n] for n, c in blockers]})
        return True

    @staticmethod
    def _is_grind(directive) -> bool:
        """A TRAVEL whose success is purely a level threshold (a grind step)."""
        s = (directive.success or {}) if directive is not None else {}
        return directive is not None and directive.intent == Intent.TRAVEL and bool(s) and set(s) == {"level"}

    def _grinding(self) -> bool:
        d = self._directive
        return (d is not None and self._is_grind(d) and self._grind_qid == d.quest_id
                and bool(self._target) and self._target.get("kind") == "grind")

    def _wedge_active(self, reason: str) -> None:
        """Wedge the active step with a specific reason (shown to L1 as why_wedged) and arm L1."""
        d = self._directive
        if d is not None and d.quest_id is not None:
            self._mark_step(d.quest_id, "wedged", reason=reason)
        self._grind_qid = None
        self._l1_event = True

    def _note_battle_end(self, *, wild: bool) -> None:
        if wild:
            self._last_wild_battle_end = self.session.step

    def _apply_stuck_to_budget(self, stuck, *, frozen: bool) -> None:
        """Feed one step's stuck verdict into the block budget (BLOCK_TRIGGER wedges the step).
        A conversation/script step is FROZEN — neither counted nor treated as progress — so a
        dialogue loop (re-talking a blocking NPC) still accumulates across its overworld steps."""
        if frozen:
            return
        if self._grinding():
            # pacing grass looks like a loop and levels come slowly: while wild battles keep coming the
            # grind IS progress; after a full window with no wild encounter, hand it back to L1
            since = max(self._grind_start, self._last_wild_battle_end or 0)
            if self.session.step - since > GRIND_ENCOUNTER_WINDOW:
                self._wedge_active(f"grind: no wild encounter in {GRIND_ENCOUNTER_WINDOW} steps on "
                                   f"{map_name(self.controller.emu.read_memory(0xD35E))}")
            self._blocked_for_n = 0
            return
        if stuck.stuck:
            # a local loop / no-objective-progress leg counts toward the block budget that feeds
            # the wedge trigger + the L1 gate (BLOCK_TRIGGER). This is the get-unstuck signal:
            # circling accumulates here until it wedges the step and L1 re-plans it.
            self._blocked_for_n += 1
        else:
            self._blocked_for_n = 0   # genuine progress this leg -> clear the block budget

    def _advance_dialog(self, obs, shot) -> ActionResult:
        from ..core.models import AdvanceDialogAction
        action = AdvanceDialogAction()
        result = self.controller.execute(action)
        rstep = ReasonStep(location="dialog", objective="advance text",
                           tried="", reasoning="advancing passive dialog", action=action)
        self._prev = rstep
        self._emit_reason(rstep, 0)
        return self._finish(obs, rstep, result, 0, {}, shot)

    def _candidate_exits(self, obs, reachable) -> list[dict]:
        """Every reachable WAY OFF this map, as coordinates L2 can route to — so it SELECTS a specific
        exit instead of us guessing the nearest door. Two kinds, both deterministic from RAM (never
        guessed): warp DOORS (from obs.exits, with their known destination map) and map-EDGE openings
        (reachable walkable tiles sitting on the map boundary — the tiles you can step off to reach the
        adjacent map). L2 may pick one of these or any other walkable tile; they are hints, not a menu."""
        out: list[dict] = []
        doors: set[tuple[int, int]] = set()
        for e in (obs.exits or []):
            try:
                x, y = int(e["x"]), int(e["y"])
            except (KeyError, TypeError, ValueError):
                continue
            doors.add((x, y))
            out.append({"x": x, "y": y, "kind": "door",
                        "dest_map": e.get("dest_map"), "dest": e.get("dest_name")})
        dims = getattr(obs, "map_dims", None)
        if dims and reachable:
            w, h = dims
            for (x, y) in reachable:
                if (x, y) in doors:
                    continue
                d = ("N" if y <= 0 else "S" if y >= h - 1 else
                     "W" if x <= 0 else "E" if x >= w - 1 else None)
                if d is not None:
                    out.append({"x": x, "y": y, "kind": "edge", "dir": d})
        return out

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
        # ground truth first: the PortalGraph's next portal (the WorldGraph only knows map-level edges
        # and e.g. vetoed the Mt. Moon door because it thinks Route 4 connects straight to Cerulean)
        portal = self._portal_next(obs.player, directive.target_map)
        if portal is not None and portal.get("dest_map") is not None:
            return portal["dest_map"]
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
        if obs.player is not None:   # a step across an elevation (tile-pair) edge is never legal
            here = (obs.player.x, obs.player.y)
            cuts = (getattr(self.world, "cut_edges", None) or {}).get(obs.player.map_id, set())
            for d, (dx, dy) in DELTA.items():
                if frozenset({here, (here[0] + dx, here[1] + dy)}) in cuts:
                    blocked.add(d.value)
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
        # standing still through a conversation/script is not being stuck (F5) — suppress the
        # detector AND freeze (not reset) the block budget, so a dialogue LOOP stays escapable
        # decided at step START (a step whose own action triggers a script is NOT frozen — it's the
        # one overworld step per dialogue-loop cycle that keeps the loop escapable); a step routed to
        # dialogue stays frozen even while the forced-wait gate is disarmed
        convo = self._step_dialogue or (self._forced_wait_armed and self._step_convo)
        if not self._step_forced_wait:
            self._forced_waits = 0   # ANY step that isn't a forced wait (dialogue, menu, battle, nav)
        stuck = self.stuck.update(
            rstep.action, result, obs.player, shot if self.vision else None,
            progress=pv, forced_movement=suppress or convo,
            objective_distance=self._objective_distance(obs.player),
        )
        if stuck.setback:
            self.memory.note(f"setback at step {self.session.step}", source="observed", step=self.session.step)
        self._apply_stuck_to_budget(stuck, frozen=convo and not suppress)
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
                "mission": (self._plan.mission if self._plan else None),
                "milestone": (self._plan.milestone if self._plan else None),
                "goals": (self._plan.goals.model_dump() if self._plan else None),
                "interrupted": (self._plan.interrupted.model_dump()
                                if self._plan and self._plan.interrupted.text else None),
                "notepad_len": (len(self._plan.notepad) if self._plan else 0),
                "plan_steps": [{"id": s.id, "map": s.map, "status": s.status,
                               "done_when": s.done_when, "kind": s.kind, "why": s.why,
                               "talk": s.talk, "who": s.who}
                               for s in self._plan_steps],
                "l1_last": self._l1_last,
                "quest_remaining": [q.reason for q in self._quest],
                "waypoint": ([tgt["x"], tgt["y"]] if tgt and tgt.get("kind") == "tile" else None),
                "target": tgt,   # the unified mid-level typed target (kind/x/y/map/sprite)
                "goal_map": self.goal_map,
                "routing_policy": (self._policy if self.pather == "policy" else self.pather),
            }
            if self._notepad_changed and self._plan is not None:
                extra["notepad"] = self._plan.notepad   # full text only on the step it changed
                self._notepad_changed = False
            self.recorder.record(step=self.session.step, obs=obs, action=rstep.action,
                                 result=result, extra=extra)
            self.capture.flush()   # persist this step's captured decisions beside the record
        self.session.step += 1
        if self.checkpoint_every and self.checkpoint_dir and self.session.step % self.checkpoint_every == 0:
            self._checkpoint()
        if self._resume_dir is not None and self.session.step % RESUME_EVERY == 0:
            self.save_resume_checkpoint()
        return result

    # --- battle sub-policy (mode dispatch routes here when in_battle) ---------
    def _sync_battle_goals(self) -> None:
        """Refresh `self._battle_goals` from L1's plan-level goals + the loop's level target
        (design §7.1). Compact shape `{"catch": [...], "level_target": int}`; empty catch ->
        GRIND. Cheap + idempotent, called each step so a plan change is picked up promptly."""
        from ..games.pokemon_red import battle_l2
        pg = getattr(self._plan, "battle_goals", None) if self._plan is not None else None
        self._battle_goals = battle_l2.battle_goals_from_plan(pg, self.level_target)

    def _battle_turn(self, obs):
        """One battle turn under the layered objective model (design §2).

        On the battle-start edge (`in_battle` false->true) run `battle_L2` to cache ONE
        objective — done ABOVE the intro-text early-return so a short intro can't skip it
        (§2.1). Then, once the FIGHT menu is up, Jev picks a typed action toward the cached
        objective and the matching macro executes it. GRIND-EXP is exactly today's fight
        path (choose_move -> use_move), so the working fight is preserved."""
        from ..core.models import AdvanceDialogAction, MenuSelectAction, WaitAction
        from ..games.pokemon_red import battle, battle_actions, battle_agent, battle_l2
        emu = self.controller.emu
        mode_before = detect_mode(emu)

        # --- battle-start edge (ABOVE the intro-text return): set the cached objective now,
        # during intro text, so a very short intro can't skip battle_L2.
        if battle.in_battle(emu) and not self._last_in_battle:
            state = battle_l2.build_state(emu, self._battle_goals)
            self._battle_objective = battle_l2.choose_objective(state, self._battle_goals)
            self._cap_det("battle_l2_objective", {"state": state, "goals": self._battle_goals},
                          {"objective": self._battle_objective})
            self.on_event("battle_objective", {
                "step": self.session.step, "objective": self._battle_objective,
                "enemy": (state.get("enemy") or {}).get("species"),
                "trainer": state.get("is_trainer")})
        self._last_in_battle = battle.in_battle(emu)

        # the party-SWITCH flow ('change POKEMON?' / 'Bring out which?' / 'already out!'): the default A
        # re-picks the active mon forever — decline / back out, or send a healthy mon after a faint
        if battle.switch_screen_showing(emu):
            ok = battle.resolve_switch_screen(emu)
            rstep = ReasonStep(location="battle", objective="resolve the switch screen",
                               reasoning="party-switch prompt: decline / back out (or replace a fainted mon)",
                               action=WaitAction(frames=1))
            return rstep, 0, {}, ActionResult(success=ok, result="completed" if ok else "blocked",
                                              mode_before=mode_before, mode_after=detect_mode(emu),
                                              detail="battle: resolved the switch screen")

        # the MOVE LIST is open without the root menu (e.g. the game refused a 0-PP move): back out
        # with B — pressing A here would just re-select the same move forever
        if not battle.fight_menu_showing(emu) and battle.move_list_showing(emu):
            ok = battle.back_to_fight_menu(emu)
            rstep = ReasonStep(location="battle", objective="back out of the move list",
                               reasoning="move list open without the FIGHT menu; pressing B",
                               action=WaitAction(frames=1))
            return rstep, 0, {}, ActionResult(success=ok, result="completed" if ok else "blocked",
                                              mode_before=mode_before, mode_after=detect_mode(emu),
                                              detail="battle: backed out of the move list")

        # intro / result text: advance until the FIGHT/PKMN/ITEM/RUN menu is interactive.
        if not battle.fight_menu_showing(emu):
            res = self.controller.execute(AdvanceDialogAction())
            rstep = ReasonStep(location="battle", objective="advance battle text",
                               reasoning="advancing battle text", action=AdvanceDialogAction())
            return rstep, 0, {}, res

        # per-turn SAFETY re-eval (design §2.1): keep the cached objective, but if CAPTURE/
        # GRIND would faint us at critical HP, override this turn to SURVIVE (trainer) / ESCAPE
        # (wild) so we heal or flee instead of throwing a ball into a KO.
        state = battle_l2.build_state(emu, self._battle_goals)
        cached = self._battle_objective or battle_l2.GRIND_EXP
        objective = battle_l2.safety_override(cached, state)
        if objective != cached:
            self.on_event("battle_safety_override", {
                "step": self.session.step, "from": cached, "to": objective,
                "hp_frac": round(battle_l2.hp_frac(state.get("active")), 3)})
        action = battle_agent.choose_action(objective, state)
        self._cap_det("battle_choose_action", {"objective": objective, "state": state}, action)
        kind = action.get("kind")

        # --- ESCAPE -> run ---
        if kind == "run":
            r = battle_actions.run(emu)
            return self._battle_result(
                objective, "run", ok=bool(r.get("ok")), mode_before=mode_before,
                events=[f"battle_action:run", f"escaped:{r.get('escaped')}"],
                detail=f"battle[{objective}]: run (escaped={r.get('escaped')})",
                action=MenuSelectAction(index=0, label="battle:run"))

        # --- CAPTURE (target weak) -> throw a ball ---
        if kind == "ball":
            # a SUCCESSFUL catch runs a long tail (wobbles -> "Gotcha!" -> nickname prompt ->
            # added to party); give the macro enough drain to resolve it fully in one turn,
            # so the battle actually ENDS rather than leaving the loop mid-catch-animation.
            r = battle_actions.throw_ball(emu, action["item"], max_advance=150)
            return self._battle_result(
                objective, "ball", ok=bool(r.get("ok")), mode_before=mode_before,
                events=[f"battle_action:ball:{action['item']}", f"caught:{r.get('caught')}"],
                detail=f"battle[{objective}]: throw {action['item']} (caught={r.get('caught')})",
                action=MenuSelectAction(index=0, label=f"battle:ball:{action['item']}"))

        # --- SURVIVE (low HP) -> use a Potion ---
        if kind == "item":
            r = battle_actions.use_item(emu, action["item"])
            return self._battle_result(
                objective, "item", ok=bool(r.get("ok")), mode_before=mode_before,
                events=[f"battle_action:item:{action['item']}",
                        f"hp:{r.get('active_hp_before')}->{r.get('active_hp_after')}"],
                detail=f"battle[{objective}]: use {action['item']}",
                action=MenuSelectAction(index=0, label=f"battle:item:{action['item']}"))

        # --- GRIND-EXP (and any fallback) -> the existing move path, UNCHANGED ---
        client = getattr(self.reasoner, "client", None)
        conf = 0.0
        if client is not None:
            slot, conf = battle_agent.choose_move(client, emu,
                                                  type_knowledge=self._battle_type_knowledge(emu),
                                                  capture=getattr(self, "capture", None))
        else:
            slot = 0
        r = battle.use_move(emu, slot)
        result = ActionResult(
            success=bool(r.get("ok")), result="completed",
            mode_before=mode_before, mode_after=detect_mode(emu),
            events=[f"battle_objective:{objective}", f"battle_action:move",
                    f"battle_move:{r.get('move')}", f"dmg:{r.get('damage_dealt')}"],
            detail=f"battle[{objective}]: {r.get('move')} dealt {r.get('damage_dealt')} (over={r.get('battle_over')})",
        )
        rstep = ReasonStep(location="battle", objective=f"use {r.get('move')}",
                           reasoning=f"[{objective}] battle move {slot} ({r.get('move')}) conf {conf:.2f}",
                           action=MenuSelectAction(index=slot, label=f"move:{r.get('move')}"))
        return rstep, 0, {"confidence": conf}, result

    def _battle_result(self, objective, kind, *, ok, mode_before, events, detail, action):
        """Build the (rstep, latency, usage, result) tuple for a non-move battle macro."""
        result = ActionResult(
            success=ok, result="completed",
            mode_before=mode_before, mode_after=detect_mode(self.controller.emu),
            events=[f"battle_objective:{objective}"] + events, detail=detail,
        )
        rstep = ReasonStep(location="battle", objective=f"battle:{kind}",
                           reasoning=f"[{objective}] {detail}", action=action)
        return rstep, 0, {}, result

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

    def _write_resume(self, dest: Path) -> None:
        """Write a resumable checkpoint pair to `dest`: latest.state (full PyBoy save) + latest.mem.json
        (world map, graph, interactions, reflection plan, map history, blocked edges)."""
        dest.mkdir(parents=True, exist_ok=True)
        self.memory.plan = self._plan
        self.controller.emu.save_state(dest / "latest.state")
        self.memory.save(dest / "latest.mem.json")

    def _checkpoint(self) -> None:
        self._write_resume(self.checkpoint_dir)
        self.on_event("checkpoint", {"step": self.session.step, "dir": str(self.checkpoint_dir)})

    def save_resume_checkpoint(self) -> None:
        """Always-on continuation snapshot into the record-dir (periodic + at run-end). No-op when the
        run isn't being recorded. Lets any run be continued later with `--resume-from <record-dir>`."""
        if self._resume_dir is None:
            return
        try:
            self._write_resume(self._resume_dir)
            self.on_event("checkpoint", {"step": self.session.step, "dir": str(self._resume_dir), "resume": True})
        except Exception as e:   # a checkpoint must never crash the run
            self.on_event("checkpoint_failed", {"step": self.session.step, "error": str(e)})

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
        try:
            for _ in range(max_steps):
                if not self.session.running:
                    break
                self.step_once()
        finally:
            # Always leave a resumable snapshot at the end (budget reached, stop, or crash), so the run
            # can be continued from exactly where it stopped rather than the last new-area state.
            self.save_resume_checkpoint()
            # distillation capture: write run metadata once + label the finished run (best-effort).
            try:
                self.capture.write_run_meta({
                    "goal_map": self.goal_map, "level_target": self.level_target,
                    "pather": self.pather, "l1_every": self.l1_every,
                    "portal_graph": getattr(getattr(self, "portals", None), "version", None),
                })
                if self.capture.enabled and self._resume_dir is not None:
                    from ..logging.outcome import label_run
                    label_run(self._resume_dir, goal_map=self.goal_map)
                self.capture.close()
            except Exception:
                pass


def _desc(action) -> str:
    d = action.model_dump()
    if d.get("type") == "move":
        return f"move {d['direction']} x{d.get('tiles', 1)}"
    if d.get("type") == "goto":
        return f"goto {d.get('label') or (d.get('x'), d.get('y'))}"
    if d.get("type") == "menu_select":
        return f"menu[{d.get('index')}] {d.get('label') or ''}"
    return d.get("type", "?") + (f" {d.get('button')}" if d.get("button") else "")
