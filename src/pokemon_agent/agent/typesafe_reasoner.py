"""A TypeSafe-backed decider — a drop-in for `Reasoner` (see reasoner.py).

Where `Reasoner` asks a generative LLM to *emit* a JSON action, this poses the
same per-step decision as a TypeSafe `Choice`: pick ONE of an enumerated set of
mutually-exclusive actions and get a *calibrated confidence* back. That buys us:

  * no JSON-parse failures / cold-start empties / reasoning-eats-the-budget,
  * ~$0.0002 per decision, fast enough to call every step,
  * confidence as a first-class control signal: exposed in `usage["confidence"]`
    so the loop can trigger an off-cycle reflection when the actor is unsure,
    instead of blindly `wait`-ing (the old confidence->wait gate is opt-in only,
    see `wait_on_low_confidence`, and off by default).

It exposes the SAME `.step()` / `.reflect()` signatures as `Reasoner`, so
`ReasoningLoop` can drive it unchanged. TypeSafe only *classifies* among options
it is given — it cannot write a plan — so `reflect()` is delegated to an optional
generative `Reasoner` (LunaRoute); without one it passes the previous plan through.

The client reads TYPESAFE_API_KEY / TYPESAFE_BASE_URL from the environment.
"""
from __future__ import annotations

import time
from typing import Protocol

from ..core.models import (
    AdvanceDialogAction,
    Direction,
    GoToAction,
    InteractAction,
    MenuSelectAction,
    MoveAction,
    PressAction,
    WaitAction,
)
from ..emulator.interface import GameButton, ImageObservation
from .plan import Directive, Intent
from .reasoner import ReasonStep, ReflectionPlan

# key -> how to build the concrete AgentAction. The keys ARE the Choice criteria.
_KIND_TO_ACTION = {
    "move_north": lambda: MoveAction(direction=Direction.NORTH),
    "move_south": lambda: MoveAction(direction=Direction.SOUTH),
    "move_east": lambda: MoveAction(direction=Direction.EAST),
    "move_west": lambda: MoveAction(direction=Direction.WEST),
    "interact": lambda: InteractAction(),
    "advance_dialog": lambda: AdvanceDialogAction(),
    "wait": lambda: WaitAction(frames=30),
    "press_start": lambda: PressAction(button=GameButton.START),
    "press_b": lambda: PressAction(button=GameButton.B),
}

_KIND_CRITERIA = {
    "move_north": "Step one tile NORTH (y decreases). Toward exits/NPCs/goal that are above you.",
    "move_south": "Step one tile SOUTH (y increases). Toward exits/NPCs/goal below you.",
    "move_east": "Step one tile EAST (x increases). Toward things to your right.",
    "move_west": "Step one tile WEST (x decreases). Toward things to your left.",
    "interact": "Press A on the person/sign/object you are FACING and standing next to "
                "(use when adjacent to and facing an NPC you want to talk to).",
    "advance_dialog": "Continue/close the open text box or accept a menu prompt. "
                      "Use ONLY when DIALOG_ACTIVE is true.",
    "wait": "Do nothing this step. Use when a cutscene / forced movement is playing and "
            "your input is being ignored, or the screen is still loading.",
    "press_start": "Open the START menu.",
    "press_b": "Press B to cancel / back out of a menu or dialog.",
}

_KIND_INSTRUCTIONS = (
    "You are driving Pokémon Red one button-press at a time toward PRIMARY_GOAL, "
    "following CURRENT_PLAN.next_objective. Choose the ONE action that makes the most "
    "progress right now, grounded in the structured state (trust PLAYER x/y, EXITS, and "
    "GAME_STATE over any guess). Rules: "
    "(a) If GAME_STATE.dialog_active is true, do NOT walk away — advance_dialog, or press "
    "A/B / move the cursor to answer. "
    "(b) To reach an EXIT tile, move so PLAYER x,y approaches that exit's x,y; x increases "
    "EAST, y increases SOUTH; route around '#' walls in MAP_VIEW. "
    "(c) To talk to someone, be on a tile ADJACENT to the NPC and facing it, then interact. "
    "(d) Do NOT repeat a move that RECENT shows was just blocked, and if MAP_HISTORY "
    "alternates between two maps you are oscillating — take a different exit. "
    "(e) If nothing you press is changing the screen (a cutscene is moving you), wait. "
    "(f) PREFER a 'goto_*' option to reach a specific person, Poké Ball, object, or exit: "
    "it walks there and (for objects/people) presses A for you, routing around walls — "
    "use the raw move_* directions only for open-ended exploration. "
    "(g) ROUTE_HINT (when present in the state) is the compass direction toward your "
    "current destination map — when you are just traveling/exploring with no better "
    "target, strongly prefer move_<ROUTE_HINT> to head that way (walk to the map's edge)."
)


def _apply_directive(criteria: dict, target_map: dict, directive: Directive) -> dict:
    """Shape the executor's Choice set FROM the directive (spec §5).

    * Inject the directive's concrete target (if it has a tile) as a first-class
      ``goto_directive_target`` option, so "pursue the directive" is a button, not a hope.
    * Mask raw ``move_*`` for reach-a-thing intents (talk_to / grab_item / enter) — but
      ONLY when a target/goto option actually exists, so the model can still explore to
      find a path when the deterministic servo found none (the BFS-impossible residual).
    * Honor ``allowed_options`` as an explicit whitelist (never leaving zero options).
    """
    crit = dict(criteria)
    xy = directive.target_xy
    if xy is not None:
        interact = directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM)
        key = "goto_directive_target"
        label = (directive.target or {}).get("kind") or "directive target"
        desc = (f"Go to the directive target at ({xy[0]},{xy[1]})"
                + (" and press A." if interact else " (step onto it)."))
        target_map[key] = {"key": key, "label": label, "desc": desc,
                           "x": xy[0], "y": xy[1], "interact": interact}
        crit[key] = desc

    has_goto = any(k.startswith("goto_") for k in crit)
    if directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM, Intent.ENTER) and has_goto:
        for d in ("north", "south", "east", "west"):
            crit.pop(f"move_{d}", None)

    if directive.allowed_options is not None:
        whitelisted = {k: v for k, v in crit.items() if k in directive.allowed_options}
        if whitelisted:  # never hand the model an empty Choice set
            crit = whitelisted
    return crit


class GenerativeReasoner(Protocol):
    def reflect(self, **kwargs) -> tuple[ReflectionPlan, int, dict]: ...


class TypeSafeReasoner:
    def __init__(
        self,
        *,
        client=None,
        model: str | None = None,
        reflector: GenerativeReasoner | None = None,
        min_confidence: float = 0.45,
        wait_on_low_confidence: bool = False,
        max_consecutive_waits: int = 2,
    ):
        if client is None:
            from typesafe_sdk import TypeSafeClient

            client = TypeSafeClient(model=model)
        self.client = client
        self.reflector = reflector
        self.min_confidence = min_confidence
        self.wait_on_low_confidence = wait_on_low_confidence
        self.max_consecutive_waits = max_consecutive_waits
        self._low_conf_waits = 0
        self.capture = None    # optional distillation Capture (mounted by the loop; §3)

    def _cap_jev(self, layer, state, ans, confidence, latency_ms=0, extra=None) -> None:
        """Best-effort per-decision capture for a Jev Choice (§3): records the calibrated
        confidence + probabilities. Never raises / never changes behavior."""
        cap = getattr(self, "capture", None)
        if cap is None:
            return
        ex = {"probabilities": getattr(ans, "probabilities", None)}
        if extra:
            ex.update(extra)
        cap.record(layer, model=getattr(self.client, "model", "typesafe"),
                   input=state, output_raw=str(getattr(ans, "choice", None)),
                   parsed={"choice": getattr(ans, "choice", None)},
                   confidence=confidence, latency_ms=latency_ms, extra=ex)

    # --- reflection is generative: delegate or pass through --------------------
    def reflect(self, **kwargs) -> tuple[ReflectionPlan, int, dict]:
        if self.reflector is not None:
            return self.reflector.reflect(**kwargs)
        previous = kwargs.get("previous")
        return (previous or ReflectionPlan(next_objective="explore toward the goal"), 0, {})

    # --- the per-step decision, as a constrained Choice ------------------------
    def step(
        self,
        *,
        primary_goal: str,
        image: ImageObservation | None = None,  # ignored: TypeSafe takes structured state
        local_map: list[str] | None = None,
        map_view: list[str] | None = None,
        player_desc: str = "",
        exits: list[dict] | None = None,
        game_state: dict | None = None,
        social_memory: dict | None = None,
        map_history: list[int] | None = None,
        recent: list[str] | None = None,
        previous: ReasonStep | None = None,
        plan: ReflectionPlan | None = None,
        targets: list[dict] | None = None,
        route_hint: str | None = None,
        blocked_dirs: set[str] | None = None,
        directive: Directive | None = None,
    ) -> tuple[ReasonStep, int, dict]:
        from typesafe_sdk import Choice

        dialog_active = bool(game_state and game_state.get("dialog_active"))
        target_map = {t["key"]: t for t in (targets or [])}
        blocked_dirs = blocked_dirs or set()
        state = {
            "primary_goal": primary_goal,
            "current_plan": plan.model_dump() if plan else None,
            "player": player_desc,
            "game_state": game_state,
            "social_memory": social_memory,
            "map_history": map_history or [],
            "exits": exits or [],
            "local_map": local_map,
            "map_view": map_view,
            "recent": recent or [],
            "previous_notes": (
                {"objective": previous.objective, "tried": previous.tried} if previous else None
            ),
        }
        # --- a selectable menu is open: choose an OPTION, not an overworld action ---
        menu = ((game_state or {}).get("context") or {}).get("menu") or {}
        if menu.get("open"):
            return self._menu_step(menu, state, previous)

        criteria = {**_KIND_CRITERIA, **{k: t["desc"] for k, t in target_map.items()}}
        # DIRECTIVE TEETH (spec §5): the executor's Choice is derived FROM the active
        # directive — inject its target as a first-class option, mask intent-illegal actions,
        # and honor an explicit whitelist. Generalizes the blocked_dirs wall-masking below.
        if directive is not None:
            criteria = _apply_directive(criteria, target_map, directive)
            state["directive"] = {
                "intent": directive.intent.value, "target": directive.target,
                "success": directive.success, "reason": directive.reason,
            }
        # mask out moves into known walls (dead-end ledger) so the model can't re-bash them
        for d in blocked_dirs:
            criteria.pop(f"move_{d}", None)
        state["blocked_directions"] = sorted(blocked_dirs)
        state["available_targets"] = [
            {"option": t["key"], "label": t["label"], "x": t["x"], "y": t["y"]}
            for t in (targets or [])
        ]
        state["route_hint"] = route_hint
        questions = {"action": Choice(instructions=_KIND_INSTRUCTIONS, criteria=criteria)}

        t = time.time()
        resp = self.client.system_one(state=state, questions=questions)
        latency = int((time.time() - t) * 1000)
        ans = resp.answers["action"]
        choice, confidence = ans.choice, float(getattr(ans, "confidence", 0.0) or 0.0)

        # SAYCAN bias (spec §5): argmax over model-probability × affordance prior. Only
        # when the directive names biased options AND we have a probability distribution —
        # so it can only re-rank among still-legal options, never invent one.
        probs = getattr(ans, "probabilities", None)
        if directive is not None and directive.option_bias and isinstance(probs, dict) and probs:
            boost = {k: float(p) * (2.0 if k in directive.option_bias else 1.0) for k, p in probs.items()}
            best = max(boost, key=boost.get)
            if best in criteria:
                choice = best

        if choice in target_map:
            tgt = target_map[choice]
            action = GoToAction(x=tgt["x"], y=tgt["y"], interact=tgt["interact"], label=tgt["label"])
        else:
            action = _KIND_TO_ACTION.get(choice, lambda: WaitAction(frames=15))()
        note = f"typesafe: {choice} (conf {confidence:.2f})"

        # confidence gate: a low-confidence MOVE during (likely) a cutscene thrashes;
        # wait instead — but bounded, so we never stall forever if it's just uncertain.
        if (
            self.wait_on_low_confidence
            and confidence < self.min_confidence
            and isinstance(action, MoveAction)
            and not dialog_active
            and self._low_conf_waits < self.max_consecutive_waits
        ):
            self._low_conf_waits += 1
            action = WaitAction(frames=20)
            note = f"typesafe: {choice} @ {confidence:.2f} < {self.min_confidence} -> wait ({self._low_conf_waits})"
        else:
            self._low_conf_waits = 0

        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        usage["confidence"] = confidence
        usage["probabilities"] = getattr(ans, "probabilities", None)

        step = ReasonStep(
            location=(player_desc or "")[:80],
            objective=(plan.next_objective if plan else primary_goal),
            tried=(previous.tried if previous else ""),
            reasoning=note,
            action=action,
        )
        # capture the executor decision — parsed={"choice": raw model pick}; the EXECUTED choice can
        # differ (SAYCAN option_bias re-rank / low-conf->wait gate), so surface it in extra for replay.
        self._cap_jev("jev_action", state, ans, confidence, latency_ms=latency,
                      extra={"executed": choice})
        return step, latency, usage

    # --- L3 PATHING via Jev: step-by-step direction choice toward a target tile ----
    def path_step(self, *, player, target, neighbors, recent=None, blocked_dirs=None, objective="",
                  bfs_suggestion=None, hp_frac=None):
        """Jev as the pathfinder: choose ONE step direction (1-of-4) toward TARGET (x, y).

        Context is deliberately MINIMAL — an A/B over map sizes showed the full map is
        oversaturation (no accuracy gain, ~2x tokens, LOWER confidence). Jev gets only what an L3
        step decision needs: the bearing to the target, its 4 immediate NEIGHBOR tiles (semantic
        class), the BFS shortest-path suggestion, and a short trail. The global routing is BFS's
        and L2's job. Returns (Direction | None, confidence, probabilities)."""
        from typesafe_sdk import Choice
        from ..core.models import Direction

        blocked_dirs = set(blocked_dirs or ())
        opts = {
            "move_north": "step north — up, toward SMALLER y",
            "move_south": "step south — down, toward LARGER y",
            "move_east": "step east — right, toward LARGER x",
            "move_west": "step west — left, toward SMALLER x",
        }
        criteria = {k: v for k, v in opts.items() if k[len("move_"):] not in blocked_dirs}
        if not criteria:
            return None, 0.0, None
        dx, dy = int(target[0]) - int(player["x"]), int(target[1]) - int(player["y"])
        compass = ("N" if dy < 0 else "S" if dy > 0 else "") + ("E" if dx > 0 else "W" if dx < 0 else "")
        state = {
            "task": "Move one tile toward TARGET on a walkable tile.",
            "bearing": f"target is {compass or 'here'} (dx={dx:+d}, dy={dy:+d})",
            "neighbors": neighbors or {},  # {north/south/east/west: floor|wall|grass|water|ledge_*|door|counter}
            "bfs_suggestion": (f"move_{bfs_suggestion}" if bfs_suggestion else None),
            "recent_trail": recent or [],
            "objective": objective,
        }
        if hp_frac is not None:  # decision-relevant: whether it's safe to risk a wild encounter
            state["party_hp_frac"] = round(float(hp_frac), 2)
        instr = (
            "You are the step-by-step NAVIGATOR. Pick the direction that moves one tile toward TARGET "
            "(see BEARING) onto a walkable NEIGHBOR — 'floor'/'grass' are walkable, 'wall'/'water' are "
            "not, ledges are one-way. BFS_SUGGESTION is the deterministic shortest-path hint: usually "
            "take it. A 'grass' tile can trigger a WILD BATTLE. When PARTY_HP_FRAC is LOW (roughly "
            "< 0.35), PREFER a non-grass walkable neighbor that still heads toward TARGET — override "
            "BFS to avoid a fight you might lose. When HP is healthy, grass is fine (it also lets you "
            "grind). Do NOT reverse your RECENT_TRAIL (no oscillating)."
        )
        t = time.time()
        resp = self.client.system_one(state=state, questions={"dir": Choice(instructions=instr, criteria=criteria)})
        latency = int((time.time() - t) * 1000)
        ans = resp.answers["dir"]
        conf = float(getattr(ans, "confidence", 0.0) or 0.0)
        probs = getattr(ans, "probabilities", None)
        choice = ans.choice if isinstance(ans.choice, str) else ""
        self._cap_jev("jev_path", state, ans, conf, latency_ms=latency)
        try:
            return Direction(choice[len("move_"):]), conf, probs
        except Exception:
            return None, conf, probs

    # --- Jev routes the control FLOW (N parallel calibrated yes/no questions in one call) ----
    def choose_flow(self, *, screen_text="", has_text=False, text_box_id=0, last_action=""):
        """Route the control FLOW as N short, focused, CALIBRATED yes/no questions asked in ONE
        parallel `system_one` call (battle is handled deterministically upstream; menu is cross-checked
        against RAM). Returns {"dialogue": (ans, conf), "menu": (ans, conf)} with ans in {"yes","no"}.

        The decisive feature is SCREEN_TEXT (decoded from the on-screen text-box region, rows 12-17):
        empty or a repeated-char blob usually means no box. Wording/features tuned via the flow-gate
        probe (scripts/probe_flow_gate.py) before this is wired into the loop."""
        from typesafe_sdk import Choice
        state = {
            "screen_text": (screen_text or "")[:120],
            "has_decoded_text": bool(has_text),
            "text_box_id": int(text_box_id or 0),
            "player_last_action": last_action or "",
            "note": "SCREEN_TEXT is decoded from the on-screen text-box area; it may be empty (no box) "
                    "or a repeated-character blob (a background picture, not real text).",
        }
        # NB: the questions describe the OBSERVABLE screen state only — never our action ("press A").
        # Jev classifies what's on screen; our code decides how to act on each flow.
        questions = {
            "dialogue": Choice(
                instructions="Is a dialogue / message text box currently shown on screen — a character "
                             "speaking, or a message being displayed to the player?",
                criteria={"yes": "a dialogue / message text box IS on screen (SCREEN_TEXT is real prose)",
                          "no": "no message box — SCREEN_TEXT is empty or a background-picture blob"}),
            "menu": Choice(
                instructions="Is a selectable menu or list (with a cursor) currently shown on screen?",
                criteria={"yes": "a menu / selectable list IS on screen",
                          "no": "no menu is on screen"}),
        }
        resp = self.client.system_one(state=state, questions=questions)
        out = {}
        for name in questions:
            ans = resp.answers.get(name)
            choice = ans.choice if (ans is not None and isinstance(ans.choice, str)) else "no"
            conf = float(getattr(ans, "confidence", 0.0) or 0.0) if ans is not None else 0.0
            out[name] = (choice if choice in ("yes", "no") else "no", conf)
        # jev_flow has TWO calibrated answers (dialogue + menu); record the combined result (§3).
        cap = getattr(self, "capture", None)
        if cap is not None:
            cap.record("jev_flow", model=getattr(self.client, "model", "typesafe"),
                       input=state, output_raw=str({k: v[0] for k, v in out.items()}),
                       parsed={k: v[0] for k, v in out.items()},
                       confidence=out.get("dialogue", (None, None))[1],
                       extra={"dialogue": list(out["dialogue"]), "menu": list(out["menu"])})
        return out

    # --- Jev picks the ROUTING POLICY for a leg (a calibrated 1-of-N objective choice) ----
    def choose_policy(self, *, hp_frac=None, level=None, level_target=0, objective="",
                      area="", grass_nearby=True):
        """Jev chooses the routing OBJECTIVE for this leg — one meaningful call, not per-step. The
        weighted router then executes it deterministically. Returns (policy, confidence)."""
        from typesafe_sdk import Choice
        criteria = {
            "shortest": "take the shortest path; ignore grass (normal travelling)",
            "dodge-grass": "avoid tall grass / wild battles — for when the party is HURT or you're "
                           "on an important errand (delivering, low HP) and can't risk a fight",
            "farm-exp": "deliberately route THROUGH grass to trigger wild battles and GRIND — for "
                        "when the party is UNDER-LEVELED for what's ahead and healthy enough to fight",
        }
        state = {
            "task": "Pick the routing objective for the next stretch toward the destination.",
            "party_hp_frac": (round(float(hp_frac), 2) if hp_frac is not None else None),
            "party_level": level, "level_target_for_next_gym": level_target or None,
            "objective": objective, "area": area, "grass_on_this_map": bool(grass_nearby),
        }
        instr = (
            "Choose the routing objective. Prefer 'dodge-grass' when PARTY_HP_FRAC is low or the "
            "objective is a delivery/errand you shouldn't risk. Prefer 'farm-exp' when PARTY_LEVEL "
            "is below LEVEL_TARGET_FOR_NEXT_GYM and HP is healthy (grind on the way). Otherwise "
            "'shortest'. If GRASS_ON_THIS_MAP is false the choice barely matters — pick 'shortest'."
        )
        resp = self.client.system_one(state=state, questions={"pol": Choice(instructions=instr, criteria=criteria)})
        ans = resp.answers["pol"]
        conf = float(getattr(ans, "confidence", 0.0) or 0.0)
        pol = ans.choice if isinstance(ans.choice, str) and ans.choice in criteria else "shortest"
        self._cap_jev("jev_policy", state, ans, conf)
        return pol, conf

    def choose_npc(self, *, objective, candidates):
        """Jev picks WHICH NPC on the map best fits the OBJECTIVE — a calibrated 1-of-N over the
        candidate people (used to disambiguate when several are present and the name is ambiguous).
        ``candidates`` is a list of {sprite, x, y, talked_to}. Returns (index | None, confidence)."""
        from typesafe_sdk import Choice
        if not candidates:
            return None, 0.0
        criteria = {
            str(i): (f"{c.get('sprite') or 'person'} at ({c.get('x')},{c.get('y')})"
                     + (" — already talked to" if c.get("talked_to") else ""))
            for i, c in enumerate(candidates)
        }
        state = {"task": "Pick the person to walk up to and talk to for the current objective.",
                 "objective": objective, "candidates": candidates}
        instr = ("Choose the ONE person who best fits OBJECTIVE — the specific NPC the plan needs "
                 "(e.g. Professor Oak for a lab errand, a shop CLERK to buy/collect, the NURSE to heal). "
                 "Prefer someone NOT already talked to unless the objective needs them again.")
        resp = self.client.system_one(state=state, questions={"npc": Choice(instructions=instr, criteria=criteria)})
        ans = resp.answers["npc"]
        conf = float(getattr(ans, "confidence", 0.0) or 0.0)
        self._cap_jev("jev_npc", state, ans, conf, extra={"candidates": candidates})
        try:
            idx = int(ans.choice)
        except (TypeError, ValueError):
            return None, conf
        return (idx if 0 <= idx < len(candidates) else None), conf

    # --- menu handling: choose an option in an open list/yes-no menu ---------
    def _menu_step(self, menu: dict, state: dict, previous):
        from typesafe_sdk import Choice

        opts = menu.get("options") or []
        n = max(int(menu.get("num_options") or 0), len(opts), 1)
        criteria = {str(i): (opts[i] if i < len(opts) else f"option {i}") for i in range(n)}
        instr = (
            "A selectable MENU is open (SCREEN_TEXT shows the prompt). Choose the option that "
            "best advances PRIMARY_GOAL — pick YES to accept/confirm what you want and NO to "
            "decline; otherwise pick the item/Pokémon/action you intend. Options are indexed 0=top."
        )
        t = time.time()
        resp = self.client.system_one(state=state, questions={"opt": Choice(instructions=instr, criteria=criteria)})
        latency = int((time.time() - t) * 1000)
        ans = resp.answers["opt"]
        conf = float(getattr(ans, "confidence", 0.0) or 0.0)
        try:
            idx = int(ans.choice)
        except (TypeError, ValueError):
            idx = 0
        label = opts[idx] if idx < len(opts) else None
        step = ReasonStep(
            location="menu", objective="operate the open menu",
            tried=(previous.tried if previous else ""),
            reasoning=f"menu: pick {idx} ({label}) conf {conf:.2f}",
            action=MenuSelectAction(index=idx, label=label),
        )
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        usage["confidence"] = conf
        return step, latency, usage

    # --- a reusable verification primitive (Noul): "did X happen / is X true?" --
    def judge(self, question: str, state: dict) -> float:
        """Calibrated 0-1 probability for a yes/no question about the game state.

        Handy for grounding events without heuristics, e.g.
          judge("Did the party just gain a starter Pokémon?", {...})
          judge("Is the player stuck oscillating without progress?", {...})
        """
        from typesafe_sdk import Noul

        resp = self.client.system_one(state=state, questions={"q": Noul(instructions=question)})
        return float(resp.answers["q"].noul)
