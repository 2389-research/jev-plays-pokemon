# Jev Flow Router — Design

**Date:** 2026-09-21
**Status:** Approved (design), pending implementation
**Branch:** `l1-reasoning-pipeline`

## Problem

The step loop routes between control flows (battle / dialogue / menu / navigation) with **deterministic
mode classification**, and the classification of "is a dialogue text box open?" is a brittle heuristic
(`read_screen_text`'s `has_upper` / letter-variety check). It has already failed in a live run: the
agent walked into a wandering NPC, her all-lowercase dialogue line (`"…strong, they can protect me!"`)
was not recognized as a dialog, so the loop stayed in navigation, issued `move west` into the open text
box forever, and got permanently stuck — it could not tell it was in a conversation.

Mode routing is a **calibrated 1-of-N classification**, which is exactly what Jev (the TypeSafe
calibrated Choice model) is built for and cheap enough to run every step. The current design has this
backwards: it routes modes with brittle heuristics and only uses Jev *inside* a mode. We should promote
Jev to the **top-level flow router**, feeding it the deterministic RAM facts as features rather than
using a heuristic as the router.

## Goal

Replace the heuristic mode-routing in `step_once` with a **Jev flow-router**: `choose_flow(state) →
{navigate | dialogue | menu}` (battle short-circuited deterministically), using calibration for a
principled low-confidence fallback. Retire the `has_upper`/letter-variety routing dependency. Each flow
stays a separate handler (already true: `_battle_turn`, `_advance_dialog`, `_jev_turn`, `_navigate_leg`).

Non-goals: changing the handlers themselves; changing L1/L2 planning; a per-A-press model call (the
dialogue handler still mashes A deterministically until the screen changes).

## Architecture

### Dispatch (in `step_once`)

```
1. in_battle (hard RAM bit, wIsInBattle) -> _battle_turn         # deterministic, safety-critical
2. reasoner has choose_flow (Jev wired, duck-typed)?
     yes -> ans = reasoner.choose_flow(state)                    # ONE call, {dialogue:(a,c), menu:(a,c)}
             # menu: take it when RAM says open OR Jev is confident (cross-check; A on a menu SELECTS)
             if ctx.menu.open or (ans["menu"] == ("yes", c) and c >= FLOW_MIN_CONF): -> _jev_turn (menu)
             # dialogue: Jev-yes (confident) -> advance; hedged -> _safe_flow
             elif ans["dialogue"] == "yes" and c >= FLOW_MIN_CONF:                   -> _advance_dialog
             elif conf < FLOW_MIN_CONF:  flow = _safe_flow(state) -> that flow
             else:                                                                    -> navigate
     no  -> deterministic dispatch (current: ctx_kind == dialog / menu, else navigate)  # llm-decider path
```

- **Battle AND menu stay deterministic.** `wIsInBattle` is an unambiguous bit (missing it loses the
  battle), and `menu.open` is a separate RAM-derived signal from `read_menu`. Both are facts, not
  judgments — and pressing A on an open menu MAKES A SELECTION (not a harmless no-op), so a menu must
  never be routed to the dialogue (A-mashing) flow. Menus are handled before Jev.
- **Jev arbitrates only the genuinely fuzzy boundary — navigate vs a dialogue box is up** — which is
  exactly where the ambiguity (and the bug) lives. `choose_flow` is a 1-of-2 Choice
  (`["navigate", "dialogue"]`), not 1-of-3. (In_battle / menu-open are passed as features = strong
  priors, but the deterministic short-circuits mean Jev never has to get them right.)
- **Jev-availability is duck-typed**, matching the existing optional-Jev pattern
  (`getattr(self.reasoner, "choose_flow", None) is not None`) — not a decider string. When absent
  (`--decider llm`, generative `Reasoner`), keep the current `ctx_kind`-based dispatch (with the
  already-committed lowercase-dialogue heuristic as its detector). The router degrades gracefully.

### `choose_flow` — ONE call, N parallel focused binaries (`TypeSafeReasoner.choose_flow`)

Jev's `system_one(state, questions)` answers a DICT of named questions in a single round-trip,
returning one calibrated answer per name (`SystemOneResponse.answers`). Jev works best on SHORT,
focused prompts — so instead of cramming the mode decision into one multi-class prompt, `choose_flow`
asks several tiny yes/no questions in parallel against the same state and routes on the calibrated
answers.

```python
questions = {
  "dialogue": Choice(criteria=["yes","no"],
                     instructions="A text box is open and the player must press A to continue. Yes/no?"),
  "menu":     Choice(criteria=["yes","no"],
                     instructions="A selectable menu/list with a cursor is open. Yes/no?"),
}
resp = self.client.system_one(state=state, questions=questions)     # one round-trip, calibrated per Q
return {name: (ans.choice, float(ans.confidence)) for name, ans in resp.answers.items()}
```

- **State (small, text-centric):** the decoded on-screen text-box region (rows 12–17, whatever it is —
  we still *decode* it, we just don't heuristically classify it), a `has_text` bool, `text_box_id`, and
  `last_action` (did the agent just try to move / press A?). Keep it tiny — the decisive feature is
  `screen_text`.
- **Returns** `dict[str, (answer, confidence)]` — e.g. `{"dialogue": ("yes", 0.93), "menu": ("no", .8)}`.
- **Number of questions is a design variable (tuned in Deliverable 0):** at minimum `dialogue?`; adding
  `menu?` lets Jev CROSS-CHECK the RAM `menu.open` signal (which the reviewer flagged as not-yet-fixture-
  validated) at no extra round-trip. Only asking questions that earn their keep on the scorecard.

**The exact questions, wording, and minimal features are designed EMPIRICALLY, not guessed** (see
Deliverable 0): we run this parallel call against labelled real frames and tune against a scorecard
(per-question accuracy + calibration) before wiring it into the loop.

### Low-confidence fallback (`_safe_flow`)

When `conf < FLOW_MIN_CONF`, don't trust the pick — choose the SAFE default. Menus are already handled
deterministically BEFORE this point (step 2), so `_safe_flow` only picks between dialogue and navigate,
and A-on-a-menu can't happen here:
- if there is on-screen text decoded -> `dialogue` (pressing A advances a real box; pressing A in the
  overworld is a cheap, self-correcting no-op — far safer than walking into an unread box forever);
- else -> `navigate`.

This is strictly more robust than a bare heuristic or an uncalibrated model: an unsure router fails
toward "close the box," the recoverable direction.

## Deliverable 0 — the flow-gate probe (BUILD + TUNE BEFORE WIRING)

Do not touch `step_once` until the gate's prompt + minimal context are validated empirically (mirrors
the Layer-2.5 criteria eval). A `scripts/probe_flow_gate.py` + fixtures:

- **Fixtures = labelled real frames.** Reconstruct states from the recorded runs / RAM: a genuine
  dialogue, the ALL-LOWERCASE dialogue that broke (`"…strong, they can protect me!"`), a plain
  overworld frame, a garbled/cutscene frame (repeated-char blob -> should be `navigate`), and (as a
  control) a menu frame. Each labelled with its true flow. Build them with a settable-RAM emulator
  (poke the text-box tiles) or by loading captured `.state`s.
- **Run `choose_flow`** on each and score: accuracy vs the label, and whether the confidence is
  well-calibrated (high on the clear cases, lower on the garbled one). Try context variants (screen_text
  alone vs +has_text vs +text_box_id) to find the MINIMAL context that classifies correctly.
- **Output** a scorecard; iterate the instructions/features until the gate is right — in particular the
  all-lowercase case must classify as `dialogue` and the cutscene blob as `navigate`.
- Gated behind `@pytest.mark.live` / run as a script (it costs Jev credits); the DETERMINISTIC grader
  is offline-unit-tested with canned Jev outputs.

Only after Deliverable 0 gives a passing scorecard do we implement `choose_flow` for real and wire the
dispatch.

### Constant
`FLOW_MIN_CONF` (in `reason_loop.py`, alongside the other Jev thresholds `JEV_OVERRIDE_CONF` etc.),
initial value ~0.55 — trust a clear Jev pick, fall back when it hedges.

## Testing

Layer-1 (offline, deterministic, stubbed Jev):
- **Battle short-circuit:** `in_battle` set -> `_battle_turn` runs and `choose_flow` is NOT called
  (deterministic, no gamble).
- **Flow dispatch:** a stub reasoner returning `("dialogue", 0.9)` -> `_advance_dialog`; `("menu", 0.9)`
  -> `_jev_turn` menu path; `("navigate", 0.9)` -> the nav path. (Assert via which handler ran / the
  action produced.)
- **Low-confidence fallback:** stub returns `("navigate", 0.2)` while screen text is present -> routes
  to `dialogue` (the safe default), NOT navigate.
- **Deterministic fallback:** with no Jev (`reasoner` without `choose_flow` / planner path llm) the
  current `ctx_kind` dispatch still works (regression).
- **The original bug, end-to-end at the routing layer:** an all-lowercase dialogue on screen +
  Jev picking `dialogue` -> `_advance_dialog` (it would have stuck under the old heuristic router).

Layer-3 (gated, live): the existing live smoke, plus optionally a fixture that starts the agent facing
an NPC so it opens a dialog, and asserts it advances out instead of stalling.

## Files
- **Create (Deliverable 0, first)** `scripts/probe_flow_gate.py` + `tests/fixtures/flow_frames.py` —
  labelled frames + the live probe/scorecard; `tests/unit/test_flow_gate_grader.py` — offline grader.
- **Modify** `agent/typesafe_reasoner.py` — add `choose_flow(state) -> (flow, conf)` (binary
  dialogue/navigate), prompt/features per Deliverable 0.
- **Modify** `agent/reason_loop.py` — `step_once` dispatch: battle short-circuit, menu short-circuit,
  then (if `getattr(self.reasoner, "choose_flow", None)`) Jev `choose_flow` + `_safe_flow` fallback,
  else the current deterministic `ctx_kind` dispatch; add `FLOW_MIN_CONF`; add `_safe_flow`.
- **Keep** `games/pokemon_red/game_state.py` `read_screen_text` (still DECODES the text as a feature and
  backs the deterministic fallback path — the committed lowercase fix stays as the fallback detector).
  Routing no longer *depends* on its boolean when Jev is present.
- **Modify** `tests/unit/test_executive.py` (or a new `test_flow_router.py`) — the dispatch tests below.

## Risks
- **Cost:** one extra Jev call per step. Jev is calibrated/cheap (output free, ~$0.042/M input) so this
  is affordable per-step (unlike a generative model); still, it needs credits — hence the deterministic
  fallback when Jev isn't wired.
- **Battle safety:** never routed by Jev (deterministic bit), so a misclassification can't cost a
  battle.
- **A brief mis-route** to dialogue on a non-dialog just presses A (a no-op) for a step and re-routes —
  cheap and self-correcting, which is why it's the safe fallback.

## Success criteria
- The flow router runs on `--decider typesafe`; the all-lowercase-dialogue stall cannot recur (Jev, or
  the safe fallback, routes to `_advance_dialog`).
- Battle is still entered deterministically and never depends on a classification.
- CI is green and spends zero credits (Jev stubbed in tests); `--decider llm` still works via the
  deterministic fallback.
