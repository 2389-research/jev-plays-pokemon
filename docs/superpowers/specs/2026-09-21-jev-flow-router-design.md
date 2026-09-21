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
1. in_battle (hard RAM bit, wIsInBattle) -> _battle_turn        # deterministic, safety-critical
2. Jev available (decider == typesafe)?
     yes -> flow, conf = reasoner.choose_flow(state)            # 1-of-N calibrated
             if conf < FLOW_MIN_CONF: flow = _safe_flow(state)  # low-confidence fallback
             dispatch by flow: dialogue -> _advance_dialog
                               menu     -> _jev_turn (menu)
                               navigate -> planner nav / legacy _jev_turn
     no  -> deterministic dispatch (current: ctx_kind == dialog/menu, else navigate)  # llm-decider path
```

- **Battle stays deterministic.** `wIsInBattle` is an unambiguous bit and missing it loses the battle;
  it is a fact, not a judgment. (It is also passed to Jev as a feature = a strong prior, but the
  short-circuit means Jev never has to get it right.)
- **Jev routes the fuzzy boundary** — navigate vs a text box is up vs a menu — which is where the
  ambiguity (and the bug) actually lives.
- **Deterministic fallback path** for when Jev isn't wired (`--decider llm`, or Jev unavailable): keep
  the current `ctx_kind`-based dispatch (with the already-committed lowercase-dialogue heuristic as its
  detector). So the router degrades gracefully; it does not depend on Jev being present.

### `choose_flow` (new `TypeSafeReasoner.choose_flow`)

Mirrors the existing `choose_policy`/`choose_npc` shape: one calibrated `Choice`.

- **State (features):** the decoded on-screen text (rows 12–17, whatever it is — we still *decode* it,
  we just don't heuristically classify it), a short `screen_has_text` flag, the menu-cursor/text-box
  RAM signals we already read (`text_box_id`, any menu state in `read_context`), player position +
  facing, whether the tile the player faces is an NPC, and the current objective.
- **Question:** `Choice(criteria=["navigate", "dialogue", "menu"], instructions=…)` — "Given the
  screen, which control flow is active right now?" Instructions describe each: dialogue = a text box is
  open and waiting (press A to continue); menu = a selectable list/cursor is up (choose an option);
  navigate = free overworld movement.
- **Returns** `(flow: str, confidence: float)` — parsed like the other `choose_*` (the chosen criterion
  + its calibrated probability).

### Low-confidence fallback (`_safe_flow`)

When `conf < FLOW_MIN_CONF`, don't trust the pick — choose the SAFE default:
- if there is on-screen text decoded -> `dialogue` (mashing A on a text box advances it; mashing A on a
  non-dialog is a cheap, self-correcting no-op — far safer than walking into an unread box forever);
- else -> `navigate`.

This is strictly more robust than a bare heuristic or an uncalibrated model: an unsure router fails
toward "close the box," which is the recoverable direction.

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
- **Modify** `agent/typesafe_reasoner.py` — add `choose_flow(state) -> (flow, conf)`.
- **Modify** `agent/reason_loop.py` — `step_once` dispatch: battle short-circuit, Jev `choose_flow`
  with `_safe_flow` fallback when Jev is wired, else the current deterministic dispatch; add
  `FLOW_MIN_CONF`; add `_safe_flow`.
- **Keep** `games/pokemon_red/game_state.py` `read_screen_text` (it still DECODES the text as a feature
  and still backs the deterministic fallback path — the committed lowercase fix stays as the fallback
  detector). The routing simply no longer *depends* on its boolean when Jev is present.
- **Modify** `tests/unit/test_executive.py` (or a new `test_flow_router.py`) — the Layer-1 tests above.

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
