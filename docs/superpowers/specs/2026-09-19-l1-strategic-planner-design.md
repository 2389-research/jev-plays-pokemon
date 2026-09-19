# L1 Strategic Planner — periodic, DSL-speaking, with a deterministic reconciler

**Status:** Design approved (2026-09-19). Not yet implemented.

## Problem

The agent has three intended layers — **L1** (long-term memory + "what should we be doing at this
point in the game"), **L2** (reflection: the concrete next target), **L3** (Jev + weighted router).
L2 and L3 were rebuilt in the unified-control-loop work and function. **L1 does not.**

Observed in the `rival-onward` run (620 steps): the agent reached Viridian City, hit the north gate
(which is closed until Oak's Parcel is delivered), never recognized the errand, and thrashed for
~380 steps (`local_loop` ×280) grinding against the guard. Root causes:

1. **L1 is fire-once + weakly reactive, not periodic.** The strategist produced a single quest at
   startup (1 `quest` event all run) and only re-fired on escalation.
2. **A single wedged step discards the whole plan.** The heal step wedged at step 240 and the wedge
   handler did `self._quest.clear()`, throwing away the remaining route (Route 2 → Forest → Pewter).
3. **Healing is a hardcoded arbiter branch,** not something L1 weighs.
4. **L1's durable memory isn't maintained.** Folding reflection into the L2 proposer removed the
   planner-path `_maybe_reflect`, so `AgentPlan` (mission / milestone / `tried_failed`) is never
   updated or consulted — there is no evolving long-term plan.
5. **The KB was the difference.** Runs that DID the parcel round-trip (`full-0918-194657`: maps
   `0→12→1→42→1→12→0→40`, kb=165, quests=102) had the Orrery KB attached; the strategist queried it
   ("Viridian Mart", "Oak's Parcel") and planned the errand. A control run WITH the KB
   (`runs/rival-kb`) planned the parcel errand up front and turned back toward Pallet — confirming
   the machinery works when L1 can query the KG, and that the failure was operational (no
   `--orrery-workspace`) compounded by the design gaps above.

## Goal

Make L1 a **persistent strategic planner** that runs on a cadence (and on events), reviews the quest
plan against the live game state + the Orrery KB, and revises it by **speaking the quest DSL**
(propose new quests with machine-checkable acceptance criteria; delete obsolete ones; or "no
change"). A **deterministic reconciler** applies its proposal to the live queue while preserving
progress. Needs (low HP, under-level, blocked) flow **into** L1 as signals; only a near-faint keeps
a hard deterministic reflex. L1 maintains durable long-term memory that L2 consults.

Non-goals: changing L2 (`propose_target`) or L3 (Jev/router); changing the `done_when` grammar the
executor already checks; battle/menu control.

## Architecture

```
L1  Strategic Planner (periodic + events)   → proposes quest DSL edits (add/remove/no-change),
    + Reconciler (deterministic)               maintains AgentPlan (mission/milestone/tried_failed)
NEEDS  RAM signals into L1  +  emergency heal reflex (deterministic, near-faint only)
L2  Proposer (unchanged)                     → typed short-term target from the ACTIVE quest step
L3  Jev + weighted router (unchanged)        → policy + tiles
```

### Component 1 — L1 Strategic Planner (the model)

A method on `Planner` (e.g. `revise_quests(emu, context) -> dict`), model = the tier-2 strategist
(glm-5.3), KB-grounded (it may call the Orrery search tool via the existing `_llm_with_search`).

- **Inputs (`context`):**
  - current map (id+name) + player pos; party (species/level/HP); items; badges;
  - **the current quest plan with per-step status** — each step's `id`, `map`, `talk`/`who`,
    `done_when`, and `status ∈ {done, active, pending, wedged}`;
  - **signals**: `hp_frac`, `min_level` vs `level_target`, `blocked_for_n` (legs the active target
    made no progress), `tried_failed` (approaches already shown not to work), key item/badge flags;
  - recent trail; the mission so far (`AgentPlan.mission`/`milestone`).
- **Output (strict JSON):**
  ```json
  {"assessment": "one line: what's the situation / why change or not",
   "change": true,
   "mission": "the overall goal right now",
   "milestone": "the current sub-goal",
   "add":    [ <quest step>, ... ],
   "remove": [ "<step_id>", ... ] }
  ```
  `change:false` is a first-class, cheap result (most cycles) — `add`/`remove` empty.
- **The quest DSL it must speak** (identical to the executor's existing grammar, so no executor
  change): a step is `{map:<id>, talk:<bool>, who:<npc name?>, done_when:<criterion>, why:<short>}`,
  and `done_when ∈ { on_map, talked, has_item:<item>, no_item:<item>, level>=<N>, badges>=<N>,
  hp_frac>=<F>, verify:<yes/no question> }`. The system prompt teaches this grammar explicitly and
  requires every added step to carry a valid `done_when`.
- **Validation:** parse strictly; drop any `add` step whose `done_when` doesn't parse (reuse
  `Planner._parse_done_when`); a `who` on a `talk` step maps to the directive's `target.sprite`.
  If the whole call fails/returns garbage → treat as `change:false` (keep the current plan) and log
  an `l1_failed` event (do NOT wipe the plan on a model hiccup).

### Component 2 — Reconciler (deterministic, no LLM)

`reconcile_quests(current: list[QuestStep], proposal: dict, active_idx: int) -> list[QuestStep]`.
Pure function, fully unit-testable. Rules:

1. **Preserve progress:** keep every `done` step and the currently `active` step untouched.
2. **Remove** only applies to `pending` steps whose id is in `proposal.remove`; never removes the
   active or a done step.
3. **Add** inserts `proposal.add` steps (in given order) into the pending region (after the active
   step), each assigned a fresh stable id; **dedup** against existing steps by `(map, done_when)`.
4. **Wedged step → replace, not clear:** when a step is flagged `wedged`, L1 is asked to propose a
   replacement for that objective; the reconciler swaps just that step and keeps the rest. The old
   `self._quest.clear()` path is removed.
5. Result is the new ordered queue; the executor keeps executing the active step (or advances if the
   reconcile removed/replaced it).

Stable ids: assign each quest step an `id` at creation (`q{n}`), so `remove`/status can reference it.

### Component 3 — Needs as signals + emergency reflex

- A `signals(emu, state)` helper computes `hp_frac`, `min_level`, `blocked_for_n`, badge/key-item
  flags each L1 cycle and passes them into L1's context. Healing and grinding become **L1
  decisions** (it may add a `hp_frac>=0.95` heal quest or a `level>=N` grind step), not a hardcoded
  arbiter intent.
- **One** deterministic reflex remains, for safety a slow L1 tick can't cover: **near-faint**
  (`hp_frac < HEAL_EMERGENCY` (≈0.15) or any conscious party member at 0 HP) forces an immediate
  heal directive that preempts, independent of L1. This is the only need still handled outside L1.
- The `NeedsArbiter`'s role shrinks to: detect battle mode (owned by the battle controller) and the
  emergency-heal reflex. Travel/grind/heal *planning* moves into L1. (BATTLE stays a mode controller.)

### Component 4 — Cadence & triggers

L1 runs when **any** of these holds (subject to a short cooldown to avoid double-firing):
- **periodic:** every `L1_EVERY_N_LEGS` legs (default **5**, tunable via `--l1-every`);
- **events:** a quest step completed or wedged; a map milestone reached (new map on the route);
  a party level-up; `blocked_for_n >= BLOCK_TRIGGER`; a low-HP ping crosses a threshold.
The emergency heal reflex is immediate and separate from this cadence.

### Component 5 — Durable long-term memory

Each L1 cycle updates `AgentPlan.mission` / `milestone` and appends to `tried_failed` when it drops
or replaces a failing approach. This is persisted in `AgentMemory` (already checkpointed) and
surfaced in the run recorder `extra` (so the viewer shows the live long-term plan). L2's proposer
context gains the current `milestone` so its short-term targets are framed by L1's intent.

## Data flow (one executive step)

1. Perceive (unchanged); compute `signals`.
2. **Emergency reflex:** if near-faint → force heal directive, skip L1 this step.
3. **L1 gate:** if a cadence/event trigger fires → call `revise_quests` → `reconcile_quests` →
   update the queue + `AgentPlan`; emit `l1_review` (with `change`/assessment) + `quest` events.
4. Directive management picks the active quest step (as today).
5. L2 `propose_target` (now also given `milestone`) → L3 Jev + router → move.

## Error handling

- L1 call fails / unparseable → `change:false`, keep the plan, log `l1_failed`. Never clear on error.
- All `add` steps validated; invalid `done_when` dropped with an `l1_bad_step` log.
- KB unavailable → L1 still runs on the model's own knowledge (degraded, logged), never crashes.
- Reconciler is total: any proposal (incl. empty / all-invalid) yields a valid queue (worst case
  unchanged).

## Testing

- **Reconciler (pure unit):** preserve done+active; remove only pending; insert+dedup; wedged→replace
  (not clear); empty proposal → unchanged; remove of active is ignored.
- **L1 parse/validate:** fake provider → valid DSL parsed; `change:false` honored; bad `done_when`
  dropped; garbage → `change:false` + `l1_failed`.
- **Signals + emergency reflex:** thresholds; near-faint forces heal and skips L1.
- **Cadence:** fires on N-legs and on each event; cooldown prevents double-fire.
- **Integration fixture:** blocked at Viridian → L1 adds parcel errand (Mart → get parcel → Lab →
  deliver) → reconciler inserts → executor routes toward the Mart/Oak. Wedge one step → the rest of
  the plan survives.
- **Live:** rival save + `--orrery-workspace` → Viridian → Mart → back to Oak → gate opens → north.

## Concrete surface (files)

- `src/pokemon_agent/agent/planner_llm.py` — `revise_quests` + `L1_SYSTEM` (teaches the DSL);
  reuse `_parse_done_when`, `_llm_with_search`. Keep `strategize` or refactor it into `revise_quests`.
- `src/pokemon_agent/agent/quest_reconciler.py` (new) — `reconcile_quests` + `QuestStep` id/status.
- `src/pokemon_agent/agent/reason_loop.py` — the L1 gate (cadence/events), signals, emergency-heal
  reflex, remove the `quest.clear()`-on-wedge path, feed `milestone` to L2, record L1 state.
- `src/pokemon_agent/agent/needs_arbiter.py` — shrink to battle-mode + emergency-heal.
- `scripts/run_agent.py` — `--l1-every` flag (default 5).
- Tests: `tests/unit/test_quest_reconciler.py`, `tests/unit/test_l1_planner.py`, additions to the
  executive tests + an integration fixture.

## Open tuning (defaults, changeable)

`L1_EVERY_N_LEGS=5`, `BLOCK_TRIGGER≈6 legs`, `HEAL_EMERGENCY≈0.15`. Model = glm-5.3 for L1;
"no change" cycles use a compact prompt to keep cost down.
