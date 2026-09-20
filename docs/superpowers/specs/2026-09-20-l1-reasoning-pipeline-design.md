# L1 Reasoning Pipeline & Acceptance-Criteria Contract — Design

**Date:** 2026-09-20
**Status:** Approved (design), pending implementation plan
**Branch:** `l1-reasoning-pipeline` (builds on `l1-strategic-planner`, PR #2)

## Problem

A full 1538-step run (`runs/l1-longrun-20260920-153335`, real Jev decider + Orrery KB) exposed
that L1 produces *plausible-looking* plans that never actually complete objectives. Forensic
analysis of the log established the root causes:

1. **Acceptance criteria are positional, not semantic.** A quest step's completion check compiles
   to `{"on_map": <map>}` whenever its `done_when` is absent or unparseable
   (`agent/quest_reconciler.py:22` `_criterion`, `compile_steps_to_directives`). So a step means
   "be standing on this map," never "did the thing."
   - **Consequence (traced):** the opening plan *did* contain the parcel errand — `q3 @ Viridian
     Mart (m42)` = pick up, `q4 @ Oak's Lab (m40)` = deliver. At step 67 `q3` **wedged** (cross-map
     nav could not route to m42). At step 69 `q4` was **marked done** — because its criterion was
     `on_map:40` and the agent was *standing in Oak's lab*. It "delivered" a parcel it never had.
     By step 102 both parcel steps were gone from the plan and never returned. At step 1125 the
     agent wandered into the Mart and finally *got* the parcel — but by then no delivery quest
     existed, and after acquisition **zero** quests ever targeted Pallet (0) / Lab (40). It never
     delivered.
   - The same defect hid the heal question: the agent sat in the Viridian PokéCenter (m41) for 72
     steps (751–1073) with **no recorded heal** — the heal step almost certainly completed on
     `on_map:41` (walking in), never talking to the nurse.
   - 982 of 1538 steps (64%) were spent oscillating in Viridian City; 26% of moves returned
     "blocked."

2. **L1 under-specifies.** The fix is *not* to change the reconciler default — it is that L1 must
   *emit* proper DSL criteria for every step ("get the parcel" → `has_item:oaks_parcel`, "deliver"
   → `no_item:oaks_parcel`, "heal" → `hp_frac>=1.0`). The `on_map` fallback for action steps is a
   silent papering-over that must become a loud failure.

3. **L1's single `revise_quests` call is too shallow** to reliably recognize "where am I in the
   game and what should I do." The user's directive: L1 should *reason* — read game state, search
   the knowledge base, brainstorm, and only then decide objectives — across multiple LunaRoute
   calls if that produces sound reasoning. Delivery (and every other objective) must be **recognized
   by the model from game state**, never forced by a deterministic rule.

4. **We were blind.** The recorder serialized plan steps as only `id/map/status` and `l1_last` as
   only `{change, assessment}`, so diagnosing this run required reading RAM by hand. Completion
   provenance ("*why* did this step complete?") was not captured at all.

## Goal

Replace L1's single-shot quest revision with a multi-call **reasoning pipeline** that recognizes
the game situation and emits objectives **with correct, machine-checkable acceptance criteria**,
and make the whole thing observable and testable without spending API credits in CI.

Non-goals (explicitly deferred, tracked separately): cross-map / building navigation and
building-entrance selection (the L2/L3 nav gap). This effort assumes nav remains as-is; it fixes
*what L1 decides and how we verify it*, not *how L2/L3 physically get there*.

## Architecture

**Approach A (chosen):** keep the deterministic reconciler (`agent/quest_reconciler.py`), the
`done_when` DSL (`Planner._parse_done_when`), and the RAM predicate engine
(`games/pokemon_red/predicates.py`) — they are sound and tested; the failure was L1's *output*.
Put all new intelligence in a new focused module, `agent/l1_pipeline.py`, that emits the same
`{add, remove, mission, milestone}` proposal the reconciler already consumes. `reason_loop._run_l1`
calls the pipeline instead of `Planner.revise_quests`.

Rejected: B (rebuild L1 + reconciler — throws away tested machinery) and C (bolt calls onto
`revise_quests` — tangles triage/brainstorm/decide into one function, the same under-factoring that
hid the `on_map` default).

### The pipeline (`agent/l1_pipeline.py`)

```
run_l1_pipeline(emu, context, planner, *, hard_event: bool) -> proposal | None
  │
  ├─ 1. READ STATE (deterministic)
  │      game_signals(emu): map, party, hp_frac, min_level, badges, items  (no LLM)
  │
  ├─ 2. TRIAGE  (cheap, 1 call, deepseek-4.1-flash)  — SKIPPED if hard_event
  │      inputs: current plan + state signals
  │      -> {change: bool, why: str}
  │      change == False and not hard_event  ->  return None  (the common, cheap path)
  │
  ├─ 3. BRAINSTORM  (glm-5.3, tool: search_kb via existing _llm_with_search)
  │      inputs: state signals; issues its OWN KB queries
  │      -> free-form: "where am I in the game, what should I be doing & why"
  │
  ├─ 4. DECIDE  (glm-5.3)
  │      inputs: state signals + brainstorm output + current plan + the DSL grammar
  │      -> proposal {add:[{map, talk?, who?, done_when, why}], remove:[ids],
  │                   mission?, milestone?, assessment}
  │      anchored to the existing plan: prefer minimal edits; the reconciler preserves progress.
  │
  ├─ 5. VALIDATE  (deterministic)
  │      parse every emitted done_when via _parse_done_when.
  │      missing OR unparseable OR on_map-on-an-action-step:
  │        -> ONE repair call (feed back the grammar + the exact parse error) -> re-validate
  │        -> still bad: emit `l1_invalid_criterion`, return None (keep prior plan; NEVER on_map)
  │
  └─ 6. return proposal  ->  reason_loop reconciles as today
```

**Triggering (two-tier, in `reason_loop`):**
- Every N legs (`--l1-every`, default 5): run **triage only**; escalate to brainstorm→decide when
  triage says `change:true`.
- **Hard events** skip triage and run brainstorm→decide directly: map change, key-item
  gained/lost, wedge (`BLOCK_TRIGGER`), HP emergency (`HEAL_EMERGENCY`), plan exhausted, level-up
  past target.

**Model roles:** triage = `deepseek-4.1-flash` (cheap); brainstorm + decide + repair = `glm-5.3`
(the strategist). All calls stream, `reasoning_effort="none"`.

### Acceptance-criteria contract

- The DECIDE prompt is handed the exact `done_when` DSL grammar and **must** attach a criterion to
  every step it emits. Grammar (unchanged): `on_map`, `talked`, `has_item:<name>`,
  `no_item:<name>`, `level>=<N>`, `badges>=<N>`, `hp_frac>=<F>`, `verify:<free-text>` (LLM-judged
  for facts not in RAM).
- **`on_map` is a valid criterion only for a pure-travel step** (a leg whose entire purpose is
  "reach map X"). For any action step (talk / pickup / deliver / heal / grind) a missing or
  `on_map` criterion is a validation error → repair-once-then-break.
- Canonical mappings the prompt teaches by example: pickup → `has_item:<item>`; deliver → the
  post-state `no_item:<item>`; heal → `hp_frac>=1.0`; grind → `level>=<N>`; gym badge →
  `badges>=<N>`; story beat not in RAM → `verify:<statement>`.
- The reconciler's `_criterion` **stops defaulting to `on_map`**: it takes the (now-guaranteed)
  parsed criterion. A `None` criterion reaching compile is a bug and raises (belt-and-suspenders;
  validation in step 5 should prevent it).

## Testing strategy

Hard constraint: L1 calls are non-deterministic and cost credits; acceptance-criteria checks read
RAM and are deterministic. Split them and test each where it is cheap.

**Layer 1 — Machinery (offline, deterministic, no LLM). The bulk of coverage.**
New test double `RamEmulator` (a dict of `address → value` with `set_party_hp`, `set_bag_item`,
`set_map`, `set_badges`, `set_level`) — today's `FakeEmulator` only serves X/Y/map.
- DSL round-trip: every `done_when` form parses to the right predicate; malformed → `None`.
- **Anti-regression:** compiling an *action* step with no criterion (or `on_map`) raises; a travel
  step may keep `on_map`.
- **Heal lifecycle (the spine):** a step with `done_when="hp_frac>=1.0"` is *not* satisfied at low
  HP, *is* satisfied after `set_party_hp` to full → step marks `done`.
- **Parcel lifecycle (anti-regression for the exact bug):** `has_item:oaks_parcel` unsatisfied
  until the item byte is set; `no_item:oaks_parcel` (deliver) unsatisfied *while the item is held*
  — so a "deliver" step standing in the lab with the parcel in the bag stays `active`.
- Reconciler: preserve done/active, remove pending, drop wedged, dedup — extend
  `tests/unit/test_quest_reconciler.py`.

**Layer 2 — Pipeline wiring (offline, deterministic, STUBBED LLM).** Inject a fake planner
returning canned triage/brainstorm/decide outputs; assert the control flow:
- triage `change:false` → brainstorm/decide never called, plan untouched.
- **HP emergency hard-event → skips triage → deep pipeline runs → (stub decide returns a heal
  quest) → a `hp_frac>=1.0` step lands in the plan.** (Your "HP drops → heal quest" at the
  decision layer, made deterministic.)
- decide emits a malformed `done_when` → **one repair call fires**; if fixed, applied; if still
  bad → `l1_invalid_criterion` logged, plan unchanged (never `on_map`).
- brainstorm actually issues `search_kb` queries (assert the KB tool-loop ran).

**Layer 3 — Live end-to-end (real LunaRoute + real emulator, `@pytest.mark.live`, NOT in CI).**
Skipped by default; run manually when credits are up. Load `viridian_stuck.state` / `pc_stuck.state`,
run the loop, confirm the *real* model chooses sane quests and they complete; plus the full viewer
run. The only layer that costs credits or can flake — deliberately opt-in.

Enablers in scope: the `RamEmulator` double, and a thin **`write_memory`** on the PyBoy adapter
(and `Emulator` interface) for Layer-3 live tests (poke HP to force a heal without grinding).

## Observability (pure instrumentation, no behavior change)

1. **Full quest state in the log:** serialize each step with `done_when`, `why`, `talk/who` — not
   just `id/map/status` (`logging/run_recorder.py`).
2. **Completion provenance events:** on `active→done` / `active→wedged`, emit an event naming the
   criterion that fired (`step h1 done: hp_frac>=1.0 satisfied`) or the wedge reason (`no route to
   map 42, 6 legs`). This is the single most valuable thing missing today.
3. **L1 pipeline trace:** expand `l1_last` + events with triage `{change, why}`, brainstorm summary
   + the `search_kb` queries issued, decide's `{add:[{map,done_when,why}], remove}`, and any
   `l1_invalid_criterion` / repair outcome.
4. **Viewer panels:** surface (1) and (2) in `logging/player.html` — each step shows its active
   criterion and, at transitions, the completion reason.

## Files

- **Create:** `src/pokemon_agent/agent/l1_pipeline.py` — triage / brainstorm / decide / validate+repair.
- **Create:** `tests/unit/test_l1_pipeline.py` — Layer-2 wiring (stubbed LLM).
- **Create/extend test double:** `RamEmulator` (in `tests/conftest.py` or `tests/support/`).
- **Modify:** `agent/reason_loop.py` — `_run_l1` calls the pipeline; two-tier triage/deep gating;
  completion-provenance events; emit `l1_invalid_criterion`.
- **Modify:** `agent/quest_reconciler.py` — `_criterion` no longer defaults to `on_map`; `None`
  criterion for an action step raises.
- **Modify:** `agent/planner_llm.py` — DECIDE/brainstorm/triage prompts + repair prompt; reuse
  `_llm_with_search`; keep `_parse_done_when` (extend tests).
- **Modify:** `emulator/interface.py`, `emulator/pyboy_adapter.py`, `emulator/fake_emulator.py` —
  add `write_memory`.
- **Modify:** `logging/run_recorder.py`, `logging/player.html` — observability.
- **Modify:** `tests/unit/test_quest_reconciler.py`, `test_predicates.py`, `test_l1_planner.py` —
  extend for the new contract.

## Risks & mitigations

- **Plan thrashing** from free-form brainstorm each deep run → the two-tier triage gates whether
  deep runs at all; the DECIDE call is anchored to the existing plan (prefer minimal edits) and the
  reconciler preserves `done`/`active`. Observability (l1 trace) makes thrash visible if it occurs.
- **Cost** of multi-call deep pipeline → triage is the cheap common path; deep runs only on
  `change:true` or hard events.
- **Repair loop non-termination** → exactly one repair attempt, then break (keep prior plan).
- **`verify:` overuse** (LLM-judged, non-deterministic) → prompt prefers RAM-checkable criteria;
  `verify:` only for facts genuinely not in RAM.

## Success criteria

- On the `viridian_stuck` / parcel fixtures, a "deliver Oak's Parcel" objective is emitted with
  `no_item:oaks_parcel` and does **not** complete while the parcel is held in the lab.
- A low-HP state produces a heal objective with `hp_frac>=1.0` that completes only after HP is
  restored.
- No step ever completes by arrival unless it is an explicit travel leg.
- CI (Layers 1–2) is green and spends **zero** API credits; the log/viewer show each step's
  criterion and why it completed.
