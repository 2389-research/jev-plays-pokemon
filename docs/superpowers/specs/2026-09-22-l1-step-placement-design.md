# L1 Step Placement — Design

**Goal:** Let L1 say *where* a new plan step belongs, so a step it means to do **later** (e.g. "go to Pewter City — after delivering the parcel") is not executed **before** a pending step it depends on. Structured output, not prose: L1 names the anchor as a field; the reconciler places it deterministically. When L1 says nothing, placement is exactly today's.

**Context:** Claude/Gemini-Plays-Pokémon-style run — L1 plans its own goals and edits its plan each review (triage → brainstorm → DECIDE → validate/repair → reconcile). Follows the interaction-reliability fixes (`2026-09-22-interaction-reliability-design.md`), whose live acceptance exposed this.

---

## 1. Incident (live acceptance run `runs/fix-accept-20260922-2259`)

- Step 0: L1 planned correctly — `[q1 travel→Pallet Town (active), q2 deliver Oak's Parcel @ Oak's Lab, done_when no_item:Oaks Parcel (pending)]`.
- Step 25: DECIDE's own assessment: *"Plan's delivery steps (q1, q2) are valid and active; missing only the **post-delivery** travel north to Pewter and the gym fight."* It added `q3 travel→Pewter City` ("…**after delivering the parcel**…") and `q4 badges>=1 @ Pewter Gym`.
- The reconciler placed them **before** the pending delivery: plan became `[q1, q3, q4, q2]`.
- Step 183: arriving in Pallet completed the travel step; the executive advanced to `q3` and turned north. The parcel was never delivered in 250 steps.

Two more facts from the logs: at **step 93** L1 saw the misordered plan `[q1, q3, q4, q2]` and still answered "no changes needed" — L1 does not audit order, so a wrong placement persists. At **step 138** `q1` was wedged (no active step); L1 removed it and re-added a replacement unanchored, which correctly went to the front.

The captured DECIDE input for step 25 is in `runs/fix-accept-20260922-2259/decisions.jsonl` (layer `l1_decide`, step 25) — exact plan, brainstorm, party, signals.

## 2. Root cause

`quest_reconciler.reconcile_quests` always returns `done + active + new_steps + pending`: **every** added step is inserted immediately after the active step, ahead of all pending steps. The DECIDE output schema (`add: [step objects]`) has no way to express position, so "add a later step" and "add a next step" are indistinguishable. L1's intent was unambiguous in its prose (assessment + step `why`), but prose isn't parsed.

"Insert after active" is the right default for the common add ("do this next": heal first, replace a wedged step, go buy X before continuing). It is wrong for "extend the plan past a pending step".

## 3. Design

### 3.1 Schema — optional `after` on an added step
Each object in DECIDE's `add` list MAY carry `"after"`:
- `"<plan step id>"` — place the new step after that step. A **valid anchor** is the id of a step that **survives into the result with status active or pending**. Anything else — unknown (including a guessed id of another new step — new steps get ids only after the model responds), wedged, done, removed in this proposal, or non-string — **falls back** to default placement;
- `"end"` — append at the current tail of the plan;
- omitted / `null` — **today's behavior (default placement)**: immediately after `done + active`, i.e. at the head of the pending steps. This is well-defined even when there is **no** active step (the step-138 wedge-replacement case).

`after` is placement-only; it is not stored on the `QuestStep`.

### 3.2 Reconciler (`agent/quest_reconciler.py`, pure)
`reconcile_quests(current, proposal, *, next_id, on_event=None)`:
1. Unchanged: keep done + active, drop removed pending steps and all wedged steps, dedup adds by `(map, done_when)` (an anchored add that dedups away is dropped silently, as today).
2. Normalize: an `after` equal to the active step's id means "next" → treated as `None`. Invalid anchors (per §3.1) → `None` + `on_event("l1_anchor_fallback", {"after": value, "reason": "unknown|wedged|done|removed|not_string"})`.
3. Start from `base = done + active + pending` (pending order preserved). Process adds **in emitted order**:
   - `None` → insert at the default slot (after `done + active`, after any default-placed adds already inserted) — so with no anchors the result is byte-identical to today's `done + active + new + pending`;
   - `"<id>"` → insert after that anchor **and after any adds already placed on that anchor**;
   - `"end"` → append at the current tail.
4. Never drops a surviving add; fallbacks only change placement.

### 3.3 Pipeline (`agent/l1_pipeline.py`)
- `validate_step` is unchanged (it already ignores unknown keys; placement isn't a criterion). Non-string anchors are handled by the reconciler fallback.
- **REPAIR preservation:** after a successful `l1_repair`, **always overwrite** the fixed step's `after` with the original step's value (placement is not repair's job; REPAIR may drop or echo a mangled value). `REPAIR_SYSTEM` is unchanged.
- The `decide` trace stage records the anchors (`[step.get("after") for step in add]`), so `l1_last.trace` and the viewer show them.
- `_run_l1` (reason_loop) passes `on_event=self.on_event` to `reconcile_quests`, so fallback events reach the run recorder.

### 3.4 Prompt (`planner_llm.DECIDE_SYSTEM`) — structured, minimal
- Add `"after": null` (the **default literal**, not a union string — a visible null leans toward omission) to both step-shape lines in the schema.
- One rule, stated as a field contract:
  *PLACEMENT — `after` says where a new step goes. Leave it null (the default) for something to do NEXT, before the rest of the plan — heals and replacements for a wedged step are always NEXT. Set `"after": "<id>"` only when the step must come AFTER an existing step that hasn't happened yet; the id must be from PLAN with status active or pending. Several steps with the same `after` run in the order you list them. `"end"` appends after everything.*
- *As shipped (after the first live eval):* the rule also says `after` **is not inherited** from the previous listed step (one run anchored only the first of three later steps, expecting chaining), the example shows **two** anchored steps, and it says **emergency** heals are NEXT (a planned, non-urgent heal can legitimately be anchored later — seen live at step 140 of the e2e run).
- One worked example, in a **different domain from the incident** (so the held-out replay measures generalization, not copying): plan `[q4 travel→Viridian City (active), q5 buy Potions @ Viridian Mart (pending)]`; adding grinding on Route 2 that should happen after shopping → `{"kind":"action","map":13,…,"done_when":"level>=10","after":"q5"}`.
- The return-JSON skeleton shows `after` inside the step objects. DECIDE uses `response_format: json_object` (no JSON schema), so there is no schema to update.
- Nothing else in the prompt changes — the "CHANGING NOTHING IS THE COMMON, PREFERRED OUTCOME" block is untouched (it fixed the L1 thrash).

### 3.5 Visibility
The existing `l1_review` event gains `anchors: [after values]`; `l1_anchor_fallback` events surface misuse in run logs and the viewer.

## 4. Testing

**Unit (reconciler, pure) — fail today (the key is ignored → `[q1, new, q2]`; `on_event` raises TypeError):**
| Case | Expect |
|---|---|
| no `after` anywhere | identical to today's output (regression) |
| `after: q2` (pending) with plan `[q1 active, q2 pending]` | `[q1, q2, new]` |
| two adds `after: q2` | `[q1, q2, newA, newB]` (emitted order) |
| `after: "end"` with `[q1 active, q2, q3]` | `[q1, q2, q3, new]` |
| `after: q1` (the active step) | normalized to default: right after q1, ordered with other default adds by emission |
| mixed: default add, `after: q3` (last pending), `"end"` | default after active; the q3-anchored add after q3; "end" at the tail after it |
| anchor unknown (incl. a guessed new-step id) / wedged / done / removed / non-string | default placement + one `l1_anchor_fallback` event each |
| no active step (wedge replacement) + unanchored add | the add is first among non-done steps (step-138 shape) |
| anchored add whose `(map, done_when)` dedups | dropped, no error, no event |
| **incident (pure):** captured step-25 plan + adds anchored `after: q2` | delivery `q2` precedes Pewter travel + gym |

**Integration-style (unit):** reconcile the incident plan with anchored adds, then `compile_steps_to_directives` / the loop's quest recompile: the queue is q1's remaining directives, then q2's TRAVEL + TALK_TO, then q3, then q4; the current `_directive` is unchanged.

**Unit (pipeline):** a step repaired by `l1_repair` carries the ORIGINAL `after` (overwritten even if repair emitted a different value); the decide trace stage lists the anchors.

**Live (isolation, spends strategist credits — small; strategist pinned to `glm-5.3`, the model recorded in the incident run):**
Replay mapping for captured inputs: the record's `input` is the DECIDE `state` dict → call `Planner.l1_decide(context=input, brainstorm={"assessment": input["brainstorm"]})`.
1. **Held-out incident replay** (captured step-25 input, N=5): in ≥4/5 the Pewter travel + gym reconcile **after `q2`**; every step well-formed.
2. **Negative controls** (N=5 each):
   - (a) captured **step-138** input (wedge replacement): the replacement reconciles to the front — no `after` pointing at `q2` or `"end"` — in ≥4/5;
   - (b) synthetic **emergency heal** (low-HP signals, pending plan): the heal is default-placed (no `after`) in ≥4/5;
   - (c) synthetic **steady state** (correctly ordered `[q1 travel→Pallet active, q2 deliver pending, q3 travel→Pewter pending, q4 gym pending]`, matching brainstorm): empty add/remove in ≥4/5.
3. **Regression:** `scripts/eval_l1_decide.py` (its all-wedged scenario; `score_step` already ignores extra keys — optionally also check `after` is a string or null) — well-formedness must not drop; and `scripts/eval_criteria.py` (criteria scorecard over `tests/fixtures/criteria_cases.py`) — score must not drop.

**End-to-end:** resume `runs/brock-run-20260922-1839` (parcel still in bag) ≤300 steps with `--capture distill --headless`. Note `--resume-from` restores the AgentPlan (mission/milestone/tried_failed) but not `_plan_steps` — L1 rebuilds the plan; clear stale `tried_failed` entries (e.g. "pick up Oak's Parcel…") in a **copy** of the run dir (never edit the original record). **Invariant (must always hold):** whenever the delivery step and a Pewter travel step both exist in `plan_steps`, the delivery precedes it. **Pass:** the invariant holds, the parcel is delivered, then the agent heads north. **Exercised:** only if some `l1_review` carries an anchor equal to the delivery step's id — otherwise report the run as *passed, placement not exercised* (L1 may emit delivery + Pewter in one DECIDE, where emitted order alone suffices).

## 5. Out of scope / known limitations
- Letting L1 reorder or move *existing* steps (only new steps are placed). **Known limitation:** an already-misordered plan, or a wrong anchor, can only be corrected by remove + re-add — L1 does not audit order (step 93 of the incident). Acceptable for now (YAGNI).
- Anchoring one new step to another new step (use a shared anchor instead).
- The starter-selection gap: runs start from the post-Squirtle save (`states/pallet_ready.state`, `--state pallet_ready`); the agent is not expected to pick a starter.


## 6. Results (2026-09-22, glm-5.3)

**Held-out incident replay** (captured step-25 input). Outcomes: MISORDERED = any off-delivery-path step before the pending delivery (the incident); ANCHORED = later steps added, all after the delivery; DEFERRED = no later steps added yet; failed calls (empty fallback) excluded — 0 occurred in these runs.

| Prompt | Misordered | Anchored | Deferred |
|---|---|---|---|
| old (`7e95e24`, no `after`; new reconciler reproduces old placement exactly), 2×N=10 | **4/20 (20%)** | 0 | 16 |
| new (as shipped), 3×N=10 | **0/30** | **7/30** (every later step `after: q2`) | 23 |

**Negative controls (new prompt, N=10):** emergency heal default-placed 10/10; steady state zero edits 10/10; wedge replacement re-added at the front on every successful call (A/B vs the previous prompt revision: failures were empty-fallback failed calls, not decisions). **Regression:** `eval_criteria` 5/5 (= baseline); `eval_l1_decide` 32/32 well-formed on glm-5.3 and deepseek-4.1-flash (= baseline).

**End-to-end** (`runs/placement-e2e-20260922-2324`, clean resume copy): parcel delivered at step 238; the delivery-before-Pewter invariant held on every step; one live anchor (`after: "q3"`, a planned Viridian heal) placed correctly; run ended inside Oak's Pokédex cutscene at the 300-step budget (97 Route-1 battle steps), so "heads north" was not reached. Placement on the delivery anchor itself: not exercised live (L1 emitted delivery + Pewter in one DECIDE, where emitted order suffices).

**Criterion amendment — PENDING SIGN-OFF.** §4.1 as written ("≥4/5 reconcile after q2") is not met: with the new prompt L1 usually *defers* adding post-delivery steps rather than anchoring them. Proposed: pass = **0 misordered** across N≥20, with ANCHORED and DEFERRED both acceptable (the safety property is ordering; deferral is a legitimate minimal-edit choice under the "changing nothing is preferred" framing).
