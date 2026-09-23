# L1 Step Placement — Design

**Goal:** Let L1 say *where* a new plan step belongs, so a step it means to do **later** (e.g. "go to Pewter City — after delivering the parcel") is not executed **before** a pending step it depends on. Structured output, not prose: L1 names the anchor as a field; the reconciler places it deterministically. When L1 says nothing, placement is exactly today's.

**Context:** Claude/Gemini-Plays-Pokémon-style run — L1 plans its own goals and edits its plan each review (triage → brainstorm → DECIDE → validate/repair → reconcile). Follows the interaction-reliability fixes (`2026-09-22-interaction-reliability-design.md`), whose live acceptance exposed this.

---

## 1. Incident (live acceptance run `runs/fix-accept-20260922-2259`)

- Step 0: L1 planned correctly — `[q1 travel→Pallet Town (active), q2 deliver Oak's Parcel @ Oak's Lab, done_when no_item:Oaks Parcel (pending)]`.
- Step 25: DECIDE's own assessment: *"Plan's delivery steps (q1, q2) are valid and active; missing only the **post-delivery** travel north to Pewter and the gym fight."* It added `q3 travel→Pewter City` ("…**after delivering the parcel**…") and `q4 badges>=1 @ Pewter Gym`.
- The reconciler placed them **before** the pending delivery: plan became `[q1, q3, q4, q2]`.
- Step 183: arriving in Pallet completed the travel step; the executive advanced to `q3` and turned north. The parcel was never delivered in 250 steps.

The captured DECIDE input for step 25 is in `runs/fix-accept-20260922-2259/decisions.jsonl` (layer `l1_decide`, step 25) — exact plan, brainstorm, party, signals.

## 2. Root cause

`quest_reconciler.reconcile_quests` always returns `done + active + new_steps + pending`: **every** added step is inserted immediately after the active step, ahead of all pending steps. The DECIDE output schema (`add: [step objects]`) has no way to express position, so "add a later step" and "add a next step" are indistinguishable. L1's intent was unambiguous in its prose (assessment + step `why`), but prose isn't parsed.

"Insert after active" is the right default for the common add ("do this next": heal first, replace a wedged step, go buy X before continuing). It is wrong for "extend the plan past a pending step".

## 3. Design

### 3.1 Schema — optional `after` on an added step
Each object in DECIDE's `add` list MAY carry `"after": "<id>"`:
- `"<existing step id>"` — place the new step immediately after that step (which must be **active or pending**);
- `"end"` — append after the last step of the plan;
- omitted / `null` — **today's behavior**: immediately after the active step (before all pending steps).

`after` is placement-only; it is not stored on the `QuestStep`.

### 3.2 Reconciler (`agent/quest_reconciler.py`, pure)
`reconcile_quests(current, proposal, *, next_id, on_event=None)`:
1. Unchanged: keep done + active, drop removed/wedged, dedup adds by `(map, done_when)`.
2. Start from `done + active + pending` (pending order preserved). Insert each **unanchored** add after the active step, in emitted order — so with no anchors the result is byte-identical to today's `done + active + new + pending`.
3. Insert each **anchored** add after its anchor. Several adds sharing one anchor keep their emitted relative order (each goes after the previously inserted add for that anchor).
4. **Fallbacks (never drop a step):** anchor is unknown, removed in the same proposal, a `done` step, or not a string → treat as unanchored (default placement) and emit `on_event("l1_anchor_fallback", {"after": ..., "reason": ...})` if a callback is given.
5. An anchor may not reference another add (new steps have no id yet) — that is also a fallback. (Chained later-steps are expressed by giving them the same anchor, in order.)

### 3.3 Pipeline (`agent/l1_pipeline.py`)
- `validate_step` ignores `after` (placement isn't a criterion); a non-string `after` is dropped with a trace note rather than failing the step.
- **REPAIR preservation:** `l1_repair` re-emits a step from scratch and may omit `after`. After a successful repair, re-attach the original step's `after` if the fixed step lacks one.

### 3.4 Prompt (`planner_llm.DECIDE_SYSTEM`) — structured, minimal
- Add `"after"` to both step-shape lines in the schema (`"after": "<plan step id> | \"end\" | null"`).
- One rule, stated as a field contract:
  *PLACEMENT — `after` says where the new step goes. Omit it (the default) for something to do NEXT, before the rest of the plan. Set `"after": "<id>"` when the step must come AFTER an existing active/pending step (e.g. anything that depends on a delivery or pickup that hasn't happened yet); several steps with the same `after` run in the order you list them. Use `"end"` to append after everything.*
- One worked example: plan `[q1 travel→Pallet (active), q2 deliver Oak's Parcel (pending)]`; adding the trip north → `{"kind":"travel","map":2,…,"after":"q2"}` then `{"kind":"action","map":54,…,"after":"q2"}`.
- The return-JSON skeleton shows `after` inside the step objects.
- Nothing else in the prompt changes (keeps the "changing nothing is preferred" framing that fixed the L1 thrash).

### 3.5 Visibility
The existing `l1_review` event gains `anchors: [after values]`; `l1_anchor_fallback` events surface misuse in run logs and the viewer.

## 4. Testing

**Unit (reconciler, pure):**
| Case | Expect |
|---|---|
| no `after` anywhere | identical to today's output (regression) |
| `after: q2` (pending) with plan `[q1 active, q2 pending]` | `[q1, q2, new]` |
| two adds `after: q2` | `[q1, q2, newA, newB]` (emitted order) |
| `after: "end"` with `[q1 active, q2, q3]` | `[q1, q2, q3, new]` |
| `after: q1` (the active step) | same as default: right after q1 |
| unknown id / removed id / done id / non-string | default placement + `l1_anchor_fallback` event each |
| mix of anchored + unanchored adds | unanchored after active, anchored after anchors |
| **incident replay (pure):** plan from the captured step-25 input + adds anchored `after: q2` | delivery `q2` precedes Pewter travel + gym |

**Unit (pipeline):** `after` survives `validate_step`; a step repaired by `l1_repair` keeps its original `after`; a non-string `after` doesn't fail validation.

**Live (isolation, spends strategist credits — small):**
1. **Decision replay of the incident:** feed the captured step-25 `l1_decide` input (from `decisions.jsonl`) to `Planner.l1_decide` with the new prompt, N=5. Pass: in ≥4/5, the Pewter travel and gym steps carry `after: "q2"` (or otherwise reconcile to after `q2`), and every emitted step is well-formed.
2. **Regression:** re-run `scripts/eval_l1_decide.py` (the isolation harness that confirmed 32/32 well-formed after the L1-thrash fix) — well-formedness must not drop, and the "change nothing" rate on its steady-state scenario must not drop.

**End-to-end:** resume `runs/brock-run-20260922-1839` (parcel still in bag) ≤300 steps with `--capture distill --headless`: parcel delivered, then the agent heads north (reaches Viridian City or beyond with the old man no longer blocking).

## 5. Out of scope
- Letting L1 reorder or move *existing* steps (only new steps are placed).
- Anchoring one new step to another new step (use a shared anchor instead).
- The starter-selection gap: runs start from the post-Squirtle save (`states/pallet_ready.state`, `--state pallet_ready`); the agent is not expected to pick a starter.
