# L1 Tiered Goals + Notepad — Design

**Goal:** Give L1 a Gemini-Plays-Pokémon-style goal hierarchy it owns — **primary / secondary / tertiary** goals as short text with an optional RAM-checkable criterion — plus a small, rewritten **notepad** (Claude-Plays-Pokémon-style knowledge base) for its own intentions and lessons. The harness provides broad guidance and bookkeeping; **L1 decides what matters at each point in time** and plans concrete steps within its goals, including diversions (heal, shop, story gates).

**Context:** L1 = the LunaRoute strategist running the review pipeline (triage → brainstorm → DECIDE → validate/repair → reconcile) over a queue of typed concrete steps (`QuestStep`). Follows the L1 step-placement work (`2026-09-22-l1-step-placement-design.md`), whose evals showed L1 naturally keeps a short concrete horizon (it deferred post-delivery steps in 23/30 replays) — i.e. long-range intent needs a home that isn't the step queue.

Decisions (brainstormed with the user, 2026-09-23):
1. Goals are **free text + optional check** (`done_when`); L1 decides completion — a met criterion is a signal, not an order.
2. Tiers are **simultaneous horizons**, not a sequence to finish one at a time. Diversions (heal, shop, story gate) are L1 **rewriting a tier**; L1 **re-focuses itself** afterwards (no harness push/pop). The harness keeps a one-level `interrupted` note so L1 sees what it paused.
3. A bounded notepad L1 **rewrites** each review; it **replaces** the write-only `AgentPlan.tried_failed` ledger.

---

## 1. Current state (verified)

- `AgentPlan` has `mission` (≈ primary) and `milestone` (≈ secondary) strings. DECIDE returns them; triage, brainstorm, DECIDE, and the L2 proposer read them (`milestone`); the recorder logs them.
- The concrete plan is `_plan_steps` (typed steps with `done_when`), compiled into directives the executive runs.
- `AgentPlan.tried_failed` is appended with the `why` of every step L1 removes (reason_loop `_run_l1`) and persisted — but **no L1 prompt ever reads it**. It had grown to 79 stale entries (incl. many "deliver Oak's Parcel" variants) in `runs/brock-run-20260922-1839`. `AgentPlan.hypotheses` is unused.
- A **separate** `AgentMemory.tried_failed` (via `mark_tried_failed`) is read only by the L2 travel-reroute prompt (`_choose_target_map`) — **out of scope**, unchanged.
- `AgentPlan` persists via pydantic `model_dump`/`model_validate` in `latest.mem.json`; new fields with defaults are backward-compatible.

## 2. Data model (`agent/plan.py`)

```python
class Goal(BaseModel):
    text: str = ""                 # L1's words, e.g. "Earn the Boulder Badge by beating Brock"
    done_when: str | None = None   # optional; same grammar as quest steps (has_item:/no_item:/level>=/badges>=/hp_frac>=/on_map/talked/verify:)

class AgentPlan(BaseModel):
    ...
    goals: dict[str, Goal] = {"primary": Goal(), "secondary": Goal(), "tertiary": Goal()}
    interrupted: str = ""          # the tertiary text L1 replaced before its criterion was met (one level)
    notes: str = ""                # L1's notepad, <= NOTES_MAX_CHARS (1200)
```
- **Source of truth = `goals`.** `mission`/`milestone` stay as stored fields, **kept in sync** (`mission = goals.primary.text`, `milestone = goals.secondary.text`) whenever goals change, so every existing reader (L2 proposer, recorder, viewer, older prompts) keeps working unchanged.
- **Old checkpoints:** a plan with `mission`/`milestone` but empty `goals` seeds `primary.text`/`secondary.text` from them on load.
- **`tried_failed` retired:** the field remains for loading old checkpoints, is **cleared on load**, and `_run_l1` **stops appending** removed steps to it. `hypotheses` is left as-is (unused; YAGNI).

## 3. Each L1 review

### 3.1 Inputs (added to the review `context` in `_run_l1`)
- `goals`: `{tier: {"text", "done_when"}}`;
- `goal_status`: `{tier: "met" | "unmet" | "none"}` — the harness evaluates each tier's `done_when` with the existing `predicates.evaluate` (after `Planner._parse_done_when`); `"none"` = no criterion; `verify:` criteria are reported `"none"` (not evaluated here — no extra model call);
- `interrupted`: the paused focus text (or empty);
- `notes`: the notepad.
Triage, brainstorm, and DECIDE all receive these (triage/brainstorm/DECIDE `state` dicts already carry mission/milestone — they gain the four keys above).

### 3.2 Outputs (DECIDE)
DECIDE's JSON gains two optional top-level keys (all existing keys unchanged):
- `"goals"`: `{"primary"?: {"text", "done_when"?}, "secondary"?: …, "tertiary"?: …}` — each tier present **replaces** that tier; an absent tier is unchanged; `null`/absent `goals` = no goal change.
- `"notes"`: a string that **replaces** the notepad; `null`/absent = unchanged.
`mission`/`milestone` keys, if DECIDE still emits them, are accepted as aliases for primary/secondary text when `goals` is absent for that tier (back-compat).

### 3.3 Harness bookkeeping (`reason_loop._run_l1`, pure helpers in `agent/goals.py`)
- **Validation:** a tier's `done_when` is checked with `Planner._parse_done_when`; if it doesn't parse, the **criterion is dropped and the text kept** (never reject a goal over a bad criterion); empty text for a tier = clear that tier.
- **interrupted:** when the tertiary text changes and the **old** tertiary had a criterion that was `unmet` (or no criterion), set `interrupted` = old tertiary text; when the new tertiary equals `interrupted` (L1 returned to it), clear `interrupted`. One level only.
- **notes:** trimmed to `NOTES_MAX_CHARS` (1200); a `notes_truncated` flag in the event if cut.
- **Events:** `goals_changed {before, after}` and `notes_changed {len, truncated}`; the recorder's per-step `extra` gains `goals`, `interrupted`, `notes` (the viewer can show them; `mission`/`milestone` stay).
- **Reviews that make no change** (the pipeline's "no-op DECIDE == no change" path) keep goals/notes as they are — but a DECIDE that changes **only** goals or notes (no step add/remove) is **applied** (it is a real edit, not a no-op): the pipeline's no-change short-circuit is widened to consider goals/notes.

### 3.4 Prompt guidance (broad, structured; DECIDE — plus one line each in triage/brainstorm)
- The three tiers: **primary** = the long-term why (e.g. earn the badge); **secondary** = the current chapter (e.g. get to Pewter Gym with a team that can win); **tertiary** = your immediate focus, which may be a **diversion** (heal, shop, grind, a story errand).
- To divert, rewrite the tertiary (or the secondary if the chapter itself changed); `interrupted` shows what you paused — return to it (or drop it) when the diversion is done.
- Plan concrete steps for your current focus and chapter; keep later chapters as goals/notes, not queued steps.
- `goal_status: met` means that goal's criterion holds in the game right now — a hint; **you** decide when to move on.
- `notes`: your own short notepad (≤ ~1200 chars) — intentions for later ("after the parcel: heal, then Route 2 → forest → Pewter"), lessons ("Brock is Rock-type; Water moves are strong"), things not to retry. **Rewrite** it (keep what still matters, drop what doesn't); omit to keep it unchanged.
- The "CHANGING NOTHING IS THE COMMON, PREFERRED OUTCOME" block stays, and applies to goals and notes too (don't restate them each review).

### 3.5 Unchanged
The step queue and reconciler (incl. the `after` placement anchor, now a backstop), the executive, the emergency-heal reflex (a near-faint still forces an L1 heal ping), battle/shop layers, and `AgentMemory.tried_failed` (L2 reroute).

## 4. Testing

**Unit (pure — `tests/unit/test_goals.py`):** goal merge (tier replace / absent unchanged / empty clears); invalid `done_when` dropped, text kept; `goal_status` met/unmet/none (via a FakeEmulator/RAM stub for `badges>=`, `no_item:`, `hp_frac>=`); `interrupted` set on an unmet-tertiary change, cleared on return, one level; notes replace/unchanged/truncate; `mission`/`milestone` stay in sync; old-checkpoint seeding; `tried_failed` cleared on load and not appended by `_run_l1`; a goals-only DECIDE is applied (not short-circuited as no-change).

**Live scenario evals (`scripts/eval_l1_goals.py`, glm-5.3, N=10 each; failed calls — empty fallback — excluded and counted separately):** built from captured DECIDE inputs where available, else synthetic:
1. **Diversion (heal):** mid-chapter (secondary = get to Pewter), HP low → tertiary becomes a heal and a heal step is added NEXT; primary/secondary unchanged. Pass ≥ 8/10.
2. **Return:** after healing (HP full), `interrupted` = "cross Viridian Forest" → tertiary returns to the forest (or an equivalent forward focus); no heal re-added. Pass ≥ 8/10.
3. **Story gate:** parcel in bag, north blocked → secondary becomes the delivery (or the delivery is the focus with a delivery step); no off-path step queued before the delivery. Pass ≥ 8/10, and **0** misordered.
4. **Steady state:** goals + plan already correct → no goal/notes change and no step edits. Pass ≥ 8/10.
5. **Chapter complete:** `goal_status.secondary = met` (parcel delivered) and `notes` says "after the parcel: heal, then Route 2 → forest → Pewter" → secondary advances and the next concrete step matches the note's intent. Pass ≥ 8/10.
6. **Notes hygiene:** across the scenarios, returned notes stay ≤ 1200 chars and don't just echo the input unchanged when the situation changed (report, no threshold).
**Regression:** `eval_criteria` (5/5), `eval_l1_decide` (32/32 well-formed, both models), `eval_l1_placement` (0 misordered; controls ≥ 8/10).

**End-to-end:** from `states/pallet_ready.state` (`--state pallet_ready`), ≥ 600 steps, `--capture distill --headless`. Report per-goal metrics: steps per secondary goal, diversions taken and whether L1 returned from them, goal churn (goal changes per 100 steps), notes length over time, and the coherence invariant (no off-path step executed before a pending delivery). Pass: the parcel is delivered and the agent heads north past Viridian City; goal churn is reported (no threshold in this round).

## 5. Out of scope
- Harness-enforced goal completion or push/pop goal stacks (L1 re-focuses itself).
- Step ↔ goal tagging (which goal a step serves) — revisit if coherence evals need it.
- `AgentMemory.tried_failed` (L2 reroute ledger) and `hypotheses`.
- Changing the step reconciler or executive.
