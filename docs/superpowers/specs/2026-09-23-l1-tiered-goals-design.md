# L1 Tiered Goals + Notepad — Design

**Goal:** Give L1 a Gemini-Plays-Pokémon-style goal hierarchy it owns — **primary / secondary / tertiary** goals as short text with an optional RAM-checkable criterion — plus a small, rewritten **notepad** (Claude-Plays-Pokémon-style knowledge base) for its own intentions and lessons. The harness provides broad guidance and bookkeeping; **L1 decides what matters at each point in time** and plans concrete steps within its goals, including diversions (heal, shop, story gates).

**Context:** L1 = the LunaRoute strategist running the review pipeline (triage → brainstorm → DECIDE → validate/repair → reconcile) over a queue of typed concrete steps (`QuestStep`). Follows the L1 step-placement work (`2026-09-22-l1-step-placement-design.md`), whose evals showed L1 naturally keeps a short concrete horizon (it deferred post-delivery steps in 23/30 replays) — long-range intent needs a home that isn't the step queue.

Decisions (brainstormed with the user, 2026-09-23):
1. Goals are **free text + optional check** (`done_when`); L1 decides completion — a met criterion is a signal, not an order.
2. Tiers are **simultaneous horizons**, not a sequence to finish one at a time. Diversions (heal, shop, story gate) are L1 **rewriting a tier**; L1 **re-focuses itself** afterwards (no harness push/pop). The harness keeps a one-level `interrupted` note so L1 sees what it paused.
3. A bounded notepad L1 **rewrites** each review; it **replaces** the `AgentPlan.tried_failed` ledger.

Spec review round 1 (2026-09-23) findings are folded in below (§ markers "R1-#n").

---

## 1. Current state (verified)

- `AgentPlan.mission` (≈ primary) / `milestone` (≈ secondary) strings: DECIDE returns them; triage/brainstorm/DECIDE state dicts carry them; the L2 proposer reads `milestone` (reason_loop ~823); the recorder `extra` logs both (~1863). `l1_decide` back-fills `mission`/`milestone` from the context when the model omits them (planner_llm ~634), so today they are almost always populated.
- **No-op short-circuit** (l1_pipeline): `if not validated_add and not remove: return None` — any mission/milestone/catch change in a DECIDE with no step edit is discarded. Captured data: **20 of 79 no-op DECIDEs (25%) reword the milestone** (e.g. "Deliver Oak's Parcel in Pallet Town, then…" → "Deliver Oak's Parcel to Oak in Oak's Lab, then…") — the short-circuit is what keeps that from becoming churn (R1-#1). `catch`-only DECIDEs are also discarded today (latent bug).
- The concrete plan is `_plan_steps`, compiled into directives the executive runs.
- `AgentPlan.tried_failed` is appended (only) in `_run_l1` with the `why` of each removed step and persisted (79 stale entries in `runs/brock-run-20260922-1839`). No L1 prompt reads it — **but** the menu path does: `_jev_turn` passes `plan=self._plan` to `reasoner.step`, which puts `current_plan: plan.model_dump()` into Jev's menu prompt (reasoner.py ~205, typesafe_reasoner.py ~200), so Jev currently sees all stale entries (R1 §1). `AgentPlan.hypotheses` is unused.
- A **separate** `AgentMemory.tried_failed` is read only by the L2 travel-reroute prompt (`_choose_target_map`) — out of scope, unchanged.
- `AgentPlan` persists via pydantic `model_dump`/`model_validate`; new fields with defaults are backward-compatible. The legacy reflect path (reason_loop ~393) replaces `_plan` wholesale.
- `l1_decide` (planner_llm ~629) and `run_l1_pipeline` both return **whitelisted** dicts — new keys would be silently dropped (R1-#5).

## 2. Data model (`agent/plan.py`) — sync/validation live in the model (R1-#6)

```python
GOAL_TEXT_MAX = 160
NOTES_MAX_CHARS = 1200

class Goal(BaseModel):
    text: str = ""                 # L1's words, <= GOAL_TEXT_MAX
    done_when: str | None = None   # optional, RESTRICTED grammar (below)

class Goals(BaseModel):            # typed tiers — no stray keys
    primary: Goal = Goal()
    secondary: Goal = Goal()
    tertiary: Goal = Goal()

class AgentPlan(BaseModel):
    ...
    goals: Goals = Goals()
    interrupted: Goal = Goal()     # the paused focus (text + criterion), one level
    notes: str = ""                # L1's notepad, <= NOTES_MAX_CHARS
    notes_truncated: bool = False  # surfaced to L1 on the next review
```
- **Goal criterion grammar (R1-#2):** goals have no map, so only map-independent forms are allowed: `has_item:<x>`, `no_item:<x>`, `level>=N`, `badges>=N`, `hp_frac>=F`, and `verify:<q>`. `on_map` / `talked` (map-relative) are **dropped on validation, text kept**. (An explicit `on_map:<id>` form is YAGNI for now.)
- **One `@model_validator(mode="after")` on `AgentPlan`**: seeds `goals.primary/secondary.text` from `mission`/`milestone` when goals are empty (old checkpoints); keeps `mission = goals.primary.text`, `milestone = goals.secondary.text`; clears `tried_failed`. One `apply_goals(update)` method is the only writer of goals (so every path — including the legacy reflect replacement — stays consistent).
- **Jev menu prompt:** `current_plan` uses `model_dump(exclude={"notes", "tried_failed", "hypotheses", "notes_truncated"})` — goals stay visible (useful context), the notepad does not (size, relevance).
- `_run_l1` stops appending to `tried_failed`. `hypotheses` unchanged (YAGNI).

## 3. Each L1 review

### 3.1 Inputs
- **Brainstorm + DECIDE** get: `goals` (`{tier: {text, done_when}}`), `goal_status` (`{tier: "met" | "unmet" | "none" | "unchecked"}`), `interrupted` (`{text, status}`), `notes`, `notes_truncated`.
  - `goal_status`: `"none"` = no criterion; `"unchecked"` = a `verify:` criterion (not evaluated here — no extra model call); otherwise evaluate via `Planner._parse_done_when(dw, None)` + `predicates.evaluate(..., memory=self.memory)`. Cheap RAM reads, per review only.
- **Triage** (the cheap gate most reviews end at) gets `goals`, `goal_status`, `interrupted` — **not** `notes` — plus one line: *"a goal already showing met is not by itself a reason to change; you are pinged when one becomes met."* (R1-#4)
- **Review trigger (R1-#4):** the harness sets `_l1_event = True` **once** when any tier's status goes `unmet → met` (deterministic, bounded). This replaces relying on triage to notice a finished diversion.

### 3.2 Outputs (DECIDE)
DECIDE's JSON: **`mission`/`milestone` are removed** from the schema and the "You are given" list and replaced by (R1-#1):
- `"goals"` *(optional)*: only the tier(s) that changed, e.g. `{"tertiary": {"text": "Heal at the Viridian Pokémon Center", "done_when": "hp_frac>=1.0"}}`; an absent tier is unchanged; empty `text` clears that tier;
- `"notes"` *(optional)*: the full rewritten notepad; absent = unchanged;
- `"interrupted": ""` *(optional)*: explicit "drop the paused focus".
Legacy `mission`/`milestone` keys, if a model still emits them, are honored **only alongside a real step edit** (today's behavior) — never as goal edits on their own.

### 3.3 Change detection + bookkeeping (`agent/goals.py`, pure; applied in `_run_l1`)
- **What counts as a change (R1-#1):** a no-add/no-remove DECIDE is applied only if it carries an explicit `goals`, `notes`, `interrupted`, or `catch` key **whose value differs from the current one after normalization** (whitespace-collapsed, case-insensitive compare; an identical echo is no change). Otherwise the existing short-circuit stands.
- **Plumbing (R1-#5):** `l1_decide` and `run_l1_pipeline` pass `goals`/`notes`/`interrupted` through. A goals/notes/catch-only edit skips `reconcile_quests`, `_recompile_quest`, and the `quest` event, and emits `l1_review` with `change: "goals"` (distinct from step edits, so the thrash metrics aren't inflated).
- **Validation:** criterion outside the goal grammar → dropped, text kept; `text` capped at 160 chars.
- **`interrupted` rules (R1-#3)** — one level, harness bookkeeping only:
  - **set** only when `interrupted` is empty **and** the new tertiary is non-empty **and** the replaced tertiary's status was not `met` → `interrupted` = the replaced tertiary (text + criterion);
  - **clear** when: the new tertiary matches it (normalized); **or** the tertiary being replaced was `met` (the diversion finished and L1 re-focused); **or** its own criterion is `met`; **or** DECIDE sends `"interrupted": ""`;
  - clearing the tertiary (empty text) never sets `interrupted`;
  - so forest → heal → shop keeps `interrupted = forest` (only the first pause is recorded).
- **Notes (R1-#11):** truncated at a line boundary to 1200 chars; `notes_truncated` set and shown to L1 next review; identical echo = no change.
- **Events:** `goals_changed {before, after}`, `notes_changed {len, truncated}`. Recorder per-step `extra` gains `goals` and `interrupted` every step, and `notes` **only on steps where it changed** (plus `notes_len` every step) to bound log size (R1-#10).

### 3.4 Prompt guidance (DECIDE; broad, structured)
Placed directly under the "CHANGING NOTHING IS THE COMMON, PREFERRED OUTCOME" block, and the old "Leave MISSION/MILESTONE unchanged…" line is **replaced** (R1-#7):
- The tiers: **primary** = the long-term why (e.g. earn the Boulder Badge); **secondary** = the current chapter (e.g. get to Pewter Gym with a team that can win); **tertiary** = the immediate focus, which may be a diversion (heal, shop, grind, a story errand).
- *"GOALS and NOTES follow the same rule: omit `goals` and `notes` unless something actually changed — a goal was achieved, you are diverting, or you learned something worth keeping. Rewording is not a change. When you do edit, include only the tier(s) that changed. `goal_status: met` means that criterion holds right now; you decide when to move on. A brief diversion can be just a step (e.g. a heal step) — you don't have to rewrite a goal for it."*
- Diverting / returning: rewrite the tertiary (or the secondary if the chapter itself changed); `interrupted` shows what you paused — return to it, or send `"interrupted": ""` to drop it.
- Plan concrete steps for your current focus and chapter; keep later chapters as goals/notes rather than queued steps.
- `notes`: your own short notepad — intentions for later, lessons, things not to retry. Only send it when it changes; keep it short.
- Goal criteria use only `has_item:`/`no_item:`/`level>=`/`badges>=`/`hp_frac>=`/`verify:` (no `on_map`/`talked`).
Brainstorm gets one line describing goals/notes; triage the one line in §3.1.

### 3.5 Interactions / unchanged
- **Unchanged:** the step queue + reconciler (the `after` anchor stays as a backstop), the executive, battle/shop layers, `AgentMemory.tried_failed`.
- **Emergency heal (R1-#13):** the near-faint path is a hard event (triage skipped) guarded by `_has_heal_step`; a goals-only DECIDE during an emergency that adds no heal step re-fires next step exactly as today.
- **Plan exhaustion (R1-#8):** with later chapters kept as goals, running out of queued steps becomes the normal way a chapter advances, firing the bootstrap hard-event review (and, if DECIDE adds no step, the provisional `goal_map` travel default). Accepted; tracked via L1 reviews per 100 steps in the e2e.
- **L2 proposer (R1-#12):** `milestone` now means "the chapter"; during a diversion L2 still gets the directive's own `objective`, which carries the focus. Also pass the tertiary text as a `focus` field to the L2 context (one line).

## 4. Testing

**Unit (pure — `tests/unit/test_goals.py`; fail today because `agent/goals.py` doesn't exist):** tier merge (replace / absent unchanged / empty clears); criterion grammar (allowed forms kept; `on_map`/`talked`/garbage dropped, text kept; text capped); `goal_status` met/unmet/none/unchecked against a FakeEmulator RAM stub; every `interrupted` rule above incl. forest→heal→shop, return with different-but-normalized wording, return with genuinely different wording (not cleared; met-replaced clears), tertiary cleared, explicit `"interrupted": ""`; notes replace/unchanged/line-boundary truncate + flag; normalized-echo = no change; **the captured reworded-milestone no-op DECIDE still short-circuits** (from `runs/brock-continue-20260922-2217/decisions.jsonl`); a goals-only edit is applied with `change: "goals"` and skips reconcile; a `catch`-only edit is applied; model validator seeding/sync/`tried_failed` clear; Jev `current_plan` excludes notes; `_l1_event` set once on unmet→met.

**Live scenario evals (`scripts/eval_l1_goals.py`, glm-5.3, N=10; failed calls excluded + counted).** Brainstorm is run **live** (`l1_brainstorm(None, ctx)`) or neutral — never a hand-written brainstorm that states the answer (R1-#9). Graders are structural:
1. **Diversion (heal)** — from a captured Route 1/Viridian DECIDE input with an injected emergency (as `heal_case` in eval_l1_placement): pass = an `hp_frac>=` step placed first **and** primary/secondary unchanged (≥ 8/10). Tertiary rewrite is **reported, not required**.
2. **Return** — synthetic from `states/pokecenter_lowhp` post-heal signals (via `game_signals`, no model call), `interrupted` = "cross Viridian Forest", tertiary = a heal whose criterion is now met: pass = no new heal step and the next focus/step is forward (maps in {13, 51, 2} or the interrupted focus) (≥ 8/10).
3. **Story gate** — captured input with **secondary seeded to "get to Pewter"** (so it isn't trivially passing): pass = **0** off-path steps before the delivery (reuse eval_l1_placement `ON_PATH`), and the delivery is either the secondary/tertiary or a queued step (≥ 8/10).
4. **Steady state / churn regression (key)** — replay **all captured no-op DECIDE inputs** with goals seeded consistently: goals/notes edit rate ≤ 1/10 (baseline: 25% milestone rewording). Replay the **captured `l1_triage` inputs** with goals added: report change-rate vs baseline.
5. **Chapter complete** — synthetic from `states/lab_deliver` + `lab_deliver.mem.json` after delivery, `goal_status.secondary = met`, notes = "after the parcel: heal, then Route 2 → forest → Pewter": pass = first added step's map ∈ {1, 13, 51} or a heal (≥ 8/10).
6. **Notes hygiene** — report sizes and echo rate across scenarios (no threshold).
**Regression:** `eval_criteria` (5/5), `eval_l1_decide` (32/32 well-formed, both models), `eval_l1_placement` (0 misordered; controls ≥ 8/10).

**End-to-end (R1-#10):** from `states/pallet_ready.state` (`--state pallet_ready`), ≥ 600 steps, `--capture distill --headless`. **Pass:** the coherence invariant holds (no off-path step executed before a pending delivery); L1 reviews per 100 steps ≤ baseline + 25% (baseline: placement-e2e, 23 `l1_review` / 300 steps); goals-only edits reported per 100 steps. Report separately: diversions taken + returned, steps per secondary goal, notes length over time, delivery (measure delivery from **items** — the parcel leaving the bag — not the labeler's `delivered` field, which is a known-weak heuristic and read `false` even on runs where the parcel was handed over).

## 5. Out of scope
- Harness-enforced goal completion / push-pop stacks; step ↔ goal tagging.
- `AgentMemory.tried_failed`, `hypotheses`.
- Viewer UI for goals (the viewer doesn't show mission/milestone today; showing goals is separate UI work).
- **Follow-up (noted):** fix the outcome labeler's `delivered` to key off the parcel leaving the bag.
