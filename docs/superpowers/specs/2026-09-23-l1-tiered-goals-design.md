# L1 Tiered Goals + Notepad — Design

**Goal:** Give L1 a Gemini-Plays-Pokémon-style goal hierarchy it owns — **primary / secondary / tertiary** goals as short text with an optional RAM-checkable criterion — plus a small, rewritten **notepad** (Claude-Plays-Pokémon-style knowledge base) for its own intentions and lessons. The harness provides broad guidance and bookkeeping; **L1 decides what matters at each point in time** and plans concrete steps within its goals, including diversions (heal, shop, story gates).

**Context:** L1 = the LunaRoute strategist running the review pipeline (triage → brainstorm → DECIDE → validate/repair → reconcile) over a queue of typed concrete steps (`QuestStep`). Follows the L1 step-placement work (`2026-09-22-l1-step-placement-design.md`), whose evals showed L1 naturally keeps a short concrete horizon (it deferred post-delivery steps in 23/30 replays) — long-range intent needs a home that isn't the step queue.

Decisions (brainstormed with the user, 2026-09-23):
1. Goals are **free text + optional check** (`done_when`); L1 decides completion — a met criterion is a signal, not an order.
2. Tiers are **simultaneous horizons**, not a sequence to finish one at a time. Diversions (heal, shop, story gate) are L1 **rewriting a tier**; L1 **re-focuses itself** afterwards (no harness push/pop). The harness keeps a one-level `interrupted` note so L1 sees what it paused.
3. A bounded notepad L1 **rewrites** each review; it **replaces** the `AgentPlan.tried_failed` ledger.

Spec review round 1 findings are marked "R1-#n", round 2 "R2-#n" (both 2026-09-23).

---

## 1. Current state (verified at 33ab266)

- `AgentPlan.mission` (≈ primary) / `milestone` (≈ secondary) strings: DECIDE returns them; triage/brainstorm/DECIDE state dicts carry them; the loop puts `milestone` into the L2 proposer context (reason_loop ~823) but `propose_target`'s state whitelist drops it; the recorder `extra` logs both (~1863). `l1_decide` **back-fills** `mission`/`milestone` from the context when the model omits them (planner_llm ~634), and `_run_l1` assigns them directly on every step edit (reason_loop ~508).
- **No-op short-circuit** (l1_pipeline ~83): `if not validated_add and not remove: return None` — any mission/milestone/catch change in a DECIDE with no step edit is discarded. Captured data (all runs): **20 of 79 no-op DECIDEs reword the milestone**, and **77 of 79 send `"catch": []`** (the other two set a real catch: brock-run-1839 steps 277 `['any']`, 434 `['Pidgey']`). Today `[]` with a step edit **clears** the catch goal (reason_loop ~513) — so an echoed `[]` silently erases a real catch (latent bug, R2-#1).
- `TRIAGE_SYSTEM` says change=true when "the mission/milestone is stale" (planner_llm ~223).
- The concrete plan is `_plan_steps`, compiled into directives the executive runs.
- `AgentPlan.tried_failed` is appended (only) in `_run_l1` with the `why` of each removed step and persisted (79 stale entries in `runs/brock-run-20260922-1839`). No L1 prompt reads it — but `_jev_turn` passes `plan=self._plan` to `reasoner.step`, which puts `current_plan: plan.model_dump()` into Jev's menu prompt (reasoner.py ~205, typesafe_reasoner.py ~200). `AgentPlan.hypotheses` is unused.
- A **separate** `AgentMemory.tried_failed` is read only by the L2 travel-reroute prompt — out of scope. `AgentMemory.notes` (list of `{text, source, step}`) also exists — hence the new field is named **`notepad`** (R2-#14).
- `AgentPlan` persists via pydantic `model_dump`/`model_validate` (`validate_assignment` off); new fields with defaults are backward-compatible.
- The legacy reflect path (reason_loop ~393) runs only when `planner is None` and rebuilds `AgentPlan` wholesale — goals/notepad are **not supported** on that path (R2-#13).
- `l1_decide` and `run_l1_pipeline` both return **whitelisted** dicts — new keys would be silently dropped (R1-#5).

## 2. Data model (`agent/plan.py`)

```python
GOAL_TEXT_MAX = 160
NOTEPAD_MAX_CHARS = 1200

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
    interrupted: Goal = Goal()       # the paused focus (text + criterion), one level
    notepad: str = ""                # L1's notepad, <= NOTEPAD_MAX_CHARS
    notepad_truncated: bool = False  # surfaced to L1 on the next review
```
- **Goal criterion grammar (R1-#2, R2-#12):** goals have no map, so only map-independent forms are allowed: `has_item:<x>`, `no_item:<x>`, `level>=N`, `badges>=N`, `hp_frac>=F`, `verify:<q>`. Validation parses via `Planner._parse_done_when(dw, None)` and **rejects by parsed key** (`on_map`, `talked_on_map`) — the parse does not return None for those — or when the parse is None (e.g. an unresolvable item name). Rejected criterion → dropped, text kept. `verify:` is special-cased before `predicates.evaluate` (which returns False for unknown keys).
- **Model validator (`mode="after"`) is for construction/load only:** seeds `goals.primary/secondary.text` from `mission`/`milestone` when goals are empty (old checkpoints); sets `mission = goals.primary.text`, `milestone = goals.secondary.text`; clears `tried_failed`. `validate_assignment` stays **off** (a validator that assigns `self` fields would recurse).
- **`apply_goals(update)` is the only runtime writer of goals** and does the `mission`/`milestone` sync itself (R2-#2). Nothing else assigns `mission`/`milestone`: the direct assignments in `_run_l1` (~508–511) are deleted.
- **Jev menu prompt:** `current_plan` uses `model_dump(exclude={"notepad", "notepad_truncated", "tried_failed", "hypotheses"})`.
- `_run_l1` stops appending to `tried_failed`. `hypotheses` unchanged (YAGNI).

## 3. Each L1 review

### 3.1 Inputs
- **Brainstorm + DECIDE** get: `goals` (`{tier: {text, done_when}}`), `goal_status` (`{tier: "met" | "unmet" | "none" | "unchecked"}`), `interrupted` (`{text, status}`), `notepad`, `notepad_truncated`. `mission`/`milestone` are **removed** from the triage/brainstorm/DECIDE state (no duplicate views) (R2-#6).
  - **Legacy contexts (R2-#7):** the l1_* state builders derive `goals` from `mission`/`milestone` when the context has no `goals`, so replayed captured inputs and the regression evals exercise the new prompt with sensible goals.
  - `goal_status`: `"none"` = no criterion; `"unchecked"` = a `verify:` criterion (no extra model call); otherwise the parsed criterion evaluated via `predicates.evaluate(..., memory=self.memory)`.
- **Triage** gets `goals`, `goal_status`, `interrupted` — **not** `notepad`. The "mission/milestone is stale" clause in TRIAGE_SYSTEM is **replaced** with: *"A goal that already shows met is not by itself a reason to change; change=true for goals only if a goal is clearly wrong or impossible."* (R2-#6)

### 3.2 Goal-met ping (R1-#4, R2-#3)
- Evaluated **every step** in `_manage_directive`, before the L1 gate — parsed predicates cached per `(tier, text, done_when)` so `resolve_item_id` isn't re-run; cheap RAM reads.
- State on the loop: `_goal_prev_status: dict[tier, str]` and `_goal_pinged: set[(tier, text, done_when)]`. Initialized from **current RAM** on start/resume (no spurious ping from an old checkpoint); a tier's prev status is reset to its current status whenever `apply_goals` changes it (a goal written already-met never pings).
- On `unmet → met` for a tier whose key isn't in `_goal_pinged`: set `_l1_event = True` and add the key (**at most one ping per goal** — `hp_frac>=` flips during battles/potions). Only the three tiers ping; `interrupted` never pings.
- `_l1_event` makes `hard_event` true (reason_loop ~619): the ping skips triage and runs brainstorm + DECIDE — accepted, bounded by the latch.
- Tested in a reason_loop unit test (not the pure `test_goals.py`).

### 3.3 Outputs (DECIDE)
DECIDE's JSON: **`mission`/`milestone` are removed** from the schema and the "You are given" list and replaced by (R1-#1):
- `"goals"` *(optional)*: only the tier(s) that changed, e.g. `{"tertiary": {"text": "Heal at the Viridian Pokémon Center", "done_when": "hp_frac>=1.0"}}`; an absent tier is unchanged; empty `text` clears that tier; a non-dict tier value (e.g. `"tertiary": "Heal…"`) is coerced to `{"text": <str>}`; other types ignored;
- `"notepad"` *(optional)*: the full rewritten notepad; absent = unchanged;
- `"interrupted": ""` *(optional)*: explicit "drop the paused focus". **Only `""` is meaningful** (`null` reads as "field not used" — models fill templates with null — so it is ignored, impl review #1); any other value (e.g. the echoed `{text, status}` dict) is ignored and is never a change (R2-#5). The return-schema template does not contain `interrupted`; it is described only in prose.
- **`catch` (R2-#1):** schema line becomes *"omit unless changing"*. Absent and `[]` both mean **unchanged** (everywhere, including with step edits — fixes the latent erase); `"catch": "clear"` clears. Compared as sorted, case-folded sets.
- **Legacy keys (R2-#2):** `l1_decide` **stops back-filling** `mission`/`milestone`. If a model still emits them, they map to `apply_goals({"primary"|"secondary": {"text": …}})` **only when a step edit is present and `goals` doesn't cover that tier** — a `goals` tier always wins. Existing tests asserting `mission`/`milestone` in the DECIDE output (test_l1_planner.py:20, test_planner_llm.py:233–234) are updated to the new contract.

### 3.4 Change detection + bookkeeping (`agent/goals.py`, pure; applied in `_run_l1`)
- **Plumbing (R1-#5, R2-#4):** `l1_decide` passes `goals`/`notepad`/`interrupted` through. `run_l1_pipeline` no longer returns None for a no-step-edit DECIDE that **carries** any of `goals`/`notepad`/`interrupted`/a non-empty `catch`: it returns those raw keys with empty add/remove. `_run_l1` then calls pure `goals.detect_change(plan, prop) -> GoalsChange | None`:
  - goals compared per tier as `(normalized text, validated done_when)` **after** dropping invalid criteria — a criterion-only change counts; normalized = whitespace-collapsed, case-folded; the same normalization is used for goals, notepad and `interrupted` matching (R2-#14);
  - notepad: normalized compare; `interrupted`: only `""` with a non-empty current value; criteria compared as parsed predicates (`hp_frac>=1` == `hp_frac>=1.0`); legacy `mission`/`milestone` keep the tier's criterion; `catch`: non-empty and different as a set.
  - `None` → treated exactly like today's no-op (nothing applied, no event). Otherwise apply, skip `reconcile_quests`/`_recompile_quest`/the `quest` event, set `_l1_last`, and emit `l1_review` with `change: "goals"`.
- A DECIDE whose step fails validation after repair is discarded **whole** (goals included), as today — the DECIDE is atomic (existing behavior, recorded in the pipeline trace).
- **Validation:** criterion outside the grammar → dropped, text kept; `text` capped at 160 chars.
- **`interrupted` rules (R1-#3, R2-#5)** — one level, harness bookkeeping only. Evaluated against the **pre-DECIDE** state, in this order:
  1. **clear** if: the new tertiary matches it (normalized); **or** the tertiary being replaced had status `met`; **or** its own criterion is `met` (checked at each review);
  2. **set** only if (after step 1) `interrupted` is empty **and** the new tertiary is non-empty **and** the replaced tertiary was non-empty and not `met` → `interrupted` = the replaced tertiary (text + criterion);
  3. an explicit `"interrupted": ""` is applied **last** (so "divert and abandon the old focus" in one DECIDE leaves it empty);
  - clearing the tertiary (empty text) never sets it; forest → heal → shop keeps `interrupted = forest`;
  - a `verify:` or criterion-less interrupted never auto-clears by status — L1 returns to it or sends `""` (the prompt says so).
- **Notepad (R1-#11):** truncated at the last line boundary ≤ 1200 chars; if a single line exceeds the cap, hard-cut at 1200. `notepad_truncated` set and shown next review.
- **Events:** `goals_changed {before, after}`, `notepad_changed {len, truncated}`. Recorder per-step `extra` gains `goals` and `interrupted` every step, and `notepad` **only on steps where it changed** (plus `notepad_len` every step) (R1-#10).

### 3.5 Prompt guidance (DECIDE; broad, structured)
Placed directly under the "CHANGING NOTHING IS THE COMMON, PREFERRED OUTCOME" block; the old "Leave MISSION/MILESTONE unchanged…" line is **replaced** (R1-#7):
- The tiers: **primary** = the long-term why (e.g. earn the Boulder Badge); **secondary** = the current chapter (e.g. get to Pewter Gym with a team that can win); **tertiary** = the immediate focus, which may be a diversion (heal, shop, grind, a story errand).
- *"GOALS and NOTEPAD follow the same rule: omit `goals`, `notepad` and `catch` unless something actually changed — a goal was achieved, you are diverting, or you learned something worth keeping. Rewording is not a change. When you do edit, include only the tier(s) that changed. `goal_status: met` means that criterion holds right now; you decide when to move on. A brief diversion can be just a step (e.g. a heal step) — you don't have to rewrite a goal for it."*
- Diverting / returning: rewrite the tertiary (or the secondary if the chapter itself changed); `interrupted` shows what you paused — return to it, or send `"interrupted": ""` to drop it (the harness can't auto-clear a paused focus without a checkable criterion).
- Plan concrete steps for your current focus and chapter; keep later chapters as goals/notepad rather than queued steps.
- `notepad`: your own short notepad — intentions for later, lessons, things not to retry. Only send it when it changes; keep it short.
- Goal criteria use only `has_item:`/`no_item:`/`level>=`/`badges>=`/`hp_frac>=`/`verify:` (no `on_map`/`talked`).
Brainstorm gets one line describing goals/notepad; triage the line in §3.1.

### 3.6 Interactions / unchanged
- **Unchanged:** the step queue + reconciler (the `after` anchor stays as a backstop), the executive, battle/shop layers, `AgentMemory.tried_failed`/`notes`.
- **Emergency heal (R1-#13):** the near-faint path is a hard event (triage skipped) guarded by `_has_heal_step`; a goals-only DECIDE during an emergency that adds no heal step re-fires next step exactly as today.
- **Plan exhaustion (R1-#8):** with later chapters kept as goals, running out of queued steps becomes the normal way a chapter advances, firing the bootstrap hard-event review (and, if DECIDE adds no step, the provisional `goal_map` travel default). Accepted; tracked in the e2e.
- **L2 proposer (R1-#12):** `milestone` now means "the chapter"; during a diversion L2 still gets the directive's own `objective`. Also pass the tertiary text as a `focus` field to the L2 context, added to `propose_target`'s state whitelist.
- **No-planner reflect path:** not supported (goals/notepad are lost if it rebuilds the plan); unchanged otherwise (R2-#13).

## 4. Testing

**Unit — pure (`tests/unit/test_goals.py`; fail today because `agent/goals.py` doesn't exist):**
- tier merge (replace / absent unchanged / empty clears / non-dict tier coerced);
- criterion grammar (allowed forms kept; `on_map`/`talked`/unresolvable item/garbage dropped by parsed key, text kept; text capped);
- `goal_status` met/unmet/none/unchecked against a FakeEmulator RAM stub;
- `detect_change`: normalized echo = None; criterion-only change counts; invalid-criterion-only diff = None; **a captured no-op DECIDE carrying `"catch": []` is None** (R2-#1/#15); `catch` non-empty different = change; echoed `interrupted` dict = None;
- every `interrupted` rule: forest→heal→shop; return with normalized-same wording (cleared); return with genuinely different wording (not cleared unless replaced tertiary was met); tertiary cleared (never sets); explicit `""`; **set + explicit clear in the same DECIDE** (ends empty); **interrupted's own criterion met while tertiary unrelated** (cleared); `verify:`/no-criterion interrupted never auto-clears;
- notepad replace / unchanged / line-boundary truncate + flag / single-long-line hard cut;
- model validator seeding / sync / `tried_failed` clear; `apply_goals` keeps `mission`/`milestone` in sync.

**Unit — wiring (`tests/unit/test_goals_wiring.py`):**
- goals-only DECIDE: applied, `change: "goals"`, reconcile not called, `_l1_last` set;
- goals.secondary edit + step add + legacy `milestone` ⇒ secondary is the new text and `plan.milestone` equals it (R2-#2);
- `catch: []` with a step edit leaves an existing catch goal intact; `"catch": "clear"` clears it;
- Jev `current_plan` excludes notepad/tried_failed;
- goal-met ping: fires once on unmet→met; no ping on resume when already met; no re-ping after `hp_frac` flips; no ping for a goal written already-met;
- triage/brainstorm/DECIDE state carries `goals` (derived from mission/milestone for legacy contexts) and not `mission`/`milestone`.

**Live scenario evals (`scripts/eval_l1_goals.py`, glm-5.3; failed calls excluded + counted).** Brainstorm is run **live** (`l1_brainstorm(None, ctx)`) — never a hand-written brainstorm that states the answer (R1-#9). Graders are structural. N=10 unless noted.
1. **Diversion (heal)** — captured Route 1/Viridian DECIDE input with `heal_case`'s party/signals injection but **live** brainstorm (R2-#8): pass = an `hp_frac>=` step placed right after the active step (`pos <= 1`, same rule as eval_l1_placement) **and** primary/secondary unchanged (≥ 8/10). Tertiary rewrite reported, not required.
2. **Return** — from `states/pokecenter_lowhp` with party current HP **written to max in RAM** before `game_signals` (no model call) (R2-#9); `interrupted` = "cross Viridian Forest"; tertiary = a heal (`hp_frac>=1.0`, now met); plan seeded with **only the done heal step**. Pass = no new heal step **and** an added step on maps {13, 51, 2} (with or without a tertiary rewrite to the interrupted focus) (≥ 8/10).
3. **Story gate** — captured input with **secondary seeded to "get to Pewter"**: pass = **0** off-path steps before the delivery (eval_l1_placement `ON_PATH`), and the delivery is either the secondary/tertiary or a queued step (≥ 8/10).
4. **Steady state / churn (key)** — replay **all 79 captured no-op DECIDE inputs** at **N=2**, goals derived from mission/milestone **and tertiary seeded from the active step's `why`/done_when** (R2-#10). Metric: goals/notepad edit rate (via `detect_change`), reported **overall and per run** (60/79 come from brock-run-1839); "filled an empty tier" counted separately. Pass: edit rate ≤ 10%. (The captured 25% milestone-rewording figure is context only — that field was required every call; the new-schema rate is the real baseline.) **Triage:** replay the captured `l1_triage` inputs **live, with and without goals** (same model, N=1); report both change-rates (captured: 69/248 ≈ 28%); pass = with-goals ≤ without-goals + 5 points.
5. **Chapter complete** — from `states/lab_deliver` + `lab_deliver.mem.json` with the parcel **removed from the bag in RAM** (post-delivery state), `goal_status.secondary = met`, notepad = "after the parcel: heal, then Route 2 → forest → Pewter", plan seeded with only the done delivery step: pass = first added step's map ∈ {1, 13, 51} or a heal (≥ 8/10).
6. **Notepad hygiene** — report sizes and echo rate across scenarios (no threshold).
**Regression:** `eval_criteria` (5/5), `eval_l1_decide` (32/32 well-formed, both models), `eval_l1_placement` (0 misordered; controls ≥ 8/10) — all on legacy contexts, which now get derived goals (R2-#7).

**End-to-end (R1-#10, R2-#11):** from `states/pallet_ready.state` (`--state pallet_ready`), ≥ 600 steps, `--capture distill --headless`, `--l1-every 5` (same as the baseline). **Pass:** the coherence invariant holds (no off-path step executed before a pending delivery); over **steps 0–299 only**, `l1_review` count ≤ baseline + 25% (baseline: `runs/placement-e2e-20260922-2324`, 23 / 300 steps — same Pallet→delivery stretch). Report separately: reviews per 100 steps for 300–600 (plan exhaustion is expected there), goals-only edits per 100 steps, goal-met pings, diversions taken + returned, steps per secondary goal, notepad length over time, delivery measured from **items** (the parcel leaving the bag), not the labeler's weak `delivered` field.

## 5. Out of scope
- Harness-enforced goal completion / push-pop stacks; step ↔ goal tagging.
- `AgentMemory.tried_failed`/`notes`, `hypotheses`; goals on the no-planner reflect path.
- Viewer UI for goals (separate UI work).
- **Follow-up (noted):** fix the outcome labeler's `delivered` to key off the parcel leaving the bag.
