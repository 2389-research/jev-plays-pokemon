# L1 Reasoning Pipeline & Acceptance-Criteria Contract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace L1's single-shot `revise_quests` with a multi-call reasoning pipeline (triage →
brainstorm+KB → decide → validate/repair) that emits quests carrying correct, machine-checkable
acceptance criteria, and make it testable without spending credits in CI.

**Architecture:** Keep the deterministic reconciler, the `done_when` DSL, and the RAM predicate
engine (all sound + tested). Add `agent/l1_pipeline.py` for the new reasoning; add a `kind:
travel|action` field to steps so `on_map` completion is legal only for pure-travel legs; instrument
the recorder/viewer so completions are legible. Spec:
`docs/superpowers/specs/2026-09-20-l1-reasoning-pipeline-design.md`.

**Tech Stack:** Python 3.13, `uv`, pytest, PyBoy, LunaRoute (via the existing `Planner` providers:
`provider`=`deepseek-4.1-flash`, `strategist`=`glm-5.3`), Orrery KB via `_llm_with_search`.

**Conventions:** Run tests with `uv run python -m pytest -q` (never bare `pytest`). Commit after
each task. Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Never print
API key values. Work on branch `l1-reasoning-pipeline`.

---

## File Structure

- **Create** `src/pokemon_agent/agent/l1_pipeline.py` — the orchestrator (`run_l1_pipeline`) + the
  deterministic step validator (`validate_step`, `classify_error`).
- **Create** `tests/support/__init__.py`, `tests/support/ram_emulator.py` — `RamEmulator` test
  double (settable RAM) for deterministic criterion-lifecycle tests.
- **Create** `tests/unit/test_l1_pipeline.py` — Layer-2 pipeline wiring (stubbed LLM) + validator.
- **Create** `tests/unit/test_criteria_eval.py` — offline unit tests of the eval *grader* (canned
  outputs, no LLM).
- **Create** `tests/fixtures/criteria_cases.py` — Layer-2.5 fixture situations + expected families.
- **Create** `scripts/eval_criteria.py` — Layer-2.5 live scorecard runner.
- **Modify** `src/pokemon_agent/agent/quest_reconciler.py` — `QuestStep.kind`; `_criterion(...,
  kind)`; carry `kind` through `compile_steps_to_directives` + `reconcile_quests`.
- **Modify** `src/pokemon_agent/agent/planner_llm.py` — TRIAGE/BRAINSTORM/DECIDE/REPAIR system
  prompts + `l1_triage`/`l1_brainstorm`/`l1_decide`/`l1_repair` methods.
- **Modify** `src/pokemon_agent/agent/reason_loop.py` — bootstrap step `kind="travel"`; `_run_l1`
  calls `run_l1_pipeline`; completion-provenance events; L1 trace into `l1_last`.
- **Modify** `src/pokemon_agent/emulator/interface.py`, `pyboy_adapter.py`, `fake_emulator.py` —
  add `write_memory`.
- **Modify** `src/pokemon_agent/logging/run_recorder.py`, `logging/player.html` — observability.
- **Modify** `tests/unit/test_quest_reconciler.py`, `tests/unit/test_predicates.py` — extend.

---

## Task 1: `write_memory` across the emulator layer

**Files:**
- Modify: `src/pokemon_agent/emulator/interface.py`
- Modify: `src/pokemon_agent/emulator/pyboy_adapter.py`
- Modify: `src/pokemon_agent/emulator/fake_emulator.py`
- Test: `tests/unit/test_fake_emulator.py`

- [ ] **Step 1: Write the failing test** in `tests/unit/test_fake_emulator.py`:

```python
def test_write_memory_round_trips():
    from pokemon_agent.emulator.fake_emulator import FakeEmulator
    emu = FakeEmulator()               # use existing constructor pattern in this file
    emu.write_memory(0xD16C, 42)
    assert emu.read_memory(0xD16C) == 42
```

- [ ] **Step 2: Run it, verify it fails** — `uv run python -m pytest tests/unit/test_fake_emulator.py -q` → AttributeError (`write_memory`).

- [ ] **Step 3: Implement.**
  - `interface.py`: add abstract `def write_memory(self, address: int, value: int, bank: int | None = None) -> None: ...` next to `read_memory`.
  - `pyboy_adapter.py`: implement `def write_memory(self, address, value, bank=None): self._pyboy.memory[address] = value` (the attribute is `self._pyboy`, per `pyboy_adapter.py:96` — mirror exactly how `read_memory` accesses it).
  - `fake_emulator.py`: add `self._ram: dict[int,int] = {}` in `__init__`; `write_memory` sets `self._ram[address] = value`; extend `read_memory` to consult `self._ram` first (falling back to the existing X/Y/map dict): `return self._ram.get(address, {ADDR_PLAYER_X: self.x, ...}.get(address, 0))`.

- [ ] **Step 4: Run, verify pass.** Then full suite `uv run python -m pytest -q` — still green.

- [ ] **Step 5: Commit** — `feat(emu): add write_memory to emulator interface + adapters`.

---

## Task 2: `RamEmulator` test double

**Files:**
- Create: `tests/support/__init__.py` (empty)
- Create: `tests/support/ram_emulator.py`
- Test: `tests/unit/test_predicates.py` (add cases using it)

`RamEmulator` is a pure dict-of-RAM emulator so predicate checks are deterministic. It must serve
the addresses `predicates.evaluate` reads: `WCURMAP` (0xD35E), `WPARTYCOUNT`, party HP/level bytes
(via `needs.party_hp_fraction`), bag items (`WNUMBAGITEMS`/`WBAGITEMS`), badges, money.

- [ ] **Step 1: Write the failing test** in `tests/unit/test_predicates.py`:

```python
from tests.support.ram_emulator import RamEmulator
from pokemon_agent.games.pokemon_red import predicates

def test_hp_frac_predicate_via_ram_emulator():
    emu = RamEmulator()
    emu.set_map(41); emu.set_party([("SQUIRTLE", 7, 3, 23)])   # (species, level, cur_hp, max_hp)
    assert not predicates.evaluate({"hp_frac": ">=1.0"}, emu)
    emu.set_party([("SQUIRTLE", 7, 23, 23)])
    assert predicates.evaluate({"hp_frac": ">=1.0"}, emu)

def test_has_no_item_predicate_via_ram_emulator():
    emu = RamEmulator()
    emu.set_bag_items([])                       # no items
    from pokemon_agent.agent.planner_llm import Planner
    crit = Planner._parse_done_when("no_item:oaks_parcel", 0)   # -> {"no_item": <id>}
    assert crit is not None and predicates.evaluate(crit, emu)  # deliver-done when not held
    emu.set_bag_items(["oaks_parcel"])
    assert not predicates.evaluate(crit, emu)                   # still holding -> NOT done
```

- [ ] **Step 2: Run, verify it fails** (module missing).

- [ ] **Step 3: Implement `RamEmulator`.** Read `games/pokemon_red/game_state.py` and `needs.py` for
  the exact party HP/level/bag layout (`WPARTYCOUNT`, party struct stride, HP hi/lo bytes, level
  offset, `WNUMBAGITEMS`, `WBAGITEMS` item-id/quantity stride). Back everything with a
  `dict[int,int]`; `read_memory`/`write_memory` hit the dict. Provide setters that write the correct
  bytes:
  - `set_map(map_id)` → `WCURMAP`.
  - `set_party(list_of_(species,level,cur_hp,max_hp))` → `WPARTYCOUNT` + each mon's level and
    cur/max HP hi/lo bytes at the right offsets.
  - `set_bag_items(names)` → `WNUMBAGITEMS` + `WBAGITEMS` entries via `resolve_item_id(name)`;
    terminate with 0xFF.
  - `set_badges(n)`, `set_money(n)`, `set_in_battle(bool)` as needed by tests.
  Use the constants imported from the game modules — do NOT hardcode addresses that already have
  named constants.

- [ ] **Step 4: Run, verify pass.** Full suite green.

- [ ] **Step 5: Commit** — `test: add RamEmulator settable-RAM test double`.

---

## Task 3: `QuestStep.kind` + acceptance-criteria contract in the reconciler

**Files:**
- Modify: `src/pokemon_agent/agent/quest_reconciler.py`
- Test: `tests/unit/test_quest_reconciler.py`

The core fix. `on_map` completion becomes legal only for `kind == "travel"`.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_quest_reconciler.py`:

```python
import pytest
from pokemon_agent.agent.quest_reconciler import QuestStep, compile_steps_to_directives, reconcile_quests

def test_travel_step_keeps_on_map():
    d = compile_steps_to_directives([QuestStep(id="t", map=2, kind="travel")])
    assert d[0].success == {"on_map": 2}

def test_action_step_without_criterion_raises():
    with pytest.raises(ValueError):
        compile_steps_to_directives([QuestStep(id="a", map=40, kind="action")])

def test_action_step_with_on_map_raises():
    with pytest.raises(ValueError):
        compile_steps_to_directives([QuestStep(id="a", map=40, kind="action", done_when="on_map")])

def test_grind_action_step_talk_false_still_requires_criterion():
    # the case the old talk-based inference missed
    with pytest.raises(ValueError):
        compile_steps_to_directives([QuestStep(id="g", map=13, kind="action", talk=False)])

def test_action_step_with_real_criterion_ok():
    d = compile_steps_to_directives([QuestStep(id="d", map=40, kind="action",
                                               done_when="no_item:oaks_parcel")])
    assert "no_item" in d[-1].success

def test_reconcile_carries_kind_default_action():
    out = reconcile_quests([], {"add": [{"map": 40, "done_when": "no_item:oaks_parcel"}]},
                           next_id=lambda: "q1")
    assert out[0].kind == "action"
    out2 = reconcile_quests([], {"add": [{"map": 2, "kind": "travel"}]}, next_id=lambda: "q2")
    assert out2[0].kind == "travel"
```

- [ ] **Step 2: Run, verify they fail.**

- [ ] **Step 3: Implement.**
  - Add `kind: str = "action"` to `QuestStep` (after `provisional`). Document: `"travel"` = reach a
    map (on_map ok); `"action"` = done by a state change (needs a real criterion).
  - Change `_criterion` to take `kind`:

```python
def _criterion(done_when: str | None, map_id: int, kind: str) -> dict:
    from .planner_llm import Planner
    parsed = Planner._parse_done_when(done_when, map_id)
    if kind == "travel":
        return parsed if parsed is not None else {"on_map": map_id}
    # kind == "action": a real, non-on_map criterion is REQUIRED (never complete-on-arrival)
    if parsed is None or parsed == {"on_map": map_id}:
        raise ValueError(
            f"action quest step (map {map_id}) needs a checkable done_when, got {done_when!r}")
    return parsed
```

  - In `compile_steps_to_directives`, pass `s.kind`: `crit = _criterion(s.done_when, s.map, s.kind)`.
    (The TRAVEL sub-directive of a talk step keeps its hardcoded `{"on_map": s.map}`; only the
    action/criterion directive uses `crit`.)
  - In `reconcile_quests`, thread kind onto new steps:
    `kind=str(a.get("kind") or "action")` in the `QuestStep(...)` constructor. Dedup key stays
    `(map, done_when)` — `kind` does not participate.

- [ ] **Step 4: Run, verify pass.** Full suite — expect failures in call sites that build action
  steps without criteria (the bootstrap); those are fixed in Task 4. **Known pre-existing break:**
  `tests/unit/test_quest_reconciler.py::test_compile_travel_only_step` builds
  `QuestStep(..., done_when="on_map")` which now defaults to `kind="action"` and will raise in
  `_criterion` — add `kind="travel"` to that test's step. If any *other* test breaks, it revealed a
  real complete-on-arrival step — fix that test's step to declare `kind`/criterion.

- [ ] **Step 5: Commit** — `feat(quests): kind travel|action; on_map completion only for travel`.

---

## Task 4: Bootstrap provisional step is `kind="travel"` (crash fix)

**Files:**
- Modify: `src/pokemon_agent/agent/reason_loop.py` (~line 493)
- Test: `tests/unit/test_unified_loop.py` (or `test_executive.py`)

The bootstrap default at `reason_loop.py:493` runs OUTSIDE `_run_l1`'s try/except; under Task 3 it
would raise. It is a pure travel leg.

- [ ] **Step 1: Write the failing test** — drive the loop (FakeEmulator, no planner/provider) so
  `_manage_directive` hits the empty-plan bootstrap, and assert `step_once()` does not raise and the
  synthesized step is a travel step:

```python
def test_empty_plan_bootstrap_is_travel_and_does_not_raise(...):
    loop = <build ReasoningLoop on FakeEmulator, planner=None, goal_map=2>
    loop.step_once()   # must not raise
    assert any(s.kind == "travel" for s in loop._plan_steps)
```

- [ ] **Step 2: Run, verify it fails** (raises ValueError from `_criterion`).

- [ ] **Step 3: Implement** — at `reason_loop.py:493`, add `kind="travel"` to the
  `QuestStep(...)` constructor for the provisional default.

- [ ] **Step 4: Run, verify pass.** Full suite green.

- [ ] **Step 5: Commit** — `fix(reason_loop): bootstrap provisional step is kind=travel`.

---

## Task 5: Deterministic step validator (`l1_pipeline.validate_step`)

**Files:**
- Create: `src/pokemon_agent/agent/l1_pipeline.py` (validator only in this task)
- Test: `tests/unit/test_l1_pipeline.py`

Pure function used by the pipeline's step-5 VALIDATE. No LLM.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_l1_pipeline.py`:

```python
from pokemon_agent.agent.l1_pipeline import validate_step

def test_travel_step_valid():
    ok, err = validate_step({"kind": "travel", "map": 2, "done_when": "on_map"})
    assert ok and err is None

def test_action_missing_criterion_invalid():
    ok, err = validate_step({"kind": "action", "map": 40})
    assert not ok and "done_when" in err

def test_action_on_map_invalid():
    ok, err = validate_step({"kind": "action", "map": 40, "done_when": "on_map"})
    assert not ok

def test_action_unparseable_invalid():
    ok, err = validate_step({"kind": "action", "map": 40, "done_when": "when i feel like it"})
    assert not ok

def test_action_real_criterion_valid():
    ok, err = validate_step({"kind": "action", "map": 40, "done_when": "no_item:oaks_parcel"})
    assert ok

def test_travel_with_talk_invalid():   # contradictory: travel = nothing happens on arrival
    ok, err = validate_step({"kind": "travel", "map": 2, "talk": True})
    assert not ok
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `validate_step`** in `l1_pipeline.py`:

```python
def validate_step(step: dict) -> tuple[bool, str | None]:
    """Deterministic step-5 VALIDATE. Returns (ok, error_message)."""
    from .planner_llm import Planner
    kind = str(step.get("kind") or "action")
    try:
        map_id = int(step.get("map"))
    except (TypeError, ValueError):
        return False, f"step map is missing/invalid: {step.get('map')!r}"
    dw = step.get("done_when")
    if kind == "travel":
        if step.get("talk") or step.get("who"):
            return False, "travel step cannot talk; use kind:action"
        parsed = Planner._parse_done_when(dw, map_id) if dw else {"on_map": map_id}
        if parsed is not None and parsed != {"on_map": map_id}:
            return False, "travel step criterion must be on_map"
        return True, None
    # kind == "action"
    parsed = Planner._parse_done_when(dw, map_id)
    if parsed is None:
        return False, f"action step needs a checkable done_when; {dw!r} did not parse"
    if parsed == {"on_map": map_id}:
        return False, "action step cannot complete on arrival (on_map)"
    return True, None
```

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `feat(l1): deterministic acceptance-criterion validator`.

---

## Task 6: Planner pipeline call methods + prompts

**Files:**
- Modify: `src/pokemon_agent/agent/planner_llm.py`
- Test: `tests/unit/test_planner_llm.py`

Add four methods on `Planner`, each using the existing providers and returning a parsed dict, never
raising (safe default on failure — mirror `revise_quests`). Prompts follow the DECIDE prompt
contract in the spec; **exact wording is authored here and tuned against the Task 11 eval.**

- `l1_triage(context) -> {"change": bool, "why": str}` — uses `self.provider` (fast). System
  `TRIAGE_SYSTEM`: given plan + signals, does the plan need to change? Cheap, no KB.
- `l1_brainstorm(emu, context) -> {"assessment": str, ...}` — uses `self.strategist or
  self.provider` via `self._llm_with_search(..., final_key="assessment")` (model-driven KB search).
  System `BRAINSTORM_SYSTEM`: "where am I in the game, what should I be doing & why."
- `l1_decide(context, brainstorm) -> {"add":[...], "remove":[...], "mission","milestone",
  "assessment", "change"}` — uses strategist. System `DECIDE_SYSTEM` embeds: the DSL grammar, the
  `kind: travel|action` rule, one worked example per objective class (pickup→has_item,
  deliver→no_item, heal→hp_frac>=1.0, grind→level>=N, badge→badges>=N, reach→travel/on_map,
  story→verify:), "prefer RAM-checkable over verify:", and the output schema `{add:[{kind, map,
  talk?, who?, done_when, why}], remove, mission?, milestone?, assessment}`.
- `l1_repair(context, bad_step, error) -> {step...}` — strategist. System `REPAIR_SYSTEM`: given the
  grammar + the offending step + the exact parse error, re-emit ONLY that step with a corrected
  `done_when`.

- [ ] **Step 1: Write failing tests** using a stub provider (a small class with
  `chat_json(system, state) -> (json_str, None, None)` returning canned content — follow the
  existing pattern in `test_planner_llm.py`). Assert:
  - `l1_triage` parses `{"change": true, "why": "..."}` from provider output; returns
    `{"change": False, ...}` when provider is None or raises.
  - `l1_decide` returns the add/remove structure verbatim from a canned decide payload.
  - `l1_brainstorm` calls `_llm_with_search` (KB tool-loop reused).

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** the four methods + system prompt strings. Keep each method wrapped in
  try/except returning a safe default (never raise into the pipeline). Do NOT validate criteria here
  — validation is the pipeline's job (Task 7). Author the DECIDE prompt per the contract; a first
  draft is fine (Task 11 tunes it).

- [ ] **Step 4: Run, verify pass.** Full suite green.

- [ ] **Step 5: Commit** — `feat(l1): triage/brainstorm/decide/repair planner calls + prompts`.

---

## Task 7: `run_l1_pipeline` orchestrator (Layer-2 wiring)

**Files:**
- Modify: `src/pokemon_agent/agent/l1_pipeline.py`
- Test: `tests/unit/test_l1_pipeline.py`

Orchestrates the calls, applies `validate_step`, does repair-once-then-break, returns a proposal
dict or `None` (no change / broke → keep prior plan).

Signature: `run_l1_pipeline(emu, context, planner, *, hard_event: bool, on_trace=None) -> dict | None`.
`on_trace(event: dict)` is an optional callback so `reason_loop` can record the L1 trace
(triage/brainstorm/decide/validation) — keeps the pipeline decoupled from the recorder.

Flow (matches spec §Architecture):
1. If `not hard_event`: `t = planner.l1_triage(context)`; if not `t["change"]` → emit trace, return
   `None`.
2. `b = planner.l1_brainstorm(emu, context)`.
3. `d = planner.l1_decide(context, b)`.
4. For each step in `d["add"]`: `ok, err = validate_step(step)`; if not ok → `fixed =
   planner.l1_repair(context, step, err)`; re-`validate_step(fixed)`; if still bad → emit
   `l1_invalid_criterion` trace and **return `None`** (keep prior plan). Replace the step with the
   repaired one on success.
5. Return `{"add": [...validated...], "remove": d.get("remove", []), "mission": d.get("mission"),
   "milestone": d.get("milestone"), "assessment": d.get("assessment")}`.

- [ ] **Step 1: Write failing tests** with a `StubPlanner` exposing the four methods as canned
  returns + a recording `search_kb`:
  - triage `change:false`, `hard_event=False` → returns `None`, brainstorm/decide not called.
  - `hard_event=True` → triage skipped; brainstorm+decide called; proposal returned.
  - decide emits a bad criterion; repair fixes it → proposal contains the fixed step.
  - decide emits a bad criterion; repair still bad → returns `None` and an `l1_invalid_criterion`
    trace was emitted.
  - a valid heal proposal (`hp_frac>=1.0`) flows through unchanged.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `run_l1_pipeline`.**

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `feat(l1): run_l1_pipeline orchestrator (triage/brainstorm/decide/repair)`.

---

## Task 8: Wire `_run_l1` to the pipeline

**Files:**
- Modify: `src/pokemon_agent/agent/reason_loop.py` (`_run_l1`, ~355-419)
- Test: `tests/unit/test_executive.py` (or `test_unified_loop.py`)

Replace the `self.planner.revise_quests(emu, context)` call (line 384) with `run_l1_pipeline(emu,
context, self.planner, hard_event=<bool>, on_trace=self._record_l1_trace)`. Keep everything else in
`_run_l1`: the reconcile, mission/milestone update, `tried_failed`, the provisional-supersede logic,
`_recompile_quest`. Handle the return:
- `None` → no change: set `self._l1_last = {"change": False, "assessment": <triage why if any>}`;
  do NOT reconcile.
- dict → reconcile as today (the proposal shape is identical to what `reconcile_quests` consumed).

`hard_event` is true when `_run_l1` was invoked for an event (emergency heal, wedge, map change,
item change, level-up, plan-exhausted) vs. the periodic `--l1-every` cadence. Thread a parameter
through `_run_l1(obs, emergency=..., hard_event=...)`; the existing emergency path sets it true.

- [ ] **Step 1: Write failing test** — stub `run_l1_pipeline` (monkeypatch) to return a heal
  proposal on a low-HP state and assert a `hp_frac>=1.0` action step lands in `_plan_steps` and
  compiles without raising; and that a `None` return leaves the plan unchanged.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** the wiring. Keep `revise_quests` in `planner_llm.py` for now (unused by
  the loop; remove in a later cleanup) to keep the diff focused.

- [ ] **Step 4: Run, verify pass.** Full suite green.

- [ ] **Step 5: Commit** — `feat(reason_loop): L1 uses run_l1_pipeline; two-tier gating`.

---

## Task 9: Completion-provenance events + full quest serialization

**Files:**
- Modify: `src/pokemon_agent/agent/reason_loop.py` (`_mark_step`, the recorder `_finish` extras, L1 trace)
- Modify: `src/pokemon_agent/logging/run_recorder.py`
- Test: `tests/unit/test_run_recorder.py`

- [ ] **Step 1: Write failing tests**:
  - The per-step recorded `plan_steps` include `done_when`, `kind`, `why`, `talk`, `who` (not just
    id/map/status).
  - When a step flips to `done`/`wedged`, a `step_done` / `step_wedged` event is recorded naming the
    criterion (e.g. `"hp_frac>=1.0"`) or the wedge reason.
  - `l1_last` (or an `l1_trace` field) carries triage `{change, why}` and, on a deep run, the
    decide add/remove summary + any `l1_invalid_criterion`.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement.**
  - Where `plan_steps` is serialized for the recorder, emit the full field set from `QuestStep`.
  - In `_mark_step`, when transitioning to `done`/`wedged`, `on_event` with the step id, its
    `done_when`, and (for done) the satisfied criterion / (for wedged) the reason.
  - Add `_record_l1_trace(event)` used as the pipeline's `on_trace`, appending into `self._l1_last`
    / events.

- [ ] **Step 4: Run, verify pass.** Full suite green.

- [ ] **Step 5: Commit** — `feat(obs): full quest serialization + completion-provenance + L1 trace`.

---

## Task 10: Viewer panels for criterion + completion reason

**Files:**
- Modify: `src/pokemon_agent/logging/player.html`
- (No unit test — visual; verify by loading a recorded run.)

- [ ] **Step 1:** In the player's per-step panel, render each plan step's `kind` + `done_when`, and
  when the step's `events` include `step_done`/`step_wedged`, show the completion reason inline.
- [ ] **Step 2:** Load an existing run dir in the browser (`runs/ && python -m http.server 8010` →
  `_viewer.html`) and confirm the criteria + completion reasons render. (The prior run
  `l1-longrun-20260920-153335` predates the new fields, so record a fresh short run or hand-craft a
  tiny fixture `log.jsonl` for the visual check.)
- [ ] **Step 3: Commit** — `feat(viewer): show step criteria + completion reasons`.

---

## Task 11: Criteria-quality eval (Layer 2.5)

**Files:**
- Create: `tests/fixtures/criteria_cases.py`
- Create: `scripts/eval_criteria.py`
- Create: `tests/unit/test_criteria_eval.py` (offline grader test)

The eval exercises only `l1_decide` (+ `l1_repair`) against situations and grades the emitted
criteria deterministically.

Fixture shape (each case):
```python
{"name": "deliver_parcel",
 "context": {...decide context: current_map, party, items(holding oaks_parcel), plan, signals...},
 "objective_hint": "you are holding Oak's Parcel and it must reach Prof. Oak",
 "expect_family": "no_item",                 # OR:
 "expect_exact": "no_item:oaks_parcel"}       # parcel cases assert the argument too
```

Grader (deterministic, no LLM):
- **hard:** every emitted step passes `validate_step`.
- **semantic:** the step whose objective matches the case emits a criterion whose family (the
  `_parse_done_when` key) equals `expect_family`, or whose raw string equals `expect_exact`.

- [ ] **Step 1: Write the offline grader test** in `test_criteria_eval.py` — feed the grader a
  canned decide output and assert pass/fail for a family match, an exact match, and a hard-rule
  violation. (No network.)
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `tests/fixtures/criteria_cases.py` (≥5 cases: deliver_parcel,
  pickup_parcel, heal_low_hp, reach_pewter, grind_underleveled) and the grader + runner in
  `scripts/eval_criteria.py`. The runner: for each case, build a `Planner` with real providers, call
  `l1_decide`, grade, print a per-case + aggregate scorecard. Gate live network behind an env check
  / `@pytest.mark.live` if exposed as a test; the script itself just runs.
- [ ] **Step 4: Run the offline grader test, verify pass.** (Do NOT run the live scorecard in CI.)
- [ ] **Step 5: Commit** — `feat(eval): criteria-quality scorecard (Layer 2.5) + fixtures + grader`.

---

## Task 12: Live-test gating (Layer 3 marker)

**Files:**
- Modify: `tests/conftest.py` (register `live` marker + skip-by-default unless `--run-live` / env)
- Create: a minimal `tests/integration/test_l1_live_smoke.py` marked `@pytest.mark.live`

- [ ] **Step 1:** Add the `live` marker registration and a skip hook (skip unless `RUN_LIVE=1` or
  `--run-live`). Note: `tests/conftest.py` currently holds only a `sys.path` insert — there is no
  existing marker pattern to follow; create the `pytest_configure` marker registration +
  `pytest_collection_modifyitems` skip hook from scratch.
- [ ] **Step 2:** Add a single `@pytest.mark.live` smoke: load `viridian_stuck.state`, run a handful
  of L1 deep reviews, assert the plan gains at least one action step with a checkable criterion.
- [ ] **Step 3:** Confirm `uv run python -m pytest -q` SKIPS it by default; `RUN_LIVE=1 uv run
  python -m pytest -q -m live` runs it (only when credits are up).
- [ ] **Step 4: Commit** — `test: gate live L1 smoke behind the live marker`.

---

## Final review

After all tasks: dispatch a final code-reviewer over the whole branch, run the full suite
(`uv run python -m pytest -q`, must be green, zero credits spent), then use
superpowers:finishing-a-development-branch to open the PR (base: `l1-strategic-planner` /
retarget to `main` after PR #2 merges). PR body ends with
`🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
