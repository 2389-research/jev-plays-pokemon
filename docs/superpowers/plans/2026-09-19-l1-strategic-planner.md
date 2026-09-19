# L1 Strategic Planner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make L1 a persistent, periodic, DSL-speaking strategic planner with a deterministic reconciler, so the agent keeps a durable long-term plan (e.g. recognizes the Viridian parcel errand), routes needs (HP) *through* L1, and never discards its plan on a single wedged step.

**Architecture:** A canonical `list[QuestStep]` (`_plan_steps`) is the source of truth. L1 (the strategist model) proposes add/remove edits in the executor's `done_when` DSL, or "no change". A pure `reconcile_quests` merges the proposal (preserving completed + active steps, replacing wedged ones). The pending steps compile to the existing `deque[Directive]` (TRAVEL + optional TALK_TO, tagged `quest_id`). Directive management becomes plan-driven; `NeedsArbiter` is kept as a pure library used only for signals + a near-faint emergency-heal reflex. L2 (`propose_target`) and L3 (Jev/router) are unchanged except L2 is handed the current `milestone`.

**Tech Stack:** Python 3.13, PyBoy, LunaRoute (strategist glm-5.3, KB via Orrery), TypeSafe/Jev, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-19-l1-strategic-planner-design.md` (read it — it's authoritative).

**Working copy:** `~/Documents/GitHub/jev-plays-pokemon`, branch `l1-strategic-planner` created off `unified-control-loop` (this work depends on the unified-loop code — `propose_target`, `_navigate_leg`, `_dispatch_servo` — which is not yet merged to `main`). Run tests with `uv run python -m pytest -q`. Current baseline: 222 tests pass. All commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Reference: current code (verified against the repo)
- `src/pokemon_agent/agent/plan.py` — `Directive` (pydantic: intent/target/success/reason), `AgentPlan` (mission/milestone/tried_failed/next_objective), `Intent`.
- `src/pokemon_agent/agent/planner_llm.py` — `strategize()` (expands a strategist step → TRAVEL + optional TALK_TO with `who`→`target.sprite`, planner_llm.py:206-217/228-245), `_parse_done_when` (286-314: `on_map/talked/has_item:/no_item:/level>=/badges>=/hp_frac>=/verify:`), `_llm_with_search` (KB tool loop, 179), `STRATEGIST_SYSTEM`.
- `src/pokemon_agent/agent/reason_loop.py` — `_manage_directive` (arbiter-intent driven, ~314-408), `self._quest: deque[Directive]`, `_in_quest`, `_quest_step_age`, `_INTENT_PRIORITY` suspend/stack, `_should_replan` (intent-change trigger), `_try_quest`/`_should_escalate` (escalation), `self._quest.clear()`-on-wedge (~363), heal errand branch (~374-394), `_dispatch_servo`, `_propose_target` (feeds L2), recorder `extra` in `_finish`.
- `src/pokemon_agent/agent/needs_arbiter.py` — `intent()` + `needs.*` (`any_fainted`, hp_frac, etc.). KEEP as a library.
- `src/pokemon_agent/games/pokemon_red/needs.py` — `any_fainted`, HP/level helpers.
- Tests: `tests/unit/test_needs_arbiter.py` (unchanged), `tests/unit/test_executive.py` (intent-driven tests get rewritten), `tests/unit/test_unified_loop.py`.

## File structure
- **New** `src/pokemon_agent/agent/quest_reconciler.py` — `QuestStep`, `reconcile_quests`, `compile_steps_to_directives`.
- **Modify** `src/pokemon_agent/agent/plan.py` — add `Directive.quest_id`.
- **Modify** `src/pokemon_agent/agent/planner_llm.py` — `revise_quests` + `L1_SYSTEM`.
- **New** `src/pokemon_agent/agent/signals.py` — `game_signals(emu)` + emergency-heal predicate.
- **Modify** `src/pokemon_agent/agent/reason_loop.py` — plan-driven `_manage_directive`, L1 gate, emergency reflex, `_plan_steps`, recorder, `milestone`→L2.
- **Modify** `scripts/run_agent.py` — `--l1-every` flag.
- **Tests** — `test_quest_reconciler.py`, `test_l1_planner.py`, `test_signals.py`, rewrites in `test_executive.py`, an integration fixture.

Deterministic-first: `reconcile_quests`, `compile_steps_to_directives`, and `game_signals` are pure and fully unit-tested with no LLM. `revise_quests` is tested with a fake provider. The executive integration is validated by rewritten executive tests + a fixture.

---

## Task 1: `Directive.quest_id` (backward-compatible tag)

**Files:** Modify `src/pokemon_agent/agent/plan.py`; Test `tests/unit/test_quest_reconciler.py` (new).

- [ ] **Step 1: Failing test**
```python
# tests/unit/test_quest_reconciler.py
from pokemon_agent.agent.plan import Directive, Intent

def test_directive_has_optional_quest_id():
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 1}, success={"on_map": 1})
    assert d.quest_id is None                      # default, backward-compatible
    d2 = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 1}, success={"on_map": 1}, quest_id="q3")
    assert d2.quest_id == "q3"
```
- [ ] **Step 2:** `uv run python -m pytest tests/unit/test_quest_reconciler.py -q` → FAIL (unexpected kw `quest_id`).
- [ ] **Step 3:** In `plan.py`, add to `Directive` (after `reason`):
```python
    quest_id: str | None = Field(default=None, description="id of the QuestStep this directive was compiled from")
```
- [ ] **Step 4:** rerun → PASS; `uv run python -m pytest -q` → 223 pass.
- [ ] **Step 5:** commit `plan: Directive.quest_id (tag compiled directives to their QuestStep)`.

---

## Task 2: `QuestStep` + `compile_steps_to_directives`

**Files:** New `src/pokemon_agent/agent/quest_reconciler.py`; Test `tests/unit/test_quest_reconciler.py`.

The compiler follows `strategize`'s expansion shape: each step → a TRAVEL directive to `map`, then if `talk` a TALK_TO directive (target `{"kind":"npc","map":map[,"sprite":who]}`), both tagged with the step's id. `done_when` is parsed with the existing `Planner._parse_done_when`; the talk step (or the travel step if no talk) carries that as its `success`. Two **deliberate** differences from today's `strategize` (both improvements, not bugs): (a) a non-talk step carries its *parsed* criterion rather than being forced to `on_map`; (b) an unparseable `done_when` falls back to `on_map` (travel) — for a talk step you may keep `talked` semantics if preferred, but `on_map` is acceptable. Don't claim byte-for-byte parity with `strategize`.

- [ ] **Step 1: Failing tests**
```python
from pokemon_agent.agent.quest_reconciler import QuestStep, compile_steps_to_directives
from pokemon_agent.agent.plan import Intent

def test_compile_travel_only_step():
    s = QuestStep(id="q1", map=1, talk=False, who=None, done_when="on_map", why="go")
    ds = compile_steps_to_directives([s])
    assert len(ds) == 1 and ds[0].intent == Intent.TRAVEL and ds[0].quest_id == "q1"
    assert ds[0].success == {"on_map": 1}

def test_compile_talk_step_adds_talk_to_with_sprite_and_criterion():
    s = QuestStep(id="q2", map=40, talk=True, who="Oak", done_when="no_item:Oak's Parcel", why="deliver")
    ds = compile_steps_to_directives([s])
    kinds = [d.intent for d in ds]
    assert Intent.TRAVEL in kinds and Intent.TALK_TO in kinds
    talk = [d for d in ds if d.intent == Intent.TALK_TO][0]
    assert talk.quest_id == "q2" and (talk.target or {}).get("sprite") == "Oak"
    assert "no_item" in talk.success          # model-authored criterion attached to the talk step

def test_compile_bad_done_when_falls_back_to_on_map():
    s = QuestStep(id="q3", map=2, talk=False, who=None, done_when="garbage", why="x")
    ds = compile_steps_to_directives([s])
    assert ds[0].success == {"on_map": 2}     # unparseable criterion -> safe default for a travel step
```
- [ ] **Step 2:** run → FAIL (module missing).
- [ ] **Step 3:** implement `quest_reconciler.py`:
```python
from __future__ import annotations
from dataclasses import dataclass, field
from .plan import Directive, Intent
from ..games.pokemon_red.maps import map_name


@dataclass
class QuestStep:
    id: str
    map: int
    talk: bool = False
    who: str | None = None
    done_when: str | None = None
    why: str = ""
    status: str = "pending"          # pending | active | done | wedged


def _criterion(done_when: str | None, map_id: int) -> dict:
    from .planner_llm import Planner
    parsed = Planner._parse_done_when(done_when, map_id)
    return parsed if parsed is not None else {"on_map": map_id}


def compile_steps_to_directives(steps: list[QuestStep]) -> list[Directive]:
    """Expand each QuestStep into its 1-2 Directives (TRAVEL + optional TALK_TO), tagged quest_id —
    matching strategize()'s expansion. The model-authored done_when attaches to the talk step when
    there is one, else to the travel step."""
    out: list[Directive] = []
    for s in steps:
        crit = _criterion(s.done_when, s.map)
        nm = map_name(s.map)
        if s.talk:
            out.append(Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": s.map},
                                 success={"on_map": s.map}, quest_id=s.id,
                                 reason=f"quest: go to {nm} — {s.why}"[:120]))
            tgt = {"kind": "npc", "map": s.map}
            if s.who:
                tgt["sprite"] = s.who
            out.append(Directive(intent=Intent.TALK_TO, target=tgt, success=crit, quest_id=s.id,
                                 reason=f"quest: talk to {s.who or 'someone'} in {nm} — {s.why}"[:120]))
        else:
            out.append(Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": s.map},
                                 success=crit, quest_id=s.id, reason=f"quest: go to {nm} — {s.why}"[:120]))
    return out
```
- [ ] **Step 4:** run new tests → PASS; full suite green.
- [ ] **Step 5:** commit `quest_reconciler: QuestStep + compile_steps_to_directives (mirrors strategize expansion)`.

---

## Task 3: `reconcile_quests` (the deterministic merge)

**Files:** Modify `src/pokemon_agent/agent/quest_reconciler.py`; Test `tests/unit/test_quest_reconciler.py`.

Merge L1's proposal into the canonical plan, preserving progress. Signature:
`reconcile_quests(current: list[QuestStep], proposal: dict, *, next_id) -> list[QuestStep]` where
`proposal = {"add": [step-dicts], "remove": [ids]}` and `next_id()` yields fresh ids.

Rules (from the spec): keep `done` + `active`; `remove` applies only to `pending`; a `wedged` step is **replaced** by the matching add (or dropped if L1 didn't supply one) — never left; `add` inserts after the active step, dedup by `(map, done_when)`.

- [ ] **Step 1: Failing tests**
```python
from pokemon_agent.agent.quest_reconciler import QuestStep, reconcile_quests

def _mk(id, map, status="pending", dw="on_map"): return QuestStep(id=id, map=map, done_when=dw, status=status)

def _ids():
    n = [100]
    def nxt():
        n[0] += 1; return f"q{n[0]}"
    return nxt

def test_reconcile_preserves_done_and_active_and_adds():
    cur = [_mk("q1", 0, "done"), _mk("q2", 1, "active"), _mk("q3", 2, "pending")]
    prop = {"add": [{"map": 42, "talk": True, "who": "clerk", "done_when": "has_item:Oak's Parcel", "why": "get parcel"}],
            "remove": ["q3"]}
    out = reconcile_quests(cur, prop, next_id=_ids())
    ids = [s.id for s in out]
    assert "q1" in ids and "q2" in ids          # done + active preserved
    assert "q3" not in ids                        # pending removed
    assert any(s.map == 42 and s.talk for s in out)   # add inserted
    assert out.index(next(s for s in out if s.id == "q2")) < out.index(next(s for s in out if s.map == 42))  # added after active

def test_reconcile_never_removes_active_or_done():
    cur = [_mk("q1", 0, "done"), _mk("q2", 1, "active")]
    out = reconcile_quests(cur, {"add": [], "remove": ["q1", "q2"]}, next_id=_ids())
    assert [s.id for s in out] == ["q1", "q2"]     # removes of done/active ignored

def test_reconcile_replaces_wedged_step():
    cur = [_mk("q2", 1, "active"), _mk("q3", 2, "wedged")]
    prop = {"add": [{"map": 2, "talk": False, "done_when": "on_map", "why": "retry via other route"}], "remove": ["q3"]}
    out = reconcile_quests(cur, prop, next_id=_ids())
    assert all(s.status != "wedged" for s in out)   # wedged gone
    assert any(s.map == 2 and s.status == "pending" for s in out)  # replaced by a fresh pending step

def test_reconcile_empty_proposal_is_unchanged():
    cur = [_mk("q2", 1, "active"), _mk("q3", 2, "pending")]
    out = reconcile_quests(cur, {"add": [], "remove": []}, next_id=_ids())
    assert [s.id for s in out] == ["q2", "q3"]

def test_reconcile_dedups_by_map_and_criterion():
    cur = [_mk("q2", 1, "active"), _mk("q3", 2, "pending", dw="on_map")]
    out = reconcile_quests(cur, {"add": [{"map": 2, "talk": False, "done_when": "on_map", "why": "dup"}], "remove": []}, next_id=_ids())
    assert sum(1 for s in out if s.map == 2 and (s.done_when or "on_map") == "on_map") == 1
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement:
```python
def reconcile_quests(current, proposal, *, next_id):
    add = proposal.get("add") or []
    remove = set(proposal.get("remove") or [])
    done = [s for s in current if s.status == "done"]
    active = [s for s in current if s.status == "active"]
    # pending survivors: dropped if removed; wedged are ALWAYS dropped (replaced by adds below)
    pending = [s for s in current if s.status == "pending" and s.id not in remove]
    have = {(s.map, s.done_when or "on_map") for s in done + active + pending}
    new_steps = []
    for a in add:
        try:
            mp = int(a["map"])
        except (KeyError, TypeError, ValueError):
            continue
        dw = a.get("done_when")
        key = (mp, dw or "on_map")
        if key in have:
            continue                       # dedup
        have.add(key)
        new_steps.append(QuestStep(id=next_id(), map=mp, talk=bool(a.get("talk")),
                                   who=(a.get("who") or None), done_when=dw, why=str(a.get("why") or "")[:80]))
    # order: done + active first (progress), then new adds (after active), then surviving pending
    return done + active + new_steps + pending
```
- [ ] **Step 4:** run → PASS; full suite green.
- [ ] **Step 5:** commit `quest_reconciler: reconcile_quests — preserve done/active, replace wedged, dedup adds`.

---

## Task 4: `Planner.revise_quests` (L1 model call, DSL-grounded)

**Files:** Modify `src/pokemon_agent/agent/planner_llm.py`; Test `tests/unit/test_l1_planner.py` (new).

`revise_quests(emu, context) -> dict` returns `{"assessment","change","mission","milestone","add":[...],"remove":[...]}`. Uses the strategist provider + KB via `_llm_with_search` (final_key `"assessment"` or a sentinel). Validates: drop `add` steps whose `done_when` is unparseable (`_parse_done_when`); on call/parse failure return `{"change": False}` (never wipe). Add `L1_SYSTEM` teaching the DSL + `done_when` vocabulary + that "no change" is expected most cycles.

- [ ] **Step 1: Failing tests**
```python
# tests/unit/test_l1_planner.py
import json
from pokemon_agent.agent.planner_llm import Planner

class FP:
    def __init__(self, obj): self.obj = obj
    def chat_json(self, system, state, image=None): return json.dumps(self.obj), 0, {}

def _ctx():
    return {"current_map": {"id": 1, "name": "Viridian City"}, "party": ["Squirtle L8 20/26"],
            "items": [], "badges": 0, "plan": [], "signals": {"hp_frac": 0.8, "blocked_for_n": 7},
            "mission": "", "milestone": ""}

def test_revise_quests_parses_add_and_change():
    obj = {"assessment": "blocked at Viridian north -> need parcel", "change": True,
           "mission": "reach Pewter", "milestone": "deliver Oak's Parcel",
           "add": [{"map": 42, "talk": True, "who": "clerk", "done_when": "has_item:Oak's Parcel", "why": "get parcel"}],
           "remove": []}
    p = Planner(goal_map=2, strategist=FP(obj))
    out = p.revise_quests(emu=None, context=_ctx())
    assert out["change"] and out["add"][0]["map"] == 42 and out["milestone"]

def test_revise_quests_no_change_passthrough():
    p = Planner(goal_map=2, strategist=FP({"assessment": "on track", "change": False, "add": [], "remove": []}))
    assert p.revise_quests(emu=None, context=_ctx())["change"] is False

def test_revise_quests_drops_invalid_done_when():
    obj = {"change": True, "add": [{"map": 3, "talk": False, "done_when": "nonsense", "why": "x"}], "remove": []}
    p = Planner(goal_map=2, strategist=FP(obj))
    out = p.revise_quests(emu=None, context=_ctx())
    assert out["add"] == []                     # invalid criterion dropped

def test_revise_quests_garbage_is_no_change():
    class Bad:
        def chat_json(self, s, st, image=None): return "not json", 0, {}
    p = Planner(goal_map=2, strategist=Bad())
    assert p.revise_quests(emu=None, context=_ctx())["change"] is False
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement `L1_SYSTEM` (teach the DSL + done_when vocab + "reply {\"change\":false} if the plan is fine; SEARCH the KB for story gates / locations before adding steps") and `revise_quests`. Reuse `_llm_with_search` (so L1 can query Orrery) and `_parse_done_when` for validation. Provider = `self.strategist or self.provider`; if None → `{"change": False}`. Wrap parse in try/except → `{"change": False}`. For each `add`, keep only if `_parse_done_when(step["done_when"], step["map"])` is not None; coerce `remove` to a list of strings.
- [ ] **Step 4:** run → PASS; full suite green.
- [ ] **Step 5:** commit `planner: revise_quests — periodic L1 quest review in the done_when DSL (KB-grounded, never wipes on failure)`.

---

## Task 5: signals + emergency-heal reflex

**Files:** New `src/pokemon_agent/agent/signals.py`; Test `tests/unit/test_signals.py`.

- [ ] **Step 1: Failing tests**
```python
from pokemon_agent.agent.signals import party_hp_frac, needs_emergency_heal

def test_hp_frac_and_emergency():
    party = [{"hp": 2, "max_hp": 26}, {"hp": 0, "max_hp": 20}]
    assert 0.0 < party_hp_frac(party) < 0.1
    assert needs_emergency_heal(party) is True            # a fainted member
    healthy = [{"hp": 25, "max_hp": 26}]
    assert needs_emergency_heal(healthy) is False

def test_emergency_on_low_frac():
    assert needs_emergency_heal([{"hp": 3, "max_hp": 30}]) is True   # <0.15
    assert needs_emergency_heal([{"hp": 20, "max_hp": 30}]) is False
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement `signals.py`: `party_hp_frac(party)`, `min_level(party)`, `needs_emergency_heal(party, thresh=0.15)` (True if any member `hp==0 and max_hp>0`, or overall frac < thresh). Also `game_signals(emu) -> dict` reading party/items/badges via existing `read_party/read_items/read_badges` and returning `{hp_frac, min_level, badges, items}` (the loop adds `blocked_for_n`). Reuse `games/pokemon_red/needs.py` where it already has helpers.
- [ ] **Step 4:** run → PASS; full suite green.
- [ ] **Step 5:** commit `signals: party hp/level + emergency-heal predicate (needs flow into L1)`.

---

## Task 6a: Executive — hold `_plan_steps`, compile/recompile, L1 gate (additive)

**Files:** Modify `src/pokemon_agent/agent/reason_loop.py`; Test additions in `tests/unit/test_executive.py`.

Introduce the plan machinery WITHOUT yet ripping out the arbiter path — add the L1 gate + `_plan_steps` and a `_run_l1(obs)` that reconciles+recompiles, plus helpers, and unit-test them directly. (6b flips `_manage_directive` to use them and removes the old machinery.)

- [ ] **Step 1: Failing tests** (concrete — use the existing `_loop` helper in `tests/unit/test_executive.py`, which returns `(loop, emu)`; stub `revise_quests` so no LLM runs):
```python
from collections import deque
from pokemon_agent.agent.plan import Intent

def test_run_l1_inserts_and_recompiles():
    loop, _ = _loop(map_id=1, goal_map=2)
    loop.planner.revise_quests = lambda emu, ctx: {
        "change": True, "mission": "reach Pewter", "milestone": "deliver parcel",
        "add": [{"map": 42, "talk": True, "who": "clerk", "done_when": "has_item:Oak's Parcel", "why": "get parcel"}],
        "remove": []}
    obs, _ = loop.builder.build(capture_screenshot=False)
    loop._run_l1(obs)
    assert any(s.map == 42 and s.talk for s in loop._plan_steps)          # reconciled into the plan
    assert isinstance(loop._quest, deque) and any(d.intent == Intent.TALK_TO for d in loop._quest)  # recompiled
    assert loop._plan is not None and loop._plan.milestone == "deliver parcel"   # durable memory updated

def test_run_l1_no_change_keeps_plan():
    loop, _ = _loop(map_id=1, goal_map=2)
    from pokemon_agent.agent.quest_reconciler import QuestStep
    loop._plan_steps = [QuestStep(id="q1", map=2, done_when="on_map", status="active")]
    loop.planner.revise_quests = lambda emu, ctx: {"change": False, "add": [], "remove": []}
    obs, _ = loop.builder.build(capture_screenshot=False)
    loop._run_l1(obs)
    assert [s.id for s in loop._plan_steps] == ["q1"]                     # unchanged on no-change

def test_l1_due_is_a_pure_predicate():
    loop, _ = _loop(goal_map=2)
    loop.l1_every = 5
    loop._legs_since_l1, loop._blocked_for_n, loop._l1_event = 0, 0, False
    assert loop._l1_due() is False
    loop._legs_since_l1 = 5;  assert loop._l1_due() is True                # cadence
    loop._legs_since_l1 = 0;  loop._blocked_for_n = 6;  assert loop._l1_due() is True   # blocked (>=BLOCK_TRIGGER)
    loop._blocked_for_n = 0;  loop._l1_event = True;    assert loop._l1_due() is True   # event flag
```
Note: `_l1_due()` is a PURE predicate (no side effects); `_run_l1` resets `_legs_since_l1=0`, `_l1_event=False` after it runs, and ensures `self._plan` is an `AgentPlan` (create one if `None`) before writing `mission`/`milestone`.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement, in `ReasoningLoop`:
  - `__init__`: `self._plan_steps: list[QuestStep] = []`, `self._qid = 0`, `self._legs_since_l1 = 0`, `self._blocked_for_n = 0`, `self._l1_event = False`.
  - `_next_qid()` → `self._qid += 1; return f"q{self._qid}"`.
  - `_l1_due()` → **pure predicate** (no side effects): True if `self._legs_since_l1 >= self.l1_every` or `self._l1_event` or `self._blocked_for_n >= BLOCK_TRIGGER`. Counter resets live in `_run_l1` (which sets `_legs_since_l1=0`, `_l1_event=False` after running), NOT here.
  - `_run_l1(obs)`: build context (map, party, items, badges, `_plan_steps` with status + done_when, signals incl. `blocked_for_n`, mission/milestone) → `self.planner.revise_quests(emu, ctx)`; if `change`, `self._plan_steps = reconcile_quests(self._plan_steps, {add,remove}, next_id=self._next_qid)`; update `AgentPlan.mission/milestone` and append dropped approaches to `tried_failed`; recompile the pending region into `self._quest` (keep the active directive); emit `l1_review` + `quest` events. On `revise_quests` failure/`change:false`, do nothing (keep plan). Emit `l1_failed` on exception.
  - `_recompile_quest()`: `self._quest = deque(compile_steps_to_directives([s for s in self._plan_steps if s.status == 'pending']))`.
  - Module constants: `L1_EVERY_N_LEGS_DEFAULT = 5`, `BLOCK_TRIGGER = 6`; `self.l1_every = l1_every or 5` from a new `__init__` kwarg.
- [ ] **Step 4:** run → PASS; full suite green (old path still active, new helpers additive).
- [ ] **Step 5:** commit `reason_loop: add _plan_steps + L1 gate + reconcile/recompile helpers (additive)`.

---

## Task 6b: Executive — switch `_manage_directive` to plan-driven; remove arbiter-intent machinery

**Files:** Modify `src/pokemon_agent/agent/reason_loop.py`; **rewrite** intent-driven tests in `tests/unit/test_executive.py`.

- [ ] **Step 1: Rewrite the affected executive tests** to the plan-driven model (do this first so they express the new contract, then make them pass):
  - `test_success_predicate_triggers_replan` → success on the active directive marks its step `done` and advances to the next compiled directive.
  - `test_higher_need_suspends_current_directive_on_the_stack` → **replace** with `test_emergency_heal_preempts_then_restores_plan` (near-faint injects a heal directive ahead of the plan; when HP safe, the plan resumes).
  - `test_story_gate_escalates_to_quest_and_advances_in_order`, `test_wedged_quest_step_re_strategizes_instead_of_abandoning`, `test_low_escalation_score_does_not_quest` → **replace** with L1-driven equivalents: a wedged step is marked `wedged` and L1's next review replaces it (the rest of `_plan_steps` survives); no `_INTENT_PRIORITY`/escalation-score path remains.
  - **`test_servo_walks_toward_same_map_tile_no_model_call` (test_executive.py:106) MUST be rewritten too** — it currently seeds `loop.arbiter = StubArbiter(TRAVEL)` + `loop.planner = StubPlanner(d)` and asserts `loop._directive is d` after `step_once`. Under plan-driven `_manage_directive` the arbiter/`planner.plan` path is gone and an empty `_plan_steps` synthesizes a default goal step, so both assertions fail. Rewrite it to **seed `loop._plan_steps` + recompile `loop._quest`** with the same on-map tile directive and assert the servo steps south (no model call). (Reviewer-caught unlisted break.)
  - Keep the door/exit servo tests (`test_warp_exit_dir_*`, `test_servo_steps_through_door_*`, `test_navigate_leg_*`) unchanged.
- [ ] **Step 2:** run the rewritten tests → FAIL (old `_manage_directive` still intent-driven).
- [ ] **Step 3:** rewrite `_manage_directive(obs)` to be plan-driven:
  0. **Imports:** `read_party` is only imported inline today (inside `_directive_satisfied`); add a module-scope import (or have `game_signals` surface the party list) so Step 3's reflex/signals don't `NameError`.
  1. `signals = game_signals(emu)`; `signals["blocked_for_n"] = self._blocked_for_n`.
  2. **Emergency reflex:** read `party = read_party(emu)` (or reuse `game_signals`'s party); if `needs_emergency_heal(party)` and the active directive isn't already the heal → set `self._directive` to a heal directive (`Intent.HEAL`, `success={"hp_frac": ">=0.95"}`) and return it (preempt). When HP safe again, drop back to the plan. (`signals.py`'s `needs_emergency_heal` takes a PARTY LIST — it implements its own fainted check; do NOT call `needs.any_fainted(emu)` here, which takes an emu.)
  3. **L1 gate:** if `_l1_due()` → `_run_l1(obs)`.
  4. **Bootstrap/normal:** if `_plan_steps` empty → synthesize a default step `QuestStep(id=_next_qid(), map=goal_map, done_when="on_map")` and recompile.
  5. **Termination/advance:** if `self._directive` is None or `_directive_satisfied(self._directive)` → mark the active step `done` (by `quest_id`), `self._directive = self._quest.popleft()` if any (mark its step `active`), else re-run L1 / keep default.
  6. **Wedge:** replace the `self._quest.clear()` path — when the active step is wedged (servo_fail / blocked_for_n over budget), mark its step `wedged` and set `self._l1_event = True` (L1 will replace it next gate). Do NOT clear the plan.
  - **Remove:** the `arbiter.intent`-driven directive selection, `_INTENT_PRIORITY`/`_dstack` suspend/stack, `_should_replan(intent)`'s intent-change branch, the HEAL errand branch, and `_try_quest`/`_should_escalate` escalation (superseded by periodic L1 + `blocked_for_n`). Keep `_directive_satisfied`, `_carry_plan` (now carrying `_plan_steps`), and the servo/L2/L3 path.
  - Track `blocked_for_n`: increment when a leg makes no progress (reuse the servo-fail/local-loop signal), reset on progress; feed L1.
  - **Recorder safety (pull forward from Task 8):** `_finish`'s recorder `extra` currently reads `self._in_quest` (reason_loop.py ~1312), which this task removes. In THIS task, update that `extra` dict to drop `_in_quest` and instead record `mission`/`milestone` (from `self._plan`) and `plan_steps` (id/map/status from `self._plan_steps`); keep `quest_remaining` from `self._quest`. Otherwise a live run between 6b and 8 would `AttributeError` (executive tests won't catch it — recorder is None there). Task 8 then only ADDS the L2-milestone wiring + `l1_last`.
- [ ] **Step 4:** run rewritten tests → PASS; then `uv run python -m pytest -q` → all green (fix any fallout in test_unified_loop/test_executive).
- [ ] **Step 5:** commit `reason_loop: plan-driven _manage_directive (L1 owns the plan; emergency-heal reflex; no arbiter-intent/priority/escalation path)`.

---

## Task 7: `--l1-every` flag + run_agent wiring

**Files:** Modify `scripts/run_agent.py`; Test: covered by executive tests.

- [ ] **Step 1:** add `ap.add_argument("--l1-every", type=int, default=5, help="run L1 strategic review every N legs (also runs on events)")`.
- [ ] **Step 2:** pass `l1_every=args.l1_every` into `ReasoningLoop(...)` (reason branch). Ensure `knowledge=` is already wired (it is) so L1 can query the KB.
- [ ] **Step 3:** `uv run python -m pytest -q` → green.
- [ ] **Step 4:** commit `run_agent: --l1-every flag; wire L1 cadence`.

---

## Task 8: Recorder + L2 milestone + integration fixture + live validation

**Files:** Modify `src/pokemon_agent/agent/reason_loop.py` (recorder `extra`, L2 `milestone`); Tests + fixture.

- [ ] **Step 1:** Feed the current `milestone` into `_propose_target`'s context (so L2's targets are framed by L1). The recorder `extra` already carries `mission`/`milestone`/`plan_steps`/`quest_remaining` (moved into Task 6b); here just ADD `l1_last` (last review's assessment + change bool). Unit-test that the recorder `extra` includes `mission`/`plan_steps`/`l1_last`.
- [ ] **Step 2: Integration fixture test** (no live LLM — use a stub strategist that returns the parcel add when `blocked_for_n` is high): assert that after `_run_l1`, `_plan_steps` contains the Mart→parcel→Lab→deliver steps and `_quest` compiles a TALK_TO to Oak with `no_item:Oak's Parcel`. Assert a wedged step is replaced (plan survives).
- [ ] **Step 3:** `uv run python -m pytest -q` → all green. Report the final count.
- [ ] **Step 4: Live validation** (manual; needs keys + KB):
```fish
uv run python scripts/run_agent.py --rom roms/pokemon_red.gb --load-state roms/pokemon_red.gb.state \
  --mode reason --goal "Reach Pewter and beat Brock; deliver Oak's Parcel first if blocked" \
  --goal-map 2 --level-target 12 --decider typesafe --pather policy --no-vision \
  --orrery-workspace 6d677a16 --l1-every 5 --steps 700 --record-dir runs/l1-test
```
Watch for: periodic `l1_review` events (most `change:false`); when blocked at Viridian, an `l1_review` with `change:true` adding the parcel errand; map path reaching the Mart (42) then back to Oak's Lab (40); durable `mission/milestone` in the viewer. Confirm a wedged step no longer wipes the plan.
- [ ] **Step 5:** commit `reason_loop: record L1 plan/mission + feed milestone to L2; integration fixture`.

---

## Definition of done
- L1 runs every N legs + on events; most cycles are `change:false`; it adds a heal/grind/story quest when signals/KB warrant.
- The reconciler preserves done+active, replaces wedged (never `quest.clear()`), dedups adds — pure-unit-tested.
- Needs flow into L1 as signals; only near-faint is a deterministic reflex.
- `NeedsArbiter` unchanged as a library; `test_needs_arbiter.py` still green; the intent-driven executive tests are rewritten to the plan-driven model.
- Durable `mission/milestone/tried_failed` maintained + shown in the viewer; L2 gets `milestone`.
- Full suite green; live run recognizes the parcel errand with the KB attached.
