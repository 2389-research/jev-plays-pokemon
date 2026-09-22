# Distillation-Capture — Instrumentation Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture every agent decision's `(input → output + confidence)` at every model-backed layer into a self-contained per-decision log, attach outcome labels, and anchor each to a reloadable save state — with **zero change to agent behavior**.

**Architecture:** A thin, best-effort `Capture` helper is held on the loop exactly like the existing `on_event` tee. Each decision layer (L1/L2 planner methods, the Jev reasoner, `battle_move`) calls `capture.record(...)` at the point where its input state and raw response are still in scope. Records are buffered per step and flushed to `<record-dir>/decisions.jsonl` when the step's `RunRecorder.record(...)` fires. A pure post-processor (`outcome.py`) reads the finished record-dir and writes per-step progress deltas + a per-episode `outcome.json`. Step-anchored saves reuse the recorder's existing `state_every=1` path.

**Tech stack:** Python 3.13, `uv run pytest`. Extends `src/pokemon_agent/logging/` (`run_recorder.py`) and instruments `src/pokemon_agent/agent/{planner_llm,typesafe_reasoner,reason_loop}.py` + `src/pokemon_agent/games/pokemon_red/battle_agent.py`. CLI flag in `scripts/run_agent.py`.

**Scope note (from spec §8a):** This is **Phase 1 of three**. Phase 2 (`distill_export.py`) and Phase 3 (replay/eval harness + `--swap`) are **separate follow-on plans**, written after this lands green. This plan is deliberately the standalone, risky "zero behavior change" surface — ship and verify it alone first.

**Spec:** `docs/superpowers/specs/2026-09-22-distillation-capture-design.md`

---

## Key facts verified against the code (do not re-derive)

- `provider.chat_json(system, user)` returns `(raw_content: str, latency_ms: int, usage: dict)`. Every planner call site currently discards the last two: `content, _, _ = prov.chat_json(...)`. Capture recovers them.
- `client.system_one(state=, questions=)` (TypeSafe) → `resp.answers[key]` with `.choice`, `.confidence`, `.probabilities`. `usage["confidence"]` is already surfaced by `TypeSafeReasoner`.
- The loop already builds a `_tee` for `on_event` when a `recorder` is present (`reason_loop.py:108-115`). `Capture` mounts the same way.
- The per-step record is written by `self.recorder.record(...)` inside `_finish` at `reason_loop.py:1741`. Flush `Capture` there (same place, same step number `self.session.step`).
- `RunRecorder.__init__(..., state_every: int = 0)` already writes `states/map<M>_step<N>.state` every step when `state_every=1` (`run_recorder.py:121-135`). Step-anchoring = pass `state_every=1`; the anchor is the **glob** `states/*_step<N>.state` (map id is not known at capture time).
- Planner decision methods (`planner_llm.py`): single-call `l1_triage` (503), `l1_decide` (561), `l1_repair` (597); L2 `next_waypoint` (696), `propose_target` (750); multi-round via `_llm_with_search` (369-395) → `l1_brainstorm` (511), `strategize` (396).
- Jev methods (`typesafe_reasoner.py`): `step` (158, the executor action Choice — layer `jev_action`), `path_step` (278, `jev_path`), `choose_policy` (374, `jev_policy`), `choose_flow` (334, the dialogue/menu flow router — `jev_flow`), `choose_npc` (invoked `reason_loop.py:1090` — `jev_npc`). `battle_agent.choose_move` (122, `battle_move`).
- CLI: `scripts/run_agent.py` argparse (73+), `RunRecorder(emu, rec_dir)` at 253, `ReasoningLoop(...)` at 257.

---

## File Structure

**Create:**
- `src/pokemon_agent/logging/capture.py` — the `Capture` helper: per-step buffer, `record()`, `record_round()`, `flush()`, `begin_step()`, `write_run_meta()`. One responsibility: turn a decision into a well-formed record and persist it. ~120 lines.
- `src/pokemon_agent/logging/outcome.py` — the outcome labeler: pure functions over a finished record-dir → per-step progress vector + `outcome.json`. No emulator, no network. ~130 lines.
- `tests/unit/test_capture.py`, `tests/unit/test_outcome.py`.
- `tests/integration/test_capture_live.py` — a short ROM-guarded `--capture distill` smoke.

**Modify:**
- `src/pokemon_agent/agent/reason_loop.py` — build+mount `Capture`, `begin_step`/`flush` per step, write `run_meta.json` once, call the labeler at run-end. Attach `capture` onto `self.planner` and `self.reasoner`.
- `src/pokemon_agent/agent/planner_llm.py` — `capture` attr (default `None`); record at the 5 single/round sites.
- `src/pokemon_agent/agent/typesafe_reasoner.py` — `capture` attr; record at the 4 Jev sites with confidence.
- `src/pokemon_agent/games/pokemon_red/battle_agent.py` — record `battle_move` (capture the discarded `ans`).
- `scripts/run_agent.py` — `--capture {off,decisions,distill}` flag; wire `state_every` + `capture_mode`.

**Zero-behavior-change contract (applies to every task):** `capture.record(...)` and `flush()` are side-effect-only and **never raise** (wrap the body in `try/except: pass`, like `RunRecorder`). When `capture is None` (the default, and whenever `--capture off`), every instrumented method must execute the exact same statements it does today — the record call is an added line guarded by `cap = getattr(self, "capture", None)` / `if cap is not None:`, never a change to control flow or return values.

---

## Task 1: `Capture` helper core

**Files:**
- Create: `src/pokemon_agent/logging/capture.py`
- Test: `tests/unit/test_capture.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_capture.py
import json
from pathlib import Path
from pokemon_agent.logging.capture import Capture


def test_record_flush_writes_wellformed_line(tmp_path):
    cap = Capture(tmp_path, mode="distill")
    cap.begin_step(7, anchor="states/*_step7.state")
    cap.record("l1_decide", kind="model", model="glm-5.3",
               input={"a": 1}, output_raw='{"x":1}', parsed={"x": 1},
               confidence=None, tokens=42, latency_ms=88)
    cap.record("jev_flow", kind="model", model="tsafe",
               input={"b": 2}, output_raw="up", parsed={"choice": "up"},
               confidence=0.9, tokens=0, latency_ms=12)
    cap.flush()

    lines = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert len(lines) == 2
    r0, r1 = json.loads(lines[0]), json.loads(lines[1])
    assert r0["step"] == 7 and r0["seq"] == 0 and r0["layer"] == "l1_decide"
    assert r0["anchor"] == "states/*_step7.state"
    assert r0["input"] == {"a": 1} and r0["output_parsed"] == {"x": 1}
    assert r0["tokens"] == 42 and r0["latency_ms"] == 88
    assert r1["seq"] == 1 and r1["confidence"] == 0.9


def test_record_never_raises_on_unserializable(tmp_path):
    cap = Capture(tmp_path, mode="distill")
    cap.begin_step(1, anchor=None)
    cap.record("l1_decide", kind="model", input={"bad": object()}, parsed=None)  # not JSON-able
    cap.flush()  # must not raise
    # a best-effort record still lands (with a serialization fallback) OR is skipped; file exists
    assert (tmp_path / "decisions.jsonl").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd jev-plays-pokemon && uv run python -m pytest tests/unit/test_capture.py -q`
Expected: FAIL — `ModuleNotFoundError: pokemon_agent.logging.capture`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokemon_agent/logging/capture.py
"""Per-decision capture for distillation/replay (design §3). Side-effect-only: it NEVER
raises and NEVER changes agent behavior — a decision layer calls `record(...)` with the
exact input it saw and its raw/parsed output; records buffer per step and flush to
`decisions.jsonl` beside the run recorder's `log.jsonl`.

Held on the loop like `on_event`; attached onto `planner`/`reasoner` so their methods can
reach it via `getattr(self, "capture", None)`."""
from __future__ import annotations

import json
import time
from pathlib import Path


def _safe(obj):
    """Best-effort JSON-serializable copy — fall back to `repr` for exotic values so a
    single unserializable field can never break capture (zero-behavior-change contract)."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return json.loads(json.dumps(obj, default=repr))


class Capture:
    def __init__(self, run_dir: str | Path, *, mode: str = "off"):
        self.dir = Path(run_dir)
        self.mode = mode                       # "off" | "decisions" | "distill"
        self.enabled = mode in ("decisions", "distill")
        self._step = 0
        self._anchor: str | None = None
        self._seq = 0
        self._buf: list[dict] = []
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._fh = (self.dir / "decisions.jsonl").open("a", encoding="utf-8")
        else:
            self._fh = None

    def begin_step(self, step: int, anchor: str | None) -> None:
        self._step, self._anchor, self._seq, self._buf = step, anchor, 0, []

    def record(self, layer: str, *, kind: str = "model", model: str | None = None,
               input=None, output_raw=None, parsed=None, confidence=None,
               tokens: int = 0, latency_ms: int = 0, group_id: str | None = None,
               extra: dict | None = None) -> None:
        if not self.enabled:
            return
        try:
            rec = {
                "run_id": self.dir.name, "step": self._step, "seq": self._seq,
                "layer": layer, "kind": kind, "model": model,
                "input": _safe(input), "output_raw": output_raw,
                "output_parsed": _safe(parsed), "confidence": confidence,
                "tokens": tokens, "latency_ms": latency_ms,
                "anchor": self._anchor,
            }
            if group_id:
                rec["group_id"] = group_id
            if extra:
                rec["extra"] = _safe(extra)
            self._buf.append(rec)
            self._seq += 1
        except Exception:
            pass

    def record_round(self, layer: str, *, group_id: str, **kw) -> None:
        """A single KB-search round inside a multi-round decision (kind='model_round')."""
        self.record(layer, kind="model_round", group_id=group_id, **kw)

    def flush(self) -> None:
        if not self.enabled or self._fh is None:
            return
        try:
            for rec in self._buf:
                self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()
        except Exception:
            pass
        finally:
            self._buf = []

    def write_run_meta(self, meta: dict) -> None:
        if not self.enabled:
            return
        try:
            (self.dir / "run_meta.json").write_text(json.dumps(_safe(meta), indent=2))
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m pytest tests/unit/test_capture.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/pokemon_agent/logging/capture.py tests/unit/test_capture.py
git commit -m "distill: Capture helper — per-decision buffer + decisions.jsonl (zero-behavior)"
```

---

## Task 2: Outcome labeler

**Files:**
- Create: `src/pokemon_agent/logging/outcome.py`
- Test: `tests/unit/test_outcome.py`

The labeler is a pure post-processor over a finished record-dir (`log.jsonl` + `decisions.jsonl`). Per spec §4: per-step progress vector + per-episode `outcome.json`. Party is parsed from the log strings `"SPECIES L# hp/max"` (spec "party parse contract").

**Grounding warning (verified against the real recorder).** The base `log.jsonl` does NOT contain a `caught` event, a `battle_end.result`, or a `deliver` intent — spec §4 flagged this. So derive from signals that ARE persisted:
- **caught** = the party list grew vs the previous step (the recorder writes `party` every step). This is the robust log-only catch signal; use it for `caught_count`.
- **battle_result** / **delivered** are best-effort: parse the persisted `detail` string, and (for capture runs) prefer structured flags written into the step `extra`. Mark them best-effort in Phase 1; a clean structured emission is a small follow-up, not a blocker.
The map-derived fields (`reached_goal_map`, `map_arrival_steps`, `steps`) come from real `map_id`/`step` log fields and are exact.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_outcome.py
import json
from pokemon_agent.logging.outcome import step_progress, label_run


def test_step_progress_deltas():
    prev = {"map_id": 1, "party": ["Squirtle L9 20/27"], "items": ["Poke Ball"]}
    cur = {"map_id": 2, "party": ["Squirtle L10 27/27"], "items": ["Poke Ball", "Potion"]}
    p = step_progress(prev, cur)
    assert p["map_changed"] is True
    assert p["level_delta"] == 1
    assert p["hp_delta"] == 7
    assert p["items_delta"] == 1


def test_label_run_writes_outcome(tmp_path):
    # Rows mirror the REAL recorder schema: party/items/map_id/step are top-level; a catch is a
    # party-length increase (there is no synthetic "caught" event in log.jsonl).
    log = tmp_path / "log.jsonl"
    rows = [
        {"step": 0, "map_id": 1, "party": ["Squirtle L5 18/18"], "items": [], "events": []},
        {"step": 1, "map_id": 1, "party": ["Squirtle L6 20/20"], "items": ["Potion"], "events": []},
        {"step": 2, "map_id": 2, "party": ["Squirtle L6 20/20", "Pidgey L4 12/12"],
         "items": ["Potion"], "events": []},   # party grew 1 -> 2  => a catch
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows))
    out = label_run(tmp_path, goal_map=2)
    assert out["reached_goal_map"] is True
    assert out["caught_count"] == 1            # detected from party growth
    assert out["steps"] == 3
    assert (tmp_path / "outcome.json").exists()
    # per-map arrival step recorded
    assert out["map_arrival_steps"]["2"] == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m pytest tests/unit/test_outcome.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokemon_agent/logging/outcome.py
"""Outcome labeler (design §4) — a PURE post-processor over a finished record-dir. Reads
`log.jsonl`, computes a per-step progress vector and a per-episode `outcome.json`. Runs
offline on any past run; no emulator, no network."""
from __future__ import annotations

import json
import re
from pathlib import Path

_LEVEL = re.compile(r"L(\d+)")
_HP = re.compile(r"(\d+)/(\d+)")


def _party_level_hp(party: list[str]) -> tuple[int, int]:
    """Sum of levels and current HP across a party of `"SPECIES L# hp/max"` strings."""
    lvl = hp = 0
    for s in party or []:
        m = _LEVEL.search(s or "")
        if m:
            lvl += int(m.group(1))
        h = _HP.search(s or "")
        if h:
            hp += int(h.group(1))
    return lvl, hp


def step_progress(prev: dict | None, cur: dict) -> dict:
    """The per-step progress delta (spec §4). `map_hop_delta` needs PortalGraph + component,
    so it is left out here (added by the export join when available) — this is the log-only
    directional signal."""
    prev = prev or {}
    p_party, c_party = prev.get("party") or [], cur.get("party") or []
    plvl, php = _party_level_hp(p_party)
    clvl, chp = _party_level_hp(c_party)
    events = cur.get("events") or []
    kinds = {e.get("kind") for e in events}
    # battle_result is best-effort: the base log's battle_end event carries no result key, so this
    # is None on real logs unless a structured flag is added later. Kept for forward-compat.
    battle = next((e.get("result") for e in events if e.get("kind") == "battle_end"), None)
    return {
        "map_changed": prev.get("map_id") != cur.get("map_id") and prev != {},
        "level_delta": clvl - plvl,
        "hp_delta": chp - php,
        "items_delta": len(cur.get("items") or []) - len(prev.get("items") or []),
        "battle_result": battle,
        "caught": len(c_party) > len(p_party),   # party grew => a catch (log-only signal)
        "wedged": any(k in kinds for k in ("step_wedged", "quest_step_wedged")),
    }


def iter_rows(record_dir: str | Path):
    p = Path(record_dir) / "log.jsonl"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def label_run(record_dir: str | Path, *, goal_map: int | None = None,
              terminal_reason: str = "end") -> dict:
    """Compute + write `outcome.json` for a finished run; returns the outcome dict."""
    record_dir = Path(record_dir)
    rows = list(iter_rows(record_dir))
    caught = 0
    arrivals: dict[str, int] = {}
    reached = False
    prev_party = None
    for r in rows:
        party_n = len(r.get("party") or [])
        if prev_party is not None and party_n > prev_party:
            caught += party_n - prev_party      # a catch => the party grew (log-only signal)
        prev_party = party_n
        mid = r.get("map_id")
        if mid is not None and str(mid) not in arrivals:
            arrivals[str(mid)] = r.get("step")
        if goal_map is not None and mid == goal_map:
            reached = True
    out = {
        "reached_goal_map": reached,
        "goal_map": goal_map,
        "caught_count": caught,
        # best-effort in Phase 1 (no clean persisted signal): parse detail, else structured extra.
        "delivered": any("deliver" in str(r.get("detail") or "").lower() for r in rows),
        "steps": len(rows),
        "terminal_reason": terminal_reason,
        "map_arrival_steps": arrivals,
    }
    try:
        (record_dir / "outcome.json").write_text(json.dumps(out, indent=2))
    except Exception:
        pass
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m pytest tests/unit/test_outcome.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pokemon_agent/logging/outcome.py tests/unit/test_outcome.py
git commit -m "distill: outcome labeler — per-step deltas + per-episode outcome.json"
```

---

## Task 3: Mount `Capture` on the loop (no layer records yet)

**Files:**
- Modify: `src/pokemon_agent/agent/reason_loop.py` (`__init__` ~88-207; `_finish` ~1741; run-end)
- Test: `tests/unit/test_capture_loop_mount.py`

Add a `capture_mode: str = "off"` constructor param. Build a `Capture` when a `recorder` is present, mount it on `self.capture`, and attach it onto `self.planner` and `self.reasoner`. Call `begin_step` at the top of `step_once` and `flush` inside `_finish` right after `self.recorder.record(...)`. Write `run_meta.json` once and run the labeler at run-end.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_capture_loop_mount.py
from pokemon_agent.logging.capture import Capture


def test_loop_mounts_capture_on_planner_and_reasoner(monkeypatch, tmp_path):
    # Build a ReasoningLoop with a stub recorder + capture_mode="distill"; assert the loop
    # exposes self.capture and attaches the SAME object to planner/reasoner.
    from pokemon_agent.agent import reason_loop as RL
    loop = RL.ReasoningLoop.__new__(RL.ReasoningLoop)   # bypass heavy __init__
    cap = Capture(tmp_path, mode="distill")
    loop.capture = cap
    class _P:  # noqa
        pass
    loop.planner = _P(); loop.reasoner = _P()
    RL.ReasoningLoop._mount_capture(loop)
    assert loop.planner.capture is cap and loop.reasoner.capture is cap
```

*(This tests a small extracted helper `_mount_capture(self)` that sets `self.planner.capture` / `self.reasoner.capture` from `self.capture` when each is present. Keep it tiny and call it at the end of `__init__`.)*

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m pytest tests/unit/test_capture_loop_mount.py -q`
Expected: FAIL — `_mount_capture` missing.

- [ ] **Step 3: Implement in `reason_loop.py`**

In `__init__` signature add `capture_mode: str = "off",`. After the recorder tee block (~line 115) add:

```python
# Distillation capture (design §3): a side-effect-only per-decision log, mounted like on_event.
from ..logging.capture import Capture
self.capture = Capture(recorder.dir, mode=capture_mode) if recorder is not None else Capture(".", mode="off")
```

After `self.planner`/`self.reasoner` are set (end of `__init__`), call `self._mount_capture()`, and add the method:

```python
def _mount_capture(self) -> None:
    """Attach the loop's Capture onto the planner + reasoner so their layer methods can
    reach it via getattr(self, "capture", None). No-op when a component is absent."""
    cap = getattr(self, "capture", None)
    if cap is None:
        return
    for comp in (getattr(self, "planner", None), getattr(self, "reasoner", None)):
        if comp is not None:
            try:
                comp.capture = cap
            except Exception:
                pass
```

In `step_once` (top, right after `self.session.record_position(obs.player)`):

```python
self.capture.begin_step(self.session.step, anchor=f"states/*_step{self.session.step}.state")
```

In `_finish`, immediately after the `self.recorder.record(...)` call (~1741):

```python
self.capture.flush()
```

**Flush-coverage hazard (must verify).** `begin_step` resets the per-step buffer, and `flush()` fires only in `_finish`. If ANY `step_once` return path bypasses `_finish`, that step's buffered records are silently dropped at the next `begin_step`. Before finishing this task, trace every `step_once` exit (battle turn, `_advance_dialog`/dialog paths, `_jev_turn`, servo, `_safe_flow`, early returns) and confirm each funnels through `_finish`. If any does not, either route it through `_finish` or move the flush to a single central point that always runs at end-of-step (e.g. a `try/finally` around the `step_once` body). Add a quick assertion-style check: run a demo for K steps with `--capture decisions` and confirm `decisions.jsonl` has records spanning the full step range, not gaps.

At run-end (the `finally` that saves the resume checkpoint — search `save_resume_checkpoint`), add:

```python
try:
    self.capture.write_run_meta({
        "goal_map": self.goal_map, "level_target": self.level_target,
        "pather": self.pather, "l1_every": self.l1_every,
        "portal_graph": getattr(getattr(self, "portals", None), "version", None),
    })
    if self.capture.enabled and self._resume_dir is not None:
        from ..logging.outcome import label_run
        label_run(self._resume_dir, goal_map=self.goal_map)
    self.capture.close()
except Exception:
    pass
```

- [ ] **Step 4: Run to verify it passes + no regression**

Run: `uv run python -m pytest tests/unit/test_capture_loop_mount.py -q && uv run python -m pytest -q`
Expected: PASS; full suite still green (411+ passed).

- [ ] **Step 5: Commit**

```bash
git add src/pokemon_agent/agent/reason_loop.py tests/unit/test_capture_loop_mount.py
git commit -m "distill: mount Capture on the loop (begin_step/flush/run_meta/labeler)"
```

---

## Task 4: Instrument the single-call planner sites (L1 + L2)

**Files:**
- Modify: `src/pokemon_agent/agent/planner_llm.py` (`Planner.__init__`; `l1_triage` 503, `l1_decide` 561, `l1_repair` 597, `next_waypoint` 696, `propose_target` 750)
- Test: `tests/unit/test_capture_planner.py`

Give `Planner` a `self.capture = None` in `__init__`. At each site, after the existing parse, add a guarded record capturing the **full** `chat_json` return (recover the discarded `_, _`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_capture_planner.py
from pokemon_agent.logging.capture import Capture
from pokemon_agent.agent.planner_llm import Planner


class _Prov:
    def chat_json(self, system, user, image=None):
        return ('{"kind":"tile","x":3,"y":4}', 55, {"total_tokens": 120})


def test_propose_target_records_full_return(tmp_path, monkeypatch):
    cap = Capture(tmp_path, mode="distill"); cap.begin_step(3, "states/*_step3.state")
    p = Planner(goal_map=2, level_target=0, reflector=None, provider=_Prov(), strategist=_Prov())
    p.capture = cap
    # call propose_target with a minimal ctx; it should return the parsed target AND record it
    out = p.propose_target(emu=None, context={"player": {"x": 1, "y": 1}})
    cap.flush()
    import json
    recs = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    r = [x for x in recs if x["layer"] == "l2_propose_target"][0]
    assert r["tokens"] == 120 and r["latency_ms"] == 55
    assert r["output_parsed"]["x"] == 3
    assert out["x"] == 3            # behavior unchanged: the parsed target still returns
```

*(Adapt the ctx/args to `propose_target`'s real signature; the point is: parsed return is unchanged AND a record with tokens/latency lands. Read the method body first and match its actual state-building so the call doesn't error. If `propose_target` needs `emu`, pass a minimal stub exposing only what the method touches, or test `l1_decide` instead — pick the site with the fewest deps for the unit test and cover the rest in Task 10's live smoke.)*

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m pytest tests/unit/test_capture_planner.py -q`
Expected: FAIL — no `l2_propose_target` record.

- [ ] **Step 3: Implement — the record pattern at each site**

The uniform edit: change `content, _, _ = prov.chat_json(SYS, state)` to keep all three, then record after parsing. **`chat_json` returns `(raw_content, latency_ms, usage)`** — the 2nd element is latency, the 3rd is the usage dict (token counts live inside it). Do NOT confuse them. Example for `l1_decide` (561):

```python
content, _lat, _usage = prov.chat_json(DECIDE_SYSTEM, state)
# ... existing parse into `result` ...
cap = getattr(self, "capture", None)
if cap is not None:
    _tok = _usage.get("total_tokens", 0) if isinstance(_usage, dict) else 0
    cap.record("l1_decide", model=getattr(prov, "model", None),
               input=state, output_raw=content, parsed=result,
               tokens=_tok, latency_ms=_lat, extra={"usage": _usage})
return result
```

**Confirm the usage token key** against a live LunaRoute response before relying on it (`total_tokens` is the expected key; if the dict nests it, e.g. `usage["usage"]["total_tokens"]`, adjust). The Task 4 test's `_Prov` returns `(content, 55, {"total_tokens": 120})` and asserts `tokens==120, latency_ms==55`, so this extraction is what makes it pass.

Apply the same shape (distinct `layer` string) to: `l1_triage` → `"l1_triage"`, `l1_repair` → `"l1_repair"`, `next_waypoint` → `"l2_next_waypoint"`, `propose_target` → `"l2_propose_target"`. Use the method's real input `state`/`s` variable as `input=` and its parsed return as `parsed=`. **Do not** move the `return`; record just before it.

**Multiple return points (`propose_target` has ~5: tile/enter/approach_npc/exit/unresolved; `next_waypoint` has a retry loop).** Record ONCE per call, capturing the winning `content`/latency/tokens from the `chat_json` that produced the returned target. Cleanest: bind `content, _lat, _usage` from the model call into locals, build the resolved dict, then record right before the single terminal `return` — refactor multiple returns into one resolved-dict return if that's simpler than recording at each exit. A call that returns without any model call (early cache/short-circuit) should record nothing.

**Also confirm `revise_quests` (planner_llm.py:439).** It runs `_llm_with_search` with `L1_SYSTEM` and is a model site. Check whether the live loop's L1 path (`l1_pipeline.run_l1_pipeline`) calls `revise_quests` or the triage→brainstorm→decide pipeline. If `revise_quests` is on the live path, instrument it as a multi-round site in Task 5 (`layer="l1_revise"`); if it's dead code on the current path, note that and skip.

- [ ] **Step 4: Run to verify it passes + full suite**

Run: `uv run python -m pytest tests/unit/test_capture_planner.py -q && uv run python -m pytest -q`
Expected: PASS; suite green.

- [ ] **Step 5: Commit**

```bash
git add src/pokemon_agent/agent/planner_llm.py tests/unit/test_capture_planner.py
git commit -m "distill: capture single-call L1/L2 sites (triage/decide/repair/waypoint/propose)"
```

---

## Task 5: Instrument the multi-round planner sites

**Files:**
- Modify: `src/pokemon_agent/agent/planner_llm.py` (`_llm_with_search` 369-395; callers `l1_brainstorm` 511, `strategize` 396)
- Test: `tests/unit/test_capture_multiround.py`

Per spec §3: emit **one decision record** (final input → final parsed output, `tokens` = sum across rounds, plus a `search_queries` list), AND emit each intermediate round as a `record_round` sharing a `group_id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_capture_multiround.py
import json
from pokemon_agent.logging.capture import Capture
from pokemon_agent.agent.planner_llm import Planner


class _Knowledge:               # stub KB so _llm_with_search treats a "search" list as a round.
    # Must match the REAL signature: query_texts(text, *, top_k=5, max_chars=1600) — the loop calls
    # it as self.knowledge.query_texts(q, top_k=4), so **kwargs (or the keyword-only params) is required
    # or the call raises TypeError and l1_brainstorm swallows it before recording.
    def query_texts(self, text, **kwargs): return ["(kb result)"]


class _SearchProv:
    def __init__(self): self.n = 0
    def chat_json(self, system, user, image=None):
        self.n += 1
        # A round is ONLY triggered when the parsed "search" is a non-empty LIST and knowledge is set
        # (planner_llm.py:385-386). A string does NOT trigger it.
        if self.n == 1:
            return ('{"search": ["where is Brock"]}', 30, {"total_tokens": 10})   # KB-search round
        return ('{"assessment":"go north","change":true}', 40, {"total_tokens": 20})  # final


def test_brainstorm_emits_one_decision_plus_rounds(tmp_path):
    cap = Capture(tmp_path, mode="distill"); cap.begin_step(9, None)
    p = Planner(goal_map=2, level_target=0, reflector=None,
                provider=_SearchProv(), strategist=_SearchProv(), knowledge=_Knowledge())
    p.capture = cap
    # drive l1_brainstorm (or _llm_with_search directly) so it does >=1 search round + a final
    p.l1_brainstorm(emu=None, context={"player": {"x": 1, "y": 1}})
    cap.flush()
    recs = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    decisions = [r for r in recs if r["layer"] == "l1_brainstorm" and r["kind"] == "model"]
    rounds = [r for r in recs if r["kind"] == "model_round"]
    assert len(decisions) == 1
    assert decisions[0]["tokens"] == 30              # sum of round(10) + final(20)
    assert len(rounds) >= 1                           # the search round WAS emitted
    assert all(r["group_id"] == decisions[0]["group_id"] for r in rounds)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m pytest tests/unit/test_capture_multiround.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

Thread a `layer` name + capture through `_llm_with_search`. Give it an optional `capture_layer: str | None = None`. Inside its round loop, before/after each `provider.chat_json`, accumulate `tokens`, append the round's query to `search_queries`, and `cap.record_round(capture_layer, group_id=gid, input=s, output_raw=content, tokens=tok, latency_ms=..., parsed=<round parse>)`. After the loop, `cap.record(capture_layer, group_id=gid, input=<final state>, output_raw=<final content>, parsed=<final>, tokens=<sum>, extra={"search_queries": search_queries})`. Generate `gid` once per call (e.g. `f"{capture_layer}-{id(state)}-{time.time_ns()}"`). Pass `capture_layer="l1_brainstorm"` / `"l1_strategize"` from the two callers. Guard everything on `cap = getattr(self, "capture", None)`.

- [ ] **Step 4: Run to verify it passes + suite**

Run: `uv run python -m pytest tests/unit/test_capture_multiround.py -q && uv run python -m pytest -q`
Expected: PASS; suite green.

- [ ] **Step 5: Commit**

```bash
git add src/pokemon_agent/agent/planner_llm.py tests/unit/test_capture_multiround.py
git commit -m "distill: capture multi-round L1 sites (brainstorm/strategize) — decision + rounds"
```

---

## Task 6: Instrument the Jev reasoner sites (with confidence)

**Files:**
- Modify: `src/pokemon_agent/agent/typesafe_reasoner.py` (`__init__`; `step` 158, `path_step` 278, `choose_policy` 374, `choose_flow` 334, `choose_npc`)
- Test: `tests/unit/test_capture_jev.py`

Give `TypeSafeReasoner` a `self.capture = None`. Each site already builds `state`, calls `client.system_one`, and reads `ans.choice`/`ans.confidence`/`ans.probabilities`. Record after the answer is read, capturing `confidence` and (in `extra`) `probabilities`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_capture_jev.py
import json
from pokemon_agent.logging.capture import Capture


class _Ans:
    choice = "up"; confidence = 0.83; probabilities = {"up": 0.83, "down": 0.17}
class _Resp:
    answers = {"pol": _Ans()}   # choose_policy reads resp.answers["pol"] (NOT "policy")
class _Client:
    def system_one(self, state, questions): return _Resp()


def test_choose_policy_records_confidence(tmp_path):
    from pokemon_agent.agent.typesafe_reasoner import TypeSafeReasoner
    r = TypeSafeReasoner.__new__(TypeSafeReasoner)
    r.client = _Client(); r.capture = Capture(tmp_path, mode="distill")
    r.capture.begin_step(4, None)
    # call choose_policy with its real kwargs; assert it returns (policy, conf) unchanged
    # AND a jev_policy record with confidence lands. (Match the real signature when writing.)
    r.choose_policy(hp_frac=1.0, level=9, level_target=12, objective="reach Brock")
    r.capture.flush()
    recs = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    rec = [x for x in recs if x["layer"] == "jev_policy"][0]
    assert rec["confidence"] == 0.83
    assert rec["extra"]["probabilities"]["up"] == 0.83
```

*(Adapt `__new__`-based construction to whatever minimal attributes each method touches; some methods reference more `self` state — set only what's needed, or use a real reasoner with a stub client.)*

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m pytest tests/unit/test_capture_jev.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement — the Jev record pattern**

This is **pseudocode, not a copy-paste block** — the answer key, the confidence local, and the presence of a `latency` local differ per method. Adapt at each site:

Layer names align with spec §3 (`jev_flow`, `jev_npc`, `jev_policy`), plus `jev_action`/`jev_path` for the two sites the spec's list doesn't enumerate:
- `step` (158): the executor action Choice — key `"action"`, has `confidence` + `latency` locals; layer **`"jev_action"`** (it is the per-step action chooser, distinct from the dialogue-flow router). **NB the recorded raw `choice` can differ from the executed action** (SAYCAN `option_bias` re-rank ~234, low-conf→wait gate ~249): record `parsed={"choice": ans.choice}` as the decision target, and add `extra={"executed": <the final action kind/label>}` so replay can see both. `step`'s early menu branch (`_menu_step`, return ~201) is a separate site — record it there too or note it's uncaptured.
- `path_step` (278): layer **`"jev_path"`**; use its own answer key + confidence local.
- `choose_policy` (374): key `"pol"`, confidence local is **`conf`** (not `confidence`), and there is **no `latency` local** (pass `latency_ms=0`); layer **`"jev_policy"`**.
- `choose_flow` (334): the dialogue/menu flow router — layer **`"jev_flow"`** (matches spec). Returns a dict of **two** answers (`dialogue`, `menu`) via `resp.answers.get(name)`, each with its own confidence — there is no single `ans`. Record the combined `out` dict (or emit one record per sub-answer).
- `choose_npc` (invoked at `reason_loop.py:1090`): layer **`"jev_npc"`** (spec §3).

General shape (adjust the names):

```python
cap = getattr(self, "capture", None)
if cap is not None:
    cap.record(LAYER, model=getattr(self.client, "model", "typesafe"),
               input=state, output_raw=str(ans.choice), parsed={"choice": ans.choice},
               confidence=conf, latency_ms=lat_or_0,
               extra={"probabilities": getattr(ans, "probabilities", None)})
```

Record just before each method's `return`; never alter the returned value.

- [ ] **Step 4: Run to verify it passes + suite**

Run: `uv run python -m pytest tests/unit/test_capture_jev.py -q && uv run python -m pytest -q`
Expected: PASS; suite green.

- [ ] **Step 5: Commit**

```bash
git add src/pokemon_agent/agent/typesafe_reasoner.py tests/unit/test_capture_jev.py
git commit -m "distill: capture Jev sites (action/path/policy/flow/npc) with calibrated confidence"
```

---

## Task 7: Instrument `battle_move`

**Files:**
- Modify: `src/pokemon_agent/games/pokemon_red/battle_agent.py` (`choose_move` 122)
- Test: `tests/unit/test_capture_battle_move.py`

`choose_move(client, emu, *, type_knowledge=None)` is a free function (no `self`). Add an optional `capture=None` param; record the currently-discarded raw `ans` when provided. The loop passes `self.capture` at the call site.

- [ ] **Step 1: Write the failing test** — stub `client.system_one` returning an `ans` with `choice="1"`, `confidence=0.7`; call `choose_move(client, emu_stub, capture=cap)`; assert a `battle_move` record with `confidence==0.7` and `output_parsed=={"slot":1}`, and the returned `(slot, conf)` is unchanged.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** — after computing `slot`/`confidence`:

```python
if capture is not None:
    capture.record("battle_move", model=getattr(client, "model", "typesafe"),
                   input=state, output_raw=str(ans.choice), parsed={"slot": slot},
                   confidence=float(getattr(ans, "confidence", 0.0) or 0.0))
```

Then update the loop's `choose_move(...)` call site (search `choose_move(` in `reason_loop.py`) to pass `capture=self.capture`.

- [ ] **Step 4: Run to verify it passes + suite.**

- [ ] **Step 5: Commit** — `git commit -m "distill: capture battle_move (recover discarded ans + confidence)"`

---

## Task 8: CLI flag + step-anchored saves

**Files:**
- Modify: `scripts/run_agent.py` (argparse ~73-127; `RunRecorder(...)` 253; `ReasoningLoop(...)` 257)
- Test: covered by Task 10's live smoke (CLI wiring is integration).

- [ ] **Step 1:** Add the flag:

```python
ap.add_argument("--capture", default="off", choices=["off", "decisions", "distill"],
                help="decision capture for distillation: 'decisions' logs every decision "
                     "(cheap); 'distill' also step-anchors a save state each step (disk-heavy)")
```

- [ ] **Step 2:** Wire it. When building the recorder, set `state_every=1` under `distill`:

```python
state_every = 1 if args.capture == "distill" else 0
recorder = RunRecorder(emu, rec_dir, state_every=state_every)
```

And pass `capture_mode=args.capture` into `ReasoningLoop(...)`.

**Wiring caveat (verified):** capture only exists under `--mode reason` — the recorder (line 253) and `ReasoningLoop` (line 256) are built inside `if args.mode == "reason":`. The default `--mode plan-act` builds neither, so `--capture` is a **no-op** there. `--mode reason` also forces `--provider lunaroute` (line ~149), so a fully-offline CLI demo of capture isn't possible; use a real provider (or rely on Task 10 Step 1's in-process mock-provider invariance test, which does not go through the CLI). Add a one-line help note on the flag that it applies only to `--mode reason`.

- [ ] **Step 3:** Manually verify `--capture off` (default) writes NO `decisions.jsonl`, and `--capture decisions` does. Use `--mode reason` (short run, real provider):

```bash
# --demo (or --rom <path>) is required or build_emulator raises SystemExit.
uv run python scripts/run_agent.py --demo --mode reason --steps 5 --goal-map 12    # default: no decisions.jsonl
uv run python scripts/run_agent.py --demo --mode reason --steps 5 --goal-map 12 --capture decisions   # decisions.jsonl appears
```

Expected: first run's record-dir has NO `decisions.jsonl`; second run's does. (If offline, skip this manual step and rely on Task 10.)

- [ ] **Step 4: Commit** — `git commit -m "distill: --capture {off,decisions,distill} flag + step-anchored saves"`

---

## Task 9: Deterministic-layer capture (behind `--capture distill`, YAGNI-gated)

**Files:**
- Modify: `battle_l2.py` (`choose_objective`), `battle_agent.py` (`choose_action`), `reason_loop.py` (`_portal_next`, the servo/BFS move), passing `capture` through.
- Test: `tests/unit/test_capture_deterministic.py`

Per spec §3/§8a: deterministic sites (`battle_l2_objective`, `battle_choose_action`, `portal_next`, `servo_move`) have no distillation payoff yet — capture them **only** when `mode == "distill"` (replay/attribution). These are pure functions; pass `capture` at the loop call sites and record with `kind="deterministic"`. Add a `Capture.record(..., kind="deterministic")` and gate emission so `mode=="decisions"` skips them (add `self.deterministic = mode == "distill"` in Capture; guard these records on `cap.deterministic`).

- [ ] **Step 1: Write the failing test** — call the loop's deterministic dispatch with a distill-mode capture; assert `battle_l2_objective` + `battle_choose_action` records appear with `kind=="deterministic"`; with `mode="decisions"` they do NOT.
- [ ] **Step 2-4:** Implement the `cap.deterministic` gate + the four record sites; run tests + suite.
- [ ] **Step 5: Commit** — `git commit -m "distill: deterministic-layer capture behind --capture distill"`

---

## Task 10: Zero-behavior-change verification + live smoke

**Files:**
- Create: `tests/integration/test_capture_live.py`

- [ ] **Step 1: Behavior-invariance test (no ROM needed).** Run the demo loop twice for N steps — once `--capture off`, once `--capture distill` — with a fixed seed/mock provider, and assert the **sequence of actions** (from each run's `log.jsonl` `action` fields) is identical. This is the core guarantee.

```python
def test_capture_does_not_change_actions(tmp_path):
    # build two identical demo loops (FakeEmulator, mock provider), one with capture off,
    # one distill; step both K times; assert [r["action"] for r in log_off] == [... log_distill]
    ...
```

- [ ] **Step 2:** Run it; expected PASS (identical action sequences).

- [ ] **Step 3: Live ROM-guarded smoke.** `pytest.skip` unless the ROM + an overworld fixture exist. Run ~15 steps with `--capture distill`; assert `decisions.jsonl` is non-empty, has records for at least `l2_propose_target` OR `jev_*`, every record has `step`/`seq`/`layer`/`input`, `outcome.json` exists at run-end, and `states/*_step*.state` anchors were written.

- [ ] **Step 4:** Run the full suite: `uv run python -m pytest -q`. Expected: all green.

- [ ] **Step 5: Commit** — `git commit -m "distill: zero-behavior-change + live capture smoke tests"`

---

## Definition of done (Phase 1)

- `--capture off` (default) leaves runs byte-for-byte unchanged; no `decisions.jsonl`.
- `--capture decisions` writes a well-formed `decisions.jsonl` (one line per model decision, distinct `seq` within a step, exact `input`/`output`/`confidence`/`tokens`/`latency_ms`) + `run_meta.json` + `outcome.json`.
- `--capture distill` additionally step-anchors a save each step and captures deterministic layers.
- The action sequence is provably identical with capture on vs off (Task 10 Step 1).
- Full test suite green.

**Deferred to Phase 2 (noted, not a Phase-1 gap):** the per-step progress vector (`step_progress`) is implemented + unit-tested here but is **joined onto each step's decisions at export time**, not persisted per-step in Phase 1 (spec §4 "attached to that step's decisions" is realized by the export join). `label_run` writes only the episode-level `outcome.json` in Phase 1.

**Follow-on (separate plans, not this one):** Phase 2 `scripts/distill_export.py` (per-layer datasets + filters + the per-step progress join); Phase 3 replay/eval harness (`replay --layer/--model`, behavioral replay from anchors, `--swap layer=model`).
