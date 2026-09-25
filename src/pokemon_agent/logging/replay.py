"""Replay / eval harness (design §5) — two modes over captured runs:

* **Decision replay** (no emulator): feed each stored ``input`` to a candidate model and diff
  its output against the recorded decision (and against the step's outcome). This is the eval
  harness for a distilled/candidate model — thousands of ground-truthed decision points.
* **Behavioral replay** (from a save-state anchor): reload the emulator state captured for a
  step and run the agent forward with a swapped layer (``--swap layer=model``) to observe the
  downstream effect. The swap installs a candidate that implements the layer's exact call
  interface (design §7 "L2Provider.propose_target(input)->output").

Pure/offline parts (decision replay, key extraction, anchor resolution, swap install) are here
and unit-tested; ``behavioral_replay`` needs a real emulator and is exercised ROM-guarded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .distill_export import progress_score

# layer -> (loop attribute holding the component, method name that layer calls).
# A swapped candidate implements the SAME method (design §7), so we bind it in place.
_LAYER_TARGET = {
    "l2_propose_target": ("planner", "propose_target"),
    "l2_next_waypoint": ("planner", "next_waypoint"),
    "l2_travel_reroute": ("planner", "_choose_target_map"),
    "l1_triage": ("planner", "l1_triage"),
    "l1_brainstorm": ("planner", "l1_brainstorm"),
    "l1_decide": ("planner", "l1_decide"),
    "l1_repair": ("planner", "l1_repair"),
    "jev_action": ("reasoner", "step"),
    "jev_path": ("reasoner", "path_step"),
    "jev_policy": ("reasoner", "choose_policy"),
    "jev_flow": ("reasoner", "choose_flow"),
    "jev_npc": ("reasoner", "choose_npc"),
}

# short CLI aliases (design §5 uses `--swap l2=...`, `l1=...`)
_ALIASES = {"l2": "l2_propose_target", "l1": "l1_decide", "jev": "jev_action"}


def _canonical_layer(name: str) -> str:
    return _ALIASES.get(name, name)


def decision_key(layer: str, parsed):
    """The salient decision from a parsed output, for diffing candidate vs recorded. Layer-aware
    so we compare the meaningful field (the chosen tile / slot / policy), not incidental prose."""
    parsed = parsed or {}
    if not isinstance(parsed, dict):
        return parsed
    if layer.startswith("jev_") or layer == "l1_triage":
        if "choice" in parsed:
            return parsed.get("choice")
        if "change" in parsed:
            return bool(parsed.get("change"))
    if layer == "battle_move":
        return parsed.get("slot")
    if layer == "l2_propose_target":
        return (parsed.get("kind"), parsed.get("x"), parsed.get("y"), parsed.get("map"))
    if layer == "l2_next_waypoint":
        return (parsed.get("x"), parsed.get("y"))
    # default: the whole parsed dict, order-normalized
    return json.dumps(parsed, sort_keys=True, default=str)


@dataclass
class ReplayReport:
    n: int
    agree: int
    n_progress: int
    agree_on_progress: int
    disagreements: list = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        return self.agree / self.n if self.n else 0.0

    @property
    def progress_agreement_rate(self) -> float:
        return self.agree_on_progress / self.n_progress if self.n_progress else 0.0

    def summary(self) -> dict:
        return {"n": self.n, "agree": self.agree, "agreement_rate": round(self.agreement_rate, 4),
                "n_progress": self.n_progress, "agree_on_progress": self.agree_on_progress,
                "progress_agreement_rate": round(self.progress_agreement_rate, 4),
                "disagreements": len(self.disagreements)}


def replay_decisions(rows, predict, *, key=None) -> ReplayReport:
    """Decision replay (design §5): run ``predict(input)->parsed`` over each captured row and
    diff its decision against the recorded one. Also splits agreement by whether the step made
    progress (a candidate that matches the strong model on the DECISIVE steps is what matters)."""
    key = key or decision_key
    agree = n_prog = agree_prog = 0
    disagreements = []
    rows = list(rows)
    for r in rows:
        layer = r.get("layer")
        try:
            pred = predict(r.get("input"))
        except Exception as e:
            pred = {"__error__": str(e)}
        rec_key = key(layer, r.get("output_parsed"))
        pred_key = key(layer, pred)
        ok = rec_key == pred_key
        agree += int(ok)
        if progress_score(r.get("step_outcome") or {}) > 0:
            n_prog += 1
            agree_prog += int(ok)
        if not ok:
            disagreements.append({"step": r.get("step"), "seq": r.get("seq"),
                                  "recorded": rec_key, "candidate": pred_key})
    return ReplayReport(n=len(rows), agree=agree, n_progress=n_prog,
                        agree_on_progress=agree_prog, disagreements=disagreements)


def load_layer_rows(source, layer: str) -> list:
    """Load rows for one layer, from an export dataset (``<layer>.jsonl``) or a record-dir's
    ``decisions.jsonl``. Filters to the requested layer either way."""
    source = Path(source)
    if source.is_dir():
        f = source / "decisions.jsonl"
    elif source.suffix == ".jsonl":
        f = source
    else:
        f = source
    rows = []
    if not f.exists():
        return rows
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("layer") == layer:
            rows.append(r)
    return rows


def resolve_anchor(record_dir, step: int):
    """The step-anchored save state for ``step`` (design §5) — glob by step since the recorder
    names it ``map<M>_step<N>.state`` and the map id isn't known at capture time. None if absent."""
    matches = sorted(Path(record_dir).glob(f"states/*_step{step}.state"))
    return matches[0] if matches else None


def apply_swaps(loop, swaps: dict) -> list:
    """Install candidate layers onto a live loop (design §7 swap-in). ``swaps`` maps a layer name
    (or a short alias: l1/l2/jev) to a candidate — either an object implementing that layer's
    method, or a bare callable. Returns the list of layers actually swapped."""
    done = []
    for name, cand in (swaps or {}).items():
        layer = _canonical_layer(name)
        target = _LAYER_TARGET.get(layer)
        if target is None:
            continue
        comp_name, method = target
        comp = getattr(loop, comp_name, None)
        if comp is None:
            continue
        fn = getattr(cand, method, None) or (cand if callable(cand) else None)
        if fn is None:
            continue
        try:
            setattr(comp, method, fn)
            done.append(layer)
        except Exception:
            pass
    return done


def behavioral_replay(record_dir, from_step: int, *, rom_path, build_loop, swaps=None,
                      steps: int = 20):
    """Behavioral replay (design §5, ROM-guarded): reload the anchor state for ``from_step`` and
    run the agent forward ``steps`` with any ``swaps`` installed, so downstream effects of a
    swapped layer are observable. ``build_loop(emu) -> (loop, recorder)`` wires the agent (kept
    injectable so this stays emulator/agent-agnostic and testable). Returns the new record-dir."""
    anchor = resolve_anchor(record_dir, from_step)
    if anchor is None:
        raise FileNotFoundError(f"no step anchor for step {from_step} under {record_dir}/states")
    from ..emulator.pyboy_adapter import PyBoyEmulator
    emu = PyBoyEmulator(str(rom_path), window="null")
    emu.load_state(anchor)
    emu.tick(6)
    loop, recorder = build_loop(emu)
    if swaps:
        apply_swaps(loop, swaps)
    try:
        loop.run(max_steps=steps)
    finally:
        if recorder is not None:
            recorder.close()
        emu.close()
    return getattr(recorder, "dir", None)
