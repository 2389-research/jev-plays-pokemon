"""Distillation export (design §7) — turn captured record-dirs into per-layer supervised
datasets. A PURE post-processor: reads each run's `decisions.jsonl` + `log.jsonl` +
`outcome.json`, joins every decision to its step's progress vector and its episode outcome,
applies quality filters, and writes one JSONL per layer.

Row schema (versioned via ``SCHEMA_VERSION``):
    { schema_version, run_id, step, seq, layer, kind, model,
      input, output_parsed, confidence, tokens, latency_ms, anchor,
      step_outcome,        # the per-step progress vector (design §4)
      episode_outcome }    # the run's outcome.json (joined by run_id)

CLI: ``scripts/distill_export.py`` wraps ``export`` with --layer/--only-successful-episodes/
--min-progress/--dedup.
"""
from __future__ import annotations

import json
from pathlib import Path

from .outcome import iter_rows as _iter_log_rows
from .outcome import step_progress

SCHEMA_VERSION = 1


def progress_score(vector: dict) -> int:
    """A coarse scalar progress signal for the `--min-progress` filter (design §4): +1 per
    map change, per positive level/item delta, and per catch. 0 = a step that made no visible
    forward progress."""
    v = vector or {}
    return (int(bool(v.get("map_changed")))
            + max(0, int(v.get("level_delta") or 0))
            + max(0, int(v.get("items_delta") or 0))
            + int(bool(v.get("caught"))))


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_run(record_dir: str | Path) -> dict:
    """Load a finished record-dir into {decisions, log_by_step, step_outcomes, outcome}.

    ``step_outcomes[step]`` is the per-step progress vector (``step_progress`` of the previous
    log row -> this one), so any decision can be joined to what its step achieved."""
    record_dir = Path(record_dir)
    decisions = list(_iter_decisions(record_dir))
    log_rows = list(_iter_log_rows(record_dir))
    log_by_step = {r.get("step"): r for r in log_rows}
    step_outcomes: dict = {}
    prev = None
    for r in log_rows:
        step_outcomes[r.get("step")] = step_progress(prev, r)
        prev = r
    outcome = _read_json(record_dir / "outcome.json", {})
    return {"decisions": decisions, "log_by_step": log_by_step,
            "step_outcomes": step_outcomes, "outcome": outcome}


def _iter_decisions(record_dir: Path):
    p = Path(record_dir) / "decisions.jsonl"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except Exception:
                continue


def _episode_ok(outcome: dict) -> bool:
    return bool(outcome.get("reached_goal_map"))


def iter_rows(record_dirs, *, layers=None, only_successful=False,
              min_progress=None, dedup=False, include_rounds=False):
    """Yield export rows across ``record_dirs`` (design §7), each decision joined to its step
    progress vector + episode outcome, after applying the quality filters."""
    layer_set = set(layers) if layers else None
    seen: set = set()
    for rd in record_dirs:
        run = load_run(rd)
        outcome = run["outcome"]
        if only_successful and not _episode_ok(outcome):
            continue
        for dec in run["decisions"]:
            layer = dec.get("layer")
            if layer_set is not None and layer not in layer_set:
                continue
            # the intermediate KB-search rounds are optional detail, not distillation examples
            if not include_rounds and dec.get("kind") == "model_round":
                continue
            step_outcome = run["step_outcomes"].get(dec.get("step"), {})
            if min_progress is not None and progress_score(step_outcome) < min_progress:
                continue
            if dedup:
                key = (layer, json.dumps(dec.get("input"), sort_keys=True, default=str))
                if key in seen:
                    continue
                seen.add(key)
            yield {
                "schema_version": SCHEMA_VERSION,
                "run_id": dec.get("run_id"),
                "step": dec.get("step"),
                "seq": dec.get("seq"),
                "layer": layer,
                "kind": dec.get("kind"),
                "model": dec.get("model"),
                "input": dec.get("input"),
                "output_parsed": dec.get("output_parsed"),
                "confidence": dec.get("confidence"),
                "tokens": dec.get("tokens", 0),
                "latency_ms": dec.get("latency_ms", 0),
                "anchor": dec.get("anchor"),
                "step_outcome": step_outcome,
                "episode_outcome": outcome,
            }


def export(record_dirs, out_dir: str | Path, *, layers=None, only_successful=False,
           min_progress=None, dedup=False, include_rounds=False) -> dict:
    """Write one ``<layer>.jsonl`` per layer under ``out_dir``; return {layer: row_count}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    handles: dict = {}
    counts: dict = {}
    try:
        for row in iter_rows(record_dirs, layers=layers, only_successful=only_successful,
                             min_progress=min_progress, dedup=dedup, include_rounds=include_rounds):
            layer = row["layer"]
            if layer not in handles:
                handles[layer] = (out_dir / f"{layer}.jsonl").open("w", encoding="utf-8")
                counts[layer] = 0
            handles[layer].write(json.dumps(row) + "\n")
            counts[layer] += 1
    finally:
        for h in handles.values():
            h.close()
    return counts
