import json
from pathlib import Path

from pokemon_agent.logging.distill_export import (
    SCHEMA_VERSION, export, iter_rows, load_run, progress_score,
)


def _write_run(d: Path, *, decisions, log, outcome):
    d.mkdir(parents=True, exist_ok=True)
    (d / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in decisions))
    (d / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log))
    (d / "outcome.json").write_text(json.dumps(outcome))
    return d


def _run_a(base: Path):
    # a SUCCESSFUL run: map advanced + a catch; 2 model decisions across 2 steps
    return _write_run(
        base / "runA",
        decisions=[
            {"run_id": "runA", "step": 0, "seq": 0, "layer": "l2_propose_target", "kind": "model",
             "model": "glm", "input": {"a": 1}, "output_parsed": {"kind": "tile", "x": 3},
             "confidence": None, "tokens": 20, "latency_ms": 40, "anchor": "states/*_step0.state"},
            {"run_id": "runA", "step": 1, "seq": 0, "layer": "jev_flow", "kind": "model",
             "model": "tsafe", "input": {"b": 2}, "output_parsed": {"choice": "up"},
             "confidence": 0.9, "tokens": 0, "latency_ms": 5, "anchor": "states/*_step1.state"},
        ],
        log=[
            {"step": 0, "map_id": 1, "party": ["Squirtle L5 18/18"], "items": [], "events": []},
            {"step": 1, "map_id": 2, "party": ["Squirtle L5 18/18", "Pidgey L3 10/10"],
             "items": [], "events": []},   # map advanced 1->2 AND party grew => progress
        ],
        outcome={"reached_goal_map": True, "goal_map": 2, "caught_count": 1, "steps": 2},
    )


def _run_b(base: Path):
    # a FAILED run (never reached goal), one low-progress decision
    return _write_run(
        base / "runB",
        decisions=[
            {"run_id": "runB", "step": 0, "seq": 0, "layer": "l2_propose_target", "kind": "model",
             "model": "glm", "input": {"a": 1}, "output_parsed": {"kind": "exit"},
             "confidence": None, "tokens": 10, "latency_ms": 30, "anchor": None},
        ],
        log=[{"step": 0, "map_id": 1, "party": ["Squirtle L5 18/18"], "items": [], "events": []}],
        outcome={"reached_goal_map": False, "goal_map": 2, "caught_count": 0, "steps": 1},
    )


def test_progress_score():
    assert progress_score({"map_changed": True, "level_delta": 0, "items_delta": 0, "caught": False}) == 1
    assert progress_score({"map_changed": True, "level_delta": 2, "items_delta": 1, "caught": True}) == 5
    assert progress_score({"map_changed": False, "level_delta": 0, "items_delta": 0, "caught": False}) == 0


def test_load_run_joins_step_outcome(tmp_path):
    run = load_run(_run_a(tmp_path))
    assert run["outcome"]["reached_goal_map"] is True
    # step 1's progress vector reflects the map change + catch
    assert run["step_outcomes"][1]["map_changed"] is True
    assert run["step_outcomes"][1]["caught"] is True


def test_iter_rows_layer_filter_and_join(tmp_path):
    _run_a(tmp_path)
    rows = list(iter_rows([tmp_path / "runA"], layers=["l2_propose_target"]))
    assert len(rows) == 1
    r = rows[0]
    assert r["layer"] == "l2_propose_target" and r["schema_version"] == SCHEMA_VERSION
    assert r["input"] == {"a": 1} and r["output_parsed"] == {"kind": "tile", "x": 3}
    assert r["episode_outcome"]["reached_goal_map"] is True
    assert "step_outcome" in r


def test_only_successful_episodes(tmp_path):
    _run_a(tmp_path); _run_b(tmp_path)
    rows = list(iter_rows([tmp_path / "runA", tmp_path / "runB"], only_successful=True))
    assert {r["run_id"] for r in rows} == {"runA"}   # runB dropped (failed episode)


def test_min_progress_filter(tmp_path):
    _run_a(tmp_path)
    # step 0 has no prior -> low progress; step 1 advanced the map + caught -> high progress
    rows = list(iter_rows([tmp_path / "runA"], min_progress=1))
    assert all(progress_score(r["step_outcome"]) >= 1 for r in rows)
    assert any(r["step"] == 1 for r in rows) and not any(r["step"] == 0 for r in rows)


def test_dedup(tmp_path):
    # two runs with an identical (layer, input) l2 row -> dedup keeps one
    _run_a(tmp_path); _run_b(tmp_path)
    rows = list(iter_rows([tmp_path / "runA", tmp_path / "runB"],
                          layers=["l2_propose_target"], dedup=True))
    inputs = [json.dumps(r["input"], sort_keys=True) for r in rows]
    assert len(inputs) == len(set(inputs))   # no duplicate inputs


def test_export_writes_per_layer_files(tmp_path):
    _run_a(tmp_path)
    out = tmp_path / "out"
    counts = export([tmp_path / "runA"], out)
    assert counts == {"l2_propose_target": 1, "jev_flow": 1}
    assert (out / "l2_propose_target.jsonl").exists()
    lines = (out / "jev_flow.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["output_parsed"] == {"choice": "up"}
