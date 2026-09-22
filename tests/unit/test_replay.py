import json
from pathlib import Path

from pokemon_agent.logging.replay import (
    apply_swaps, decision_key, load_layer_rows, replay_decisions, resolve_anchor,
)


def test_decision_key_is_layer_aware():
    assert decision_key("jev_policy", {"choice": "dodge-grass"}) == "dodge-grass"
    assert decision_key("battle_move", {"slot": 2}) == 2
    assert decision_key("l2_propose_target", {"kind": "tile", "x": 3, "y": 4, "note": "x"}) == ("tile", 3, 4, None)
    assert decision_key("l2_next_waypoint", {"x": 5, "y": 6, "reason": "z"}) == (5, 6)


def test_replay_decisions_agreement_and_progress_split():
    rows = [
        {"layer": "jev_policy", "input": {"a": 1}, "output_parsed": {"choice": "shortest"},
         "step_outcome": {"map_changed": True}},   # progress step
        {"layer": "jev_policy", "input": {"a": 2}, "output_parsed": {"choice": "farm-exp"},
         "step_outcome": {"map_changed": False}},   # no-progress step
    ]
    # candidate always predicts "shortest": agrees on row 0, disagrees on row 1
    report = replay_decisions(rows, lambda inp: {"choice": "shortest"})
    assert report.n == 2 and report.agree == 1
    assert report.agreement_rate == 0.5
    assert report.n_progress == 1 and report.agree_on_progress == 1   # agreed on the progress step
    assert len(report.disagreements) == 1 and report.disagreements[0]["recorded"] == "farm-exp"


def test_load_layer_rows_from_record_dir(tmp_path):
    d = tmp_path / "run"; d.mkdir()
    (d / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"layer": "jev_policy", "input": {"a": 1}, "output_parsed": {"choice": "shortest"}, "step": 0},
        {"layer": "l2_propose_target", "input": {"b": 2}, "output_parsed": {"kind": "exit"}, "step": 1},
    ]))
    rows = load_layer_rows(d, "jev_policy")
    assert len(rows) == 1 and rows[0]["output_parsed"]["choice"] == "shortest"


def test_resolve_anchor_globs_by_step(tmp_path):
    states = tmp_path / "states"; states.mkdir()
    (states / "map2_step7.state").write_bytes(b"x")
    (states / "map3_step8.state").write_bytes(b"y")
    assert resolve_anchor(tmp_path, 7).name == "map2_step7.state"
    assert resolve_anchor(tmp_path, 99) is None


def test_apply_swaps_installs_candidate_method():
    class _Loop: ...
    class _Planner: ...
    loop = _Loop(); loop.planner = _Planner(); loop.reasoner = _Planner()

    class _Candidate:
        def propose_target(self, emu, context): return {"kind": "tile", "x": 1, "y": 1}

    cand = _Candidate()
    apply_swaps(loop, {"l2_propose_target": cand})
    assert loop.planner.propose_target == cand.propose_target
    # alias form ("l2") resolves to the same target
    apply_swaps(loop, {"l2": cand})
    assert loop.planner.propose_target == cand.propose_target
