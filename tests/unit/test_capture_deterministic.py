import json
from pokemon_agent.logging.capture import Capture
from pokemon_agent.agent import reason_loop as RL


def _loop_with_capture(cap):
    loop = RL.ReasoningLoop.__new__(RL.ReasoningLoop)   # bypass heavy __init__
    loop.capture = cap
    return loop


def test_deterministic_records_only_under_distill(tmp_path):
    # distill mode: deterministic-layer records ARE emitted with kind="deterministic"
    cap = Capture(tmp_path, mode="distill"); cap.begin_step(1, None)
    loop = _loop_with_capture(cap)
    loop._cap_det("battle_l2_objective", {"state": {"hp": 1}}, {"objective": "GRIND-EXP"})
    loop._cap_det("battle_choose_action", {"objective": "GRIND-EXP"}, {"kind": "move"})
    cap.flush()
    recs = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    layers = {r["layer"] for r in recs}
    assert {"battle_l2_objective", "battle_choose_action"} <= layers
    assert all(r["kind"] == "deterministic" for r in recs)


def test_deterministic_skipped_under_decisions(tmp_path):
    # decisions mode: deterministic-layer records are NOT emitted (YAGNI gate)
    cap = Capture(tmp_path, mode="decisions"); cap.begin_step(1, None)
    loop = _loop_with_capture(cap)
    loop._cap_det("battle_l2_objective", {"state": {"hp": 1}}, {"objective": "GRIND-EXP"})
    loop._cap_det("battle_choose_action", {"objective": "GRIND-EXP"}, {"kind": "move"})
    cap.flush()
    # file may exist but must contain no deterministic records
    p = tmp_path / "decisions.jsonl"
    recs = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    assert not any(r.get("kind") == "deterministic" for r in recs)
