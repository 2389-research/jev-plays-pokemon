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
