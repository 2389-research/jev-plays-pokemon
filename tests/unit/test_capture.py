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
