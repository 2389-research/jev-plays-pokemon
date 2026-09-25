"""Distillation-capture (design §3) end-to-end verification.

Step 1 (the core guarantee, no ROM): with capture OFF vs DISTILL the loop produces the
IDENTICAL sequence of actions — capture is side-effect-only.

Step 3 (ROM-guarded live smoke): a short real run under --capture distill writes a
non-empty decisions.jsonl, an outcome.json, and step-anchored save states.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.session import Session
from pokemon_agent.core.models import GoalState
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.logging.run_recorder import RunRecorder
from pokemon_agent.observations.builder import ObservationBuilder
from tests.support.stub_reasoner import StubReasoner as _StubReasoner

ROOT = Path(__file__).resolve().parents[2]
ROM_PATH = ROOT / "roms" / "pokemon_red.gb"


def _run_demo(rec_dir: Path, *, capture_mode: str, steps: int, goal_map=None) -> list:
    emu = FakeEmulator()
    session = Session(GoalState(primary="leave", current="leave"))
    recorder = RunRecorder(emu, rec_dir, state_every=(1 if capture_mode == "distill" else 0))
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=_StubReasoner(), session=session, vision=False,
                         reflect_every=100, recorder=recorder, goal_map=goal_map,
                         capture_mode=capture_mode)
    loop.run(max_steps=steps)
    recorder.close()
    actions = [json.loads(l).get("action") for l in (rec_dir / "log.jsonl").read_text().splitlines()]
    return actions


def test_capture_does_not_change_actions(tmp_path):
    # The core zero-behavior-change guarantee: identical action sequences with capture on vs off.
    off = _run_demo(tmp_path / "off", capture_mode="off", steps=12)
    distill = _run_demo(tmp_path / "distill", capture_mode="distill", steps=12)
    assert off == distill and len(off) == 12
    # off must NOT write a decisions.jsonl; distill's dir exists
    assert not (tmp_path / "off" / "decisions.jsonl").exists()


def test_distill_writes_outcome_at_run_end(tmp_path):
    # The run-end labeler fires under distill and writes a well-formed outcome.json (no ROM needed).
    # (On-disk step anchors need a real emulator — FakeEmulator.save_state is in-memory only — so
    # anchor-writing is asserted in the ROM-guarded live smoke below.)
    _run_demo(tmp_path / "d", capture_mode="distill", steps=8, goal_map=1)
    out = json.loads((tmp_path / "d" / "outcome.json").read_text())
    assert out["steps"] == 8 and "map_arrival_steps" in out and out["goal_map"] == 1


@pytest.mark.skipif(not ROM_PATH.exists(), reason="ROM not present")
def test_live_capture_smoke(tmp_path):
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    # a FREE-ROAM overworld state: after_starter is mid-cutscene (wJoyIgnore set for ~360 frames),
    # where the agent correctly waits for the script instead of navigating (conversation gate)
    overworld = next((ROOT / "states" / f"{n}.state" for n in ("pallet_ready", "after_starter")
                      if (ROOT / "states" / f"{n}.state").exists()), None)
    if overworld is None:
        pytest.skip("no overworld fixture present")

    emu = PyBoyEmulator(str(ROM_PATH), window="null")
    emu.load_state(overworld)
    emu.tick(6)
    session = Session(GoalState(primary="reach Pewter", current="reach Pewter"))
    rec_dir = tmp_path / "live"
    recorder = RunRecorder(emu, rec_dir, state_every=1)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=_StubReasoner(), session=session, vision=False,
                         reflect_every=100, recorder=recorder, goal_map=2, capture_mode="distill")
    loop.run(max_steps=15)
    recorder.close()
    emu.close()

    dj = rec_dir / "decisions.jsonl"
    assert dj.exists() and dj.stat().st_size > 0, "distill run must write decisions.jsonl"
    recs = [json.loads(l) for l in dj.read_text().splitlines()]
    layers = {r["layer"] for r in recs}
    assert any(l == "l2_propose_target" or l.startswith("jev_") or l.startswith("l1_") for l in layers)
    for r in recs:
        assert all(k in r for k in ("step", "seq", "layer", "input"))
    assert (rec_dir / "outcome.json").exists()
    assert list((rec_dir / "states").glob("*_step*.state")), "step anchors must be written"
