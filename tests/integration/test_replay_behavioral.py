"""Behavioral replay (design §5, ROM-guarded): a step-anchored save reloads and the agent
continues forward — with a swapped layer installed — producing a fresh trajectory + capture.

Uses the deterministic stub reasoner from the live-capture test (no network) so the whole
capture -> anchor -> reload -> swap -> run-forward loop is exercised without a real model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.session import Session
from pokemon_agent.core.models import GoalState
from pokemon_agent.logging.replay import apply_swaps, behavioral_replay, resolve_anchor
from pokemon_agent.logging.run_recorder import RunRecorder
from pokemon_agent.observations.builder import ObservationBuilder
from tests.support.stub_reasoner import StubReasoner as _StubReasoner

ROOT = Path(__file__).resolve().parents[2]
ROM_PATH = ROOT / "roms" / "pokemon_red.gb"


def _overworld_state():
    # a FREE-ROAM overworld state first: after_starter is mid-cutscene (wJoyIgnore set for ~360
    # frames), where the agent correctly waits for the script instead of proposing targets
    for n in ("pallet_ready", "after_starter"):
        p = ROOT / "states" / f"{n}.state"
        if p.exists():
            return p
    return None


@pytest.mark.skipif(not ROM_PATH.exists(), reason="ROM not present")
def test_behavioral_replay_from_anchor(tmp_path):
    state = _overworld_state()
    if state is None:
        pytest.skip("no overworld fixture present")
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator

    # 1) capture a short run under distill so step anchors + decisions are written
    emu = PyBoyEmulator(str(ROM_PATH), window="null"); emu.load_state(state); emu.tick(6)
    rec_dir = tmp_path / "orig"
    rec = RunRecorder(emu, rec_dir, state_every=1)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=_StubReasoner(), session=Session(GoalState(primary="p", current="p")),
                         vision=False, reflect_every=100, recorder=rec, goal_map=2, capture_mode="distill")
    loop.run(max_steps=10); rec.close(); emu.close()

    # an anchor for a mid-run step must exist
    assert resolve_anchor(rec_dir, 5) is not None

    # 2) a candidate L2 proposer that always exits — installed via --swap l2=<candidate>
    class _ExitProposer:
        calls = 0
        def propose_target(self, emu, context):
            _ExitProposer.calls += 1
            return {"kind": "exit", "note": "swapped candidate"}

    cand = _ExitProposer()

    def build_loop(emu2):
        r2 = RunRecorder(emu2, tmp_path / "replay", state_every=1)
        lp = ReasoningLoop(builder=ObservationBuilder(emu2), controller=ActionController(emu2),
                           reasoner=_StubReasoner(), session=Session(GoalState(primary="p", current="p")),
                           vision=False, reflect_every=100, recorder=r2, goal_map=2, capture_mode="distill")
        return lp, r2

    # 3) reload the anchor at step 5 and run forward with the swap
    out_dir = behavioral_replay(rec_dir, 5, rom_path=ROM_PATH, build_loop=build_loop,
                                swaps={"l2": cand}, steps=6)
    assert out_dir is not None and (Path(out_dir) / "log.jsonl").exists()
    # the swapped candidate actually drove the replay
    assert _ExitProposer.calls > 0
    # the replay produced its own capture (fresh decisions/outcome)
    assert (Path(out_dir) / "decisions.jsonl").exists()
    assert (Path(out_dir) / "outcome.json").exists()


@pytest.mark.skipif(not ROM_PATH.exists(), reason="ROM not present")
def test_apply_swaps_on_real_loop(tmp_path):
    # apply_swaps binds the candidate's method onto the live planner (no emulator step needed)
    from pokemon_agent.emulator.fake_emulator import FakeEmulator
    emu = FakeEmulator()
    rec = RunRecorder(emu, tmp_path / "r", state_every=0)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=_StubReasoner(), session=Session(GoalState(primary="p", current="p")),
                         vision=False, reflect_every=100, recorder=rec, goal_map=2)
    class _Cand:
        def propose_target(self, emu, context): return {"kind": "exit"}
    cand = _Cand()
    done = apply_swaps(loop, {"l2_propose_target": cand})
    rec.close()
    assert done == ["l2_propose_target"]
    assert loop.planner.propose_target == cand.propose_target
