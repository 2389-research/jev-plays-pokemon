"""RunRecorder writes per-step JSONL + shots + new-area save states."""
import json
from pathlib import Path

from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.logging.run_recorder import RunRecorder
from pokemon_agent.observations.builder import ObservationBuilder


def test_records_steps_shots_and_new_area_states(tmp_path):
    emu = FakeEmulator(map_id=40)
    rec = RunRecorder(emu, tmp_path / "run", shot_every=1)
    b = ObservationBuilder(emu)
    obs, _ = b.build()
    from pokemon_agent.core.models import WaitAction, ActionResult, GameMode
    res = ActionResult(success=True, result="completed", mode_before=GameMode.OVERWORLD,
                       mode_after=GameMode.OVERWORLD)
    rec.on_event("directive", {"step": 0, "intent": "travel", "reason": "go"})
    rec.record(step=0, obs=obs, action=WaitAction(frames=1), result=res)
    emu.map_id = 41  # new area -> should trigger a save state
    obs2, _ = b.build()
    rec.record(step=1, obs=obs2, action=WaitAction(frames=1), result=res)
    rec.close()

    lines = (tmp_path / "run" / "log.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    r0 = json.loads(lines[0])
    assert r0["step"] == 0 and r0["shot"] == "shots/000000.png"
    assert r0["events"][0]["kind"] == "directive"          # event attached to the step
    assert (tmp_path / "run" / r0["shot"]).exists()         # screenshot written
    assert r0["state"] is not None                          # first map -> state saved
    assert json.loads(lines[1])["state"] is not None        # new map (41) -> another state
    assert (tmp_path / "run" / "viewer.html").exists()      # viewer shipped into the run dir
