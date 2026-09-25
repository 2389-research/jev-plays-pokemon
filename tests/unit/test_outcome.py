import json
from pokemon_agent.logging.outcome import step_progress, label_run


def test_step_progress_deltas():
    prev = {"map_id": 1, "party": ["Squirtle L9 20/27"], "items": ["Poke Ball"]}
    cur = {"map_id": 2, "party": ["Squirtle L10 27/27"], "items": ["Poke Ball", "Potion"]}
    p = step_progress(prev, cur)
    assert p["map_changed"] is True
    assert p["level_delta"] == 1
    assert p["hp_delta"] == 7
    assert p["items_delta"] == 1


def test_label_run_writes_outcome(tmp_path):
    # Rows mirror the REAL recorder schema: party/items/map_id/step are top-level; a catch is a
    # party-length increase (there is no synthetic "caught" event in log.jsonl).
    log = tmp_path / "log.jsonl"
    rows = [
        {"step": 0, "map_id": 1, "party": ["Squirtle L5 18/18"], "items": [], "events": []},
        {"step": 1, "map_id": 1, "party": ["Squirtle L6 20/20"], "items": ["Potion"], "events": []},
        {"step": 2, "map_id": 2, "party": ["Squirtle L6 20/20", "Pidgey L4 12/12"],
         "items": ["Potion"], "events": []},   # party grew 1 -> 2  => a catch
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows))
    out = label_run(tmp_path, goal_map=2)
    assert out["reached_goal_map"] is True
    assert out["caught_count"] == 1            # detected from party growth
    assert out["steps"] == 3
    assert (tmp_path / "outcome.json").exists()
    # per-map arrival step recorded
    assert out["map_arrival_steps"]["2"] == 2
