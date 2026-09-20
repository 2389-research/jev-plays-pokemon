from pokemon_agent.agent.l1_pipeline import validate_step


def test_travel_step_valid():
    ok, err = validate_step({"kind": "travel", "map": 2, "done_when": "on_map"})
    assert ok and err is None


def test_action_missing_criterion_invalid():
    ok, err = validate_step({"kind": "action", "map": 40})
    assert not ok and "done_when" in err


def test_action_on_map_invalid():
    ok, err = validate_step({"kind": "action", "map": 40, "done_when": "on_map"})
    assert not ok


def test_action_unparseable_invalid():
    ok, err = validate_step({"kind": "action", "map": 40, "done_when": "when i feel like it"})
    assert not ok


def test_action_real_criterion_valid():
    ok, err = validate_step({"kind": "action", "map": 40, "done_when": "no_item:oaks_parcel"})
    assert ok


def test_travel_with_talk_invalid():   # contradictory: travel = nothing happens on arrival
    ok, err = validate_step({"kind": "travel", "map": 2, "talk": True})
    assert not ok


def test_missing_map_invalid():
    ok, err = validate_step({"kind": "action", "done_when": "hp_frac>=1.0"})
    assert not ok
