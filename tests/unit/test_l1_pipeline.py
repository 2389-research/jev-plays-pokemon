from pokemon_agent.agent.l1_pipeline import validate_step, run_l1_pipeline


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


class StubPlanner:
    """Canned Planner stand-in for run_l1_pipeline orchestration tests. Records which
    methods were called so tests can assert on call sequencing/short-circuiting."""

    def __init__(self, *, triage=None, brainstorm=None, decide=None, repair=None):
        self._triage = triage
        self._brainstorm = brainstorm
        self._decide = decide
        self._repair = repair
        self.calls = []

    def l1_triage(self, context):
        self.calls.append("triage")
        return self._triage

    def l1_brainstorm(self, emu, context):
        self.calls.append("brainstorm")
        return self._brainstorm

    def l1_decide(self, context, brainstorm):
        self.calls.append("decide")
        return self._decide

    def l1_repair(self, context, bad_step, error):
        self.calls.append("repair")
        return self._repair


def test_no_change_triage_short_circuits():
    planner = StubPlanner(triage={"change": False, "why": "nothing new"})
    result = run_l1_pipeline(None, {}, planner, hard_event=False)
    assert result is None
    assert planner.calls == ["triage"]


def test_hard_event_skips_triage():
    planner = StubPlanner(
        brainstorm={"assessment": "need to heal"},
        decide={"add": [{"kind": "action", "map": 41, "done_when": "hp_frac>=1.0"}],
                "remove": [], "mission": "m", "milestone": "ms", "assessment": "a"},
    )
    result = run_l1_pipeline(None, {}, planner, hard_event=True)
    assert "triage" not in planner.calls
    assert planner.calls == ["brainstorm", "decide"]
    assert result is not None
    assert len(result["add"]) == 1


def test_empty_decide_is_treated_as_no_change():
    # decide adds/removes nothing -> the plan is unchanged; the pipeline returns None (keep going,
    # continue the active step) rather than a churny no-op proposal that restates the milestone.
    planner = StubPlanner(
        brainstorm={"assessment": "still travelling"},
        decide={"add": [], "remove": [], "mission": "m", "milestone": "ms", "assessment": "a"},
    )
    assert run_l1_pipeline(None, {}, planner, hard_event=True) is None


def test_triage_change_true_runs_brainstorm_and_decide():
    planner = StubPlanner(
        triage={"change": True, "why": "signal fired"},
        brainstorm={"assessment": "assess"},
        decide={"add": [{"kind": "travel", "map": 2, "done_when": "on_map"}],
                "remove": [], "mission": "m", "milestone": "ms", "assessment": "a"},
    )
    result = run_l1_pipeline(None, {}, planner, hard_event=False)
    assert planner.calls == ["triage", "brainstorm", "decide"]
    assert result is not None


def test_bad_step_gets_repaired_and_fixed_step_used():
    bad_step = {"kind": "action", "map": 40, "done_when": "on_map"}
    fixed_step = {"kind": "action", "map": 40, "done_when": "no_item:oaks_parcel"}
    planner = StubPlanner(
        triage={"change": True, "why": "x"},
        brainstorm={"assessment": "a"},
        decide={"add": [bad_step], "remove": [], "mission": "m", "milestone": "ms", "assessment": "a"},
        repair=fixed_step,
    )
    result = run_l1_pipeline(None, {}, planner, hard_event=False)
    assert planner.calls == ["triage", "brainstorm", "decide", "repair"]
    assert result is not None
    assert result["add"] == [fixed_step]


def test_bad_step_repair_still_bad_returns_none_and_traces():
    bad_step = {"kind": "action", "map": 40, "done_when": "on_map"}
    still_bad_step = {"kind": "action", "map": 40, "done_when": "on_map"}
    planner = StubPlanner(
        triage={"change": True, "why": "x"},
        brainstorm={"assessment": "a"},
        decide={"add": [bad_step], "remove": [], "mission": "m", "milestone": "ms", "assessment": "a"},
        repair=still_bad_step,
    )
    traces = []
    result = run_l1_pipeline(None, {}, planner, hard_event=False, on_trace=traces.append)
    assert result is None
    invalid_events = [t for t in traces if t.get("stage") == "invalid_criterion"]
    assert len(invalid_events) == 1
    assert invalid_events[0]["step"] == bad_step


def test_valid_heal_step_flows_through_unchanged():
    heal_step = {"kind": "action", "map": 41, "done_when": "hp_frac>=1.0"}
    planner = StubPlanner(
        triage={"change": True, "why": "x"},
        brainstorm={"assessment": "a"},
        decide={"add": [heal_step], "remove": [], "mission": "m", "milestone": "ms", "assessment": "a"},
    )
    result = run_l1_pipeline(None, {}, planner, hard_event=False)
    assert result is not None
    assert result["add"] == [heal_step]
    assert "repair" not in planner.calls
