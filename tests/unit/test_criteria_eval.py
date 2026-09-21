"""Offline unit tests for the Layer 2.5 criteria-quality GRADER (scripts/eval_criteria.py).

These exercise `grade_case` on CANNED `add` steps only — no network, no LLM. The live
scorecard in `scripts/eval_criteria.py:main()` is exercised manually (it spends credits) and
is out of scope for CI.
"""
from scripts.eval_criteria import grade_case
from tests.fixtures.criteria_cases import CASES


def _case(name: str) -> dict:
    return next(c for c in CASES if c["name"] == name)


def test_fixture_cases_cover_the_required_situations():
    assert len(CASES) >= 5
    names = {c["name"] for c in CASES}
    for expected in ("deliver_parcel", "pickup_parcel", "heal_low_hp", "reach_pewter",
                      "grind_underleveled"):
        assert expected in names
    for c in CASES:
        assert "context" in c and isinstance(c["context"], dict)
        assert c.get("expect_exact") or c.get("expect_family")


def test_deliver_step_grades_hard_true_and_semantic_true():
    step = {"kind": "action", "map": 0, "done_when": "no_item:oaks_parcel"}
    grade = grade_case(_case("deliver_parcel"), [step])
    assert grade["hard"] is True
    assert grade["semantic"] is True


def test_pickup_step_matches_exact_case_insensitively():
    step = {"kind": "action", "map": 5, "done_when": " Has_Item:oaks_parcel "}
    grade = grade_case(_case("pickup_parcel"), [step])
    assert grade["hard"] is True
    assert grade["semantic"] is True


def test_heal_step_grades_semantic_true_by_family():
    step = {"kind": "action", "map": 41, "done_when": "hp_frac>=1.0"}
    grade = grade_case(_case("heal_low_hp"), [step])
    assert grade["hard"] is True
    assert grade["semantic"] is True


def test_travel_step_satisfies_on_map_family():
    step = {"kind": "travel", "map": 2, "done_when": "on_map"}
    grade = grade_case(_case("reach_pewter"), [step])
    assert grade["hard"] is True
    assert grade["semantic"] is True


def test_hard_rule_violation_grades_hard_false():
    step = {"kind": "action", "map": 0, "done_when": "on_map"}
    grade = grade_case(_case("deliver_parcel"), [step])
    assert grade["hard"] is False


def test_wrong_family_step_grades_semantic_false():
    # a level-up step doesn't satisfy the heal case's expected hp_frac family
    step = {"kind": "action", "map": 30, "done_when": "level>=12"}
    grade = grade_case(_case("heal_low_hp"), [step])
    assert grade["hard"] is True
    assert grade["semantic"] is False


def test_no_add_steps_semantic_false_hard_vacuously_true():
    grade = grade_case(_case("grind_underleveled"), [])
    assert grade["hard"] is True
    assert grade["semantic"] is False
