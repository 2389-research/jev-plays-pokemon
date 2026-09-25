"""Canned-output tests for scripts/eval_l1_placement.judge (pure) — review issues 2 and 3."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "eval_l1_placement", Path(__file__).resolve().parents[2] / "scripts" / "eval_l1_placement.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

CTX = {"plan": [
    {"id": "q1", "map": 0, "kind": "travel", "talk": False, "done_when": "on_map", "status": "active"},
    {"id": "q2", "map": 40, "kind": "action", "talk": True, "done_when": "no_item:Oaks Parcel", "status": "pending"},
]}


def _t(map, **kw):
    return {"kind": "travel", "map": map, "talk": False, "who": None, "done_when": "on_map", "why": "", **kw}


def test_any_off_path_step_before_the_delivery_is_misordered():
    for north_map in (1, 13, 51, 2):     # Viridian, Route 2, Viridian Forest, Pewter
        ok, why = E.judge("held-out", CTX, {"add": [_t(north_map)], "remove": []})
        assert not ok and why.startswith("MISORDERED"), (north_map, why)


def test_on_path_or_heal_steps_before_the_delivery_are_fine():
    heal = {"kind": "action", "map": 41, "talk": True, "who": "Nurse", "done_when": "hp_frac>=1.0", "why": ""}
    ok, why = E.judge("held-out", CTX, {"add": [_t(12), heal], "remove": []})
    assert why.startswith("DEFERRED"), why


def test_north_steps_after_the_delivery_are_anchored():
    ok, why = E.judge("held-out", CTX, {"add": [_t(2, after="q2"), _t(51, after="q2")], "remove": []})
    assert ok and why.startswith("ANCHORED"), why


def test_removed_delivery_is_its_own_outcome_not_a_crash():
    ok, why = E.judge("held-out", CTX, {"add": [_t(2)], "remove": ["q2"]})
    assert not ok and why.startswith("DELIVERY_REMOVED"), why


def test_an_empty_fallback_is_a_failed_call_not_a_decision():
    assert E.call_failed({"add": [], "remove": []})
    assert not E.call_failed({"assessment": "plan is fine", "add": [], "remove": []})
    assert not E.call_failed({"add": [_t(2)], "remove": []})
