"""L1 step placement (spec docs/superpowers/specs/2026-09-22-l1-step-placement-design.md).

Incident (runs/fix-accept-20260922-2259): at step 25 L1 added "go to Pewter City — after delivering
the parcel" + the gym fight, but the reconciler inserted every add right after the ACTIVE step, ahead
of the still-pending delivery: [q1, q3, q4, q2]. The agent reached Pallet, then turned north with the
parcel still in the bag. An optional `after` anchor lets L1 place a later step after a pending one;
with no anchor, placement is exactly today's.
"""
from __future__ import annotations

from pokemon_agent.agent.quest_reconciler import QuestStep, reconcile_quests


def _mk(id, map, status="pending", dw="on_map", kind="travel"):
    return QuestStep(id=id, map=map, done_when=dw, status=status, kind=kind)


def _ids(start=10):
    n = [start]

    def nxt():
        n[0] += 1
        return f"q{n[0]}"
    return nxt


def _add(map, dw="on_map", kind="travel", **kw):
    return {"map": map, "done_when": dw, "kind": kind, "talk": False, "why": "", **kw}


def _order(steps):
    return [(s.id, s.map) for s in steps]


# the incident plan (captured step-25 DECIDE input): travel to Pallet active, delivery pending
def _incident_plan():
    return [_mk("q1", 0, "active"),
            _mk("q2", 40, dw="no_item:Oaks Parcel", kind="action")]


def test_no_anchor_is_identical_to_today():
    cur = _incident_plan()
    out = reconcile_quests(cur, {"add": [_add(2)], "remove": []}, next_id=_ids())
    assert _order(out) == [("q1", 0), ("q11", 2), ("q2", 40)]   # new step right after the active one


def test_incident_later_steps_anchored_after_the_pending_delivery():
    cur = _incident_plan()
    prop = {"add": [_add(2, after="q2"), _add(54, dw="badges>=1", kind="action", after="q2")], "remove": []}
    out = reconcile_quests(cur, prop, next_id=_ids())
    assert _order(out) == [("q1", 0), ("q2", 40), ("q11", 2), ("q12", 54)]


def test_end_appends_at_the_tail():
    cur = [_mk("q1", 0, "active"), _mk("q2", 1), _mk("q3", 2)]
    out = reconcile_quests(cur, {"add": [_add(54, dw="badges>=1", kind="action", after="end")], "remove": []},
                           next_id=_ids())
    assert [s.id for s in out] == ["q1", "q2", "q3", "q11"]


def test_anchor_to_the_active_step_means_next_in_emitted_order():
    cur = [_mk("q1", 0, "active"), _mk("q2", 1)]
    prop = {"add": [_add(5), _add(6, after="q1")], "remove": []}
    out = reconcile_quests(cur, prop, next_id=_ids())
    assert [s.id for s in out] == ["q1", "q11", "q12", "q2"]


def test_mixed_default_anchor_and_end():
    cur = [_mk("q1", 0, "active"), _mk("q2", 1), _mk("q3", 2)]
    prop = {"add": [_add(5), _add(6, after="q3"), _add(7, after="end")], "remove": []}
    out = reconcile_quests(cur, prop, next_id=_ids())
    assert [s.id for s in out] == ["q1", "q11", "q2", "q3", "q12", "q13"]


def test_invalid_anchors_fall_back_to_default_with_an_event_each():
    cur = [_mk("q0", 9, "done"), _mk("q1", 0, "active"), _mk("q2", 1), _mk("q3", 2, "wedged"), _mk("q4", 3)]
    events = []
    prop = {"add": [_add(20, after="q99"),      # unknown
                    _add(21, after="q3"),       # wedged (always dropped)
                    _add(22, after="q0"),       # done
                    _add(23, after="q4"),       # removed in this proposal
                    _add(24, after=7),          # not a string
                    _add(25, after="q11")],     # a guessed id of a NEW step (ids are assigned after the model responds)
            "remove": ["q4"]}
    out = reconcile_quests(cur, prop, next_id=_ids(), on_event=lambda k, p: events.append((k, p)))
    assert [s.id for s in out] == ["q0", "q1", "q11", "q12", "q13", "q14", "q15", "q16", "q2"]
    reasons = [p["reason"] for k, p in events if k == "l1_anchor_fallback"]
    assert reasons == ["unknown", "wedged", "done", "removed", "not_string", "unknown"]


def test_no_active_step_default_add_goes_first_among_live_steps():
    """The step-138 shape: the active step wedged, L1 replaces it (unanchored) -> it goes first."""
    cur = [_mk("q5", 0, "wedged"), _mk("q3", 2), _mk("q4", 54, dw="badges>=1", kind="action")]
    out = reconcile_quests(cur, {"add": [_add(0)], "remove": ["q5"]}, next_id=_ids())
    assert [s.id for s in out] == ["q11", "q3", "q4"]


def test_anchored_add_that_dedups_is_dropped_quietly():
    cur = _incident_plan()
    events = []
    out = reconcile_quests(cur, {"add": [_add(0, after="q2")], "remove": []}, next_id=_ids(),
                           on_event=lambda k, p: events.append(k))
    assert [s.id for s in out] == ["q1", "q2"] and events == ["l1_add_deduped"]   # no anchor fallback


def test_anchored_steps_compile_behind_the_delivery():
    """Integration: the executive's queue (compile order) delivers the parcel before heading north."""
    from pokemon_agent.agent.plan import Intent
    from pokemon_agent.agent.quest_reconciler import compile_steps_to_directives
    cur = [_mk("q1", 0, "active"),
           QuestStep(id="q2", map=40, talk=True, who="Oak", done_when="no_item:Oaks Parcel", kind="action")]
    prop = {"add": [_add(2, after="q2"), _add(54, dw="badges>=1", kind="action", after="q2")], "remove": []}
    out = reconcile_quests(cur, prop, next_id=_ids())
    directives = compile_steps_to_directives([s for s in out if s.status != "done"])
    qids = [d.quest_id for d in directives]
    first_talk = next(i for i, d in enumerate(directives) if d.intent == Intent.TALK_TO)
    assert qids.index("q2") < qids.index("q11") < qids.index("q12")
    assert directives[first_talk].quest_id == "q2"


# ---- implementation-review follow-ups ---------------------------------------------------------
def test_anchored_add_emitted_before_a_default_add():
    cur = [_mk("q1", 0, "active"), _mk("q2", 1), _mk("q3", 2)]
    prop = {"add": [_add(6, after="q2"), _add(5)], "remove": []}
    out = reconcile_quests(cur, prop, next_id=_ids())
    assert [s.id for s in out] == ["q1", "q12", "q2", "q11", "q3"]


def test_interleaved_anchors_keep_per_anchor_order():
    cur = [_mk("q1", 0, "active"), _mk("q2", 1), _mk("q3", 2)]
    prop = {"add": [_add(5, after="q2"), _add(6, after="q3"), _add(7, after="q2")], "remove": []}
    out = reconcile_quests(cur, prop, next_id=_ids())
    assert [s.id for s in out] == ["q1", "q2", "q11", "q13", "q3", "q12"]


def test_end_emitted_before_an_add_anchored_on_the_last_step():
    cur = [_mk("q1", 0, "active"), _mk("q2", 1), _mk("q3", 2)]
    prop = {"add": [_add(7, after="end"), _add(6, after="q3")], "remove": []}
    out = reconcile_quests(cur, prop, next_id=_ids())
    assert [s.id for s in out] == ["q1", "q2", "q3", "q12", "q11"]


def test_invalid_anchor_without_an_event_sink_still_falls_back():
    cur = _incident_plan()
    out = reconcile_quests(cur, {"add": [_add(2, after="q99")], "remove": []}, next_id=_ids())
    assert [s.id for s in out] == ["q1", "q11", "q2"]


def test_duplicate_id_anchor_never_lands_among_done_steps():
    """A done step and a pending step sharing an id: the anchor must resolve to the LIVE one."""
    cur = [_mk("q2", 9, "done"), _mk("q1", 0, "active"), _mk("q2", 40, dw="no_item:X", kind="action")]
    out = reconcile_quests(cur, {"add": [_add(2, after="q2")], "remove": []}, next_id=_ids())
    assert [(s.id, s.status) for s in out] == [("q2", "done"), ("q1", "active"), ("q2", "pending"),
                                               ("q11", "pending")]


def test_fallback_reason_for_a_non_live_status_is_not_called_removed():
    cur = [_mk("q1", 0, "active"), _mk("q7", 3, "blocked")]   # an unexpected status
    events = []
    reconcile_quests(cur, {"add": [_add(2, after="q7")], "remove": []}, next_id=_ids(),
                     on_event=lambda k, p: events.append(p["reason"]))
    assert events == ["not_live"]


def test_a_repeat_errand_is_not_deduped_against_a_done_step():
    """runs/brock-goals4-20260923 steps 1308-1419: at 4/31 HP L1 added 'heal at Pewter Pokecenter
    (58, hp_frac>=1.0)' ~60 times; every add was silently dropped because the earlier heal q5 with the
    same key was DONE. Heals / shopping / returning to a map are repeatable."""
    cur = [_mk("q5", 58, "done", dw="hp_frac>=1.0", kind="action"), _mk("q14", 13, "active", dw="level>=13", kind="action")]
    out = reconcile_quests(cur, {"add": [_add(58, dw="hp_frac>=1.0", kind="action")], "remove": []}, next_id=_ids())
    assert [(s.id, s.status) for s in out] == [("q5", "done"), ("q14", "active"), ("q11", "pending")]


def test_a_dropped_duplicate_add_emits_an_event():
    cur = [_mk("q1", 0, "active"), _mk("q2", 40, dw="no_item:Oaks Parcel", kind="action")]
    events = []
    reconcile_quests(cur, {"add": [_add(40, dw="no_item:Oaks Parcel", kind="action")], "remove": []},
                     next_id=_ids(), on_event=lambda k, p: events.append((k, p)))
    assert [k for k, _ in events] == ["l1_add_deduped"] and events[0][1]["against"] == "q2"
