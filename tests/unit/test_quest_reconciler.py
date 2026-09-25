import pytest

from pokemon_agent.agent.plan import Directive, Intent


def test_directive_has_optional_quest_id():
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 1}, success={"on_map": 1})
    assert d.quest_id is None                      # default, backward-compatible
    d2 = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 1}, success={"on_map": 1}, quest_id="q3")
    assert d2.quest_id == "q3"


from pokemon_agent.agent.quest_reconciler import QuestStep, compile_steps_to_directives


def test_compile_travel_only_step():
    s = QuestStep(id="q1", map=1, talk=False, who=None, done_when="on_map", why="go", kind="travel")
    ds = compile_steps_to_directives([s])
    assert len(ds) == 1 and ds[0].intent == Intent.TRAVEL and ds[0].quest_id == "q1"
    assert ds[0].success == {"on_map": 1}


def test_compile_talk_step_adds_talk_to_with_sprite_and_criterion():
    s = QuestStep(id="q2", map=40, talk=True, who="Oak", done_when="no_item:Oak's Parcel", why="deliver")
    ds = compile_steps_to_directives([s])
    kinds = [d.intent for d in ds]
    assert Intent.TRAVEL in kinds and Intent.TALK_TO in kinds
    talk = [d for d in ds if d.intent == Intent.TALK_TO][0]
    assert talk.quest_id == "q2" and (talk.target or {}).get("sprite") == "Oak"
    assert "no_item" in talk.success          # model-authored criterion attached to the talk step


def test_compile_bad_done_when_falls_back_to_on_map():
    s = QuestStep(id="q3", map=2, talk=False, who=None, done_when="garbage", why="x", kind="travel")
    ds = compile_steps_to_directives([s])
    assert ds[0].success == {"on_map": 2}     # unparseable criterion -> safe default for a travel step


from pokemon_agent.agent.quest_reconciler import reconcile_quests

def _mk(id, map, status="pending", dw="on_map"): return QuestStep(id=id, map=map, done_when=dw, status=status)

def _ids():
    n = [100]
    def nxt():
        n[0] += 1; return f"q{n[0]}"
    return nxt

def test_reconcile_preserves_done_and_active_and_adds():
    cur = [_mk("q1", 0, "done"), _mk("q2", 1, "active"), _mk("q3", 2, "pending")]
    prop = {"add": [{"map": 42, "talk": True, "who": "clerk", "done_when": "has_item:Oak's Parcel", "why": "get parcel"}],
            "remove": ["q3"]}
    out = reconcile_quests(cur, prop, next_id=_ids())
    ids = [s.id for s in out]
    assert "q1" in ids and "q2" in ids          # done + active preserved
    assert "q3" not in ids                        # pending removed
    assert any(s.map == 42 and s.talk for s in out)   # add inserted
    assert out.index(next(s for s in out if s.id == "q2")) < out.index(next(s for s in out if s.map == 42))  # added after active

def test_reconcile_never_removes_active_or_done():
    cur = [_mk("q1", 0, "done"), _mk("q2", 1, "active")]
    out = reconcile_quests(cur, {"add": [], "remove": ["q1", "q2"]}, next_id=_ids())
    assert [s.id for s in out] == ["q1", "q2"]     # removes of done/active ignored

def test_reconcile_replaces_wedged_step():
    cur = [_mk("q2", 1, "active"), _mk("q3", 2, "wedged")]
    prop = {"add": [{"map": 2, "talk": False, "done_when": "on_map", "why": "retry via other route"}], "remove": ["q3"]}
    out = reconcile_quests(cur, prop, next_id=_ids())
    assert all(s.status != "wedged" for s in out)   # wedged gone
    assert any(s.map == 2 and s.status == "pending" for s in out)  # replaced by a fresh pending step

def test_reconcile_empty_proposal_is_unchanged():
    cur = [_mk("q2", 1, "active"), _mk("q3", 2, "pending")]
    out = reconcile_quests(cur, {"add": [], "remove": []}, next_id=_ids())
    assert [s.id for s in out] == ["q2", "q3"]

def test_reconcile_dedups_by_map_and_criterion():
    cur = [_mk("q2", 1, "active"), _mk("q3", 2, "pending", dw="on_map")]
    out = reconcile_quests(cur, {"add": [{"map": 2, "talk": False, "done_when": "on_map", "why": "dup"}], "remove": []}, next_id=_ids())
    assert sum(1 for s in out if s.map == 2 and (s.done_when or "on_map") == "on_map") == 1


def test_travel_step_keeps_on_map():
    d = compile_steps_to_directives([QuestStep(id="t", map=2, kind="travel")])
    assert d[0].success == {"on_map": 2}

def test_action_step_without_criterion_raises():
    with pytest.raises(ValueError):
        compile_steps_to_directives([QuestStep(id="a", map=40, kind="action")])

def test_action_step_with_on_map_raises():
    with pytest.raises(ValueError):
        compile_steps_to_directives([QuestStep(id="a", map=40, kind="action", done_when="on_map")])

def test_grind_action_step_talk_false_still_requires_criterion():
    with pytest.raises(ValueError):
        compile_steps_to_directives([QuestStep(id="g", map=13, kind="action", talk=False)])

def test_action_step_with_real_criterion_ok():
    d = compile_steps_to_directives([QuestStep(id="d", map=40, kind="action",
                                               done_when="no_item:oaks_parcel")])
    assert "no_item" in d[-1].success

def test_reconcile_carries_kind_default_action():
    out = reconcile_quests([], {"add": [{"map": 40, "done_when": "no_item:oaks_parcel"}]},
                           next_id=lambda: "q1")
    assert out[0].kind == "action"
    out2 = reconcile_quests([], {"add": [{"map": 2, "kind": "travel"}]}, next_id=lambda: "q2")
    assert out2[0].kind == "travel"


def test_the_active_step_is_removable_only_with_an_approved_interrupt():
    from pokemon_agent.agent.quest_reconciler import QuestStep, reconcile_quests
    cur = [QuestStep(id="q1", map=42, talk=True, who="clerk", done_when="has_item:Potion", status="active")]
    add = [{"kind": "action", "map": 41, "talk": True, "who": "the Nurse", "done_when": "hp_frac>=1.0"}]
    kept = reconcile_quests(cur, {"add": add, "remove": ["q1"]}, next_id=lambda: "q2")
    assert [s.id for s in kept] == ["q1", "q2"]
    cur[0].status = "active"
    swapped = reconcile_quests(cur, {"add": add, "remove": ["q1"]}, next_id=lambda: "q3", allow_active_removal=True)
    assert [s.id for s in swapped] == ["q3"]
