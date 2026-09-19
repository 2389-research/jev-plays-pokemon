from pokemon_agent.agent.plan import Directive, Intent


def test_directive_has_optional_quest_id():
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 1}, success={"on_map": 1})
    assert d.quest_id is None                      # default, backward-compatible
    d2 = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 1}, success={"on_map": 1}, quest_id="q3")
    assert d2.quest_id == "q3"


from pokemon_agent.agent.quest_reconciler import QuestStep, compile_steps_to_directives


def test_compile_travel_only_step():
    s = QuestStep(id="q1", map=1, talk=False, who=None, done_when="on_map", why="go")
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
    s = QuestStep(id="q3", map=2, talk=False, who=None, done_when="garbage", why="x")
    ds = compile_steps_to_directives([s])
    assert ds[0].success == {"on_map": 2}     # unparseable criterion -> safe default for a travel step
