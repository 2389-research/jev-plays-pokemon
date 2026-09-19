from pokemon_agent.agent.plan import Directive, Intent


def test_directive_has_optional_quest_id():
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 1}, success={"on_map": 1})
    assert d.quest_id is None                      # default, backward-compatible
    d2 = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 1}, success={"on_map": 1}, quest_id="q3")
    assert d2.quest_id == "q3"
