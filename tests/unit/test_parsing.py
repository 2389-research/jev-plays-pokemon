import pytest

from pokemon_agent.providers.parsing import parse_decision, strip_fences


def test_plain_json():
    d = parse_decision('{"action":{"type":"move","direction":"north","tiles":1},"decision_note":"go"}')
    assert d.action.type == "move"
    assert d.action.direction.value == "north"


def test_markdown_fenced_json():
    raw = '```json\n{"action":{"type":"press","button":"a"},"decision_note":"tap"}\n```'
    d = parse_decision(raw)
    assert d.action.type == "press"
    assert d.action.button.value == "a"


def test_json_with_surrounding_prose():
    raw = 'Sure! Here is my choice:\n{"action":{"type":"wait","frames":10},"decision_note":"pause"} hope that helps'
    d = parse_decision(raw)
    assert d.action.type == "wait"


def test_empty_content_raises():
    with pytest.raises(ValueError):
        parse_decision("")


def test_invalid_json_raises():
    with pytest.raises(ValueError):
        parse_decision("{not json")


def test_bad_schema_raises():
    with pytest.raises(ValueError):
        parse_decision('{"action":{"type":"fly"},"decision_note":"nope"}')


def test_strip_fences_idempotent_on_clean():
    assert strip_fences('{"a":1}') == '{"a":1}'
