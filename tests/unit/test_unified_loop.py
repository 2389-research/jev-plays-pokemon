from pokemon_agent.agent.planner_llm import Planner


class FakeProvider:
    def __init__(self, content): self.content = content
    def chat_json(self, system, state, image=None): return self.content, 0, {}


def _ctx(**kw):
    base = {"player": {"x": 3, "y": 3, "map_id": 1}, "reachable": {(3, 3), (3, 4), (4, 3)},
            "exit_tile": (4, 3), "map_view": ["..."], "objective": "go", "destination": "map 0",
            "goal_dir": "south", "npcs": [], "recent_trail": [], "recent_targets": [],
            "default": {"kind": "exit"}, "stuck": False, "why": "pick"}
    base.update(kw); return base


def test_propose_target_no_provider_returns_deterministic_default():
    # OFFLINE / tests: running with no LLM is intentional, not an error -> deterministic default.
    p = Planner(goal_map=0, provider=None)
    assert p.propose_target(emu=None, context=_ctx()) == {"kind": "exit"}


def test_propose_target_parses_tile():
    p = Planner(goal_map=0, provider=FakeProvider('{"kind":"tile","x":3,"y":4,"note":"south"}'))
    t = p.propose_target(emu=None, context=_ctx())
    assert t["kind"] == "tile" and (t["x"], t["y"]) == (3, 4) and t["note"] == "south"


def test_propose_target_configured_but_unreachable_tile_returns_unresolved():
    # a provider IS wired (live run) but it named an unreachable tile -> this is a model failure;
    # BREAK LOUDLY, do NOT silently substitute the deterministic default.
    p = Planner(goal_map=0, provider=FakeProvider('{"kind":"tile","x":9,"y":9,"note":"bad"}'))
    t = p.propose_target(emu=None, context=_ctx())
    assert t["kind"] == "unresolved" and "note" in t


def test_propose_target_passes_through_exit_and_approach_and_enter():
    for content, kind in [('{"kind":"exit","note":"leave"}', "exit"),
                          ('{"kind":"approach_npc","sprite":"Oak","note":"talk"}', "approach_npc"),
                          ('{"kind":"enter","map":1,"note":"door"}', "enter")]:
        p = Planner(goal_map=0, provider=FakeProvider(content))
        assert p.propose_target(emu=None, context=_ctx())["kind"] == kind


def test_propose_target_configured_but_bad_json_returns_unresolved():
    # provider wired but returns garbage -> model failure -> unresolved (flagged), not a guess.
    p = Planner(goal_map=0, provider=FakeProvider("not json"))
    assert p.propose_target(emu=None, context=_ctx())["kind"] == "unresolved"


def test_propose_target_approach_npc_without_sprite_returns_unresolved():
    p = Planner(goal_map=0, provider=FakeProvider('{"kind":"approach_npc","note":"talk"}'))
    assert p.propose_target(emu=None, context=_ctx())["kind"] == "unresolved"


def test_propose_target_enter_without_map_returns_unresolved():
    p = Planner(goal_map=0, provider=FakeProvider('{"kind":"enter","note":"door"}'))
    assert p.propose_target(emu=None, context=_ctx())["kind"] == "unresolved"


def test_propose_target_unresolved_payload_has_reason_and_note():
    p = Planner(goal_map=0, provider=FakeProvider("not json"))
    t = p.propose_target(emu=None, context=_ctx())
    assert t["kind"] == "unresolved" and t.get("reason") and t.get("note")
