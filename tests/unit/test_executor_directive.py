"""Executor teeth: the Choice set is derived from the active directive (spec §5)."""
from types import SimpleNamespace

from pokemon_agent.agent.plan import Directive, Intent
from pokemon_agent.agent.typesafe_reasoner import TypeSafeReasoner
from pokemon_agent.core.models import GoToAction


class RecClient:
    """Records the criteria it was handed and returns a scripted choice."""

    def __init__(self, choice="goto_directive_target", confidence=0.9, probabilities=None):
        self.choice, self.confidence, self.probabilities = choice, confidence, probabilities

    def system_one(self, *, state, questions):
        self.criteria = questions["action"].criteria
        self.state = state
        ans = SimpleNamespace(choice=self.choice, confidence=self.confidence,
                              probabilities=self.probabilities)
        return SimpleNamespace(answers={"action": ans}, usage=None)


def _step(client, directive, **kw):
    r = TypeSafeReasoner(client=client)
    return r.step(primary_goal="g", player_desc="x=5 y=5 map=2",
                  game_state={"dialog_active": False}, directive=directive, **kw)


def test_talk_to_masks_raw_moves_and_injects_target():
    d = Directive(intent=Intent.TALK_TO, target={"kind": "oak", "map": 2, "x": 7, "y": 6},
                  success={"talked_to": [2, 7, 6]})
    c = RecClient(choice="goto_directive_target")
    step, _, _ = _step(c, d)
    # raw moves masked (a target exists), directive target injected as a goto option
    assert not any(k.startswith("move_") for k in c.criteria)
    assert "goto_directive_target" in c.criteria
    assert isinstance(step.action, GoToAction)
    assert (step.action.x, step.action.y, step.action.interact) == (7, 6, True)


def test_travel_keeps_moves_for_exploration():
    # travel with a cross-map target (no x,y tile) -> moves stay available to explore
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 2}, success={"on_map": 2})
    c = RecClient(choice="move_north")
    _step(c, d)
    assert "move_north" in c.criteria  # not masked; no goto tile to head to


def test_allowed_options_whitelist():
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 2}, success={"on_map": 2},
                  allowed_options=["move_north", "move_south"])
    c = RecClient(choice="move_north")
    _step(c, d)
    assert set(c.criteria) == {"move_north", "move_south"}


def test_saycan_bias_reranks_among_legal_options():
    # model prefers move_south (0.5) but bias doubles move_north (0.4 -> 0.8) -> north wins
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 2}, success={"on_map": 2},
                  option_bias=["move_north"])
    probs = {"move_south": 0.5, "move_north": 0.4, "wait": 0.1}
    c = RecClient(choice="move_south", probabilities=probs)
    step, _, _ = _step(c, d)
    from pokemon_agent.core.models import Direction, MoveAction
    assert isinstance(step.action, MoveAction) and step.action.direction == Direction.NORTH


def test_no_directive_is_legacy_behavior():
    # a plain call (no directive) keeps all move_* options — unchanged from before
    c = RecClient(choice="move_north")
    r = TypeSafeReasoner(client=c)
    r.step(primary_goal="g", player_desc="p", game_state={"dialog_active": False})
    assert {"move_north", "move_south", "move_east", "move_west"} <= set(c.criteria)
