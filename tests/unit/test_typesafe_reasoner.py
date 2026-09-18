"""TypeSafeReasoner with a fake TypeSafe client (no network)."""
from dataclasses import dataclass, field

from pokemon_agent.agent.typesafe_reasoner import TypeSafeReasoner
from pokemon_agent.core.models import (
    AdvanceDialogAction,
    Direction,
    GoToAction,
    MoveAction,
    WaitAction,
)


@dataclass
class _Ans:
    choice: str
    confidence: float
    probabilities: dict = field(default_factory=dict)


@dataclass
class _Usage:
    input_tokens: int = 10
    output_tokens: int = 1


@dataclass
class _Resp:
    answers: dict
    usage: _Usage = field(default_factory=_Usage)


class FakeClient:
    """Returns a scripted choice+confidence for the 'action' question."""

    def __init__(self, choice: str, confidence: float):
        self.choice, self.confidence = choice, confidence
        self.last_state = None

    def system_one(self, *, state, questions):
        self.last_state = state
        return _Resp(answers={"action": _Ans(self.choice, self.confidence)})


def _step(client, game_state=None, **ctor):
    r = TypeSafeReasoner(client=client, **ctor)
    return r.step(primary_goal="Leave the lab", player_desc="x=5 y=5 map=37",
                  game_state=game_state or {"dialog_active": False})


def test_confident_move_maps_to_move_action():
    step, latency, usage = _step(FakeClient("move_north", 0.92))
    assert isinstance(step.action, MoveAction) and step.action.direction == Direction.NORTH
    assert usage["confidence"] == 0.92


def test_low_confidence_move_stays_a_move_by_default():
    # the confidence->wait gate is opt-in; by default a low-conf move is still taken
    step, _, usage = _step(FakeClient("move_east", 0.10), min_confidence=0.45)
    assert isinstance(step.action, MoveAction)
    assert usage["confidence"] == 0.10


def test_wait_gate_when_enabled_downgrades_low_conf_move():
    step, _, usage = _step(FakeClient("move_east", 0.10), min_confidence=0.45,
                           wait_on_low_confidence=True)
    assert isinstance(step.action, WaitAction)  # opt-in thrash guard kicked in
    assert usage["confidence"] == 0.10


def test_low_confidence_wait_is_bounded():
    # after max_consecutive_waits, it stops waiting and takes the move anyway
    r = TypeSafeReasoner(client=FakeClient("move_east", 0.10), min_confidence=0.45,
                         wait_on_low_confidence=True, max_consecutive_waits=2)
    kinds = []
    for _ in range(4):
        s, _, _ = r.step(primary_goal="g", player_desc="p", game_state={"dialog_active": False})
        kinds.append(type(s.action).__name__)
    assert kinds == ["WaitAction", "WaitAction", "MoveAction", "WaitAction"]


def test_dialog_choice_not_downgraded_even_if_low_conf():
    step, _, _ = _step(FakeClient("advance_dialog", 0.20),
                       game_state={"dialog_active": True})
    assert isinstance(step.action, AdvanceDialogAction)


def test_goto_choice_maps_to_gotoaction():
    targets = [{"key": "goto_poke_ball_11_7", "label": "Poke Ball (11,7)",
                "desc": "go to ball", "x": 11, "y": 7, "interact": True}]
    r = TypeSafeReasoner(client=FakeClient("goto_poke_ball_11_7", 0.88))
    step, _, usage = r.step(primary_goal="pick a starter", player_desc="x=8 y=5",
                            game_state={"dialog_active": False}, targets=targets)
    assert isinstance(step.action, GoToAction)
    assert (step.action.x, step.action.y, step.action.interact) == (11, 7, True)
    assert step.action.label == "Poke Ball (11,7)"
    assert usage["confidence"] == 0.88


def test_state_carries_structured_context():
    c = FakeClient("wait", 0.9)
    r = TypeSafeReasoner(client=c)
    r.step(primary_goal="Leave", player_desc="x=1", exits=[{"x": 7, "y": 1, "dest_map": 0}],
           game_state={"dialog_active": False})
    assert c.last_state["exits"] == [{"x": 7, "y": 1, "dest_map": 0}]
    assert c.last_state["primary_goal"] == "Leave"


def test_blocked_dirs_are_masked_from_choices():
    """A move into a known wall must be removed from the executor's Choice set."""
    from types import SimpleNamespace

    class RecClient:
        def system_one(self, *, state, questions):
            self.criteria = questions["action"].criteria
            return SimpleNamespace(answers={"action": SimpleNamespace(choice="move_south", confidence=0.9)})

    c = RecClient()
    r = TypeSafeReasoner(client=c)
    step, _, _ = r.step(primary_goal="go", player_desc="p",
                        game_state={"dialog_active": False}, blocked_dirs={"north", "east"})
    assert "move_north" not in c.criteria and "move_east" not in c.criteria
    assert "move_south" in c.criteria and "move_west" in c.criteria
