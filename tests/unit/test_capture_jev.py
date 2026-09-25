import json
from pokemon_agent.logging.capture import Capture


class _Ans:
    choice = "up"; confidence = 0.83; probabilities = {"up": 0.83, "down": 0.17}
class _Resp:
    answers = {"pol": _Ans()}   # choose_policy reads resp.answers["pol"] (NOT "policy")
class _Client:
    def system_one(self, state, questions): return _Resp()


def test_choose_policy_records_confidence(tmp_path):
    from pokemon_agent.agent.typesafe_reasoner import TypeSafeReasoner
    r = TypeSafeReasoner.__new__(TypeSafeReasoner)
    r.client = _Client(); r.capture = Capture(tmp_path, mode="distill")
    r.capture.begin_step(4, None)
    # call choose_policy with its real kwargs; assert it returns (policy, conf) unchanged
    # AND a jev_policy record with confidence lands. (Match the real signature when writing.)
    r.choose_policy(hp_frac=1.0, level=9, level_target=12, objective="reach Brock")
    r.capture.flush()
    recs = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    rec = [x for x in recs if x["layer"] == "jev_policy"][0]
    assert rec["confidence"] == 0.83
    assert rec["extra"]["probabilities"]["up"] == 0.83
