import json
from pokemon_agent.logging.capture import Capture
from pokemon_agent.games.pokemon_red import battle_agent


class _Ans:
    choice = "1"; confidence = 0.7
class _Resp:
    answers = {"move": _Ans()}
class _Client:
    model = "tsafe"
    def system_one(self, state, questions): return _Resp()


def test_choose_move_records_and_returns_unchanged(tmp_path, monkeypatch):
    # stub the emu-reading helpers so a trivial emu suffices
    monkeypatch.setattr(battle_agent.battle, "active_moves", lambda emu: ["Tackle", "Ember"])
    monkeypatch.setattr(battle_agent, "read_battle", lambda emu: {"active": {}, "enemy": {}})

    cap = Capture(tmp_path, mode="distill"); cap.begin_step(2, None)
    slot, conf = battle_agent.choose_move(_Client(), object(), capture=cap)
    cap.flush()

    assert slot == 1 and conf == 0.7                 # behavior unchanged
    recs = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    rec = [x for x in recs if x["layer"] == "battle_move"][0]
    assert rec["confidence"] == 0.7
    assert rec["output_parsed"] == {"slot": 1}
