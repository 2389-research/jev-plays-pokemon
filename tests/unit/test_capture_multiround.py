import json
from pokemon_agent.logging.capture import Capture
from pokemon_agent.agent.planner_llm import Planner


class _Knowledge:               # stub KB — MIRRORS the real signature (knowledge.py:57) so the real
    # call works: _llm_with_search does self.knowledge.query_texts(q, top_k=4) (planner_llm.py:390).
    def query_texts(self, text, *, top_k=5, max_chars=1600):
        return ["(kb result)"]


class _SearchProv:
    def __init__(self): self.n = 0
    def chat_json(self, system, user, image=None):
        self.n += 1
        # A round is ONLY triggered when the parsed "search" is a non-empty LIST and knowledge is set
        # (planner_llm.py:385-386). A string does NOT trigger it.
        if self.n == 1:
            return ('{"search": ["where is Brock"]}', 30, {"total_tokens": 10})   # KB-search round
        return ('{"assessment":"go north","change":true}', 40, {"total_tokens": 20})  # final


def test_brainstorm_emits_one_decision_plus_rounds(tmp_path):
    cap = Capture(tmp_path, mode="distill"); cap.begin_step(9, None)
    p = Planner(goal_map=2, level_target=0, reflector=None,
                provider=_SearchProv(), strategist=_SearchProv(), knowledge=_Knowledge())
    p.capture = cap
    # drive l1_brainstorm (or _llm_with_search directly) so it does >=1 search round + a final
    p.l1_brainstorm(emu=None, context={"player": {"x": 1, "y": 1}})
    cap.flush()
    recs = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    decisions = [r for r in recs if r["layer"] == "l1_brainstorm" and r["kind"] == "model"]
    rounds = [r for r in recs if r["kind"] == "model_round"]
    assert len(decisions) == 1
    assert decisions[0]["tokens"] == 30              # sum of round(10) + final(20)
    assert len(rounds) >= 1                           # the search round WAS emitted
    assert all(r["group_id"] == decisions[0]["group_id"] for r in rounds)
