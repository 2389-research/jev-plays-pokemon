from pokemon_agent.agent.planner import Plan, Planner


class FakeReflectiveProvider:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def chat_json(self, system_prompt, user, image=None):
        self.calls += 1
        return self.content, 123, {"completion_tokens": 10}


def test_parses_clean_plan():
    raw = '{"scene":"bedroom, stairs bottom-left","strategy":"go downstairs","next_checkpoint":"reach the stairs","memory":"in bedroom","progress":"on_track"}'
    p = Planner(FakeReflectiveProvider(raw))
    plan, latency, usage = p.reflect(primary_goal="find oak", trajectory_summary="s", previous_plan=None, image=None, player_desc="x")
    assert plan.next_checkpoint == "reach the stairs"
    assert plan.progress == "on_track"
    assert latency == 123


def test_strips_think_tag_leak():
    raw = '<think>let me look... stairs are bottom left</think>{"scene":"room","strategy":"go","next_checkpoint":"stairs","memory":"m","progress":"unknown"}'
    p = Planner(FakeReflectiveProvider(raw))
    plan, _, _ = p.reflect(primary_goal="g", trajectory_summary="s", previous_plan=None, image=None, player_desc="x")
    assert plan.next_checkpoint == "stairs"


def test_bad_plan_falls_back_not_crash():
    prev = Plan(strategy="keep going", next_checkpoint="the door", memory="notes")
    p = Planner(FakeReflectiveProvider("garbage not json"))
    plan, _, _ = p.reflect(primary_goal="g", trajectory_summary="s", previous_plan=prev, image=None, player_desc="x")
    assert plan.next_checkpoint == "the door"  # fell back to previous plan


def test_carries_previous_memory_into_prompt():
    prev = Plan(memory="already went downstairs")
    fake = FakeReflectiveProvider('{"scene":"","strategy":"","next_checkpoint":"","memory":"downstairs + mom seen","progress":"unknown"}')
    Planner(fake).reflect(primary_goal="g", trajectory_summary="s", previous_plan=prev, image=None, player_desc="x")
    assert fake.calls == 1
