"""Episode memory, unexplored list, stall monitor, and the explore step (spec 2026-09-24).

runs/ss-anne-20260923: L1 re-issued "travel to Vermilion" ~10 times — each removal erased the
failure from its view, and why_wedged was the step's own text. The harness now keeps an attempt
ledger, explains wedges, lists what's unexplored, and can run an explore step."""
from __future__ import annotations

from types import SimpleNamespace

from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.episode_log import EpisodeLog
from pokemon_agent.agent.exploration import StallMonitor, pick_target, unexplored
from pokemon_agent.agent.plan import Directive, Intent, ReflectionPlan
from pokemon_agent.agent.portal_graph import PortalGraph
from pokemon_agent.agent.quest_reconciler import QuestStep, compile_steps_to_directives, reconcile_quests
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep
from pokemon_agent.agent.session import Session
from pokemon_agent.agent.l1_pipeline import validate_step
from pokemon_agent.core.models import GoalState, WaitAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.observations.builder import ObservationBuilder


def _step(i, **kw):
    return QuestStep(id=f"q{i}", map=kw.pop("map", 5), kind=kw.pop("kind", "travel"),
                     done_when=kw.pop("done_when", "on_map"), **kw)


def test_attempt_ledger_survives_removal_and_aggregates_retries():
    e = EpisodeLog()
    for k in range(3):
        s = _step(k)
        e.attempt(10 * k, s, "started")
        e.attempt(10 * k + 5, s, "wedged", "no known way from this part of Cerulean City to Vermilion City")
    e.attempt(40, _step(9), "removed", "replaced")
    rows = e.attempts_view()
    assert rows[0].startswith("travel to Vermilion City: tried 3x (done 0, wedged 3, removed 1)")
    assert "no known way" in e.attempts["travel|5||on_map"]["last_reason"] or "replaced" in rows[0]


def test_since_reports_a_bounce_and_plan_changes():
    e = EpisodeLog()
    for i in range(8):
        e.record(100 + i, "map_enter", map=3 if i % 2 else 15, map_name="Cerulean City" if i % 2 else "Route 4")
    e.attempt(108, _step(1), "wedged", "kept moving between Cerulean City (4x), Route 4 (4x)")
    out = e.since(90, now=110)
    assert out["entered_repeatedly"] == {"Route 4": 4, "Cerulean City": 4}
    assert "wedged: travel to Vermilion City — kept moving" in out["plan_changes"][0]


def test_unexplored_lists_the_trashed_house_door_and_untalked_people():
    pg = PortalGraph.load()
    comp = pg._grid(3)[(19, 18)]
    npcs = [{"x": 15, "y": 18, "sprite": "Cooltrainer M", "kind": "person", "slot": 3, "talked_to": True},
            {"x": 9, "y": 21, "sprite": "Super Nerd", "kind": "person", "slot": 4, "talked_to": False}]
    out = unexplored(pg, 3, comp, visited_maps={3, 64, 35, 15, 65}, npcs=npcs, used_tiles=set())
    labels = [c["label"] for c in out]
    assert any("Cerulean Trashed House" in lb for lb in labels)
    assert any("Super Nerd" in lb for lb in labels) and not any("Cooltrainer" in lb for lb in labels)
    assert not any("Cerulean Pokecenter" in lb for lb in labels)            # visited
    assert "Trashed House" in pick_target(out, (19, 18), prefer="(27,11)")["label"]          # coordinates
    copied = next(lb for lb in labels if "Trashed House" in lb)
    assert pick_target(out, (19, 18), prefer=copied)["label"] == copied                     # a copied entry
    idx = next(i for i, c in enumerate(out) if "Trashed House" in c["label"])
    seen = []
    chosen = pick_target(out, (19, 18), prefer="the house by the Rocket",
                         choose=lambda p, labels: seen.append(p) or idx)                     # free text -> model
    assert "Trashed House" in chosen["label"] and seen == ["the house by the Rocket"]
    assert pick_target(out, (9, 20), prefer=None)["kind"] in ("portal", "npc")              # nearest


def test_stall_monitor_counts_bouncing_as_no_progress():
    m = StallMonitor(threshold=10)
    sig = (1, 2, 3)
    for step in range(30):                                  # two tiles back and forth
        m.observe(step, pos=(3, 0, 18 + step % 2), signature=sig)
    assert m.stalled(30) and m.stalled_for(30) >= 28
    m.observe(31, pos=(62, 2, 7), signature=sig)            # a new tile = progress
    assert not m.stalled(31)


def test_explore_step_compiles_validates_and_dedups():
    assert validate_step({"kind": "explore", "map": 3, "who": "the trashed house"}) == (True, None)
    steps = reconcile_quests([], {"add": [{"kind": "explore", "map": 3, "who": "the trashed house",
                                           "why": "find the way south"}]}, next_id=lambda: "q1")
    assert steps[0].kind == "explore" and steps[0].done_when == "explored"
    ds = compile_steps_to_directives(steps)
    assert [d.intent for d in ds] == [Intent.TRAVEL, Intent.EXPLORE]
    assert ds[1].target["prefer"] == "the trashed house" and ds[1].success == {"explored": 3}


class Stub:
    def reflect(self, **kw):
        return ReflectionPlan(next_objective="go"), 0, {}

    def step(self, **kw):
        return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}


def _loop(map_id=3):
    emu = FakeEmulator(map_id=map_id)
    events = []
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu), reasoner=Stub(),
                         session=Session(GoalState(primary="g", current="g")), vision=False, reflect_every=100,
                         goal_map=2, on_event=lambda k, p: events.append((k, p)))
    return loop, events


def test_explore_ends_on_a_new_place_or_something_new_heard():
    loop, events = _loop()
    loop._map_history[:] = [3]
    d = Directive(intent=Intent.EXPLORE, target={"kind": "explore", "map": 3}, success={"explored": 3},
                  quest_id="q1")
    assert loop._directive_satisfied(d) is False            # first look sets the baseline
    assert loop._directive_satisfied(d) is False
    loop._map_history.append(62)
    assert loop._directive_satisfied(d) is True
    assert any(k == "explore_end" and "Cerulean Trashed House" in p["note"] for k, p in events)
    loop._directive_satisfied(d)                            # new baseline
    loop.heard.observe(5, map_id=62, map_name="Cerulean Trashed House", active=True,
                       lines=["Those miserable ROCKETS!", ""], speaker="Fishing Guru")
    loop.heard.observe(6, map_id=62, map_name="Cerulean Trashed House", active=False, lines=[])
    assert loop._directive_satisfied(d) is True


def test_a_wedge_is_explained_and_recorded_with_the_attempt():
    loop, events = _loop()
    loop._plan_steps = [QuestStep(id="q7", map=5, kind="travel", done_when="on_map", status="active")]
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 5}, success={"on_map": 5}, quest_id="q7")
    loop._wedge_maps = ["Route 4", "Cerulean City"] * 3
    obs = SimpleNamespace(player=None)
    why = loop._explain_wedge(obs, d)
    assert "kept moving between Route 4 (3x), Cerulean City (3x)" in why
    loop._mark_step("q7", "wedged", reason=why)
    assert loop.episode.attempts["travel|5||on_map"]["wedged"] == 1
    assert "kept moving" in loop.episode.attempts_view()[0]


def test_l1_is_told_what_happened_to_its_edits_and_what_the_active_step_is_doing():
    """runs/fresh-squirtle: L1 read an ACTIVE Mart step (still travelling there) as "buying on Route 1"
    and re-issued remove+add 15x — the remove of an active step and the duplicate add were silently
    ignored every time. L1 now sees "doing" and the fate of its last edits."""
    loop, events = _loop(map_id=12)
    loop._plan_steps = [QuestStep(id="q2", map=42, kind="action", talk=True, who="the Mart clerk",
                                  done_when="has_item:Poke Ball>=5", status="active")]
    loop._directive = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 42},
                                success={"on_map": 42}, quest_id="q2")
    obs = SimpleNamespace(player=SimpleNamespace(map_id=12, x=10, y=30), game_state={})
    assert loop._active_doing(obs) == "travelling to Viridian Mart"

    class P:
        strategist = provider = object()
    loop.planner = P()
    import pokemon_agent.agent.reason_loop as rl
    prop = {"assessment": "reorder", "remove": ["q2"],
            "add": [{"kind": "action", "map": 42, "talk": True, "who": "the Mart clerk",
                     "done_when": "has_item:Poke Ball>=5", "after": None}]}
    orig = rl.run_l1_pipeline
    rl.run_l1_pipeline = lambda *a, **k: prop
    try:
        loop._plan = rl.AgentPlan()
        loop._run_l1(obs, hard_event=True)
    finally:
        rl.run_l1_pipeline = orig
    fb = loop._memory_context(obs)["since_last_review"]["your_last_edits"]
    assert any("remove q2 IGNORED" in f and "travelling to Viridian Mart" in f for f in fb)
    assert any("IGNORED: step q2 already covers it" in f for f in fb)
