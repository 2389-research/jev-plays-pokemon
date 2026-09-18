"""Executive dispatch: deterministic servo, success-replan, stack suspension (spec §9)."""
from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.plan import Directive, Intent, ReflectionPlan
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep
from pokemon_agent.agent.session import Session
from pokemon_agent.core.models import GoalState, WaitAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.observations.builder import ObservationBuilder


class StubReasoner:
    def reflect(self, **kw):
        return ReflectionPlan(next_objective="go"), 0, {}

    def step(self, **kw):
        self.last_directive = kw.get("directive")
        return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}


class StubPlanner:
    def __init__(self, *directives):
        self.queue = list(directives)
        self.calls = 0

    def plan(self, intent, emu, memory, *, why="", context=None):
        self.calls += 1
        self.last_why = why
        self.last_context = context
        return self.queue.pop(0) if self.queue else self.queue[-1]


class StubArbiter:
    def __init__(self, intent):
        self._intent = intent

    def intent(self, emu):
        return self._intent


def _loop(goal_map=99, map_id=0, events=None):
    emu = FakeEmulator(map_id=map_id)
    session = Session(GoalState(primary="reach pewter", current="reach pewter"))
    sink = (lambda k, p: events.append((k, p))) if events is not None else None
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=StubReasoner(), session=session, vision=False,
                         reflect_every=100, goal_map=goal_map, on_event=sink)
    return loop, emu


def test_servo_walks_toward_same_map_tile_no_model_call():
    loop, emu = _loop(map_id=0)
    # target is open floor south of the start (2,2) -> servo should step south, no executor
    loop.arbiter = StubArbiter(Intent.TRAVEL)
    d = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 0, "x": 2, "y": 4},
                  success={"on_map": 99})
    loop.planner = StubPlanner(d)
    start = (emu.x, emu.y)
    loop.step_once()
    assert (emu.x, emu.y) != start and emu.y > start[1]  # moved south toward the target
    assert loop._directive is d  # committed to the directive (no replan)


def test_success_predicate_triggers_replan():
    events = []
    loop, emu = _loop(map_id=5, events=events)
    loop.arbiter = StubArbiter(Intent.TRAVEL)
    done = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 5}, success={"on_map": 5})
    nxt = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 9, "x": 2, "y": 4},
                    success={"on_map": 9})
    loop.planner = StubPlanner(nxt)
    loop._directive = done  # already-satisfied directive (emu is on map 5)
    loop.step_once()
    kinds = [k for k, _ in events]
    assert "directive_done" in kinds
    assert loop.planner.calls == 1 and loop._directive is nxt  # replanned to the next directive


class StubQuestPlanner(StubPlanner):
    def __init__(self, *directives, quest=()):
        super().__init__(*directives)
        self._quest = list(quest)

    def strategize(self, emu, memory, *, why=""):
        self.strategize_why = why
        return list(self._quest)

    def plan(self, intent, emu, memory, *, why="", context=None):
        # real planner always returns a directive; the stub falls back to a benign one when its
        # scripted queue is exhausted (e.g. the normal replan after a quest completes).
        if self.queue:
            return self.queue.pop(0)
        return Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 0}, success={"on_map": -1})


class JudgingReasoner(StubReasoner):
    def __init__(self, score):
        self.score = score

    def judge(self, question, state):
        return self.score


def test_story_gate_escalates_to_quest_and_advances_in_order():
    loop, emu = _loop(map_id=0)
    loop.reasoner = JudgingReasoner(0.9)               # Jev router: "needs a sub-quest"
    loop.arbiter = StubArbiter(Intent.TRAVEL)
    q1 = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 5}, success={"on_map": 5})
    q2 = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 9}, success={"on_map": 9})
    loop.planner = StubQuestPlanner(quest=[q1, q2])
    # an impossible active directive forces the escalation path
    loop._directive = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 2}, success={"on_map": 2})
    loop._servo_fail = 99

    loop._manage_directive(loop.builder.build(capture_screenshot=False)[0])
    assert loop._in_quest and loop._directive is q1     # escalated -> first quest step

    emu.map_id = 5                                        # q1 success -> advance to q2
    loop._manage_directive(loop.builder.build(capture_screenshot=False)[0])
    assert loop._in_quest and loop._directive is q2

    emu.map_id = 9                                        # q2 success -> quest complete
    loop._manage_directive(loop.builder.build(capture_screenshot=False)[0])
    assert not loop._in_quest


def test_wedged_quest_step_re_strategizes_instead_of_abandoning():
    # a quest step that wedges too long re-plans from current state (force), staying on the
    # errand rather than falling back to grind — even with a LOW escalation score.
    loop, emu = _loop(map_id=1)
    loop.reasoner = JudgingReasoner(0.1)               # router says "not a gate" ...
    loop.arbiter = StubArbiter(Intent.TRAVEL)
    q1 = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 0}, success={"on_map": 0})
    deliver = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 40}, success={"on_map": 40})
    loop.planner = StubQuestPlanner(quest=[deliver])   # ... but re-strategize returns the delivery quest
    loop._directive = q1
    loop._in_quest = True
    loop._quest_step_age = 999                          # wedged
    from pokemon_agent.agent.reason_loop import QUEST_STEP_BUDGET
    loop._manage_directive(loop.builder.build(capture_screenshot=False)[0])
    assert loop._in_quest and loop._directive is deliver  # re-strategized, still on the errand


def test_low_escalation_score_does_not_quest():
    loop, emu = _loop(map_id=0)
    loop.reasoner = JudgingReasoner(0.1)               # Jev router: "just reroute"
    loop.arbiter = StubArbiter(Intent.TRAVEL)
    normal = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 9}, success={"on_map": 9})
    loop.planner = StubQuestPlanner(normal, quest=[Directive(intent=Intent.TRAVEL,
                                                             target={"kind": "map", "map": 5}, success={"on_map": 5})])
    loop._directive = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 2}, success={"on_map": 2})
    loop._servo_fail = 99
    loop._manage_directive(loop.builder.build(capture_screenshot=False)[0])
    assert not loop._in_quest and loop._directive is normal  # normal replan, no quest


def test_higher_need_suspends_current_directive_on_the_stack():
    loop, emu = _loop(map_id=0)
    travel = Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": 9}, success={"on_map": 9})
    heal = Directive(intent=Intent.HEAL, target=None, success={"hp_frac": ">=0.8"})
    loop._directive = travel
    loop.arbiter = StubArbiter(Intent.HEAL)   # a higher-priority need preempts
    loop.planner = StubPlanner(heal)
    loop._manage_directive(loop.builder.build(capture_screenshot=False)[0])
    assert loop._directive is heal
    assert travel in loop._dstack  # the lower-priority directive was suspended, not dropped
