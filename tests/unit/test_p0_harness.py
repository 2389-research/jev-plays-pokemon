"""P0 harness: the loop uses persistent memory, checkpoints it, and learns the graph."""
from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.memory import AgentMemory
from pokemon_agent.agent.reason_loop import ReasoningLoop
from pokemon_agent.agent.reasoner import ReasonStep, ReflectionPlan
from pokemon_agent.agent.session import Session
from pokemon_agent.core.models import Direction, GoalState, MoveAction
from pokemon_agent.emulator.fake_emulator import FakeEmulator
from pokemon_agent.observations.builder import ObservationBuilder


class Stub:
    def reflect(self, **k):
        return ReflectionPlan(next_objective="go"), 0, {}

    def step(self, **k):
        return ReasonStep(location="", objective="", reasoning="",
                          action=MoveAction(direction=Direction.SOUTH)), 0, {}


def _loop(emu, tmp_path=None, memory=None):
    return ReasoningLoop(
        builder=ObservationBuilder(emu), controller=ActionController(emu),
        reasoner=Stub(), session=Session(GoalState()), vision=False, reflect_every=100,
        memory=memory, checkpoint_every=(2 if tmp_path else 0), checkpoint_dir=tmp_path,
    )


def test_checkpoint_writes_and_memory_round_trips(tmp_path):
    emu = FakeEmulator(start=(2, 2), map_id=40)
    loop = _loop(emu, tmp_path)
    for _ in range(4):
        loop.step_once()

    memfile = tmp_path / "latest.mem.json"
    assert memfile.exists()
    mem = AgentMemory.load(memfile)
    assert 40 in mem.map_history                 # the loop's map history was persisted
    assert mem.graph.route(0, 2)                  # seeded Pallet->Pewter graph survived the round-trip
    assert str(tmp_path / "latest.state") in emu.saved_states  # emu-state checkpoint requested


def test_loop_writes_into_shared_memory():
    emu = FakeEmulator(start=(2, 2), map_id=40)
    m = AgentMemory()
    loop = _loop(emu, memory=m)
    loop.step_once()
    assert loop.world is m.world                  # same objects — what the loop learns is what gets saved
    assert loop.interactions is m.interactions
