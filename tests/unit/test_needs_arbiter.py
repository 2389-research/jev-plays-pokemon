"""Needs arbitration: priority, hysteresis, grind budget (pure, no ROM)."""
from pokemon_agent.agent.needs_arbiter import NeedsArbiter
from pokemon_agent.agent.plan import Intent

B = 0xD16B


class MemFake:
    def __init__(self, mem):
        self.mem = dict(mem)

    def read_memory(self, addr, bank=None):
        return self.mem.get(addr, 0)

    def set(self, **kw):
        self.mem.update(kw)
        return self


def _party(hp, maxhp, level):
    return {0xD163: 1, B: 33, B + 1: hp >> 8, B + 2: hp & 0xFF,
            B + 0x21: level, B + 0x22: maxhp >> 8, B + 0x23: maxhp & 0xFF}


def _mem(hp=20, maxhp=20, level=14, map_id=12, in_battle=0):
    return MemFake({**_party(hp, maxhp, level), 0xD35E: map_id, 0xD057: in_battle})


def test_priority_progress_when_ready():
    arb = NeedsArbiter(goal_map=2, level_target=12)
    assert arb.current(_mem(level=14, map_id=12)).name == "progress"
    assert arb.intent(_mem(level=14, map_id=12)) == Intent.TRAVEL


def test_readiness_maps_to_grind_intent():
    arb = NeedsArbiter(goal_map=2, level_target=12)
    assert arb.current(_mem(level=7)).name == "readiness"
    # readiness -> GRIND intent; the planner turns that into a level-target success predicate
    assert arb.intent(_mem(level=7)) == Intent.GRIND


def test_survive_maps_to_heal_intent():
    arb = NeedsArbiter(goal_map=2, level_target=12, low_hp=0.30, heal_hp=0.80)
    assert arb.intent(_mem(hp=5, maxhp=20)) == Intent.HEAL


def test_survive_latches_with_hysteresis():
    arb = NeedsArbiter(goal_map=2, level_target=12, low_hp=0.30, heal_hp=0.80)
    # drops below low_hp -> survive latches
    assert arb.current(_mem(hp=5, maxhp=20)).name == "survive"
    # partial recovery (50%) is NOT enough to unlatch -> still survive (no thrash)
    assert arb.current(_mem(hp=10, maxhp=20)).name == "survive"
    # recovered past heal_hp -> unlatches, back to normal priority
    assert arb.current(_mem(hp=18, maxhp=20, level=7)).name == "readiness"


def test_battle_preempts_readiness_and_progress():
    arb = NeedsArbiter(goal_map=2, level_target=12)
    assert arb.current(_mem(level=7, in_battle=1)).name == "battle"


def test_grind_budget_yields_to_progress():
    arb = NeedsArbiter(goal_map=2, level_target=99, grind_budget=3)  # target unreachable
    got = [arb.current(_mem(level=7, map_id=12)).name for _ in range(6)]
    assert got[:3] == ["readiness", "readiness", "readiness"]
    assert got[3] == "progress"  # budget spent -> stop grinding, push toward goal
