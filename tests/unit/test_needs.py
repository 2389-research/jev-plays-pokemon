"""RAM-verifiable needs assessment (pure, no ROM)."""
from pokemon_agent.games.pokemon_red import needs

# party struct: count at 0xD163; mon0 at 0xD16B; +1 hp(BE2), +0x21 level, +0x22 maxHP(BE2)
B = 0xD16B


class MemFake:
    def __init__(self, mem):
        self.mem = dict(mem)

    def read_memory(self, addr, bank=None):
        return self.mem.get(addr, 0)


def _party(hp, maxhp, level, species=33):
    return {0xD163: 1, B: species, B + 1: hp >> 8, B + 2: hp & 0xFF,
            B + 0x21: level, B + 0x22: maxhp >> 8, B + 0x23: maxhp & 0xFF}


def test_survive_is_top_when_hp_low():
    emu = MemFake({**_party(5, 20, 7), 0xD35E: 12})   # 25% HP
    n = needs.top_need(emu, goal_map=2, level_target=12)
    assert n.name == "survive"


def test_battle_beats_readiness_and_progress():
    emu = MemFake({**_party(18, 20, 7), 0xD35E: 12, 0xD057: 1})  # healthy, in battle
    assert needs.top_need(emu, goal_map=2, level_target=12).name == "battle"


def test_readiness_when_underleveled_and_safe():
    emu = MemFake({**_party(18, 20, 7), 0xD35E: 12})   # healthy, not in battle, lvl 7 < 12
    assert needs.top_need(emu, goal_map=2, level_target=12).name == "readiness"


def test_progress_when_leveled_and_not_at_goal():
    emu = MemFake({**_party(18, 20, 14), 0xD35E: 12})  # lvl 14 >= 12, on map 12 != goal 2
    assert needs.top_need(emu, goal_map=2, level_target=12).name == "progress"


def test_no_need_when_at_goal_healthy_leveled():
    emu = MemFake({**_party(20, 20, 14), 0xD35E: 2})   # at Pewter, healthy, leveled
    assert needs.top_need(emu, goal_map=2, level_target=12) is None


def test_fainted_triggers_survive():
    emu = MemFake({**_party(0, 20, 7), 0xD35E: 12})
    assert needs.any_fainted(emu) is True
    assert needs.top_need(emu, goal_map=2, level_target=12).name == "survive"
