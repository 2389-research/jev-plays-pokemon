"""RAM success/impossibility predicates (pure, no ROM)."""
from pokemon_agent.games.pokemon_red import predicates

from tests.support.ram_emulator import RamEmulator

B = 0xD16B


class MemFake:
    def __init__(self, mem):
        self.mem = dict(mem)

    def read_memory(self, addr, bank=None):
        return self.mem.get(addr, 0)


def _mem(hp=20, maxhp=20, level=14, map_id=12, party=1, in_battle=0, badges=0, money=0):
    m = {0xD163: party, 0xD35E: map_id, 0xD057: in_battle, 0xD356: badges}
    if party:
        m.update({B: 33, B + 1: hp >> 8, B + 2: hp & 0xFF, B + 0x21: level,
                  B + 0x22: maxhp >> 8, B + 0x23: maxhp & 0xFF})
    # money is 3-byte BCD big-endian; encode a small value in the low byte
    m[0xD347], m[0xD348], m[0xD349] = 0, (money // 100) % 100, money % 100
    return MemFake(m)


def test_on_map_equality():
    assert predicates.evaluate({"on_map": 2}, _mem(map_id=2))
    assert not predicates.evaluate({"on_map": 2}, _mem(map_id=12))


def test_party_size_and_level_comparisons():
    assert predicates.evaluate({"party_size": ">=1"}, _mem(party=1))
    assert not predicates.evaluate({"party_size": ">=1"}, _mem(party=0))
    assert predicates.evaluate({"level": ">=12"}, _mem(level=14))
    assert not predicates.evaluate({"level": ">=12"}, _mem(level=7))


def test_hp_frac_threshold():
    assert predicates.evaluate({"hp_frac": ">=0.8"}, _mem(hp=20, maxhp=20))
    assert not predicates.evaluate({"hp_frac": ">=0.8"}, _mem(hp=10, maxhp=20))


def test_in_battle_and_badges():
    assert predicates.evaluate({"in_battle": 0}, _mem(in_battle=0))
    assert not predicates.evaluate({"in_battle": 0}, _mem(in_battle=1))
    assert predicates.evaluate({"badges": ">=1"}, _mem(badges=0b1))
    assert not predicates.evaluate({"badges": ">=1"}, _mem(badges=0))


def test_multiple_clauses_are_anded():
    m = _mem(map_id=2, level=14)
    assert predicates.evaluate({"on_map": 2, "level": ">=12"}, m)
    assert not predicates.evaluate({"on_map": 2, "level": ">=20"}, m)


def test_empty_predicate_never_satisfied():
    assert not predicates.evaluate({}, _mem())
    assert not predicates.evaluate(None, _mem())


def test_unknown_key_is_unverifiable():
    assert not predicates.evaluate({"mystery": 1}, _mem())


def test_at_xy_within_one_tile():
    m = MemFake({0xD35E: 1, 0xD362: 11, 0xD361: 3})
    assert predicates.evaluate({"at_xy": [1, 11, 3]}, m)       # exact
    assert predicates.evaluate({"at_xy": [1, 11, 4]}, m)       # within 1 tile
    assert not predicates.evaluate({"at_xy": [1, 11, 6]}, m)   # too far
    assert not predicates.evaluate({"at_xy": [2, 11, 3]}, m)   # wrong map


def test_talked_to_consults_memory():
    class Interactions:
        talked = {(2, 5, 6)}

    class Mem:
        interactions = Interactions()

    assert predicates.evaluate({"talked_to": [2, 5, 6]}, _mem(), memory=Mem())
    assert not predicates.evaluate({"talked_to": [2, 9, 9]}, _mem(), memory=Mem())
    assert not predicates.evaluate({"talked_to": [2, 5, 6]}, _mem())  # no memory -> False


def test_has_item_checks_bag():
    m = MemFake({0xD31D: 2, 0xD31E: 0x46, 0xD31F: 1, 0xD320: 0x14, 0xD321: 3})
    assert predicates.evaluate({"has_item": 0x46}, m)   # Oak's Parcel present
    assert predicates.evaluate({"has_item": 0x14}, m)
    assert not predicates.evaluate({"has_item": 0x99}, m)


def test_talked_on_map():
    class Interactions:
        talked = {(1, 5, 6), (42, 3, 4)}

    class Mem:
        interactions = Interactions()

    assert predicates.evaluate({"talked_on_map": 42}, MemFake({}), memory=Mem())
    assert not predicates.evaluate({"talked_on_map": 7}, MemFake({}), memory=Mem())
    assert not predicates.evaluate({"talked_on_map": 42}, MemFake({}))  # no memory -> False


def test_hp_frac_predicate_via_ram_emulator():
    emu = RamEmulator()
    emu.set_map(41)
    emu.set_party([("SQUIRTLE", 7, 3, 23)])  # (species, level, cur_hp, max_hp)
    assert not predicates.evaluate({"hp_frac": ">=1.0"}, emu)
    emu.set_party([("SQUIRTLE", 7, 23, 23)])
    assert predicates.evaluate({"hp_frac": ">=1.0"}, emu)


def test_has_no_item_predicate_via_ram_emulator():
    emu = RamEmulator()
    emu.set_bag_items([])
    from pokemon_agent.agent.planner_llm import Planner

    # "oaks_parcel" resolves via resolve_item_id's alnum-lowercase fuzzy match to the
    # same id (70) as "Oak's Parcel" / "Oaks Parcel" — see game_state.resolve_item_id.
    crit = Planner._parse_done_when("no_item:oaks_parcel", 0)  # -> {"no_item": 70}
    assert crit is not None and predicates.evaluate(crit, emu)  # deliver-done when not held
    emu.set_bag_items(["oaks_parcel"])
    assert not predicates.evaluate(crit, emu)  # still holding -> NOT done
