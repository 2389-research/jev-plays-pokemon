"""RAM success/impossibility predicates (pure, no ROM)."""
from pokemon_agent.games.pokemon_red import predicates

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


def test_talked_to_consults_memory():
    class Interactions:
        talked = {(2, 5, 6)}

    class Mem:
        interactions = Interactions()

    assert predicates.evaluate({"talked_to": [2, 5, 6]}, _mem(), memory=Mem())
    assert not predicates.evaluate({"talked_to": [2, 9, 9]}, _mem(), memory=Mem())
    assert not predicates.evaluate({"talked_to": [2, 5, 6]}, _mem())  # no memory -> False
