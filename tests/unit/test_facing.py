"""Sprite kind tagging + 'what the player is facing' state."""
from pokemon_agent.games.pokemon_red.game_state import _sprite_kind, read_facing

B1, B2 = 0xC100, 0xC200


class MemFake:
    def __init__(self, mem):
        self.mem = dict(mem)

    def read_memory(self, addr, bank=None):
        return self.mem.get(addr, 0)


def test_sprite_kind():
    assert _sprite_kind("Poke Ball") == "item"
    assert _sprite_kind("Oak") == "person"
    assert _sprite_kind("Scientist") == "person"


def _sprite(mem, slot, pic, x, y, facing=0):
    mem[B1 + slot * 16] = pic
    mem[B1 + slot * 16 + 9] = facing
    mem[B2 + slot * 16 + 5] = x + 4   # +4 map-border offset
    mem[B2 + slot * 16 + 4] = y + 4


def test_read_facing_detects_adjacent_npc():
    # player at (5,5) facing north (4); an NPC directly north at (5,4)
    mem = {0xD362: 5, 0xD361: 5, 0xC109: 4}
    _sprite(mem, 1, 3, 5, 4)  # Oak (person) at (5,4)
    f = read_facing(MemFake(mem))
    assert f["direction"] == "north" and f["front_tile"] == [5, 4]
    assert f["can_interact"] and f["facing_sprite"]["sprite"] == "Oak"


def test_read_facing_across_counter_distance_two():
    # facing north; nobody adjacent, but a clerk 2 tiles north (across a counter)
    mem = {0xD362: 5, 0xD361: 5, 0xC109: 4}
    _sprite(mem, 1, 3, 5, 3)  # at (5,3) = 2 north
    f = read_facing(MemFake(mem))
    assert f["can_interact"] and (f["facing_sprite"]["x"], f["facing_sprite"]["y"]) == (5, 3)


def test_read_facing_nothing_there():
    mem = {0xD362: 5, 0xD361: 5, 0xC109: 4}
    f = read_facing(MemFake(mem))
    assert f["can_interact"] is False
