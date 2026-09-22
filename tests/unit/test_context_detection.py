"""read_screen_text / read_context: real dialogue vs a cutscene picture.

A memory-backed fake lets us feed exact tile bytes without a ROM. The key case is
the graphics false-positive: a picture fills the tilemap with 0xA0-0xB9 tiles, which
the font maps to a-z ("aaaaa"); that must NOT read as dialogue.
"""
from pokemon_agent.games.pokemon_red.game_state import (
    WTILEMAP,
    read_context,
    read_screen_text,
)

# Gen-1 font tiles
SP = 0x7F
UP = {c: 0x80 + (ord(c) - ord("A")) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
LOW = {c: 0xA0 + (ord(c) - ord("a")) for c in "abcdefghijklmnopqrstuvwxyz"}
COLON = 0x9C


class MemFake:
    """read_memory backed by a dict; everything unset reads 0."""

    def __init__(self, mem=None):
        self.mem = dict(mem or {})

    def read_memory(self, addr, bank=None):
        return self.mem.get(addr, 0)


def _row(mem, row, tiles):
    for i, t in enumerate(tiles):
        mem[WTILEMAP + row * 20 + i] = t


def test_real_dialogue_is_detected():
    mem = {}
    # "OAK: HI" on row 12 — has uppercase font tiles + a space
    _row(mem, 12, [UP["O"], UP["A"], UP["K"], COLON, SP, UP["H"], UP["I"]])
    text, active = read_screen_text(MemFake(mem))
    assert active is True and "OAK" in text


def test_all_lowercase_dialogue_is_detected():
    # a continuation line with NO uppercase (e.g. "...strong, they can protect me!") must still read
    # as dialogue. Otherwise the agent gets permanently stuck: it walked into an NPC, a text box is up
    # so it can't move, but it thinks it's overworld and never presses A to close the conversation.
    mem = {}
    _row(mem, 12, [SP if c == " " else LOW[c] for c in "strong they can"])
    text, active = read_screen_text(MemFake(mem))
    assert active is True and "strong" in text


def test_cutscene_graphics_are_not_dialogue():
    mem = {}
    # a picture: rows of 0xA0 (font maps to 'a'), NO uppercase band anywhere
    for r in range(12, 18):
        _row(mem, r, [0xA0] * 20)
    text, active = read_screen_text(MemFake(mem))
    assert active is False and text == ""


def test_context_kinds():
    # battle
    ctx = read_context(MemFake({0xD057: 1}))
    assert ctx["kind"] == "battle" and ctx["battle_kind"] == "wild"
    # overworld (nothing set)
    assert read_context(MemFake({}))["kind"] == "overworld"
    # dialogue
    mem = {}
    _row(mem, 12, [UP["H"], UP["I"], SP, UP["M"], UP["O"], UP["M"]])
    assert read_context(MemFake(mem))["kind"] == "dialog"


def test_forced_movement_flag():
    assert read_context(MemFake({0xD730: 0x40}))["forced_movement"] is True
    assert read_context(MemFake({0xD730: 0x25}))["forced_movement"] is False
