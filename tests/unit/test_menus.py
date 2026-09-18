"""Menu detection + base-agnostic selection (pure, no ROM)."""
from pathlib import Path

import pytest

from pokemon_agent.games.pokemon_red import menus
from pokemon_agent.games.pokemon_red.game_state import WTILEMAP

CURSOR = menus.CURSOR_TILE  # 0xED
UP = {c: 0x80 + (ord(c) - ord("A")) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}


class RecFake:
    """Records button presses; reads memory from a dict."""

    def __init__(self, mem=None):
        self.mem = dict(mem or {})
        self.presses = []

    def read_memory(self, addr, bank=None):
        return self.mem.get(addr, 0)

    def press(self, button):
        self.presses.append(getattr(button, "value", button))

    def tick(self, n=1, render=True):
        pass


def _row(mem, row, text_tiles, col=0):
    for i, t in enumerate(text_tiles):
        mem[WTILEMAP + row * 20 + col + i] = t


def test_menu_open_via_cursor_tile():
    assert menus.menu_open(RecFake({})) is False
    assert menus.menu_open(RecFake({WTILEMAP + 14 * 20 + 9: CURSOR})) is True


def test_read_menu_detects_yesno():
    mem = {WTILEMAP + 12 * 20 + 8: CURSOR}  # cursor
    _row(mem, 12, [UP[c] for c in "YES"], col=9)
    _row(mem, 14, [UP[c] for c in "NO"], col=9)
    mem[0xCC26] = 0
    mem[0xCC28] = 1
    m = menus.read_menu(RecFake(mem))
    assert m["open"] and m["kind"] == "yesno" and m["num_options"] == 2


def test_select_option_sequence_is_base_agnostic():
    emu = RecFake({WTILEMAP + 10 * 20 + 5: CURSOR})  # a menu is open
    res = menus.select_option(emu, 2, max_options=5)
    assert res["ok"]
    # top (UP x max_options) then DOWN x index then confirm A
    assert emu.presses == ["up"] * 5 + ["down"] * 2 + ["a"]


def test_answer_yesno_maps_to_indices():
    emu_yes = RecFake({WTILEMAP + 10 * 20 + 5: CURSOR})
    menus.answer_yesno(emu_yes, True)
    assert emu_yes.presses.count("down") == 0 and emu_yes.presses[-1] == "a"   # YES = top
    emu_no = RecFake({WTILEMAP + 10 * 20 + 5: CURSOR})
    menus.answer_yesno(emu_no, False)
    assert emu_no.presses.count("down") == 1 and emu_no.presses[-1] == "a"      # NO = one down


def test_select_option_refuses_when_no_menu():
    assert menus.select_option(RecFake({}), 0)["ok"] is False


ROM = Path("roms/pokemon_red.gb")


@pytest.mark.skipif(not (ROM.exists() and Path("states/battle_menu.state").exists()),
                    reason="needs ROM + battle_menu fixture")
def test_menu_open_true_on_real_menu_false_on_overworld():
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    e1 = PyBoyEmulator(str(ROM), window="null", speed=0); e1.load_state(Path("states/battle_menu.state")); e1.tick(3)
    e2 = PyBoyEmulator(str(ROM), window="null", speed=0); e2.load_state(Path("states/after_starter.state")); e2.tick(3)
    assert menus.menu_open(e1) is True
    assert menus.menu_open(e2) is False
    e1.close(); e2.close()


@pytest.mark.skipif(not (ROM.exists() and Path("states/starter_prompt.state").exists()),
                    reason="needs ROM + starter_prompt fixture (the YES/NO 'want SQUIRTLE?')")
def test_answer_yes_gets_the_starter():
    from pokemon_agent.core.models import GameButton
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.games.pokemon_red.progress import read_progress
    emu = PyBoyEmulator(str(ROM), window="null", speed=0)
    emu.load_state(Path("states/starter_prompt.state")); emu.tick(3)
    assert menus.read_menu(emu)["kind"] == "yesno"
    assert read_progress(emu).party_size == 0
    menus.answer_yesno(emu, True)                       # confirm the starter
    for _ in range(20):                                 # advance the give-mon cutscene
        emu.press(GameButton.A); emu.tick(24)
    pr = read_progress(emu)
    assert pr.party_size == 1 and pr.milestones["got_starter"] is True
    emu.close()
