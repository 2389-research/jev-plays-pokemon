"""Out-of-PP moves (runs/verify-mtmoon2-20260923, ~560 steps stuck vs a Lass on Route 3).

Water Gun had 0 PP (PP [35, 30, 30, 0]); Jev was never shown PP and kept choosing it; the game refused
("No PP left for this move!") and left the MOVE LIST open; the battle path only recognised the root
FIGHT menu, so it pressed A — re-selecting Water Gun — forever.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pokemon_agent.games.pokemon_red import battle, battle_agent

ROM = Path("roms/pokemon_red.gb")
STUCK = Path("runs/verify-mtmoon2-20260923/states/map14_step60.state")


class Mem:
    def __init__(self, moves, pp):
        self.m = {0xD01C + i: mv for i, mv in enumerate(moves)}
        self.m.update({0xD02D + i: p for i, p in enumerate(pp)})

    def read_memory(self, a, bank=None):
        return self.m.get(a, 0)


def test_active_pp_reads_the_low_six_bits():
    emu = Mem([33, 39, 145, 55], [35, 30 | 0x40, 30, 0])      # PP-up bits set on slot 1
    assert battle.active_pp(emu) == [35, 30, 30, 0]


def test_choose_move_never_offers_a_move_without_pp(monkeypatch):
    emu = Mem([33, 39, 145, 55], [35, 30, 30, 0])
    seen = {}

    class Client:
        def system_one(self, state, questions):
            seen["criteria"] = questions["move"].criteria
            return SimpleNamespace(answers={"move": SimpleNamespace(choice="2", confidence=0.9)})
    monkeypatch.setattr(battle_agent, "battle_state_summary", lambda e: {})
    slot, _ = battle_agent.choose_move(Client(), emu)
    assert set(seen["criteria"]) == {"0", "1", "2"} and "3" not in seen["criteria"]
    assert "PP" in seen["criteria"]["2"] and slot == 2


def test_usable_slot_falls_back_from_a_zero_pp_choice():
    assert battle.usable_slot([35, 30, 30, 0], 3) in (0, 1, 2)
    assert battle.usable_slot([35, 30, 30, 0], 2) == 2
    assert battle.usable_slot([0, 0, 0, 0], 3) == 3            # all empty -> the game uses Struggle


@pytest.mark.skipif(not (ROM.exists() and STUCK.exists()), reason="ROM / stuck state not present")
def test_recovers_from_the_stuck_move_list_and_uses_a_move_with_pp():
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    emu = PyBoyEmulator(str(ROM), window="null")
    emu.load_state(STUCK)
    emu.tick(2)
    assert battle.move_list_showing(emu) and not battle.fight_menu_showing(emu)
    assert battle.back_to_fight_menu(emu) and battle.fight_menu_showing(emu)
    r = battle.use_move(emu, 3)                                  # asks for Water Gun (0 PP)
    assert r["ok"] and r["move"] != "Water Gun" and emu.read_memory(0xCCDC) != 55
    emu.close()
