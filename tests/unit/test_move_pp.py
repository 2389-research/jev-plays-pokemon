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


SWITCH = Path("runs/verify-mtmoon5-20260923/states/map59_step123.state")


@pytest.mark.skipif(not (ROM.exists() and SWITCH.exists()), reason="ROM / switch-loop state not present")
def test_backs_out_of_a_voluntary_switch_menu():
    """verify-mtmoon5 steps 121-499: the battle sat in 'Bring out which POKEMON?' / 'WARTORTLE is already
    out!' while the default A kept re-picking the active mon."""
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    emu = PyBoyEmulator(str(ROM), window="null")
    emu.load_state(SWITCH)
    emu.tick(2)
    assert battle.switch_screen_showing(emu)
    assert battle.resolve_switch_screen(emu) and battle.fight_menu_showing(emu)
    emu.close()


def test_switch_screen_picks_a_healthy_mon_when_the_active_one_fainted():
    assert battle.replacement_slot([{"hp": 0}, {"hp": 12}, {"hp": 5}]) == 1
    assert battle.replacement_slot([{"hp": 0}, {"hp": 0}]) is None


# ---- cerulean-team3: fainted Wartortle never replaced; every catch nicknamed "AAAAAAAAAA" ----------------
FAINT = Path("runs/cerulean-team3-20260923/states/map3_step362.state")
CAPTURE = Path("states/capture_wild.state")


@pytest.mark.skipif(not (ROM.exists() and FAINT.exists()), reason="ROM / faint state not present")
def test_forced_switch_sends_out_a_healthy_mon():
    """'Bring out which POKEMON?' after a faint: the party screen isn't a menu to menu_open(), so the
    handler pressed B (can't cancel a forced switch) and the default A re-picked the fainted Wartortle
    ('There's no will to fight!') for ~2,100 steps."""
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    emu = PyBoyEmulator(str(ROM), window="null")
    emu.load_state(FAINT)
    emu.tick(4)
    assert battle.switch_screen_showing(emu)
    battle.resolve_switch_screen(emu)
    assert battle._u16(emu, battle.ACTIVE_HP) > 0          # a healthy mon is out
    emu.close()


@pytest.mark.skipif(not (ROM.exists() and CAPTURE.exists()), reason="ROM / capture fixture not present")
def test_a_catch_is_not_nicknamed_aaaa():
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.games.pokemon_red import battle_actions
    from pokemon_agent.games.pokemon_red.game_state import read_party
    for attempt in range(6):
        emu = PyBoyEmulator(str(ROM), window="null")
        emu.load_state(CAPTURE)
        emu.tick(6 + attempt * 3)
        emu.write_memory(0xCFE6, 0); emu.write_memory(0xCFE7, 1)   # enemy at 1 HP: a near-certain catch
        r = battle_actions.throw_ball(emu, "Poke Ball", max_advance=150)
        if r.get("caught"):
            party = read_party(emu)
            new = party[-1]
            assert "AAAA" not in new["nickname"] and new["nickname"].upper() == new["species"].upper()
            emu.close()
            return
        emu.close()
    pytest.skip("no catch in 6 tries")


def test_loop_declines_a_nickname_prompt_it_sees():
    from pokemon_agent.games.pokemon_red import menus

    class Scr:
        def __init__(self):
            self.pressed = []
    s = Scr()
    assert menus.nickname_action("Do you want to give a nickname to SPEAROW?") == "B"
    assert menus.nickname_action("A B C D E F G H I\nJ K L M N O P Q R\nS T U V W X Y Z") == "START"
    assert menus.nickname_action("Wild ZUBAT appeared!") is None


def test_party_screen_nicknames_are_not_mistaken_for_the_keyboard():
    from pokemon_agent.games.pokemon_red import menus
    party = "WARTORTLE 21 FNT\n 0 58\nAAAAAAAAAA5\n 19 19\nAAAAAAAAAA9\nAAAAAAAAAA6\nBring out which POKEMON?"
    assert menus.nickname_action(party) is None


# ---- A: per-move effectiveness for the move chooser (cerulean-team4: Water Gun into Misty's Water types) ----
class BattleMem(Mem):
    def __init__(self, moves, pp, my_types, enemy_types):
        super().__init__(moves, pp)
        self.m.update({0xD019: my_types[0], 0xD01A: my_types[1], 0xCFEA: enemy_types[0], 0xCFEB: enemy_types[1]})


def test_move_analysis_vs_starmie_prefers_tackle_over_resisted_water():
    WATER, PSYCHIC = 0x15, 0x18
    emu = BattleMem([33, 39, 145, 55], [35, 30, 30, 25], (WATER, WATER), (WATER, PSYCHIC))
    a = {m["move"]: m for m in battle.move_analysis(emu)}
    assert a["Water Gun"]["effectiveness"] == 0.5 and a["Water Gun"]["stab"] is True
    assert a["Tackle"]["effectiveness"] == 1.0 and a["Tail Whip"]["expected"] == 0
    best = max(a.values(), key=lambda m: m["expected"])
    assert best["move"] == "Tackle"


def test_choose_move_shows_the_numbers_to_jev(monkeypatch):
    WATER, PSYCHIC = 0x15, 0x18
    emu = BattleMem([33, 55], [35, 25], (WATER, WATER), (WATER, PSYCHIC))
    seen = {}

    class Client:
        def system_one(self, state, questions):
            seen.update(state=state, criteria=questions["move"].criteria)
            return SimpleNamespace(answers={"move": SimpleNamespace(choice="0", confidence=0.9)})
    monkeypatch.setattr(battle_agent, "battle_state_summary", lambda e: {})
    battle_agent.choose_move(Client(), emu)
    assert "0.5x" in seen["criteria"]["1"] and "expected" in seen["criteria"]["1"]
    assert seen["state"]["move_analysis"][0]["move"] == "Tackle"


# ---- verify-train2: 'Delete an older move to make room for BITE?' — the move-list back-out cancelled it ----
LEARN = Path("runs/verify-train2-20260923/states/map65_step8.state")


def test_forget_choice_is_the_weakest_move():
    # Tackle 35, Tail Whip (status), Bubble 20, Water Gun 40 -> forget Tail Whip for Bite (60)
    assert battle.move_to_forget([33, 39, 145, 55]) == 1
    assert battle.worth_learning("BITE", [33, 39, 145, 55]) is True
    assert battle.worth_learning("TAIL WHIP", [33, 145, 55, 44]) is False


@pytest.mark.skipif(not (ROM.exists() and LEARN.exists()), reason="ROM / learn-move state not present")
def test_learns_bite_by_forgetting_the_weakest_move():
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.emulator.interface import GameButton
    emu = PyBoyEmulator(str(ROM), window="null")
    emu.load_state(LEARN)
    emu.tick(2)
    for _ in range(30):
        if "Bite" in battle.active_moves(emu):
            break
        if not battle.handle_learn_move(emu):
            emu.press(GameButton.A)
            emu.tick(30)
    assert "Bite" in battle.active_moves(emu) and "Tail Whip" not in battle.active_moves(emu)
    emu.close()
