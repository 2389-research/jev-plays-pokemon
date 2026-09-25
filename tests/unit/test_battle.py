"""Battle reads (pure, no ROM) + a real puppeteer test against a battle fixture."""
from pathlib import Path

import pytest

from pokemon_agent.games.pokemon_red import battle

ROM = Path("roms/pokemon_red.gb")
FIXTURE = Path("states/battle_menu.state")

# Gen-1 font tiles for FIGHT
_UP = {c: 0x80 + (ord(c) - ord("A")) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}


class MemFake:
    def __init__(self, mem=None):
        self.mem = dict(mem or {})

    def read_memory(self, addr, bank=None):
        return self.mem.get(addr, 0)


def test_reads_active_moves_and_enemy_hp():
    mem = {0xD057: 2, 0xD01C: 33, 0xD01D: 39, 0xCFE6: 0, 0xCFE7: 18}
    emu = MemFake(mem)
    assert battle.in_battle(emu)
    assert battle.move_count(emu) == 2
    assert battle.active_moves(emu) == ["Tackle", "Tail Whip"]
    assert battle.enemy_hp(emu) == 18


def test_fight_menu_detection():
    from pokemon_agent.games.pokemon_red.battle import WTILEMAP
    mem = {WTILEMAP + 14 * 20 + i: _UP[c] for i, c in enumerate("FIGHT")}
    assert battle.fight_menu_showing(MemFake(mem)) is True
    assert battle.fight_menu_showing(MemFake({})) is False


def test_use_move_rejects_when_not_in_battle():
    assert battle.use_move(MemFake({0xD057: 0}))["ok"] is False


def test_bot_chooses_a_move_slot():
    from types import SimpleNamespace

    from pokemon_agent.games.pokemon_red import battle_agent

    class FakeClient:
        def __init__(self, choice):
            self.choice = choice
            self.state = None

        def system_one(self, *, state, questions):
            self.state = state
            ans = SimpleNamespace(choice=self.choice, confidence=0.9)
            return SimpleNamespace(answers={"move": ans})

    emu = MemFake({0xD057: 2, 0xD01C: 33, 0xD01D: 39, 0xD02D: 35, 0xD02E: 40})   # both moves have PP
    client = FakeClient("1")
    slot, conf = battle_agent.choose_move(client, emu)
    assert slot == 1 and conf == 0.9
    # it was handed the real move list to choose from
    assert client.state["your_moves"] == {"0": "Tackle", "1": "Tail Whip"}
    # out-of-range choice is clamped to a valid slot
    assert battle_agent.choose_move(FakeClient("9"), emu)[0] == 1
    # retrieved type-effectiveness knowledge is injected into Jev's decision state
    c2 = FakeClient("0")
    battle_agent.choose_move(c2, emu, type_knowledge=["Water is super effective vs Rock"])
    assert c2.state["type_knowledge"] == ["Water is super effective vs Rock"]


@pytest.mark.skipif(not (ROM.exists() and FIXTURE.exists()),
                    reason="needs the ROM and a real battle fixture (states/battle_menu.state)")
def test_puppeteer_a_turn_deals_damage():
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator

    emu = PyBoyEmulator(str(ROM), window="null", speed=0)
    emu.load_state(FIXTURE)
    emu.tick(3)
    before = battle.enemy_hp(emu)
    result = battle.use_move(emu, 0)  # Tackle
    assert result["ok"] and result["move"] == "Tackle"
    assert result["enemy_hp_before"] == before
    assert result["damage_dealt"] > 0  # Tackle actually damaged the enemy
    emu.close()
