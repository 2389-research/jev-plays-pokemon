"""Puppeteer a Pokémon Red battle from the FIGHT menu.

Validated against a real rival-battle fixture (states/battle_menu.state):
  * enemy mon at 0xCFE5 (species) / 0xCFE6 (HP, BE) / 0xCFF3 (level) / 0xCFF4 (maxHP)
  * active mon at 0xD014 / 0xD015 / 0xD022 / 0xD023, its move ids at 0xD01C..0x1F
  * the FIGHT/PKMN/ITEM/RUN menu opens with FIGHT selected; pressing A opens the
    move list, where the cursor variable CC26 = (move slot + 1), wrapping within the
    live moves. Move slot i is selected when CC26 == i + 1.

Timing matters: the battle engine polls input on specific frames and animates each
turn, so presses need a generous settle (~35-40 frames) between them — spamming A
faster than that gets inputs dropped or cancels the action.
"""
from __future__ import annotations

from ...emulator.interface import Emulator, GameButton
from .constants import MOVES
from .game_state import WTILEMAP, _decode_byte

WISINBATTLE = 0xD057
ENEMY_SPECIES = 0xCFE5
ENEMY_HP = 0xCFE6          # big-endian, 2 bytes
ACTIVE_MOVES = 0xD01C      # 4 move ids of the active battle mon
CC26 = 0xCC26              # move-menu cursor: slot + 1
ACTIVE_PP = 0xD02D         # wBattleMonPP: 4 bytes, low 6 bits = current PP (top 2 = PP Ups)


def _u16(emu: Emulator, addr: int) -> int:
    return (emu.read_memory(addr) << 8) | emu.read_memory(addr + 1)


def in_battle(emu: Emulator) -> bool:
    return emu.read_memory(WISINBATTLE) != 0


def is_trainer_battle(emu: Emulator) -> bool:
    """True for a trainer battle, False for a wild one (or not in battle).

    `wIsInBattle` (0xD057) is 1 for a wild encounter and 2 for a trainer battle
    (0xFF is the Safari lost-battle sentinel). Trainer battles can't be run from and
    are always GRIND/SURVIVE — never CAPTURE — so the battle layer branches on this.
    """
    return emu.read_memory(WISINBATTLE) == 2


def enemy_hp(emu: Emulator) -> int:
    return _u16(emu, ENEMY_HP)


def active_move_ids(emu: Emulator) -> list[int]:
    return [emu.read_memory(ACTIVE_MOVES + i) for i in range(4)]


def active_moves(emu: Emulator) -> list[str]:
    return [MOVES.get(m, f"#{m}") for m in active_move_ids(emu) if m]


def move_count(emu: Emulator) -> int:
    return sum(1 for m in active_move_ids(emu) if m)


def active_pp(emu: Emulator) -> list[int]:
    """Current PP of each of the active mon's moves (same order as active_moves)."""
    return [emu.read_memory(ACTIVE_PP + i) & 0x3F for i in range(move_count(emu))]


def usable_slot(pp: list[int], slot: int) -> int:
    """``slot`` if it still has PP; otherwise the move with the most PP left. With every move at 0
    PP the choice doesn't matter — the game uses Struggle."""
    if not pp or not (0 <= slot < len(pp)) or pp[slot] > 0 or not any(pp):
        return slot
    return max(range(len(pp)), key=lambda i: pp[i])


def move_list_showing(emu: Emulator) -> bool:
    """True when the MOVE LIST (not the root FIGHT/PKMN/ITEM/RUN menu) is on screen — e.g. after the
    game refused a move with "No PP left for this move!"."""
    if fight_menu_showing(emu):
        return False
    names = [m.upper() for m in active_moves(emu)]
    for r in range(8, 18):
        row = "".join(_decode_byte(emu.read_memory(WTILEMAP + r * 20 + c)) for c in range(20))
        if "No PP left" in row or any(n and n in row for n in names):
            return True
    return False


def back_to_fight_menu(emu: Emulator, tries: int = 6) -> bool:
    """Back out of the move list (B) until the root FIGHT menu is up again."""
    for _ in range(tries):
        if fight_menu_showing(emu):
            return True
        _press(emu, GameButton.B, 30)
    return fight_menu_showing(emu)


def fight_menu_showing(emu: Emulator) -> bool:
    """True if the FIGHT/PKMN/ITEM/RUN menu is on screen (turn ready for input)."""
    for r in range(14, 18):
        row = "".join(_decode_byte(emu.read_memory(WTILEMAP + r * 20 + c)) for c in range(20))
        if "FIGHT" in row:
            return True
    return False


def _press(emu: Emulator, button: GameButton, settle: int = 38) -> None:
    emu.press(button)
    emu.tick(settle)


def use_move(emu: Emulator, slot: int = 0, *, max_advance: int = 28) -> dict:
    """From the FIGHT menu, select and execute the active mon's move in `slot`
    (0-based), then advance the turn's messages until control returns or the battle
    ends. Returns a summary including enemy HP before/after and whether it's over.
    """
    if not in_battle(emu):
        return {"ok": False, "reason": "not in battle"}
    n = move_count(emu)
    if n == 0:
        return {"ok": False, "reason": "no moves"}
    slot = max(0, min(slot, n - 1))
    slot = usable_slot(active_pp(emu), slot)   # never select a move the game will refuse (0 PP)
    before = enemy_hp(emu)
    move_name = active_moves(emu)[slot]

    # The FIGHT/PKMN/ITEM/RUN menu is a 2x2 and the cursor can be left on another
    # option from a previous turn (CC26 tracks the row). Force it to FIGHT (top-left)
    # before selecting, or A would open ITEM/RUN and no move fires.
    _press(emu, GameButton.UP, 16)
    _press(emu, GameButton.LEFT, 16)
    _press(emu, GameButton.A, 40)          # FIGHT -> move list
    target = slot + 1                      # CC26 == slot + 1
    for _ in range(n + 2):
        if emu.read_memory(CC26) == target:
            break
        _press(emu, GameButton.DOWN, 28)
    _press(emu, GameButton.A, 50)          # execute the move

    # advance the turn's result text until we're back at the menu or the battle ends
    for _ in range(max_advance):
        if not in_battle(emu) or fight_menu_showing(emu):
            break
        _press(emu, GameButton.A, 40)

    return {
        "ok": True,
        "move": move_name,
        "enemy_hp_before": before,
        "enemy_hp_after": enemy_hp(emu),
        "damage_dealt": max(0, before - enemy_hp(emu)),
        "battle_over": not in_battle(emu),
    }
