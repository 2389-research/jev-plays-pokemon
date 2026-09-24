"""Reorder the party through the in-game menu (START > POKEMON > <mon> > SWITCH > slot 1).

Used to put L1's chosen LEAD at the front so it starts battles and earns EXP (training the bench).
Every press is held (1-frame taps can alias with the game's joypad sampling); the result is verified
from RAM (party order), never assumed.
"""
from __future__ import annotations

from ...emulator.interface import Emulator, GameButton
from . import menus
from .game_state import read_party

WCURMENUITEM = 0xCC26


def _hold(emu: Emulator, button: GameButton, hold: int = 10, settle: int = 40) -> None:
    emu.hold(button)
    emu.tick(hold)
    emu.release(button)
    emu.tick(settle)


def clear_text(emu: Emulator, presses: int = 6) -> None:
    """Close any leftover dialogue / menu (B) so the START menu can open."""
    for _ in range(presses):
        _hold(emu, GameButton.B)
    emu.tick(40)


def _cursor_to(emu: Emulator, index: int, tries: int = 8) -> None:
    for _ in range(tries):
        cur = emu.read_memory(WCURMENUITEM)
        if cur == index:
            return
        _hold(emu, GameButton.UP if cur > index else GameButton.DOWN, 6, 14)


def swap_to_front(emu: Emulator, slot: int) -> bool:
    """Move party member ``slot`` to the front. True when RAM confirms it; always backs out of the menus."""
    party = read_party(emu)
    if not (0 < slot < len(party)):
        return slot == 0
    want = party[slot].get("nickname") or party[slot].get("species")
    try:
        _hold(emu, GameButton.START)
        if not menus.menu_open(emu):
            return False
        text = menus.screen_text(emu)
        _cursor_to(emu, 1 if "DEX" in text else 0)          # POKEMON sits below POKEDEX once you have it
        _hold(emu, GameButton.A)
        emu.tick(20)
        _cursor_to(emu, slot)
        _hold(emu, GameButton.A)                               # the mon's submenu (field moves / STATS / SWITCH / CANCEL)
        for _ in range(6):                                     # find SWITCH by the cursor row, not a fixed index
            _cx, cy = menus.cursor_pos(emu)
            rows = menus.screen_text(emu).splitlines()
            if cy is not None and cy < len(rows) and "SWITCH" in rows[cy]:
                break
            _hold(emu, GameButton.DOWN, 6, 14)
        _hold(emu, GameButton.A)                               # "Move POKEMON where?"
        _cursor_to(emu, 0)
        _hold(emu, GameButton.A)
        emu.tick(30)
    finally:
        for _ in range(3):
            _hold(emu, GameButton.B)
    now = read_party(emu)
    return bool(now) and (now[0].get("nickname") or now[0].get("species")) == want
