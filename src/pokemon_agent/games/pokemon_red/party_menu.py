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


WPARTYMON0 = 0xD16B      # first party struct (44 bytes each); moves at +8..+11
PARTY_STRUCT = 44


def _back_out(emu: Emulator, tries: int = 16) -> None:
    """B until no menu (and no text box) is left on screen — a fixed count leaves the bag open after a
    'learned CUT!' message swallows presses."""
    from .game_state import read_screen_text
    for _ in range(tries):
        if not menus.menu_open(emu) and not read_screen_text(emu)[1]:
            return
        _hold(emu, GameButton.B)


def _party_move_ids(emu: Emulator, slot: int) -> list[int]:
    return [emu.read_memory(WPARTYMON0 + slot * PARTY_STRUCT + 8 + i) for i in range(4)]


def knows(emu: Emulator, slot: int, move: str) -> bool:
    party = read_party(emu)
    if not (0 <= slot < len(party)):
        return False
    def norm(m):
        return "".join(ch for ch in str(m).split(" (")[0].lower() if ch.isalnum())
    return any(norm(m) == norm(move) for m in party[slot].get("moves") or [])


def _open_start_entry(emu: Emulator, entry: str) -> bool:
    """START, then the POKEMON or ITEM entry (their index shifts once you have a Pokédex)."""
    for _ in range(3):                    # a press during the last step's walk/turn animation is ignored
        emu.tick(24)
        _hold(emu, GameButton.START)
        if menus.menu_open(emu):
            break
    else:
        return False
    dex = "DEX" in menus.screen_text(emu)
    _cursor_to(emu, {"POKEMON": 0, "ITEM": 1}[entry] + (1 if dex else 0))
    _hold(emu, GameButton.A)
    emu.tick(20)
    return True


def teach(emu: Emulator, item_index: int, slot: int, move: str, forget: str | None = None) -> tuple[bool, str]:
    """Teach the TM/HM at bag position ``item_index`` to party ``slot`` (START > ITEM > it > USE > the
    Pokémon). With 4 moves known it forgets ``forget`` (a move name) or else the weakest non-HM move.
    Success is read from RAM (the move is in the Pokémon's move list), never assumed; always backs out."""
    from .battle import move_to_forget
    from .constants import MOVES
    from .tmhm import HMS
    if knows(emu, slot, move):
        return True, "already knows it"
    detail = "the menus didn't reach the teach prompt"
    try:
        if not _open_start_entry(emu, "ITEM"):
            return False, "the START menu didn't open"
        for _ in range(24):                                    # to the top of the bag list
            _hold(emu, GameButton.UP, 6, 12)
        for _ in range(item_index):
            _hold(emu, GameButton.DOWN, 6, 12)
        if emu.read_memory(WCURMENUITEM) + emu.read_memory(0xCC36) != item_index:   # cursor + list scroll
            return False, "couldn't select the item in the bag"
        _hold(emu, GameButton.A)                               # USE / TOSS
        _hold(emu, GameButton.A)                               # USE
        for _ in range(30):
            if knows(emu, slot, move):
                return True, f"learned {move}"
            text = " ".join(menus.screen_text(emu).split())
            if "which POK" in text:                            # the party list, each marked ABLE / NOT ABLE
                marks = [m for m in text.replace("NOT ABLE", "NOT_ABLE").split() if m in ("ABLE", "NOT_ABLE")]
                if slot < len(marks) and marks[slot] == "NOT_ABLE":
                    return False, f"that Pokémon can't learn {move} (the game marks it NOT ABLE)"
                _cursor_to(emu, slot)
                _hold(emu, GameButton.A)
            elif ("make room for" in text or "should be forgotten" in text or "Which move should" in text
                  or ("YES" in text and "NO" in text)) and not menus.menu_open(emu):
                emu.tick(30)                             # a YES/NO box / move list draws after its text finishes
            elif "make room for" in text:
                menus.answer_yesno(emu, True)
            elif "should be forgotten" in text or "Which move should" in text:
                ids = _party_move_ids(emu, slot)
                names = [str(MOVES.get(i, "")).lower() for i in ids]
                idx = names.index(forget.lower()) if forget and forget.lower() in names else None
                if idx is None:
                    keep = [i for i, n in enumerate(names) if n.title() not in HMS]
                    idx = keep[move_to_forget([ids[i] for i in keep])] if keep else 0
                menus.select_option(emu, idx, max_options=4)
            elif "not compatible" in text:          # NOT "can't learn more than 4 moves" (that's the forget flow)
                return False, f"that Pokémon can't learn {move}"
            elif "YES" in text and "NO" in text:
                menus.answer_yesno(emu, True)                  # "Teach CUT to a POKéMON?"
            else:
                _hold(emu, GameButton.A)                       # "Booted up an HM!" / "It contained CUT!" ...
        detail = "no confirmation that the move was learned"
    finally:
        _back_out(emu)
    return knows(emu, slot, move), detail


def use_field_move(emu: Emulator, slot: int, move: str) -> tuple[bool, str]:
    """Use a field move from the party menu (START > POKEMON > the Pokémon > the move): the player must
    already be facing what it acts on. Returns (the game accepted it, the text it showed)."""
    said = ""
    try:
        if not _open_start_entry(emu, "POKEMON"):
            return False, "the START menu didn't open"
        _cursor_to(emu, slot)
        _hold(emu, GameButton.A)                               # the submenu: field moves / STATS / SWITCH / CANCEL
        for _ in range(6):
            _cx, cy = menus.cursor_pos(emu) or (None, None)
            rows = menus.screen_text(emu).splitlines()
            if cy is not None and cy < len(rows) and move.upper() in rows[cy].upper():
                break
            _hold(emu, GameButton.DOWN, 6, 14)
        else:
            return False, f"{move} isn't in that Pokémon's menu"
        _hold(emu, GameButton.A)
        for _ in range(8):
            emu.tick(30)
            text = " ".join(menus.screen_text(emu).split())
            if any(k in text for k in ("hacked away", "isn't anything", "BADGE is required", "No!")):
                said = text
                break
        ok = "hacked away" in said or ("used" in said and move.upper() in said.upper())
        return ok, said or "no reply"
    finally:
        _back_out(emu)
