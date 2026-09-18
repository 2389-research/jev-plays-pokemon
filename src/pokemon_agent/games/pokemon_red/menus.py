"""Read and operate Pokémon Red menus (P1).

Menu detection is grounded in the CURSOR ARROW tile (0xED, the "▶"): it appears in
the tilemap exactly at the selected option and is absent in the free overworld and on
graphics/cutscene screens — so its presence is a reliable "a menu is open" signal, and
its row is the current selection. `wCurrentMenuItem`/`wMaxMenuItem` (CC26/CC28) give
the cursor index and option count, but persist stale when no menu is up, so they are
only trusted once the arrow confirms a menu is open.

Selection is done robustly WITHOUT depending on the CC26 index base (which varies by
menu): go to the top of the list (press UP n times), then press DOWN `index` times,
then A. Timing needs a generous settle (the menu engine polls input on specific frames).
"""
from __future__ import annotations

from ...emulator.interface import Emulator, GameButton
from .game_state import WTILEMAP, _decode_byte

CURSOR_TILE = 0xED
WISINBATTLE = 0xD057
WCURMENUITEM = 0xCC26
WMAXMENUITEM = 0xCC28


def _tiles(emu: Emulator) -> list[int]:
    return [emu.read_memory(WTILEMAP + i) for i in range(20 * 18)]


def _rows_text(emu: Emulator) -> list[str]:
    return ["".join(_decode_byte(emu.read_memory(WTILEMAP + r * 20 + c)) for c in range(20)) for r in range(18)]


def menu_open(emu: Emulator) -> bool:
    """True if a selectable menu is on screen (the cursor arrow is drawn)."""
    return any(emu.read_memory(WTILEMAP + i) == CURSOR_TILE for i in range(20 * 18))


def cursor_pos(emu: Emulator) -> tuple[int, int] | None:
    for i in range(20 * 18):
        if emu.read_memory(WTILEMAP + i) == CURSOR_TILE:
            return (i % 20, i // 20)
    return None


def _looks_like_name_entry(rows: list[str]) -> bool:
    # the name-entry keyboard is a grid of single letters — several rows each with many
    # spread-out single alpha chars (A B C D E ...). Heuristic: >=2 rows with >=7 letters.
    lettery = sum(1 for r in rows if sum(ch.isalpha() for ch in r) >= 7)
    return lettery >= 2


def read_menu(emu: Emulator) -> dict:
    """Structured menu state. `open` False means no menu is up (fields omitted)."""
    if not menu_open(emu):
        return {"open": False}
    rows = _rows_text(emu)
    joined = "\n".join(rows)
    cx, cy = cursor_pos(emu)
    # options: the non-empty text lines to the right of the cursor column (best-effort)
    options = [rows[r][cx:].strip() for r in range(18) if rows[r][cx:].strip()]
    if emu.read_memory(WISINBATTLE):
        kind = "battle"
    elif "YES" in joined and "NO" in joined:
        kind = "yesno"
        options = ["YES", "NO"]  # canonical: index 0 = YES, 1 = NO
    elif _looks_like_name_entry(rows):
        kind = "name_entry"
    else:
        kind = "list"
    return {
        "open": True,
        "kind": kind,
        "cursor_xy": (cx, cy),
        "cursor_index": emu.read_memory(WCURMENUITEM),   # trust only while a menu is open
        "num_options": emu.read_memory(WMAXMENUITEM) + 1,
        "options": options[:8],
    }


def _press(emu: Emulator, button: GameButton, settle: int = 26) -> None:
    emu.press(button)
    emu.tick(settle)


def select_option(emu: Emulator, index: int, *, max_options: int = 8) -> dict:
    """Move a vertical LIST menu's cursor to `index` (0=top) and confirm with A.

    Base-agnostic: goes to the top of the list, then down `index`. Returns the menu
    read taken just before confirming (so the caller can see what it selected)."""
    if not menu_open(emu):
        return {"ok": False, "reason": "no menu open"}
    index = max(0, index)
    for _ in range(max_options):          # guarantee cursor at the top
        _press(emu, GameButton.UP, 18)
    for _ in range(index):                # step down to the target option
        _press(emu, GameButton.DOWN, 18)
    chosen = read_menu(emu)
    _press(emu, GameButton.A, 30)         # confirm
    return {"ok": True, "selected_index": index, "menu": chosen}


def answer_yesno(emu: Emulator, yes: bool) -> dict:
    """YES is the top option (index 0), NO is index 1."""
    return select_option(emu, 0 if yes else 1)


def advance(emu: Emulator) -> None:
    _press(emu, GameButton.A, 20)


def cancel(emu: Emulator) -> None:
    _press(emu, GameButton.B, 20)
