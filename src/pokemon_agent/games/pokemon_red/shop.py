"""Deterministic Mart BUY macro (P1 of the battle/shop subsystem — design §6.1).

Built on the generic `menus.py` primitives. From an open Mart counter (the clerk's
"…may I help you?" dialogue OR the BUY/SELL/QUIT menu), `shop_buy` drives one purchase:

    advance → BUY → pick item → set quantity → confirm YES → back out to the overworld.

**Fixed vs. looked-up indices (design §3).** Structural menu indices are constant and
encoded here — BUY is the root menu's top option (index 0), YES is the yes/no top option
(handled by `menus.answer_yesno`). But the index of a *specific item* in the BUY list is
DATA-DEPENDENT (a mart's list order varies), so it is resolved every call from the live
shop list read from RAM (`wListPointer`), never hardcoded (`resolve_shop_index`).

Every outcome is RAM-checkable by the existing readers (`read_items`, `read_money`), so
the macro carries no verification of its own — the caller/tests assert against RAM.
"""
from __future__ import annotations

from ...emulator.interface import Emulator, GameButton
from . import menus
from .constants import ITEMS
from .game_state import _decode_byte, resolve_item_id

WTILEMAP = 0xC3A0
# wListPointer — address of the currently-displayed list menu's item list. While the BUY
# item list is up it points at the mart's for-sale list (count byte, then item ids, 0xFF end).
WLISTPOINTER = 0xCF8B


def _norm(s: object) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _tilemap_text(emu: Emulator) -> str:
    return "\n".join(
        "".join(_decode_byte(emu.read_memory(WTILEMAP + r * 20 + c)) for c in range(20))
        for r in range(18)
    )


def at_shop_menu(emu: Emulator) -> bool:
    """True when the Mart root BUY/SELL/QUIT menu is on screen (the deterministic signal the
    executive uses to fire the buy macro)."""
    if not menus.menu_open(emu):
        return False
    t = _tilemap_text(emu)
    return "BUY" in t and ("SELL" in t or "QUIT" in t)


def read_shop_list(emu: Emulator) -> list[dict]:
    """The mart's for-sale list in menu order, read from `wListPointer` RAM. Only meaningful
    while the BUY item list is displayed. Returns `[{"id", "item"}, …]`."""
    ptr = emu.read_memory(WLISTPOINTER) | (emu.read_memory(WLISTPOINTER + 1) << 8)
    if not (0xC000 <= ptr <= 0xFFFF):
        return []
    count = emu.read_memory(ptr)
    if count > 20:
        return []
    out: list[dict] = []
    for i in range(count):
        iid = emu.read_memory(ptr + 1 + i)
        if iid == 0xFF:
            break
        out.append({"id": iid, "item": ITEMS.get(iid, f"#{iid}")})
    return out


def resolve_shop_index(shop_items: list, item_name: str) -> int | None:
    """The CURRENT index of `item_name` in the shop's ordered list (data-dependent, §3).

    Accepts a list of `{"id","item"}` dicts (as `read_shop_list` returns) or plain name
    strings. Matches by canonical item id when resolvable, then by normalized name, then by
    substring; returns None when the item isn't sold here."""
    target_id = resolve_item_id(item_name)
    key = _norm(item_name)

    def name_of(entry: object) -> str:
        return entry.get("item") if isinstance(entry, dict) else entry  # type: ignore[return-value]

    def id_of(entry: object):
        return entry.get("id") if isinstance(entry, dict) else None

    for i, entry in enumerate(shop_items):
        if target_id is not None and id_of(entry) == target_id:
            return i
        if key and _norm(name_of(entry)) == key:
            return i
    for i, entry in enumerate(shop_items):
        if key and key in _norm(name_of(entry)):
            return i
    return None


def _wait_menu(emu: Emulator, tries: int = 8) -> bool:
    for _ in range(tries):
        emu.tick(8)
        if menus.menu_open(emu):
            return True
    return menus.menu_open(emu)


def close_shop(emu: Emulator, tries: int = 8) -> None:
    """Back out of any shop sub-menu to the overworld — press B until the counter menu is gone and no
    menu is open. Used on EVERY exit path (success and failure) so we never leave a menu half-open for
    the next loop step to re-enter (a not-sold item or an unaffordable qty otherwise stalls the counter)."""
    for _ in range(tries):
        if not at_shop_menu(emu) and not menus.menu_open(emu):
            return
        menus.cancel(emu)
        emu.tick(12)


_close_shop = close_shop   # internal name used by the macro's exit paths


def shop_buy(emu: Emulator, item_name: str, qty: int, *, max_advance: int = 14) -> dict:
    """Buy `qty` of `item_name` from an open Mart counter. Deterministic keypress macro:

    1. advance the clerk dialogue until the BUY/SELL/QUIT menu is up (no-op if already there);
    2. select BUY (fixed index 0) → the item list opens;
    3. resolve the item's current index from the live shop list (RAM) → select it;
    4. quantity selector (starts at 1): press UP `qty-1` times → A;
    5. advance to the "That'll be $N. OK?" YES/NO prompt → answer YES;
    6. back out (B) through the item list and root menu to the overworld.

    Returns `{"ok": True, "item", "qty", "index", "price"}` on success, else
    `{"ok": False, "reason": …}`. Outcomes are verified by the caller via `read_items` /
    `read_money` (design §5) — this macro asserts nothing itself."""
    qty = max(1, int(qty))

    # 1) advance the counter dialogue to the root BUY/SELL/QUIT menu
    for _ in range(max_advance):
        if at_shop_menu(emu):
            break
        menus.advance(emu)
        emu.tick(6)
    if not at_shop_menu(emu):
        return {"ok": False, "reason": "shop menu did not open"}

    # 2) BUY is the root menu's top option (fixed structural index)
    menus.select_option(emu, 0)
    _wait_menu(emu)

    # 3) resolve the target item's CURRENT index from the live shop list (data-dependent, §3)
    shop = read_shop_list(emu)
    idx = resolve_shop_index(shop, item_name)
    if idx is None:
        _close_shop(emu)
        return {"ok": False, "reason": f"{item_name!r} not sold here",
                "shop": [s["item"] for s in shop]}

    # 4) select the item → the quantity selector opens
    menus.select_option(emu, idx)
    emu.tick(20)

    # 5) quantity: the selector starts at 1, press UP (qty-1) times, then A to confirm
    for _ in range(qty - 1):
        emu.press(GameButton.UP)
        emu.tick(18)
    emu.press(GameButton.A)
    emu.tick(28)

    # 6) advance to the "That'll be $N. OK?" prompt and answer YES
    confirmed = False
    for _ in range(max_advance):
        m = menus.read_menu(emu)
        if m.get("open") and m.get("kind") == "yesno":
            menus.answer_yesno(emu, True)
            emu.tick(30)
            confirmed = True
            break
        menus.advance(emu)
    if not confirmed:
        # Insufficient funds (or the qty was rejected) — no YES/NO ever appeared. Back all the way
        # out so the next loop step doesn't re-enter a half-open counter (#4).
        _close_shop(emu)
        return {"ok": False, "reason": "no purchase confirmation appeared"}

    # 7) advance the "Here you are! Thank you!" text until the item list is interactive again,
    #    then back out through the item list and root menu to leave the shop (SEE YA / QUIT).
    for _ in range(max_advance):
        emu.tick(6)
        if menus.cursor_pos(emu) is not None:
            break
        menus.advance(emu)
    _close_shop(emu)   # item list -> root -> close the counter (every exit path funnels here, #3)
    return {"ok": True, "item": item_name, "qty": qty, "index": idx}
