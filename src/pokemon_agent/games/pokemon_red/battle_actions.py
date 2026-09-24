"""Deterministic in-battle action macros (P2 of the battle subsystem — design §6.2).

Built on the generic `menus.py` primitives and the `battle.py` readers. Three macros —
`throw_ball`, `use_item`, `run` — each a fixed keypress sequence with a RAM-checkable
outcome (design §5), so tests assert against RAM, not screenshots.

**The battle menu is a 2×2 grid, not a vertical list:**

    FIGHT   PKMN
    ITEM    RUN

The cursor starts on FIGHT (top-left), so `menus.select_option` (which walks a VERTICAL
list) does NOT apply here — the 2×2 is navigated with directional presses. Verified
empirically on the ROM (a Viridian-Forest wild battle):

    from FIGHT:  DOWN -> ITEM,  RIGHT -> PKMN,  DOWN then RIGHT -> RUN;
    UP and LEFT at FIGHT are edge no-ops (no wrap), so pressing UP+LEFT always
    re-homes the cursor to FIGHT regardless of where a previous turn left it.

The bag / item sub-menu that opens on ITEM→A IS a vertical list, so `select_option`
drives it — with the item's index resolved at runtime from the live bag (§3), never
hardcoded (`resolve_item_index`).

**Fixed vs. looked-up indices (design §3).** The 2×2 structural moves are constant and
encoded here (pinned by a fixture test, §5). But the index of a *specific item* — the
ball in `throw_ball`, the Potion in `use_item` — is DATA-DEPENDENT (bag order varies),
so it is resolved every call from `game_state.read_items`. When a name matches several
bag entries the macro surfaces all candidates and picks the first (a later phase lets
Jev choose); no accent-folding or fuzzy heuristics — mechanics stay deterministic.
"""
from __future__ import annotations

from ...emulator.interface import Emulator, GameButton
from . import battle, menus
from .game_state import read_items, read_party, resolve_item_id


def _norm(s: object) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _name_of(entry: object) -> str:
    return entry.get("item") if isinstance(entry, dict) else entry  # type: ignore[return-value]


def resolve_item_index(items: list, item_name: str) -> dict | None:
    """The CURRENT index of `item_name` in the live bag order (data-dependent, §3).

    `items` is a list of `{"item","qty"}` dicts (as `read_items` returns) or plain name
    strings, in bag/menu order. Matching is tiered — canonical item id, then normalized
    name equality, then substring — and every plausible match is collected as a candidate
    (in priority order, de-duplicated) so a later phase (Jev) can disambiguate. Returns
    `{"index", "item", "candidates": [{"index","item"}, …]}` (index = candidates[0]) or
    None when the item isn't in the bag. No accent-folding heuristics."""
    target_id = resolve_item_id(item_name)
    key = _norm(item_name)

    tier_id, tier_name, tier_sub = [], [], []
    for i, entry in enumerate(items):
        name = _name_of(entry)
        if target_id is not None and resolve_item_id(name) == target_id:
            tier_id.append(i)
        elif key and _norm(name) == key:
            tier_name.append(i)
        elif key and key in _norm(name):
            tier_sub.append(i)

    ordered: list[int] = []
    for tier in (tier_id, tier_name, tier_sub):
        for i in tier:
            if i not in ordered:
                ordered.append(i)
    if not ordered:
        return None
    candidates = [{"index": i, "item": _name_of(items[i])} for i in ordered]
    return {"index": ordered[0], "item": _name_of(items[ordered[0]]), "candidates": candidates}


def read_items_qty(emu: Emulator, name: str) -> int:
    """Live quantity of `name` in the bag by name (survives index shifts after a catch)."""
    key = _norm(name)
    return next((i["qty"] for i in read_items(emu) if _norm(i["item"]) == key), 0)


def _press(emu: Emulator, button: GameButton, settle: int = 24) -> None:
    emu.press(button)
    emu.tick(settle)


def _ensure_fight_menu(emu: Emulator, max_advance: int = 20) -> bool:
    """Advance intro / result text until the FIGHT/PKMN/ITEM/RUN menu is up and interactive.

    Guards against pressing A when the menu is ALREADY showing (which would open FIGHT's
    move list); it only presses A while the menu is not yet interactive."""
    for _ in range(max_advance):
        if battle.fight_menu_showing(emu) and menus.menu_open(emu):
            return True
        menus.advance(emu)
        emu.tick(6)
    return battle.fight_menu_showing(emu) and menus.menu_open(emu)


# The 2×2 battle menu, as directional moves FROM the FIGHT (top-left) home position.
_BATTLE_MENU_MOVES = {
    "FIGHT": (),
    "PKMN": (GameButton.RIGHT,),
    "ITEM": (GameButton.DOWN,),
    "RUN": (GameButton.DOWN, GameButton.RIGHT),
}


def _goto_battle_option(emu: Emulator, option: str) -> None:
    """Home the 2×2 cursor to FIGHT (UP+LEFT are no-op edges), then step to `option`."""
    _press(emu, GameButton.UP, 16)
    _press(emu, GameButton.LEFT, 16)
    for button in _BATTLE_MENU_MOVES[option]:
        _press(emu, button, 20)


def _drain_turn(emu: Emulator, *, max_advance: int, done) -> None:
    """Press A to advance the turn's animation/result text until `done(emu)` is true, the
    FIGHT menu returns, or the battle ends."""
    for _ in range(max_advance):
        if done(emu) or not battle.in_battle(emu):
            return
        if battle.fight_menu_showing(emu) and menus.menu_open(emu):
            return
        if menus.handle_nickname(emu):      # after a catch: answer NO, never type "AAAAAAAAAA"
            continue
        _press(emu, GameButton.A, 30)


def throw_ball(emu: Emulator, ball_name: str, *, max_advance: int = 30) -> dict:
    """Throw `ball_name` at the wild Pokémon from the battle menu. Deterministic macro:

    1. ensure the FIGHT/PKMN/ITEM/RUN menu is up;
    2. resolve the ball's CURRENT bag index at runtime (`read_items`, §3);
    3. 2×2: home to FIGHT → DOWN to ITEM → A (open the bag);
    4. `select_option(index)` picks the ball → the throw fires;
    5. advance the throw animation/text until control returns or the battle ends.

    Outcome (RAM-checkable by the caller): ball count −1; on a successful catch party +1
    and `in_battle` false. Returns `{"ok", "item", "index", "candidates", "balls_before",
    "balls_after", "party_before", "party_after", "caught", "battle_over"}`, or
    `{"ok": False, "reason": …}` (e.g. ball not in bag — nothing pressed)."""
    if not battle.in_battle(emu):
        return {"ok": False, "reason": "not in battle"}
    if not _ensure_fight_menu(emu):
        return {"ok": False, "reason": "battle menu did not open"}

    match = resolve_item_index(read_items(emu), ball_name)
    if match is None:
        return {"ok": False, "reason": f"{ball_name!r} not in bag",
                "bag": [i["item"] for i in read_items(emu)]}

    balls_before = read_items_qty(emu, ball_name)
    party_before = len(read_party(emu))

    _goto_battle_option(emu, "ITEM")
    _press(emu, GameButton.A, 40)              # open the bag (vertical list)
    menus.select_option(emu, match["index"], max_options=20)  # pick the ball -> throws
    emu.tick(30)

    _drain_turn(emu, max_advance=max_advance,
                done=lambda e: not battle.in_battle(e))

    balls_after = read_items_qty(emu, ball_name)
    party_after = len(read_party(emu))
    return {
        "ok": True,
        "item": match["item"],
        "index": match["index"],
        "candidates": match["candidates"],
        "balls_before": balls_before,
        "balls_after": balls_after,
        "party_before": party_before,
        "party_after": party_after,
        "caught": party_after > party_before,
        "battle_over": not battle.in_battle(emu),
    }


def use_item(emu: Emulator, item_name: str, *, max_advance: int = 30) -> dict:
    """Use `item_name` from the bag in battle on the active (first) Pokémon. Macro:

    1. ensure the battle menu is up;
    2. resolve the item's CURRENT bag index (`read_items`, §3);
    3. 2×2: home to FIGHT → DOWN to ITEM → A (open the bag);
    4. `select_option(index)` picks the item; if it needs a target (e.g. a Potion) the
       party screen opens with the active mon highlighted — the advance-loop's first A
       selects it — then the effect text advances until control returns.

    Outcome (RAM-checkable): item count −1 and its effect (a Potion raises current HP).
    Returns `{"ok", "item", "index", "candidates", "qty_before", "qty_after",
    "active_hp_before", "active_hp_after"}` or `{"ok": False, "reason": …}`."""
    if not battle.in_battle(emu):
        return {"ok": False, "reason": "not in battle"}
    if not _ensure_fight_menu(emu):
        return {"ok": False, "reason": "battle menu did not open"}

    match = resolve_item_index(read_items(emu), item_name)
    if match is None:
        return {"ok": False, "reason": f"{item_name!r} not in bag",
                "bag": [i["item"] for i in read_items(emu)]}

    def active_hp() -> int:
        party = read_party(emu)
        return party[0]["hp"] if party else 0

    qty_before = read_items_qty(emu, item_name)
    hp_before = active_hp()

    _goto_battle_option(emu, "ITEM")
    _press(emu, GameButton.A, 40)              # open the bag
    menus.select_option(emu, match["index"], max_options=20)  # pick the item
    emu.tick(30)

    # advance: the first A selects the target mon on the party screen, the rest advance
    # the effect text, until the item is consumed and the FIGHT menu returns.
    _drain_turn(emu, max_advance=max_advance,
                done=lambda e: read_items_qty(e, item_name) < qty_before)

    return {
        "ok": True,
        "item": match["item"],
        "index": match["index"],
        "candidates": match["candidates"],
        "qty_before": qty_before,
        "qty_after": read_items_qty(emu, item_name),
        "active_hp_before": hp_before,
        "active_hp_after": active_hp(),
    }


def run(emu: Emulator, *, max_advance: int = 20) -> dict:
    """Attempt to flee the wild battle. Macro: ensure the menu is up → 2×2 home to FIGHT →
    DOWN, RIGHT to RUN → A → advance the result text.

    A run can FAIL and cost the turn (the enemy attacks and the menu returns), so the
    outcome is reported both ways. Returns `{"ok", "escaped", "battle_over"}` — `escaped`
    (== `in_battle` now false) is the RAM-checkable success signal."""
    if not battle.in_battle(emu):
        return {"ok": False, "reason": "not in battle"}
    if not _ensure_fight_menu(emu):
        return {"ok": False, "reason": "battle menu did not open"}

    _goto_battle_option(emu, "RUN")
    _press(emu, GameButton.A, 40)
    _drain_turn(emu, max_advance=max_advance,
                done=lambda e: not battle.in_battle(e))

    escaped = not battle.in_battle(emu)
    return {"ok": True, "escaped": escaped, "battle_over": escaped}
