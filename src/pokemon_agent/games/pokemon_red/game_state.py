"""Comprehensive Pokémon Red state read from RAM.

Every section is defensive (try/except) and returns partial data rather than
crashing the loop. Addresses are pokered WRAM symbols. Some (battle struct, NPC
slots) are best-effort and only meaningful in the right mode; they're marked.
"""
from __future__ import annotations

from ...emulator.interface import Emulator
from .constants import ITEMS, MOVES, SPECIES, SPRITES

# --- Gen-1 text charmap (for names + on-screen dialog) --------------------
# Kept conservative so non-text tiles decode to nothing (avoids garbage like "PKMN").
_SPECIAL = {   # pokered constants/charmap.asm (tiles as rendered on screen)
    0x7F: " ", 0x4E: " ", 0x9C: ":", 0xE8: ".", 0xE6: "?", 0xE7: "!",
    0xE3: "-", 0xE0: "'", 0xF4: ",", 0x9A: "(", 0x9B: ")", 0x9D: ";", 0x9E: "[", 0x9F: "]",
    0xBA: "é", 0xBB: "'d", 0xBC: "'l", 0xBD: "'s", 0xBE: "'t", 0xBF: "'v", 0xE4: "'r", 0xE5: "'m",
    0xE1: "PK", 0xE2: "MN", 0x75: "…", 0xF3: "/", 0xEF: "♂", 0xF5: "♀", 0xF1: "×", 0xF0: "¥",
    0xF2: ".",
}

_FACING = {0: "south", 4: "north", 8: "west", 12: "east"}

# Sprite entity tables. Map coords live in wSpriteStateData2 (+4 Y, +5 X), absolute
# and in the same frame as the player — correct even for off-screen sprites.
WSPRITE1 = 0xC100  # 16 bytes/sprite; +0 picture id, +9 facing
WSPRITE2 = 0xC200  # 16 bytes/sprite; +4 map Y, +5 map X (each carries a +4 map-border offset)
SPRITE_COORD_OFFSET = 4  # wSpriteStateData2 stores map coords shifted by the 4-tile border
# Missable ("toggleable") objects: the game HIDES them by setting a bit, but leaves the sprite slot
# populated — e.g. the intro Oak in Pallet, the second lab Oak, starter balls already taken.
WMISSABLE_LIST = 0xD5CE   # (sprite slot, missable index) pairs for the current map, 0xFF-terminated
WMISSABLE_FLAGS = 0xD5A6  # bitfield indexed by missable index; bit set = hidden


def hidden_sprite_slots(emu: Emulator) -> set[int]:
    """Sprite slots (1-15) the game has hidden via its missable-object flags. Bounded parse;
    any read error means "nothing hidden" so a bad read can never make real NPCs disappear."""
    hidden: set[int] = set()
    try:
        for i in range(16):
            slot = emu.read_memory(WMISSABLE_LIST + 2 * i)
            if slot == 0xFF:
                break
            idx = emu.read_memory(WMISSABLE_LIST + 2 * i + 1)
            if 1 <= slot <= 15 and idx < 256 and (emu.read_memory(WMISSABLE_FLAGS + idx // 8) >> (idx % 8)) & 1:
                hidden.add(slot)
    except Exception:
        return set()
    return hidden


def _decode_byte(b: int) -> str:
    if 0x80 <= b <= 0x99:
        return chr(ord("A") + b - 0x80)
    if 0xA0 <= b <= 0xB9:
        return chr(ord("a") + b - 0xA0)
    if 0xF6 <= b <= 0xFF:
        return chr(ord("0") + b - 0xF6)
    return _SPECIAL.get(b, "")


def _decode(emu: Emulator, addr: int, length: int, stop_at_terminator: bool = True) -> str:
    out = []
    for i in range(length):
        b = emu.read_memory(addr + i)
        if b == 0x50 and stop_at_terminator:
            break
        out.append(_decode_byte(b))
    return "".join(out).strip()


def _u16(emu: Emulator, addr: int) -> int:  # big-endian, as Gen 1 stores stats
    return (emu.read_memory(addr) << 8) | emu.read_memory(addr + 1)


# --- addresses ------------------------------------------------------------
WPARTYCOUNT = 0xD163
WPARTYMON0 = 0xD16B      # first party struct; each is 44 (0x2C) bytes
WPARTYNICKS = 0xD2B5     # 11 bytes each
PARTY_STRUCT = 0x2C
WOBTAINEDBADGES = 0xD356
WPLAYERMONEY = 0xD347    # 3 bytes, BCD, big-endian
WNUMBAGITEMS = 0xD31D
WBAGITEMS = 0xD31E       # pairs (itemid, qty), 0xFF terminator
WISINBATTLE = 0xD057
WPLAYERNAME = 0xD158
WRIVALNAME = 0xD34A
WTILEMAP = 0xC3A0        # 20x18 on-screen tiles (decodes to visible text)

_STATUS_BITS = {0x08: "poison", 0x10: "burn", 0x20: "freeze", 0x40: "paralyze"}


def _status_name(s: int) -> str:
    if s == 0:
        return "ok"
    if s & 0x07:  # low bits = sleep turns remaining
        return "sleep"
    for bit, name in _STATUS_BITS.items():
        if s & bit:
            return name
    return f"status:{s}"


def read_party(emu: Emulator) -> list[dict]:
    out: list[dict] = []
    try:
        n = emu.read_memory(WPARTYCOUNT)
        if n > 6:
            return []
        for i in range(n):
            b = WPARTYMON0 + i * PARTY_STRUCT
            species = emu.read_memory(b + 0x00)
            move_ids = [emu.read_memory(b + 0x08 + j) for j in range(4)]
            pp = [emu.read_memory(b + 0x1D + j) for j in range(4)]
            out.append({
                "species": SPECIES.get(species, f"#{species}"),
                "nickname": _decode(emu, WPARTYNICKS + i * 11, 11),
                "level": emu.read_memory(b + 0x21),
                "hp": _u16(emu, b + 0x01),
                "max_hp": _u16(emu, b + 0x22),
                "status": _status_name(emu.read_memory(b + 0x04)),
                "moves": [f"{MOVES.get(mid, mid)} (PP {p})" for mid, p in zip(move_ids, pp) if mid],
            })
    except Exception:
        pass
    return out


def read_badges(emu: Emulator) -> dict:
    try:
        bits = emu.read_memory(WOBTAINEDBADGES)
        return {"count": bin(bits).count("1"), "bitfield": bits}
    except Exception:
        return {"count": 0, "bitfield": 0}


def read_money(emu: Emulator) -> int:
    try:  # 3-byte BCD
        total = 0
        for i in range(3):
            byte = emu.read_memory(WPLAYERMONEY + i)
            total = total * 100 + (byte >> 4) * 10 + (byte & 0x0F)
        return total
    except Exception:
        return 0


_ITEM_ID_BY_NAME = {
    "".join(ch for ch in name.lower() if ch.isalnum()): iid for iid, name in ITEMS.items()
}


def resolve_item_id(name: str) -> int | None:
    """Fuzzy item name -> id (e.g. "Oak's Parcel" -> 70). Case/punctuation-insensitive."""
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    if key in _ITEM_ID_BY_NAME:
        return _ITEM_ID_BY_NAME[key]
    return next((iid for k, iid in _ITEM_ID_BY_NAME.items() if key and (key in k or k in key)), None)


def read_items(emu: Emulator) -> list[dict]:
    out: list[dict] = []
    try:
        n = emu.read_memory(WNUMBAGITEMS)
        if n > 20:
            return []
        for i in range(n):
            item_id = emu.read_memory(WBAGITEMS + i * 2)
            if item_id == 0xFF:
                break
            out.append({"item": ITEMS.get(item_id, f"#{item_id}"), "qty": emu.read_memory(WBAGITEMS + i * 2 + 1)})
    except Exception:
        pass
    return out


def read_battle(emu: Emulator) -> dict | None:
    """Best-effort enemy/active summary; only meaningful mid-battle."""
    try:
        if emu.read_memory(WISINBATTLE) == 0:
            return None
        return {
            "enemy": {
                "species": SPECIES.get(emu.read_memory(0xCFE5), "?"),
                "level": emu.read_memory(0xCFF3),
                "hp": _u16(emu, 0xCFE6),
                "max_hp": _u16(emu, 0xCFF4),
                "status": _status_name(emu.read_memory(0xCFE9)),
            },
            "active": {
                "species": SPECIES.get(emu.read_memory(0xD014), "?"),
                "level": emu.read_memory(0xD022),
                "hp": _u16(emu, 0xD015),
                "max_hp": _u16(emu, 0xD023),
                "status": _status_name(emu.read_memory(0xD018)),
            },
        }
    except Exception:
        return None


def _uppercase_run(emu: Emulator, addr: int, length: int) -> bool:
    """True if the tile row contains a real uppercase letter (font tiles 0x80-0x99).

    This is the key discriminator between actual dialogue and a cutscene PICTURE:
    picture tiles land in the 0xA0-0xB9 range (which the font maps to a-z), so a
    graphics screen decodes to garbage like "aaaaa" — but real text almost always
    carries a capital (names in caps, sentence starts), and pictures rarely use the
    0x80-0x99 band. Requiring one uppercase tile rejects the graphics false-positive.
    """
    return any(0x80 <= emu.read_memory(addr + i) <= 0x99 for i in range(length))


def decode_textbox_raw(emu: Emulator) -> str:
    """The letters decoded from the bottom text-box region (rows 12-17), WITHOUT any dialog/no-dialog
    classification — just the raw ≥3-alpha lines joined. Empty => no text at all. The Jev flow router
    feeds THIS to the model (so it sees an all-lowercase line the has_upper heuristic would zero); the
    model, not a string heuristic, decides whether it's a dialogue box or a background-picture blob."""
    try:
        lines = []
        for row in range(12, 18):
            s = _decode(emu, WTILEMAP + row * 20, 20, stop_at_terminator=False).strip()
            if sum(c.isalpha() for c in s) >= 3:
                lines.append(s)
        return " ".join(lines).strip()
    except Exception:
        return ""


def read_screen_text(emu: Emulator) -> tuple[str, bool]:
    """Decode the on-screen textbox region into text, and whether a real dialog/menu
    is active. Text boxes occupy the bottom rows (12-17). A row counts as text only
    if it has a run of letters AND at least one uppercase font tile somewhere in the
    region — otherwise a full-screen cutscene picture decodes to garbage ("aaaaa")
    and is falsely read as dialogue (which makes the agent mash A forever).

    Returns (text, dialog_active)."""
    try:
        lines = []
        has_upper = False
        for row in range(12, 18):  # bottom textbox region
            base = WTILEMAP + row * 20
            s = _decode(emu, base, 20, stop_at_terminator=False).strip()
            if sum(c.isalpha() for c in s) >= 3:
                lines.append(s)
                has_upper = has_upper or _uppercase_run(emu, base, 20)
        text = " ".join(lines).strip()
        # A real dialog/menu box has an uppercase font tile OR reads like natural language. The
        # uppercase check ALONE wrongly rejects an all-lowercase dialogue line (a continuation like
        # "...strong, they can protect me!"), leaving the agent stuck: a box is up so it can't move,
        # but it reads as overworld and never presses A to close it. A full-screen cutscene picture,
        # by contrast, decodes to a REPEATED char ("aaaa…") — so real language is distinguished by
        # letter VARIETY (many distinct letters), not by casing.
        words = [w for w in text.split() if sum(c.isalpha() for c in w) >= 2]
        distinct = len({c for c in text.lower() if c.isalpha()})
        looks_like_language = len(words) >= 2 and distinct >= 5
        return (text, True) if (text and (has_upper or looks_like_language)) else ("", False)
    except Exception:
        return "", False


def read_dialog_lines(emu: Emulator) -> list[str]:
    """The two text lines of the standard bottom text box (tile rows 14 and 16), in order. The
    HeardLog stitches these frames into complete messages (typing grows a line; a scroll moves the
    lower line up)."""
    try:
        return [_decode(emu, WTILEMAP + row * 20 + 1, 18, stop_at_terminator=False).strip()
                for row in (14, 16)]
    except Exception:
        return []


def _sprite_kind(name: str) -> str:
    """'item' = a pickup you press A to grab (Poké Ball / item on the ground); else 'person'
    (an NPC you talk to). Lets the executor route grab_item vs talk_to correctly."""
    low = name.lower()
    if "ball" in low or "fossil" in low or low in ("item", "boulder"):
        return "item"
    return "person"


def read_npcs(emu: Emulator) -> list[dict]:
    """NPCs/objects on the current map as {x, y, facing, sprite, sprite_id, kind}, from
    RAM map coordinates (correct even when off-screen). ``kind`` is 'item' (a pickup) or
    'person' (an NPC to talk to)."""
    out: list[dict] = []
    hidden = hidden_sprite_slots(emu)
    try:
        for i in range(1, 16):  # slot 0 is the player
            pic = emu.read_memory(WSPRITE1 + i * 16)  # picture id: 0 = empty slot
            if pic == 0 or i in hidden:   # empty slot, or a missable the game has hidden
                continue
            b2 = WSPRITE2 + i * 16
            name = SPRITES.get(pic, f"sprite#{pic}")
            out.append({
                "x": emu.read_memory(b2 + 5) - SPRITE_COORD_OFFSET,
                "y": emu.read_memory(b2 + 4) - SPRITE_COORD_OFFSET,
                "facing": _FACING.get(emu.read_memory(WSPRITE1 + i * 16 + 9), "?"),
                "sprite": name,
                "sprite_id": pic,
                "slot": i,                      # stable identity for a wandering NPC (sprite slot)
                "kind": _sprite_kind(name),
            })
    except Exception:
        pass
    return out


WXCOORD = 0xD362
WYCOORD = 0xD361
WPLAYERFACING = 0xC109


def read_facing(emu: Emulator, npcs: list[dict] | None = None) -> dict:
    """What the player is currently facing / would interact with by pressing A: the front tile
    and any sprite there, PLUS the tile one beyond (a sprite there is talkable across a counter).
    Grounds 'is an interaction available right now'."""
    try:
        x, y = emu.read_memory(WXCOORD), emu.read_memory(WYCOORD)
        face = _FACING.get(emu.read_memory(WPLAYERFACING))
        if face is None:
            return {}
        dx, dy = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}[face]
        npcs = npcs if npcs is not None else read_npcs(emu)
        by_xy = {(n["x"], n["y"]): n for n in npcs}
        front = (x + dx, y + dy)              # distance 1
        beyond = (x + 2 * dx, y + 2 * dy)     # distance 2 (across a counter)
        facing_sprite = by_xy.get(front) or by_xy.get(beyond)
        return {
            "direction": face,
            "front_tile": list(front),
            "facing_sprite": facing_sprite,
            "can_interact": facing_sprite is not None,
        }
    except Exception:
        return {}


# --- interaction CONTEXT: "what kind of moment is this?" ------------------
# Addresses cross-checked against the pokered disassembly / DataCrystal RAM map.
WISINBATTLE_ADDR = 0xD057   # 0 none / 1 wild / 2 trainer
WBATTLETYPE = 0xD05A        # 0 normal / 1 old-man tutorial / 2 safari
WTEXTBOXID = 0xD125         # id of the text box currently set up
WCURMENUITEM = 0xCC26       # selected menu index (0-based) — PERSISTS when no menu
WMAXMENUITEM = 0xCC28       # index of the last menu item
WMENUCURSORY = 0xCC24
WMENUCURSORX = 0xCC25
WMENUWATCHEDKEYS = 0xCC29   # bitmask of keys this menu reacts to
WD730 = 0xD730              # bit 6 set during forced/scripted movement (input ignored)

_BATTLE_KIND = {0: "none", 1: "wild", 2: "trainer"}


def read_context(emu: Emulator) -> dict:
    """A coarse but reliable read of WHAT KIND of moment the agent is in, so it can
    pick an appropriate action (walk / advance text / operate a menu / wait).

    Reliable today: battle + battle kind, and real-dialogue detection (graphics-proofed
    by read_screen_text). The raw menu cursor fields are included but PERSIST stale when
    no menu is open, so they're only meaningful once we confirm a menu is up — that
    confirmation (a menu-open signal + option decoding) is the next piece to validate
    against a real menu fixture, so `kind` stays coarse for now.
    """
    try:
        inb = emu.read_memory(WISINBATTLE_ADDR)
    except Exception:
        inb = 0
    text, text_active = read_screen_text(emu)
    from .menus import read_menu  # lazy import avoids a load-time cycle (menus imports game_state)
    menu = read_menu(emu)
    if inb and not menu.get("open"):
        kind = "battle"
    elif menu.get("open"):
        kind = "menu"          # a selectable menu is up (cursor arrow present) — needs a choice, not mashing A
    elif text_active:
        kind = "dialog"        # passive text box — advance it
    else:
        kind = "overworld"
    ctx = {
        "kind": kind,
        "in_battle": bool(inb),
        "battle_kind": _BATTLE_KIND.get(inb, f"#{inb}"),
        "text_active": text_active,
        "screen_text": text,
        "screen_text_raw": decode_textbox_raw(emu),  # raw decode (unclassified) for the Jev flow router
        "menu": menu,          # {open, kind, cursor_index, num_options, options} when a menu is up
    }
    try:
        ctx["battle_type"] = emu.read_memory(WBATTLETYPE)
        ctx["text_box_id"] = emu.read_memory(WTEXTBOXID)
        ctx["forced_movement"] = bool(emu.read_memory(WD730) & 0x40)
        # raw menu cursor — only trustworthy once a menu is confirmed open (see docstring)
        ctx["menu_raw"] = {
            "cursor_item": emu.read_memory(WCURMENUITEM),
            "last_item": emu.read_memory(WMAXMENUITEM),
            "cursor_yx": (emu.read_memory(WMENUCURSORY), emu.read_memory(WMENUCURSORX)),
            "watched_keys": emu.read_memory(WMENUWATCHEDKEYS),
        }
    except Exception:
        pass
    return ctx


def read_game_state(emu: Emulator) -> dict:
    """One rich state block, fed alongside the map/exits/trajectory."""
    text, dialog_active = read_screen_text(emu)
    npcs = read_npcs(emu)
    return {
        "context": read_context(emu),
        "party": read_party(emu),
        "battle": read_battle(emu),
        "npcs": npcs,
        "facing": read_facing(emu, npcs),
        "badges": read_badges(emu),
        "money": read_money(emu),
        "items": read_items(emu),
        "player_name": _decode(emu, WPLAYERNAME, 11),
        "rival_name": _decode(emu, WRIVALNAME, 11),
        "dialog_active": dialog_active,
        "screen_text": text,
        "dialog_lines": read_dialog_lines(emu) if dialog_active else [],
    }
