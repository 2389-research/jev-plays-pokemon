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


ACTIVE_TYPES = 0xD019      # wBattleMonType1/2
ENEMY_TYPES = 0xCFEA       # wEnemyMonType1/2


def move_analysis(emu: Emulator) -> list[dict]:
    """Per move slot: type, power, PP, the type multiplier against the enemy's types (Gen 1 chart incl.
    its quirks), same-type bonus, and expected damage = power x multiplier x (1.5 if STAB) x accuracy.
    Status moves (power 0) score 0 — a decision aid for the move chooser, not a rule."""
    from .battle_data import MOVE_DATA, TYPE_EFFECTS, TYPE_NAMES
    mine = {TYPE_NAMES.get(emu.read_memory(ACTIVE_TYPES + i)) for i in range(2)}
    theirs = [TYPE_NAMES.get(emu.read_memory(ENEMY_TYPES + i)) for i in range(2)]
    theirs = list(dict.fromkeys(t for t in theirs if t))          # a mono-type mon repeats its type
    pp = active_pp(emu)
    out = []
    for slot, mid in enumerate(m for m in active_move_ids(emu) if m):
        const, mtype, power, acc, _maxpp = MOVE_DATA.get(mid, ("?", None, 0, 100, 0))
        mult = 1.0
        for t in theirs:
            mult *= TYPE_EFFECTS.get((mtype, t), 1.0)
        stab = mtype in mine
        expected = round(power * mult * (1.5 if stab else 1.0) * acc / 100, 1) if power else 0
        out.append({"slot": slot, "move": MOVES.get(mid, const), "type": mtype, "power": power,
                    "pp": pp[slot] if slot < len(pp) else None, "effectiveness": mult, "stab": stab,
                    "expected": expected})
    return out


PARTY1_MOVES = 0xD173      # wPartyMon1Moves (learning outside battle)


def _move_power(move_id: int) -> int:
    from .battle_data import MOVE_DATA
    return (MOVE_DATA.get(move_id) or ("", "", 0))[2]


def move_to_forget(move_ids: list[int]) -> int:
    """The slot to give up for a new move: a status move (no damage) first, else the lowest power."""
    ids = [m for m in move_ids if m]
    return min(range(len(ids)), key=lambda i: (_move_power(ids[i]), i))


def worth_learning(new_move: str, move_ids: list[int]) -> bool:
    """Learn when the new move out-hits the weakest known move (status moves count as 0)."""
    from .battle_data import MOVE_DATA
    key = new_move.strip().upper().replace(" ", "_")
    power = next((v[2] for v in MOVE_DATA.values() if v[0] == key), 0)
    ids = [m for m in move_ids if m]
    return power > min(_move_power(m) for m in ids) if ids else True


def handle_learn_move(emu: Emulator) -> bool:
    """The 'learn a new move' flow when 4 moves are known: learn it if it beats the weakest move
    (forgetting that one), otherwise decline. The default A / move-list back-out otherwise loops or
    cancels it. True if it acted."""
    from . import menus
    s = _screen(emu)
    ids = active_move_ids(emu) if in_battle(emu) else [emu.read_memory(PARTY1_MOVES + i) for i in range(4)]
    if not menus.menu_open(emu):
        return False
    if "make room for" in s.replace("\n", " "):
        flat = " ".join(s.split())
        new = flat.split("make room for", 1)[1].split("?", 1)[0].strip()
        menus.answer_yesno(emu, worth_learning(new, ids))
        return True
    if "Abandon learning" in " ".join(s.split()):
        menus.answer_yesno(emu, True)                  # we chose not to learn it
        return True
    if "should be forgotten" in " ".join(s.split()) or "Which move should" in s:
        menus.select_option(emu, move_to_forget(ids), max_options=4)
        return True
    return False


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


ACTIVE_HP = 0xD015         # wBattleMonHP (big-endian)


def _screen(emu: Emulator) -> str:
    return "\n".join("".join(_decode_byte(emu.read_memory(WTILEMAP + r * 20 + c)) for c in range(20))
                     for r in range(18))


def switch_screen_showing(emu: Emulator) -> bool:
    """The party-switch flow is on screen: 'Will ... change POKEMON?' (YES/NO), 'Use next POKEMON?',
    'Bring out which POKEMON?' or '<mon> is already out!'."""
    if not in_battle(emu) or fight_menu_showing(emu):
        return False
    s = _screen(emu)
    return any(k in s for k in ("Bring out which", "already out", "change", "Use next", "no will"))


def replacement_slot(party: list[dict]) -> int | None:
    """First party member that can still fight (the forced pick after the active mon faints)."""
    return next((i for i, p in enumerate(party) if int(p.get("hp") or 0) > 0), None)


def resolve_switch_screen(emu: Emulator, tries: int = 10) -> bool:
    """Leave the switch flow the way a player would: decline a voluntary 'change POKEMON?' (NO), accept
    'Use next POKEMON?' (YES, after a faint), back out of 'Bring out which POKEMON?' with B unless the
    active mon has fainted (then send out the first healthy one). True once the FIGHT menu is back (or
    the battle ended)."""
    from . import menus
    from .game_state import read_party
    for _ in range(tries):
        if not in_battle(emu) or fight_menu_showing(emu):
            return True
        s = _screen(emu)
        fainted = _u16(emu, ACTIVE_HP) == 0
        if "Use next" in s and menus.menu_open(emu):
            menus.answer_yesno(emu, True)
        elif "change" in s and menus.menu_open(emu):
            menus.answer_yesno(emu, False)
        elif fainted and ("Bring out which" in s or "no will" in s):
            # forced pick after a faint (the party screen isn't a menu to menu_open(): drive the RAM
            # cursor directly) — dismiss "There's no will to fight!", then send out a healthy mon
            slot = replacement_slot(read_party(emu))
            if "no will" in s:
                _press(emu, GameButton.B, 30)
                continue
            if slot is None:
                return False
            pick_party_slot(emu, slot)
        else:
            _press(emu, GameButton.B, 30)
        emu.tick(10)
    return not in_battle(emu) or fight_menu_showing(emu)


WCURMENUITEM = 0xCC26


def _hold_press(emu: Emulator, button: GameButton, hold: int = 10, settle: int = 30) -> None:
    """A reliable press: 1-frame taps can alias with the game's joypad sampling and be dropped."""
    emu.hold(button)
    emu.tick(hold)
    emu.release(button)
    emu.tick(settle)


def pick_party_slot(emu: Emulator, slot: int) -> None:
    """On the battle party screen: move the cursor to ``slot`` (RAM wCurrentMenuItem) and send it out
    (A opens the SWITCH/STATS/CANCEL submenu, A again picks SWITCH)."""
    for _ in range(7):
        if emu.read_memory(WCURMENUITEM) == 0:
            break
        _hold_press(emu, GameButton.UP, 6, 12)
    for _ in range(slot):
        _hold_press(emu, GameButton.DOWN, 6, 12)
    _hold_press(emu, GameButton.A)
    _hold_press(emu, GameButton.A)


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
