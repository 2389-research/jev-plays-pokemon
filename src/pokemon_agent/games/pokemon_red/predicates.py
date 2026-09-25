"""Machine-checkable success/impossibility predicates over RAM (the verifier).

A ``Directive.success`` is a predicate dict; this module evaluates it against live
emulator RAM every step. Deterministic — never a model judgment (the false-completion
guard). This is the option's termination β: it owns *detection*; the planner owns what
to do next.

A predicate is a dict of one or more clauses, ANDed together. Each clause is
``{key: spec}`` where ``spec`` is either a bare value (equality) or a comparison
string like ``">=1"`` / ``"<0.8"``. Supported keys:

    on_map      current map id            {"on_map": 2}
    party_size  number of party mons      {"party_size": ">=1"}
    hp_frac     lowest party hp/max_hp    {"hp_frac": ">=0.8"}
    level       max party level           {"level": ">=12"}
    badges      badge count (popcount)    {"badges": ">=1"}
    money       player money              {"money": ">=1000"}
    in_battle   1 if in a battle else 0   {"in_battle": 0}
    talked_to   faced tile talked-to      {"talked_to": [map, x, y]}  (needs memory)

An empty predicate is never satisfied (a directive with no termination never commits —
that is a planner bug we surface rather than silently completing).
"""
from __future__ import annotations

import operator

from ...emulator.interface import Emulator
from . import needs
from .game_state import WNUMBAGITEMS, WBAGITEMS, WPARTYCOUNT, read_badges, read_money, read_party
from .needs import WCURMAP, WISINBATTLE

WPLAYERX = 0xD362
WPLAYERY = 0xD361


def _bag_item_ids(emu) -> set[int]:
    ids: set[int] = set()
    try:
        n = emu.read_memory(WNUMBAGITEMS)
        if n > 20:
            return ids
        for i in range(n):
            iid = emu.read_memory(WBAGITEMS + i * 2)
            if iid == 0xFF:
                break
            ids.add(iid)
    except Exception:
        pass
    return ids

def _bag_item_qty(emu, item_id: int) -> int:
    total = 0
    try:
        n = emu.read_memory(WNUMBAGITEMS)
        if n > 20:
            return 0
        for i in range(n):
            iid = emu.read_memory(WBAGITEMS + i * 2)
            if iid == 0xFF:
                break
            if iid == item_id:
                total += emu.read_memory(WBAGITEMS + i * 2 + 1)
    except Exception:
        pass
    return total


_OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt,
        "<": operator.lt, "==": operator.eq, "!=": operator.ne}


def _cmp(value: float, spec) -> bool:
    """Compare ``value`` against ``spec`` (a bare value = equality, or a "<op><num>" string)."""
    if isinstance(spec, str):
        for token, op in _OPS.items():
            if spec.startswith(token):
                try:
                    return op(value, float(spec[len(token):].strip()))
                except ValueError:
                    return False
        # bare numeric string
        try:
            return value == float(spec)
        except ValueError:
            return False
    return value == spec


def cut_tree_sites(map_id: int) -> set[tuple[int, int]]:
    """Where the game's Cut trees stand on a map (scripts/gen_cut_trees.py)."""
    import json
    from functools import lru_cache
    from pathlib import Path

    @lru_cache(maxsize=1)
    def _load():
        try:
            return json.loads((Path(__file__).with_name("cut_trees.json")).read_text())
        except Exception:
            return {}
    return {(t["x"], t["y"]) for t in _load().get(str(int(map_id)), [])}


def _clause(key: str, spec, emu: Emulator, memory=None) -> bool:
    if key == "on_map":
        return _cmp(emu.read_memory(WCURMAP), spec)
    if key == "party_size":
        return _cmp(emu.read_memory(WPARTYCOUNT), spec)
    if key == "hp_frac":
        return _cmp(needs.party_hp_fraction(emu), spec)
    if key == "level":
        return _cmp(needs.max_party_level(emu), spec)
    if key == "badges":
        return _cmp(read_badges(emu)["count"], spec)
    if key == "money":
        return _cmp(read_money(emu), spec)
    if key == "in_battle":
        return _cmp(1 if emu.read_memory(WISINBATTLE) else 0, spec)
    if key == "tree_cut":
        # spec = [map, x, y]: on that map, the Cut tree tile at (x, y) is gone (it regrows on reload,
        # so this is only true while we're on the map having cut it)
        from .map_reader import read_collision_map
        mid, x, y = (int(v) for v in spec)
        if (x, y) not in cut_tree_sites(mid):
            return False                     # not a tree at all (runs/sleeves-east: "done" instantly)
        coll = read_collision_map(emu) if emu.read_memory(WCURMAP) == mid else None
        return bool(coll) and coll["terrain"].get((x, y)) not in (None, "cut_tree")
    if key == "has_item":
        # spec = item id (int): true when that item is in the bag (e.g. Oak's Parcel 0x46).
        try:
            return int(spec) in _bag_item_ids(emu)
        except (TypeError, ValueError):
            return False
    if key == "no_item":
        # spec = item id: true when the item is NOT in the bag (e.g. delivered/consumed).
        try:
            return int(spec) not in _bag_item_ids(emu)
        except (TypeError, ValueError):
            return False
    if key == "member_level":
        # spec = [nickname or species, N]: that party member reached level N (training one Pokemon)
        try:
            name, n = str(spec[0]).strip().lower(), int(spec[1])
        except (TypeError, ValueError, IndexError):
            return False
        for m in read_party(emu):
            if name in (str(m.get("nickname") or "").strip().lower(), str(m.get("species") or "").lower()):
                return int(m.get("level") or 0) >= n
        return False
    if key == "item_count":
        # spec = [item id, N]: at least N of that item in the bag (e.g. 5 Poke Balls bought)
        try:
            iid, n = int(spec[0]), int(spec[1])
        except (TypeError, ValueError, IndexError):
            return False
        return _bag_item_qty(emu, iid) >= n
    if key == "talked_on_map":
        # spec = map id: true once we've had a real dialog with an NPC on that map (from
        # interaction memory) — the machine-checkable "did the talk_to step happen" signal.
        if memory is None or not hasattr(memory, "interactions"):
            return False
        try:
            mp = int(spec)
        except (TypeError, ValueError):
            return False
        return any(t[0] == mp for t in memory.interactions.talked)
    if key == "at_xy":
        # spec = [map, x, y]: player has reached (within 1 tile of) that map cell — used as a
        # waypoint's termination so an LLM-chosen unstuck target actually commits.
        try:
            mp, x, y = (int(v) for v in spec)
        except (TypeError, ValueError):
            return False
        return (emu.read_memory(WCURMAP) == mp
                and abs(emu.read_memory(WPLAYERX) - x) + abs(emu.read_memory(WPLAYERY) - y) <= 1)
    if key == "talked_to":
        # spec = [map, x, y] faced tile; verified against interaction memory's talked set.
        if memory is None or not hasattr(memory, "interactions"):
            return False
        try:
            return tuple(int(v) for v in spec) in memory.interactions.talked
        except (TypeError, ValueError):
            return False
    return False  # unknown key -> unverifiable -> not satisfied


def evaluate(pred: dict | None, emu: Emulator, *, memory=None) -> bool:
    """True iff every clause in ``pred`` holds against current RAM. Empty/None -> False."""
    if not pred:
        return False
    return all(_clause(k, spec, emu, memory) for k, spec in pred.items())
