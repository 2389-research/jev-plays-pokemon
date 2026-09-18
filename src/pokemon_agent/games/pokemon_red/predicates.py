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
from .game_state import WPARTYCOUNT, read_badges, read_money
from .needs import WCURMAP, WISINBATTLE

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
