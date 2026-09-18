"""RAM-verifiable "hierarchy of needs" for the agent (foundation for the needs stack).

Each need is a deterministic predicate read from RAM (never a model judgment — the
CPP false-completion guard). The agent pursues the highest-priority ACTIVE need:

    SURVIVE   (heal when HP is low / a mon has fainted)   — reactive, top priority
    BATTLE    (fight when in a battle)                    — reactive
    READINESS (grind to a level target before the gym)    — proactive
    PROGRESS  (navigate toward the objective map)         — proactive, default

The arbitration mechanism (priority stack vs behavior tree vs utility) is wired on
top of this; hysteresis lives there so needs don't flip every step. This module only
answers "what is true right now" from RAM.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...emulator.interface import Emulator
from .game_state import read_party

WISINBATTLE = 0xD057
WCURMAP = 0xD35E

# priorities (higher wins)
P_SURVIVE, P_BATTLE, P_READINESS, P_PROGRESS = 100, 90, 50, 10


@dataclass
class Need:
    name: str
    active: bool
    priority: int
    detail: str


def party_hp_fraction(emu: Emulator) -> float:
    """Lowest hp/max_hp across the party (1.0 if no party / unknown)."""
    fracs = [m["hp"] / m["max_hp"] for m in read_party(emu) if m.get("max_hp")]
    return min(fracs) if fracs else 1.0


def any_fainted(emu: Emulator) -> bool:
    return any(m.get("hp") == 0 for m in read_party(emu))


def max_party_level(emu: Emulator) -> int:
    return max((m.get("level", 0) for m in read_party(emu)), default=0)


def has_party(emu: Emulator) -> bool:
    return bool(read_party(emu))


def assess_needs(emu: Emulator, *, goal_map: int | None = None,
                 level_target: int = 0, low_hp: float = 0.3) -> list[Need]:
    """All needs with their current active state, highest priority first."""
    party = read_party(emu)
    hp_frac = party_hp_fraction(emu)
    in_battle = emu.read_memory(WISINBATTLE) != 0
    mx = max_party_level(emu)
    cur_map = emu.read_memory(WCURMAP)
    needs = [
        Need("survive", bool(party) and (hp_frac < low_hp or any_fainted(emu)),
             P_SURVIVE, f"hp_frac={hp_frac:.2f} fainted={any_fainted(emu)}"),
        Need("battle", in_battle, P_BATTLE, "in battle"),
        Need("readiness", bool(party) and mx < level_target, P_READINESS,
             f"level {mx}/{level_target}"),
        Need("progress", goal_map is None or cur_map != goal_map, P_PROGRESS,
             f"map {cur_map} -> {goal_map}"),
    ]
    return sorted(needs, key=lambda n: -n.priority)


def top_need(emu: Emulator, **kw) -> Need | None:
    """The highest-priority ACTIVE need — what the agent should be doing right now."""
    for n in assess_needs(emu, **kw):
        if n.active:
            return n
    return None
