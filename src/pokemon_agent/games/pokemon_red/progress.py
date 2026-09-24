"""RAM-only milestone / progress detector for Pokémon Red.

This module is the single source of truth for "how far along is the run" and
"which milestones are done". It is PURE: every value is read deterministically
from emulator RAM (no model, no heuristics, no side effects). It reuses the
existing low-level readers in ``game_state`` and the map-name table in ``maps``
rather than re-deriving anything.

Milestone map ids are resolved by reverse-looking-up human names in
``constants.MAP_NAMES_RAW`` (id -> name), so a name that does not exist in the
table yields a permanently-False milestone instead of a guessed id.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .constants import MAP_NAMES_RAW
from .game_state import WPARTYCOUNT, read_badges, read_money, read_party
from .maps import map_name

# --- addresses ------------------------------------------------------------
WCURMAP = 0xD35E  # current map id

# name -> id, built once by reversing the id -> name table. If two ids ever
# shared a name the lowest id wins (deterministic); none do in practice.
_NAME_TO_ID: dict[str, int] = {}
for _mid, _name in sorted(MAP_NAMES_RAW.items()):
    _NAME_TO_ID.setdefault(_name, _mid)


def _map_id_for(name: str) -> int | None:
    """Reverse-lookup a map id by its human name, or None if the name is absent."""
    return _NAME_TO_ID.get(name)


# No story-specific milestones (they were Pallet->Pewter-era: "reached_pewter", "beat_brock"): the
# agent sets its own goals; progress here is only what's true for any point in the game.
_MILESTONE_MAP_NAMES: dict[str, str] = {}

# Names we failed to resolve at import time (their milestones are forced False).
UNRESOLVED_MILESTONES: dict[str, str] = {
    key: name
    for key, name in _MILESTONE_MAP_NAMES.items()
    if _map_id_for(name) is None
}


@dataclass
class Progress:
    badges: int              # popcount of wObtainedBadges (0xD356)
    map_id: int              # 0xD35E
    map_name: str            # via maps.map_name(map_id)
    party_size: int          # 0xD163
    party_levels: list[int]  # level of each party mon
    money: int               # 3-byte BCD at 0xD347
    milestones: dict[str, bool] = field(default_factory=dict)


def read_progress(emu) -> Progress:
    """Read a full deterministic Progress snapshot from RAM."""
    badges = read_badges(emu)["count"]
    map_id = emu.read_memory(WCURMAP)
    party_size = emu.read_memory(WPARTYCOUNT)
    party_levels = [mon["level"] for mon in read_party(emu)]
    money = read_money(emu)

    milestones: dict[str, bool] = {"got_starter": party_size >= 1}
    for n in range(1, 9):                       # one per badge, whichever order they're earned in
        milestones[f"badge_{n}"] = badges >= n
    for key, name in _MILESTONE_MAP_NAMES.items():
        target = _map_id_for(name)
        # Unresolvable name -> milestone can never be True (no guessed id).
        milestones[key] = target is not None and map_id == target

    return Progress(
        badges=badges,
        map_id=map_id,
        map_name=map_name(map_id),
        party_size=party_size,
        party_levels=party_levels,
        money=money,
        milestones=milestones,
    )


def progress_vector(emu) -> dict:
    """Flat, log-friendly scalar summary of progress."""
    p = read_progress(emu)
    return {
        "badges": p.badges,
        "map_id": p.map_id,
        "party_size": p.party_size,
        "max_party_level": max(p.party_levels) if p.party_levels else 0,
        "money": p.money,
    }
