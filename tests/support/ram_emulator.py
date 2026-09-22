"""RamEmulator: a pure dict-of-RAM test double.

Unlike ``MemFake`` in ``tests/unit/test_predicates.py`` (which only reads raw
addresses handed to it), this double exposes semantic setters
(``set_party``, ``set_bag_items``, ...) that write the CORRECT bytes at the
CORRECT addresses/encodings so that ``predicates.evaluate`` (and the
``needs``/``game_state`` helpers it calls through, e.g.
``needs.party_hp_fraction`` -> ``game_state.read_party``) read them back
correctly. Addresses/encodings are imported from the real source modules —
never hardcoded — so this stays in lockstep with the production RAM layout.
"""
from __future__ import annotations

from pokemon_agent.games.pokemon_red.game_state import (
    PARTY_STRUCT,
    WBAGITEMS,
    WNUMBAGITEMS,
    WOBTAINEDBADGES,
    WPARTYCOUNT,
    WPARTYMON0,
    WPLAYERMONEY,
    resolve_item_id,
)
from pokemon_agent.games.pokemon_red.needs import WCURMAP, WISINBATTLE


def _bcd_byte(two_digit: int) -> int:
    """Encode a 0-99 decimal value as a single BCD byte (hi nibble tens, lo nibble ones)."""
    tens, ones = divmod(two_digit % 100, 10)
    return (tens << 4) | ones


class RamEmulator:
    """Settable-RAM emulator double: back everything with a plain ``dict[int, int]``."""

    def __init__(self) -> None:
        self._ram: dict[int, int] = {}

    # --- Emulator protocol (the two methods predicates/needs/game_state use) --
    def read_memory(self, address: int, bank: int | None = None) -> int:
        return self._ram.get(address, 0)

    def write_memory(self, address: int, value: int, bank: int | None = None) -> None:
        self._ram[address] = value & 0xFF

    # --- semantic setters ------------------------------------------------
    def set_map(self, map_id: int) -> None:
        self.write_memory(WCURMAP, map_id)

    def set_in_battle(self, in_battle: bool) -> None:
        self.write_memory(WISINBATTLE, 1 if in_battle else 0)

    def set_badges(self, count: int) -> None:
        """Set the badge count (popcount) `read_badges` reports, via a bitfield with
        that many low bits set — the actual bit pattern doesn't matter to predicates,
        only ``bin(bits).count("1")`` does."""
        bits = (1 << count) - 1 if count > 0 else 0
        self.write_memory(WOBTAINEDBADGES, bits)

    def set_money(self, amount: int) -> None:
        """3-byte BCD, big-endian, two decimal digits per byte (see `read_money`)."""
        hi = (amount // 10000) % 100
        mid = (amount // 100) % 100
        lo = amount % 100
        self.write_memory(WPLAYERMONEY, _bcd_byte(hi))
        self.write_memory(WPLAYERMONEY + 1, _bcd_byte(mid))
        self.write_memory(WPLAYERMONEY + 2, _bcd_byte(lo))

    def set_party(self, party: list[tuple[str, int, int, int]]) -> None:
        """``party`` = [(species, level, cur_hp, max_hp), ...].

        Species is accepted for test readability but isn't required by any predicate
        (`needs.party_hp_fraction`/`max_party_level` only read level + HP words via
        `game_state.read_party`), so it's written as a placeholder byte (0) — the real
        species id is irrelevant to what's under test here.
        """
        self.write_memory(WPARTYCOUNT, len(party))
        for i, (_species, level, cur_hp, max_hp) in enumerate(party):
            b = WPARTYMON0 + i * PARTY_STRUCT
            self.write_memory(b + 0x00, 0)  # species placeholder (unused by predicates)
            self.write_memory(b + 0x01, (cur_hp >> 8) & 0xFF)
            self.write_memory(b + 0x02, cur_hp & 0xFF)
            self.write_memory(b + 0x21, level)
            self.write_memory(b + 0x22, (max_hp >> 8) & 0xFF)
            self.write_memory(b + 0x23, max_hp & 0xFF)

    def set_bag_items(self, names: list[str], qty: int = 1) -> None:
        """``names`` are resolved via `resolve_item_id` (fuzzy, e.g. "oaks_parcel")."""
        ids = [resolve_item_id(n) for n in names]
        if any(iid is None for iid in ids):
            missing = [n for n, iid in zip(names, ids) if iid is None]
            raise ValueError(f"unresolvable item name(s): {missing!r}")
        self.write_memory(WNUMBAGITEMS, len(ids))
        for i, iid in enumerate(ids):
            self.write_memory(WBAGITEMS + i * 2, iid)
            self.write_memory(WBAGITEMS + i * 2 + 1, qty)
