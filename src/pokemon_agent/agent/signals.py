from __future__ import annotations

HEAL_EMERGENCY = 0.15   # party HP fraction at/below which we force an immediate heal


def party_hp_frac(party: list[dict]) -> float:
    """Fraction of total party HP remaining; 1.0 for an empty/unknown party (no false emergency)."""
    tot = sum(int(p.get("max_hp") or 0) for p in party)
    if tot <= 0:
        return 1.0
    return sum(int(p.get("hp") or 0) for p in party) / tot


def min_level(party: list[dict]) -> int | None:
    lv = [int(p.get("level")) for p in party if p.get("level") is not None]
    return min(lv) if lv else None


def needs_emergency_heal(party: list[dict], thresh: float = HEAL_EMERGENCY) -> bool:
    """True if any conscious-capable member is fainted (hp==0, max_hp>0) or overall HP < thresh.
    Empty party -> False (nothing to heal)."""
    if not party:
        return False
    if any(int(p.get("hp") or 0) == 0 and int(p.get("max_hp") or 0) > 0 for p in party):
        return True
    return party_hp_frac(party) < thresh


def game_signals(emu) -> dict:
    """RAM-derived signals fed INTO L1 each cycle (the loop adds blocked_for_n). Includes the party
    list so the emergency reflex can reuse it without a second read."""
    from ..games.pokemon_red.game_state import read_party, read_items, read_badges, read_money
    party = read_party(emu)
    return {"party": party, "hp_frac": party_hp_frac(party), "min_level": min_level(party),
            "money": read_money(emu),
            "badges": (read_badges(emu) or {}).get("count", 0),
            "items": [f"{it.get('item')} x{it.get('qty')}" if (it.get("qty") or 1) > 1 else it.get("item")
                      for it in (read_items(emu) or [])]}
