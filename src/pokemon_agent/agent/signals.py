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
    from ..games.pokemon_red.evolution import evolution_line
    party = read_party(emu)
    for m in party:                       # player knowledge: what each member becomes
        line = evolution_line(m.get("species"))
        if line:
            m["evolves"] = line
    return {"party": party, "hp_frac": party_hp_frac(party), "min_level": min_level(party),
            "money": read_money(emu),
            "badges": (read_badges(emu) or {}).get("count", 0),
            "items": [f"{it.get('item')} x{it.get('qty')}" if (it.get("qty") or 1) > 1 else it.get("item")
                      for it in (read_items(emu) or [])]}


def catch_status(battle_goals: dict | None, items: list[dict], party: list[dict]) -> dict:
    """Whether L1's standing catch goal can actually fire in the next wild battle (the battle layer
    only switches to CAPTURE with a ball in the bag and a free party slot). Shown to L1 as
    SIGNALS.catch so an inert goal is visible instead of silently grinding."""
    from ..games.pokemon_red.battle_l2 import PARTY_MAX, is_ball
    goal = [str(x) for x in ((battle_goals or {}).get("catch") or [])]
    balls = sum(int(it.get("qty") or 0) for it in (items or []) if is_ball(it.get("item")))
    size = len(party or [])
    if not goal:
        why = "no catch goal set"
    elif balls == 0:
        why = "no Poké Balls in the bag"
    elif size >= PARTY_MAX:
        why = f"party is full ({size}/{PARTY_MAX})"
    else:
        why = ""
    return {"goal": goal, "ready": bool(goal) and why == "", "why": why,
            "balls": balls, "party_size": size, "party_max": PARTY_MAX}
