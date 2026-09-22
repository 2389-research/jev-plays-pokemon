"""battle_L2 — the battle OBJECTIVE layer (P3 of the battle subsystem — design §2.1/§2.2).

Mirrors the overworld L2 proposer: run ONCE on the battle-start edge, it reads L1's
standing goals + the encounter and picks ONE objective from the closed set
{GRIND-EXP, CAPTURE, ESCAPE, SURVIVE}, cached for the whole fight (the loop re-runs it
only on real triggers — a critical-HP flip, or CAPTURE reaching the catch band).

`choose_objective` is a small, PURE, testable function over a plain state dict, so it can
be exercised with crafted encounters and no ROM. `build_state` assembles that dict from
the live RAM readers for the loop. Decisions here are DETERMINISTIC (a clear policy);
a Jev-calibrated refinement can replace it later without changing the wiring.
"""
from __future__ import annotations

from ...emulator.interface import Emulator
from . import battle
from .game_state import read_battle, read_items, read_party

# The closed set of battle objectives (design §2.2).
GRIND_EXP = "GRIND-EXP"
CAPTURE = "CAPTURE"
ESCAPE = "ESCAPE"
SURVIVE = "SURVIVE"

# HP bands (fraction of max) that drive the deterministic policy.
CRITICAL_HP_FRAC = 0.15    # wild + our mon this low -> ESCAPE (cut losses)
ENDANGERED_HP_FRAC = 0.35  # our mon this low -> SURVIVE (heal, but we must win)
PARTY_MAX = 6              # a party slot is free below this (box handling is later)


def _norm(s: object) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def hp_frac(mon: dict | None) -> float:
    """Current-HP fraction of a mon dict (`{"hp","max_hp"}`); 1.0 when unknown/full."""
    if not mon:
        return 1.0
    mx = mon.get("max_hp") or 0
    if mx <= 0:
        return 1.0
    return max(0.0, min(1.0, (mon.get("hp") or 0) / mx))


def is_ball(item_name: object) -> bool:
    """A Poké/Great/Ultra/Master/Safari Ball — anything whose name ends in 'ball'."""
    return _norm(item_name).endswith("ball")


def has_ball(items: list) -> bool:
    return any(is_ball(it.get("item") if isinstance(it, dict) else it) for it in items)


def slot_free(party: list) -> bool:
    return len(party) < PARTY_MAX


def catch_match(enemy_species: object, goals: dict | None) -> bool:
    """True when L1's standing catch goal covers this encounter's species.

    `goals["catch"]` is a small list of species names (or the wildcards "any"/"*"),
    matched case/punctuation-insensitively against the enemy species."""
    catch = (goals or {}).get("catch") or []
    if not catch:
        return False
    enemy = _norm(enemy_species)
    for want in catch:
        w = _norm(want)
        if w == "any" or want == "*":     # explicit wildcards only — an empty "" is NOT a catch-all
            return True
        if w and w == enemy:
            return True
    return False


def choose_objective(state: dict, goals: dict | None = None) -> str:
    """Pick ONE cached battle objective from the encounter + L1 goals (deterministic §2.2).

    Policy:
      * trainer battle (can't run): our mon endangered -> SURVIVE, else GRIND-EXP;
      * wild + our mon critically low -> ESCAPE;
      * wild + a catch goal matches + a slot is free + we have a ball -> CAPTURE;
      * otherwise GRIND-EXP (the default).
    `state`: `{"enemy","active","party","items","is_trainer"}` (see `build_state`)."""
    goals = goals or {}
    enemy = state.get("enemy") or {}
    active = state.get("active") or {}
    party = state.get("party") or []
    items = state.get("items") or []
    active_frac = hp_frac(active)

    if state.get("is_trainer"):
        # Trainer battles are win-or-lose (no running); protect an endangered mon.
        if active_frac <= ENDANGERED_HP_FRAC:
            return SURVIVE
        return GRIND_EXP

    # Wild encounter.
    if active_frac <= CRITICAL_HP_FRAC:
        return ESCAPE
    if catch_match(enemy.get("species"), goals) and slot_free(party) and has_ball(items):
        return CAPTURE
    return GRIND_EXP


def safety_override(objective: str, state: dict) -> str:
    """Per-turn safety override (design §2.1 critical-HP trigger). Pure + testable.

    Don't faint while trying to catch or grind: if the cached objective is CAPTURE or
    GRIND-EXP and our active mon is at CRITICAL HP, flip to SURVIVE (trainer — must win, so
    heal) or ESCAPE (wild — cut losses). Any other objective (already SURVIVE/ESCAPE, or a
    healthy mon) is returned unchanged. This is a per-turn override, NOT a re-plan: the loop
    keeps the initial cached objective and re-derives this each turn from the live state."""
    if objective not in (CAPTURE, GRIND_EXP):
        return objective
    if hp_frac(state.get("active")) <= CRITICAL_HP_FRAC:
        return SURVIVE if state.get("is_trainer") else ESCAPE
    return objective


def battle_goals_from_plan(plan_goals: dict | None, level_target: int = 0) -> dict:
    """Derive the compact battle-goals dict the battle layer reads (design §7.1) from L1's
    plan-level goals + the loop's level target. Shape: `{"catch": [...], "level_target": int}`.

    `plan_goals` is the (optional) `battle_goals` field L1 sets on the plan — currently a
    `{"catch": [species|"any", ...]}` map; anything absent defaults empty (→ GRIND)."""
    catch = list((plan_goals or {}).get("catch") or [])
    return {"catch": catch, "level_target": int(level_target or 0)}


def build_state(emu: Emulator, goals: dict | None = None) -> dict:
    """Assemble the pure `choose_objective`/`choose_action` state dict from live RAM."""
    b = read_battle(emu) or {}
    party = read_party(emu)
    return {
        "enemy": b.get("enemy") or {},
        "active": b.get("active") or (party[0] if party else {}),
        "party": party,
        "items": read_items(emu),
        "is_trainer": battle.is_trainer_battle(emu),
    }
