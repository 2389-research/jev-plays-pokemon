"""Let a model choose the move in battle, instead of a fixed slot.

The active mon's moves are a small, enumerated, mutually-exclusive set — a natural
TypeSafe `Choice`. We hand it the battle state (both mons' species/level/HP/status
and the available moves) and it returns the slot to use, with a calibrated
confidence. `use_move` (battle.py) then executes it.
"""
from __future__ import annotations

from ...emulator.interface import Emulator
from . import battle
from .battle_l2 import (
    CAPTURE,
    ENDANGERED_HP_FRAC,
    ESCAPE,
    SURVIVE,
    hp_frac,
    is_ball,
)
from .game_state import read_battle

# CAPTURE: throw once the target is at/below this HP fraction; weaken (a move) above it — but only
# if the wild is big enough to survive a hit (we have no move-power data to pick a WEAK move, so the
# chooser would one-shot a small wild and lose the catch). Small wilds are thrown at directly.
CATCH_HP_BAND = 0.5
CATCH_SMALL_MAX_HP = 30   # wilds with max HP <= this are thrown at directly (a hit would likely KO)

CHOOSE_MOVE_INSTRUCTIONS = (
    "You are choosing the best move in a Pokémon Red battle. You are given both "
    "Pokémon's species, level, HP and status, your available moves, and TYPE_KNOWLEDGE "
    "(retrieved type-effectiveness guidance — use it to judge which of your moves is "
    "super effective against the opponent). Pick the move that best progresses toward "
    "winning THIS battle: usually the SUPER-EFFECTIVE / highest-damage move for the "
    "type match-up, but consider status/setup moves when they help. Prefer a damaging "
    "move when the foe is low on HP and you can knock it out. Each option states its COMPUTED "
    "effectiveness against this opponent (the game's own Gen 1 type chart), same-type bonus and "
    "expected damage (also in MOVE_ANALYSIS) — trust those numbers: a resisted move of your own "
    "type often does less than a neutral one."
)


def battle_state_summary(emu: Emulator) -> dict:
    b = read_battle(emu) or {}
    return {
        "your_pokemon": b.get("active"),
        "opponent": b.get("enemy"),
        "your_moves": {str(i): m for i, m in enumerate(battle.active_moves(emu))},
    }


def battle_lookup_query(emu: Emulator) -> str:
    """The knowledge-base query for the current match-up (enemy + your move options)."""
    b = read_battle(emu) or {}
    enemy = (b.get("enemy") or {}).get("species", "the opponent")
    moves = ", ".join(battle.active_moves(emu)) or "my moves"
    return (f"Pokemon Red type effectiveness: which move types are super effective against "
            f"{enemy}? Which of these moves is best: {moves}?")


def _norm(s: object) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _first_ball(items: list) -> str | None:
    """First ball in the live bag order (candidates[0] equivalent — Jev disambiguation later)."""
    for it in items:
        name = it.get("item") if isinstance(it, dict) else it
        if is_ball(name):
            return name
    return None


def _first_potion(items: list) -> str | None:
    """First HP-restore item (anything named '…Potion') in the live bag order."""
    for it in items:
        name = it.get("item") if isinstance(it, dict) else it
        if "potion" in _norm(name):
            return name
    return None


def choose_action(objective: str, state: dict) -> dict:
    """Map (objective, live state) -> ONE typed per-turn battle action (design §2.3).

    Pure and testable over the `battle_l2.build_state` dict. Returns a small typed action:
      * ``{"kind": "move"}``            — fight this turn (GRIND-EXP; also the fallback);
      * ``{"kind": "run"}``             — flee (ESCAPE);
      * ``{"kind": "item", "item": …}`` — use a Potion (SURVIVE when endangered + have one);
      * ``{"kind": "ball", "item": …}`` — throw a ball (CAPTURE once the target is weak).

    The move choice itself (which slot) stays with `choose_move`/`use_move` — this only
    picks the action TYPE, so GRIND-EXP is exactly today's fight path. Item/ball NAMES are
    the first live-bag candidate; the deterministic macro resolves the current index."""
    enemy = state.get("enemy") or {}
    active = state.get("active") or {}
    items = state.get("items") or []

    if objective == ESCAPE:
        return {"kind": "run"}

    if objective == SURVIVE:
        potion = _first_potion(items)
        if potion is not None and hp_frac(active) <= ENDANGERED_HP_FRAC:
            return {"kind": "item", "item": potion}
        return {"kind": "move"}

    if objective == CAPTURE:
        ball = _first_ball(items)
        if ball is None:                        # no ball on hand -> keep fighting
            return {"kind": "move"}
        if hp_frac(enemy) <= CATCH_HP_BAND:     # already in the catch band -> throw
            return {"kind": "ball", "item": ball}
        try:
            enemy_max = int(enemy.get("max_hp") or 0)
        except (TypeError, ValueError):
            enemy_max = 0
        if enemy_max and enemy_max <= CATCH_SMALL_MAX_HP:
            return {"kind": "ball", "item": ball}   # a hit would likely KO -> throw instead of weakening
        return {"kind": "move"}                     # big enough to survive weakening

    # GRIND-EXP and any unknown objective -> today's fight path.
    return {"kind": "move"}


def choose_move(client, emu: Emulator, *, type_knowledge: list[str] | None = None,
                capture=None) -> tuple[int, float]:
    """Ask the TypeSafe client which move slot to use. ``type_knowledge`` is optional retrieved
    type-effectiveness guidance (from the knowledge base) injected into Jev's decision state.
    ``capture`` is an optional distillation Capture (§3) — recording is best-effort and never
    changes behavior. Returns (slot, confidence)."""
    from typesafe_sdk import Choice

    moves = battle.active_moves(emu)
    if not moves:
        return 0, 0.0
    pp = battle.active_pp(emu)
    usable = [i for i in range(len(moves)) if i >= len(pp) or pp[i] > 0]
    if not usable:
        return 0, 0.0          # every move is out of PP: the game uses Struggle
    # only moves with PP left can be chosen (the game refuses a 0-PP move and the menu loops); each option
    # carries its computed effectiveness vs this opponent so the choice is informed, not guessed
    analysis = {m["slot"]: m for m in battle.move_analysis(emu)}

    def label(i):
        a = analysis.get(i)
        left = pp[i] if i < len(pp) else "?"
        if not a:
            return f"Use {moves[i]} ({left} PP left)."
        if not a["power"]:
            return f"Use {moves[i]} (status move, no damage; {left} PP left)."
        return (f"Use {moves[i]} ({a['type']}, power {a['power']}, {a['effectiveness']:g}x vs the opponent"
                f"{', same-type bonus' if a['stab'] else ''}; expected ~{a['expected']:g} damage; {left} PP left).")
    criteria = {str(i): label(i) for i in usable}
    state = battle_state_summary(emu)
    state["move_analysis"] = [analysis[i] for i in sorted(analysis)]
    if type_knowledge:
        state["type_knowledge"] = type_knowledge
    resp = client.system_one(
        state=state, questions={"move": Choice(instructions=CHOOSE_MOVE_INSTRUCTIONS, criteria=criteria)}
    )
    ans = resp.answers["move"]
    try:
        slot = int(ans.choice)
    except (TypeError, ValueError):
        slot = 0
    slot = max(0, min(slot, len(moves) - 1))
    if slot not in usable:
        slot = battle.usable_slot(pp, slot)
    confidence = float(getattr(ans, "confidence", 0.0) or 0.0)
    if capture is not None:
        capture.record("battle_move", model=getattr(client, "model", "typesafe"),
                       input=state, output_raw=str(getattr(ans, "choice", None)),
                       parsed={"slot": slot}, confidence=confidence)
    return slot, confidence
