"""Let a model choose the move in battle, instead of a fixed slot.

The active mon's moves are a small, enumerated, mutually-exclusive set — a natural
TypeSafe `Choice`. We hand it the battle state (both mons' species/level/HP/status
and the available moves) and it returns the slot to use, with a calibrated
confidence. `use_move` (battle.py) then executes it.
"""
from __future__ import annotations

from ...emulator.interface import Emulator
from . import battle
from .game_state import read_battle

CHOOSE_MOVE_INSTRUCTIONS = (
    "You are choosing the best move in a Pokémon Red battle. You are given both "
    "Pokémon's species, level, HP and status, and your available moves. Pick the move "
    "that best progresses toward winning THIS battle: usually the move that deals the "
    "most damage given the type match-up, but consider status/setup moves when they "
    "help (e.g. lowering the foe's stats, or when you can safely set up). Prefer a "
    "damaging move when the foe is low on HP and you can knock it out."
)


def battle_state_summary(emu: Emulator) -> dict:
    b = read_battle(emu) or {}
    return {
        "your_pokemon": b.get("active"),
        "opponent": b.get("enemy"),
        "your_moves": {str(i): m for i, m in enumerate(battle.active_moves(emu))},
    }


def choose_move(client, emu: Emulator) -> tuple[int, float]:
    """Ask the TypeSafe client which move slot to use. Returns (slot, confidence)."""
    from typesafe_sdk import Choice

    moves = battle.active_moves(emu)
    if not moves:
        return 0, 0.0
    criteria = {str(i): f"Use {m}." for i, m in enumerate(moves)}
    state = battle_state_summary(emu)
    resp = client.system_one(
        state=state, questions={"move": Choice(instructions=CHOOSE_MOVE_INSTRUCTIONS, criteria=criteria)}
    )
    ans = resp.answers["move"]
    try:
        slot = int(ans.choice)
    except (TypeError, ValueError):
        slot = 0
    slot = max(0, min(slot, len(moves) - 1))
    return slot, float(getattr(ans, "confidence", 0.0) or 0.0)
