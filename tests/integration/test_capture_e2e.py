"""CAPTURE end-to-end (design §6 Phase 4): L1 standing catch goal -> battle_L2 CAPTURE ->
Jev throw -> the ball macro -> a REAL catch, driven through the loop's `_battle_turn`.

Fixture `states/capture_wild.state` (built by scripts/make_capture_fixture.py): the Viridian
Forest wild battle with the enemy at 1 HP + FROZEN + catch-rate 255 (a Caterpie/Weedle/Pidgey-
class early wild) and a bag of ONLY Poke Balls + Potions — the real early-game inventory, no
fake Ultra/Master Ball. Under the restored RNG the loop reliably catches within a few throws.
states/*.state are gitignored, so these skip when the ROM/fixture is absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROM_PATH = ROOT / "roms" / "pokemon_red.gb"
STATE_PATH = ROOT / "states" / "capture_wild.state"

ACTIVE_HP = 0xD015  # player's active battle mon current HP (big-endian u16)


def _guard():
    if not ROM_PATH.exists() or not STATE_PATH.exists():
        pytest.skip("ROM/capture fixture not present")


def _emu():
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator

    emu = PyBoyEmulator(str(ROM_PATH), window="null")
    emu.load_state(STATE_PATH)
    emu.tick(6)
    return emu


def _loop(emu, events=None):
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.plan import AgentPlan
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import ReasonStep, ReflectionPlan
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import GoalState, WaitAction
    from pokemon_agent.observations.builder import ObservationBuilder

    class Stub:  # no `client` attr -> GRIND uses move slot 0 (no network)
        def reflect(self, **k):
            return ReflectionPlan(), 0, {}

        def step(self, **k):
            return ReasonStep(location="", objective="", reasoning="",
                              action=WaitAction(frames=1)), 0, {}

    on_event = (lambda kind, payload: events.append((kind, payload))) if events is not None else None
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=Stub(), session=Session(GoalState(primary="p", current="p")),
                         vision=False, reflect_every=100, on_event=on_event)
    # L1 would set this standing goal; here we inject it directly on the plan.
    loop._plan = AgentPlan(battle_goals={"catch": ["Kakuna"]})
    return loop


def _qty(items, name):
    return next((it["qty"] for it in items if it["item"].lower() == name.lower()), 0)


def test_capture_goal_catches_the_wild_pokemon():
    """The full Phase-4 path: catch goal -> CAPTURE -> throw -> caught. party +1, ball −1,
    battle ends — the deterministic proof that a real catch happens through the loop."""
    _guard()
    from pokemon_agent.games.pokemon_red import battle
    from pokemon_agent.games.pokemon_red.battle_l2 import CAPTURE
    from pokemon_agent.games.pokemon_red.game_state import read_items, read_party

    emu = _emu()
    try:
        events = []
        loop = _loop(emu, events)
        balls_before = _qty(read_items(emu), "Poke Ball")
        party_before = len(read_party(emu))
        assert balls_before > 0 and battle.in_battle(emu)

        caught = False
        for _ in range(8):
            if not battle.in_battle(emu):
                break
            loop.step_once()
            if len(read_party(emu)) > party_before:
                caught = True
                break

        # the objective was CAPTURE (goal -> battle_L2), and we actually caught it.
        assert loop._battle_objective == CAPTURE
        assert any(k == "battle_objective" and p["objective"] == CAPTURE for k, p in events)
        assert caught, "the loop should have caught the wild Pokémon"
        assert len(read_party(emu)) == party_before + 1          # party +1
        # at least one ball was consumed (a Poké Ball can break free, so the loop may throw a few
        # before it sticks — the realistic early-game ball, not a fake high-tier one).
        assert 0 < _qty(read_items(emu), "Poke Ball") < balls_before
        assert not battle.in_battle(emu)                         # the battle ended
    finally:
        emu.close()


def test_capture_at_critical_hp_does_not_throw_and_flees():
    """Safety re-eval (§2.1): CAPTURE + our active mon at CRITICAL HP -> the per-turn override
    flips to ESCAPE (wild), so the loop RUNS rather than throwing a ball into a KO. No ball is
    consumed and the safety override is recorded."""
    _guard()
    from pokemon_agent.games.pokemon_red import battle
    from pokemon_agent.games.pokemon_red.battle_l2 import CAPTURE, ESCAPE
    from pokemon_agent.games.pokemon_red.game_state import read_items

    emu = _emu()
    try:
        events = []
        loop = _loop(emu, events)
        # pin CAPTURE cached + mark the battle-start edge already seen, then force critical HP.
        loop._battle_objective = CAPTURE
        loop._last_in_battle = True
        emu.write_memory(ACTIVE_HP, 0)
        emu.write_memory(ACTIVE_HP + 1, 2)   # 2/27 HP -> below the critical band

        balls_before = _qty(read_items(emu), "Poke Ball")
        loop.step_once()

        # the per-turn override flipped CAPTURE -> ESCAPE this turn (recorded), and NO ball flew.
        assert any(k == "battle_safety_override" and p["from"] == CAPTURE and p["to"] == ESCAPE
                   for k, p in events)
        assert _qty(read_items(emu), "Poke Ball") == balls_before, "must not throw at critical HP"
        # the cached objective is untouched (per-turn override, not a re-plan).
        assert loop._battle_objective == CAPTURE
    finally:
        emu.close()
