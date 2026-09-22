"""Executive wiring (design §6.1): a SHOP directive whose route reached the Mart and opened the
counter must run the deterministic buy macro — not hand the menu to Jev, which flails.

Mirrors test_gate_crossing: deterministic (no network), ROM/fixture-guarded. The fixture is the
Viridian Mart (map 42) BUY/SELL/QUIT menu; states/*.state are gitignored, so this skips if absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROM_PATH = ROOT / "roms" / "pokemon_red.gb"
STATE_PATH = ROOT / "states" / "mart_counter.state"


def _loop(emu):
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import ReasonStep, ReflectionPlan
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import GoalState, WaitAction
    from pokemon_agent.observations.builder import ObservationBuilder

    class Stub:
        def reflect(self, **k):
            return ReflectionPlan(), 0, {}

        def step(self, **k):
            return ReasonStep(location="", objective="", reasoning="",
                              action=WaitAction(frames=1)), 0, {}

    return ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=Stub(), session=Session(GoalState(primary="p", current="p")),
                         vision=False, reflect_every=100, goal_map=42)


def test_shop_directive_runs_buy_macro_at_the_counter():
    if not ROM_PATH.exists() or not STATE_PATH.exists():
        pytest.skip("ROM/mart-counter fixture not present")
    from pokemon_agent.agent.plan import Directive, Intent
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.games.pokemon_red.game_state import read_items, read_money

    emu = PyBoyEmulator(str(ROM_PATH), window="null")
    try:
        emu.load_state(STATE_PATH)
        emu.tick(6)
        assert emu.read_memory(0xD35E) == 42  # at the Viridian Mart

        loop = _loop(emu)
        # the executive has already routed the SHOP directive to the clerk and the counter is open
        loop._directive = Directive(intent=Intent.SHOP,
                                    target={"kind": "npc", "map": 42, "item": "Antidote", "qty": 3},
                                    success={"has_item": "Antidote"})

        money_before = read_money(emu)
        loop.step_once()   # flow router sees the BUY menu -> _maybe_shop -> buy macro

        def qty_of(items, name):
            return next((it["qty"] for it in items if it["item"].lower() == name.lower()), 0)

        assert qty_of(read_items(emu), "Antidote") == 3
        assert money_before - read_money(emu) == 3 * 100
    finally:
        emu.close()
