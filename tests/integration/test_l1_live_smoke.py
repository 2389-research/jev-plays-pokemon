"""Layer 3 live smoke test: real LunaRoute + real emulator, gated behind the `live` marker.

Skipped by default (see tests/conftest.py) — run manually with `RUN_LIVE=1` or `--run-live`
when LunaRoute credits are available. This is a SHAPE smoke test, not a determinism test: it
only asserts that a few L1 strategic reviews against a stuck-agent fixture state produce at
least one action step with a parseable, non-on_map done_when criterion (i.e. L1 did real
strategic work rather than falling back to a bootstrap travel default).
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROM_PATH = ROOT / "roms" / "pokemon_red.gb"
STATE_PATH = ROOT / "states" / "viridian_stuck.state"


@pytest.mark.live
def test_l1_strategic_review_produces_checkable_action_step():
    if not ROM_PATH.exists() or not STATE_PATH.exists():
        pytest.skip("ROM/state fixture not present")

    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.knowledge import KnowledgeBase
    from pokemon_agent.agent.loop import AgentLoop  # noqa: F401  (sanity: module imports cleanly)
    from pokemon_agent.agent.planner_llm import Planner
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import Reasoner
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import GoalState
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.observations.builder import ObservationBuilder
    from pokemon_agent.providers.lunaroute import LunaRouteProvider

    emu = PyBoyEmulator(str(ROM_PATH), window="null", speed=0)
    try:
        emu.load_state(STATE_PATH)

        strategist = LunaRouteProvider(model="glm-5.3")
        reasoner = Reasoner(strategist)
        knowledge = KnowledgeBase.from_env(None, None) or KnowledgeBase(
            "http://localhost:8100", "6d677a16"
        )

        session = Session(GoalState(primary="Leave Viridian", current="Leave Viridian"))
        builder = ObservationBuilder(emu)
        loop = ReasoningLoop(
            builder=builder,
            controller=ActionController(emu),
            reasoner=reasoner,
            session=session,
            vision=False,
            goal_map=2,  # Pewter City — gives the planner a live goal to strategize toward
            strategist_provider=strategist,
            knowledge=knowledge,
            l1_every=1,
        )
        assert loop.planner is not None, "Planner did not construct (goal_map/level_target wiring)"

        found_checkable_action = False
        for _ in range(3):
            obs, _ = builder.build()
            loop._run_l1(obs, hard_event=True)
            for step in loop._plan_steps:
                if step.kind == "action" and step.done_when:
                    parsed = Planner._parse_done_when(step.done_when, step.map)
                    if parsed is not None and parsed != {"on_map": step.map}:
                        found_checkable_action = True
                        break
            if found_checkable_action:
                break

        assert found_checkable_action, (
            "expected at least one action-kind plan step with a parseable, "
            f"non-on_map done_when after live L1 reviews; got plan_steps={loop._plan_steps!r}"
        )
    finally:
        emu.close()
