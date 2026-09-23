"""Conversation/script gate + per-step objective budget (interaction-reliability spec F4/F5).

Incident (runs/brock-continue-20260922-2217): during Oak's parcel-delivery cutscene the screen
briefly shows no text between boxes, the flow router said "navigate", and the full nav path ran —
wedging the step, re-planning L1 six times, issuing moves mid-cutscene. Standing still in a
conversation also fed the position-oscillation check. And the stuck detector's objective state was
never reset per plan step, so fresh steps wedged after exactly BLOCK_TRIGGER steps.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pokemon_agent.actions.stuck_detector import StuckDetector
from pokemon_agent.core.models import (
    ActionResult, GameMode, InteractAction, PlayerState, WaitAction,
)

JOYIGNORE = 0xCD6B


def _loop():
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import ReasonStep, ReflectionPlan
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import GoalState
    from pokemon_agent.emulator.fake_emulator import FakeEmulator
    from pokemon_agent.observations.builder import ObservationBuilder

    class Stub:
        def reflect(self, **k): return ReflectionPlan(), 0, {}
        def step(self, **k): return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}
    emu = FakeEmulator(map_id=40)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=Stub(), session=Session(GoalState(primary="p", current="p")),
                         vision=False, reflect_every=100, goal_map=2)
    return loop, emu


# ---------------------------------------------------------------- F4
def test_commit_directive_resets_the_objective_budget():
    from pokemon_agent.agent.plan import Directive, Intent
    loop, _ = _loop()
    loop._directive = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 40, "sprite": "Oak"},
                                success={"no_item": "Oaks Parcel"})
    calls = {"n": 0}
    loop.stuck.reset_objective = lambda: calls.__setitem__("n", calls["n"] + 1)
    loop._commit_directive("new step")
    assert calls["n"] == 1


# ---------------------------------------------------------------- F5: in_conversation
def test_in_conversation_sources():
    from pokemon_agent.agent.reason_loop import CONVO_GRACE_STEPS
    loop, emu = _loop()
    loop.session.step = 100
    assert loop._in_conversation("dialog")                      # a text box is up
    assert not loop._in_conversation("overworld")               # plain overworld
    loop._last_dialog_step = 100 - CONVO_GRACE_STEPS            # the gap between two text boxes
    assert loop._in_conversation("overworld")
    loop._last_dialog_step = 100 - CONVO_GRACE_STEPS - 1        # the conversation is over
    assert not loop._in_conversation("overworld")
    emu.write_memory(JOYIGNORE, 0xFC)                           # a script owns the controls
    assert loop._in_conversation("overworld")


# ---------------------------------------------------------------- F5: forced-wait timeout
def test_forced_waits_time_out_disarm_and_rearm():
    from pokemon_agent.agent.reason_loop import CONVO_GRACE_STEPS, FORCED_WAIT_MAX_STEPS
    loop, emu = _loop()
    events = []
    loop.on_event = lambda kind, payload: events.append(kind)
    emu.write_memory(JOYIGNORE, 0xFC)   # a flag stuck on
    loop.session.step = 1000
    for _ in range(FORCED_WAIT_MAX_STEPS):
        assert loop._should_defer_to_script("overworld") is True
    assert loop._should_defer_to_script("overworld") is False     # timed out -> fall through
    assert "script_wait_timeout" in events
    assert loop._should_defer_to_script("overworld") is False     # disarmed while the flag persists
    emu.write_memory(JOYIGNORE, 0)
    loop.session.step += CONVO_GRACE_STEPS + 1
    assert loop._should_defer_to_script("overworld") is False     # nothing to wait for; re-armed
    emu.write_memory(JOYIGNORE, 0xFC)
    assert loop._should_defer_to_script("overworld") is True      # armed again


def test_a_non_forced_step_resets_the_forced_wait_counter():
    from pokemon_agent.agent.reason_loop import FORCED_WAIT_MAX_STEPS
    loop, emu = _loop()
    emu.write_memory(JOYIGNORE, 0xFC)
    loop.session.step = 500
    for _ in range(FORCED_WAIT_MAX_STEPS - 1):
        loop._should_defer_to_script("overworld")
    loop._note_dialogue_step()          # the cutscene advanced a text box -> not a stall
    for _ in range(FORCED_WAIT_MAX_STEPS):
        assert loop._should_defer_to_script("overworld") is True


def test_grace_window_overworld_step_waits_without_managing_the_directive():
    loop, _ = _loop()
    loop._last_dialog_step = loop.session.step       # a text box was up on the previous step
    loop._manage_directive = lambda obs: pytest.fail("must not run the executive mid-conversation")
    res = loop.step_once()
    assert isinstance(loop._prev.action, WaitAction)
    assert res.result == "completed"


# ---------------------------------------------------------------- F5: freeze, don't reset
def test_conversation_steps_freeze_the_block_budget():
    loop, _ = _loop()
    loop._blocked_for_n = 4
    loop._apply_stuck_to_budget(SimpleNamespace(stuck=False), frozen=True)
    assert loop._blocked_for_n == 4                  # frozen: neither reset nor incremented
    loop._apply_stuck_to_budget(SimpleNamespace(stuck=True), frozen=False)
    assert loop._blocked_for_n == 5
    loop._apply_stuck_to_budget(SimpleNamespace(stuck=False), frozen=False)
    assert loop._blocked_for_n == 0


def test_a_repeated_same_npc_dialogue_cycle_still_wedges():
    """Re-talking a blocking NPC must stay escapable: one overworld interact per cycle feeds the
    oscillation check while the dialogue steps are frozen. Accepted latency: <= 15 cycles."""
    from pokemon_agent.agent.reason_loop import BLOCK_TRIGGER
    loop, _ = _loop()
    det = StuckDetector()
    here = PlayerState(x=18, y=9, map_id=1)
    done = ActionResult(success=True, result="completed", mode_before=GameMode.OVERWORLD,
                        mode_after=GameMode.DIALOG)
    for cycle in range(1, 16):
        s = det.update(InteractAction(), done, here)                       # overworld: talk
        loop._apply_stuck_to_budget(s, frozen=False)
        for _ in range(6):                                                 # dialogue + grace waits
            s = det.update(WaitAction(frames=12), done, here, forced_movement=True)
            loop._apply_stuck_to_budget(s, frozen=True)
        if loop._blocked_for_n >= BLOCK_TRIGGER:
            break
    assert loop._blocked_for_n >= BLOCK_TRIGGER, "a dialogue loop became unescapable"
    assert cycle <= 15
