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


def _legacy_loop(on_step=None):
    """No planner (no goal) -> the legacy Jev path; `on_step(emu)` runs inside the reasoner's step,
    i.e. as part of THIS step's action (to simulate an action that triggers a script)."""
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import ReasonStep, ReflectionPlan
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import GoalState
    from pokemon_agent.emulator.fake_emulator import FakeEmulator
    from pokemon_agent.observations.builder import ObservationBuilder
    emu = FakeEmulator(map_id=40)

    class Stub:
        def reflect(self, **k): return ReflectionPlan(), 0, {}
        def step(self, **k):
            if on_step:
                on_step(emu)
            return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=Stub(), session=Session(GoalState(primary="p", current="p")),
                         vision=False, reflect_every=100)
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
    """Only a step ROUTED to dialogue (and its short grace window) or a script owning the controls
    count — never the raw text detector alone (the flow router may confidently disagree with it)."""
    from pokemon_agent.agent.reason_loop import CONVO_GRACE_STEPS
    loop, emu = _loop()
    loop.session.step = 100
    assert not loop._in_conversation()                          # plain overworld
    loop._note_dialogue_step()                                  # this step was routed to dialogue
    assert loop._in_conversation()
    loop.session.step = 100 + CONVO_GRACE_STEPS                 # the gap between two text boxes
    assert loop._in_conversation()
    loop.session.step = 100 + CONVO_GRACE_STEPS + 1             # the conversation is over
    assert not loop._in_conversation()
    emu.write_memory(JOYIGNORE, 0xFC)                           # a script owns the controls
    assert loop._in_conversation()


def test_false_positive_text_routed_to_navigate_never_forces_waits():
    """Review issue 1: a persistent false 'dialog' text read that the flow router confidently
    routes to NAVIGATE must not be overridden into ~95% forced waits."""
    loop, _ = _loop()
    loop.session.step = 300
    assert not any(loop._should_defer_to_script() for _ in range(50))


# ---------------------------------------------------------------- F5: forced-wait timeout
def test_forced_waits_time_out_disarm_and_rearm():
    from pokemon_agent.agent.reason_loop import CONVO_GRACE_STEPS, FORCED_WAIT_MAX_STEPS
    loop, emu = _loop()
    events = []
    loop.on_event = lambda kind, payload: events.append(kind)
    emu.write_memory(JOYIGNORE, 0xFC)   # a flag stuck on
    loop.session.step = 1000
    for _ in range(FORCED_WAIT_MAX_STEPS):
        assert loop._should_defer_to_script() is True
    assert loop._should_defer_to_script() is False     # timed out -> fall through
    assert "script_wait_timeout" in events
    assert loop._should_defer_to_script() is False     # disarmed while the flag persists
    emu.write_memory(JOYIGNORE, 0)
    loop.session.step += CONVO_GRACE_STEPS + 1
    assert loop._should_defer_to_script() is False     # nothing to wait for; re-armed
    emu.write_memory(JOYIGNORE, 0xFC)
    assert loop._should_defer_to_script() is True      # armed again


def test_a_non_forced_step_resets_the_forced_wait_counter():
    from pokemon_agent.agent.reason_loop import FORCED_WAIT_MAX_STEPS
    loop, emu = _loop()
    emu.write_memory(JOYIGNORE, 0xFC)
    loop.session.step = 500
    for _ in range(FORCED_WAIT_MAX_STEPS - 1):
        loop._should_defer_to_script()
    loop._note_dialogue_step()          # the cutscene advanced a text box -> not a stall
    for _ in range(FORCED_WAIT_MAX_STEPS):
        assert loop._should_defer_to_script() is True


def test_a_menu_step_also_resets_the_forced_wait_counter():
    """Review issue 6: ANY non-forced step (menu, battle...) resets the counter, not just dialogue."""
    loop, _ = _legacy_loop()
    loop._forced_waits = 39
    loop._route_flow = lambda obs, ctx: "menu"
    loop.step_once()
    assert loop._forced_waits == 0


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


def test_a_step_whose_own_action_triggers_a_script_is_not_frozen():
    """Review issue 2: conversation-ness is decided at STEP START. A step whose action triggers a
    script (e.g. a coordinate trigger that pushes you back) must still feed the block budget — it
    is the one overworld step per loop cycle that keeps a dialogue loop escapable."""
    from pokemon_agent.core.models import StuckInfo
    loop, emu = _legacy_loop(on_step=lambda e: e.write_memory(JOYIGNORE, 0xFC))
    loop._blocked_for_n = 3
    loop.stuck.update = lambda *a, **k: StuckInfo(stuck=True, kind="local_loop")
    loop.step_once()
    assert loop._blocked_for_n == 4          # counted, not frozen


def test_the_step_after_a_script_starts_is_a_frozen_forced_wait():
    from pokemon_agent.core.models import StuckInfo
    loop, emu = _legacy_loop()
    emu.write_memory(JOYIGNORE, 0xFC)        # the script is running at step start
    loop._blocked_for_n = 3
    loop.stuck.update = lambda *a, **k: StuckInfo(stuck=True, kind="local_loop")
    loop.step_once()
    assert isinstance(loop._prev.action, WaitAction)
    assert loop._blocked_for_n == 3          # frozen


def test_routed_dialogue_stays_frozen_even_while_the_gate_is_disarmed():
    """Review issue 7: a disarmed gate must not re-expose real dialogue steps to the detector."""
    from pokemon_agent.core.models import StuckInfo
    loop, _ = _legacy_loop()
    loop._forced_wait_armed = False
    loop._route_flow = lambda obs, ctx: "dialogue"
    loop._blocked_for_n = 3
    loop.stuck.update = lambda *a, **k: StuckInfo(stuck=True, kind="local_loop")
    loop.step_once()
    assert loop._blocked_for_n == 3
