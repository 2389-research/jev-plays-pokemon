"""The critic: a second opinion when the agent has stopped making progress.

The stuck model can't be trusted to notice it's stuck (runs/ss-anne: ~10 identical "travel to
Vermilion" re-issues; the Continual Harness paper reports a tool call repeated 842 times). The
deterministic ``StallMonitor`` decides WHEN; this model reads what actually happened — the attempt
ledger, the recent account, what was heard, what's unexplored — and writes a short diagnosis for L1.
It never edits the plan: L1 reads the critique at its next review and decides.
"""
from __future__ import annotations

import json

from ..providers.parsing import strip_fences

CRITIC_SYSTEM = """You review an agent playing Pokémon Red that has STOPPED MAKING PROGRESS: nothing new
(no new places, nothing new heard, no items/badges/levels, no goals met) for STALLED_FOR steps. You see
its GOALS, its PLAN, ATTEMPTS (every step it tried and how each ended — repeats included), RECENT (what
happened lately: maps entered, plan changes and why), HEARD (what people and signs told it — hints
matter), UNEXPLORED_HERE (doors, people and objects on this map it hasn't tried), and NOTEPAD.

Say plainly what's going wrong and what to try instead. Look for: the same step retried with the same
result; bouncing between the same maps; beliefs in the NOTEPAD or plan that the evidence contradicts;
hints in HEARD that point somewhere; unexplored doors/people that could hold the way forward. Don't
invent game facts — if the way forward is unknown, say that exploring (and which unexplored things
first) is the move. Two or three sentences.

Return ONLY JSON: {"diagnosis": "<what's going wrong>", "suggestion": "<what to try instead>"}"""


INTERRUPT_SYSTEM = """An agent playing Pokémon Red has a PLANNER that wants to INTERRUPT the step currently
in progress (ACTIVE, with what it's DOING right now) and replace it (PROPOSED). You are the reviewer.
Approve only when the REASON is concrete and it really can't wait — finishing the active step first
would be worse — e.g. the party is about to faint and the active step walks through wild grass, the
active step became impossible or obsolete, or something urgent appeared. Reject churn: reordering for
its own sake, misreading an active step that is simply still travelling to its map, a reason the
PARTY / SIGNALS contradict, or repeating an interrupt that was already rejected (RECENT_FEEDBACK).
Return ONLY JSON: {"approve": <true|false>, "why": "<one sentence>"}"""


class Critic:
    def __init__(self, provider, *, cooldown: int = 150, on_event=None, capture=None):
        self.provider = provider
        self.cooldown = cooldown
        self.on_event = on_event or (lambda k, p: None)
        self.capture = capture
        self.last_step = -10**9
        self.latest: dict | None = None               # {"step", "diagnosis", "suggestion"}

    def due(self, step: int, stalled: bool) -> bool:
        return self.provider is not None and stalled and step - self.last_step >= self.cooldown

    def judge_interrupt(self, step: int, state: dict) -> tuple[bool, str]:
        """Review a planner's request to replace the ACTIVE step. No reviewer configured -> approve
        (the planner's reason is still logged); a failed call -> reject (keep the step)."""
        if self.provider is None:
            return True, "no reviewer configured"
        try:
            raw, _lat, _usage = self.provider.chat_json(INTERRUPT_SYSTEM, state)
            data = json.loads(strip_fences(raw))
        except Exception:
            return False, "the reviewer call failed, so the active step was kept"
        if self.capture is not None:
            try:
                self.capture.record("interrupt_review", model=getattr(self.provider, "model", None), input=state,
                                    output_raw=json.dumps(data), parsed=data)
            except Exception:
                pass
        ok, why = bool(data.get("approve")), str(data.get("why") or "")[:240]
        self.on_event("interrupt_review", {"approve": ok, "why": why, "reason": state.get("reason")})
        return ok, why

    def review(self, step: int, state: dict) -> dict | None:
        self.last_step = step
        try:
            raw, _lat, _usage = self.provider.chat_json(CRITIC_SYSTEM, state)
            data = json.loads(strip_fences(raw))
        except Exception:
            return None
        if self.capture is not None:
            try:
                self.capture.record("critic", model=getattr(self.provider, "model", None), input=state,
                                    output_raw=json.dumps(data), parsed=data)
            except Exception:
                pass
        if not isinstance(data, dict) or not (data.get("diagnosis") or data.get("suggestion")):
            return None
        self.latest = {"step": step, "diagnosis": str(data.get("diagnosis") or "")[:400],
                       "suggestion": str(data.get("suggestion") or "")[:400]}
        self.on_event("critique", dict(self.latest))
        return self.latest
