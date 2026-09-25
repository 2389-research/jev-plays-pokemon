"""L1 tiered goals + notepad — pure bookkeeping (spec 2026-09-23-l1-tiered-goals-design §3.3–3.4).

L1 owns its goals; this module only (1) validates a goal's criterion against the map-independent
grammar, (2) evaluates goal status from RAM, (3) decides whether a DECIDE output *really* changes
goals / notepad / interrupted / catch (a reworded echo is not a change — that's what keeps the no-op
short-circuit from turning into churn), and (4) applies a change with the one-level ``interrupted``
bookkeeping. No model calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from ..games.pokemon_red import predicates
from .plan import GOAL_TEXT_MAX, GOAL_TIERS, NOTEPAD_MAX_CHARS, AgentPlan, Goal, goal_text

__all__ = ["GOAL_TEXT_MAX", "NOTEPAD_MAX_CHARS", "GOAL_TIERS", "GoalsChange", "norm", "clean_criterion",
           "clean_goal", "goal_status", "statuses", "truncate_notepad", "detect_change", "apply_change",
           "refresh_interrupted", "goals_view"]

# parsed keys of the map-RELATIVE criteria (goals have no map): rejected by key, since the parse
# returns {"on_map": None} / {"talked_on_map": None} rather than None for them
_MAP_RELATIVE = {"on_map", "talked_on_map"}


def norm(s) -> str:
    """Comparison form: whitespace-collapsed, case-folded (rewording-by-spacing/case is no change)."""
    return re.sub(r"\s+", " ", str(s or "")).strip().casefold()


@lru_cache(maxsize=256)
def _parse(dw: str) -> dict | None:
    from .planner_llm import Planner
    return Planner._parse_done_when(dw, None)


def clean_criterion(dw) -> str | None:
    """``dw`` if it is in the goal grammar, else None (the caller keeps the goal's text)."""
    if not isinstance(dw, str) or not dw.strip():
        return None
    parsed = _parse(dw.strip())
    if not parsed or set(parsed) & _MAP_RELATIVE:
        return None
    return dw.strip()


def clean_goal(raw) -> Goal | None:
    """Coerce a DECIDE tier value into a Goal: a dict {text, done_when}, or a bare string (text only).
    Other types -> None (ignored). Invalid criterion dropped, text capped."""
    if isinstance(raw, str):
        raw = {"text": raw}
    if not isinstance(raw, dict):
        return None
    text = goal_text(raw.get("text"))
    return Goal(text=text, done_when=clean_criterion(raw.get("done_when")) if text else None)


def goal_status(goal: Goal, emu, memory=None) -> str:
    """"none" (no goal / no criterion), "unchecked" (verify: — not evaluated here), "met", "unmet"."""
    if not goal.text or not goal.done_when:
        return "none"
    parsed = _parse(goal.done_when)
    if not parsed:
        return "none"
    if "verify" in parsed:
        return "unchecked"
    try:
        return "met" if predicates.evaluate(parsed, emu, memory=memory) else "unmet"
    except Exception:
        return "unmet"


def statuses(plan: AgentPlan, emu, memory=None) -> dict[str, str]:
    return {t: goal_status(getattr(plan.goals, t), emu, memory) for t in GOAL_TIERS}


def goals_view(plan: AgentPlan, emu=None, memory=None) -> dict:
    """The goal fields shown to L1 (triage gets all but the notepad)."""
    g = {t: getattr(plan.goals, t).model_dump() for t in GOAL_TIERS}
    st = statuses(plan, emu, memory) if emu is not None else {t: "none" for t in GOAL_TIERS}
    intr = ({"text": plan.interrupted.text,
             "status": goal_status(plan.interrupted, emu, memory) if emu is not None else "none"}
            if plan.interrupted.text else None)
    return {"goals": g, "goal_status": st, "interrupted": intr,
            "notepad": plan.notepad, "notepad_truncated": plan.notepad_truncated}


def truncate_notepad(text: str) -> tuple[str, bool]:
    """Cap at NOTEPAD_MAX_CHARS, cutting at the last line boundary (hard cut if one line is longer)."""
    text = str(text or "").strip()
    if len(text) <= NOTEPAD_MAX_CHARS:
        return text, False
    head = text[:NOTEPAD_MAX_CHARS + 1]
    cut = head.rfind("\n")
    if cut > 0:
        return text[:cut].rstrip(), True
    return text[:NOTEPAD_MAX_CHARS], True


def _catch_set(v) -> frozenset[str]:
    return frozenset(norm(x) for x in (v or []) if norm(x))


@dataclass
class GoalsChange:
    goals: dict[str, Goal] = field(default_factory=dict)   # only the tiers that really changed
    notepad: str | None = None
    notepad_truncated: bool = False
    drop_interrupted: bool = False
    catch: list[str] | str | None = None                   # list = set it; "clear"; None = unchanged
    lead: str | None = None                                # a party nickname to put first; "clear" removes
    train: list[str] | str | None = None                   # switch-train these members; "clear" removes
    teach: dict | None = None                              # {"move", "who", "forget"?}: teach a TM/HM once


def _crit(dw: str | None) -> dict | None:
    """Criterion identity = the parsed predicate, so `hp_frac>=1` echoes `hp_frac>=1.0`."""
    return _parse(dw.strip()) if dw else None


def _same(a: Goal, b: Goal) -> bool:
    return norm(a.text) == norm(b.text) and _crit(a.done_when) == _crit(b.done_when)


def detect_change(plan: AgentPlan, prop: dict, *, step_edit: bool) -> GoalsChange | None:
    """What a DECIDE output really changes, after validation + normalization; None = nothing.

    Legacy ``mission``/``milestone`` keys count only alongside a step edit and never override a
    ``goals`` tier. ``catch``: absent / [] = unchanged; "clear" clears; a list sets (compared as a set).
    ``interrupted``: only "" (drop) is meaningful."""
    ch = GoalsChange()
    raw = prop.get("goals") if isinstance(prop.get("goals"), dict) else {}
    proposed: dict[str, Goal] = {}
    for tier in GOAL_TIERS:
        if tier in raw:
            g = clean_goal(raw[tier])
            if g is not None:
                proposed[tier] = g
    if step_edit:
        for key, tier in (("mission", "primary"), ("milestone", "secondary")):
            if tier not in raw and isinstance(prop.get(key), str) and prop[key].strip():
                # legacy text never carries a criterion: keep the tier's own (same text -> no change)
                cur, g = getattr(plan.goals, tier), clean_goal(prop[key])
                proposed[tier] = Goal(text=g.text, done_when=cur.done_when if g.text else None)
    for tier, g in proposed.items():
        if not _same(g, getattr(plan.goals, tier)):
            ch.goals[tier] = g

    if isinstance(prop.get("notepad"), str):
        text, cut = truncate_notepad(prop["notepad"])
        if norm(text) != norm(plan.notepad):
            ch.notepad, ch.notepad_truncated = text, cut

    # an explicit drop is always honoured (it may cancel a pause recorded by this same DECIDE), but on
    # its own it is only a change when something is actually paused
    # (only "" — models write null for "field not used", so null is NOT a drop)
    ch.drop_interrupted = prop.get("interrupted") == ""

    cur = _catch_set((plan.battle_goals or {}).get("catch"))
    c = prop.get("catch")
    if isinstance(c, str) and norm(c) == "clear":
        if cur:
            ch.catch = "clear"
    elif isinstance(c, list) and _catch_set(c) and _catch_set(c) != cur:
        ch.catch = [str(x) for x in c if norm(x)]
    lead = prop.get("lead")
    cur_lead = norm((plan.battle_goals or {}).get("lead"))
    if isinstance(lead, str) and lead.strip():
        if norm(lead) == "clear":
            if cur_lead:
                ch.lead = "clear"
        elif norm(lead) != cur_lead:
            ch.lead = lead.strip()
    train = prop.get("train")
    cur_train = [norm(t) for t in ((plan.battle_goals or {}).get("train") or [])]
    if isinstance(train, str) and norm(train) == "clear":
        if cur_train:
            ch.train = "clear"
    elif isinstance(train, list):
        names = [str(t).strip() for t in train if str(t).strip()]
        if names and [norm(t) for t in names] != cur_train:
            ch.train = names
    teach = prop.get("teach")
    if isinstance(teach, dict) and str(teach.get("move") or "").strip() and str(teach.get("who") or "").strip():
        ch.teach = {k: str(teach[k]).strip() for k in ("move", "who", "forget") if str(teach.get(k) or "").strip()}
    if not (ch.goals or ch.notepad is not None or ch.catch is not None or ch.lead is not None or ch.train is not None
            or ch.teach is not None
            or (ch.drop_interrupted and plan.interrupted.text)):
        return None
    return ch


def apply_change(plan: AgentPlan, ch: GoalsChange, *, pre_status: dict[str, str]) -> None:
    """Apply a detected change. ``interrupted`` rules are evaluated against the PRE-DECIDE state
    (``pre_status`` = tier statuses before the edit), clear before set, explicit drop last."""
    old_t, new_t = plan.goals.tertiary, ch.goals.get("tertiary")
    intr_before = plan.interrupted
    replaced_met = pre_status.get("tertiary") == "met"
    if new_t is not None:
        if intr_before.text and (norm(new_t.text) == norm(intr_before.text) or replaced_met):
            plan.interrupted = Goal()
        elif (not intr_before.text and new_t.text and old_t.text and not replaced_met
              and norm(new_t.text) != norm(old_t.text)):
            plan.interrupted = old_t.model_copy()
    if ch.drop_interrupted:
        plan.interrupted = Goal()
    if ch.goals:
        plan.apply_goals(ch.goals)
    if ch.notepad is not None:
        plan.notepad, plan.notepad_truncated = ch.notepad, ch.notepad_truncated
    if ch.catch is not None:
        plan.battle_goals = {**(plan.battle_goals or {}), "catch": [] if ch.catch == "clear" else list(ch.catch)}
    if ch.train is not None:
        bg = dict(plan.battle_goals or {})
        if ch.train == "clear":
            bg.pop("train", None)
        else:
            bg["train"] = list(ch.train)
        plan.battle_goals = bg
    if ch.teach is not None:
        plan.battle_goals = {**(plan.battle_goals or {}), "teach": dict(ch.teach)}
    if ch.lead is not None:
        bg = dict(plan.battle_goals or {})
        if ch.lead == "clear":
            bg.pop("lead", None)
        else:
            bg["lead"] = ch.lead
        plan.battle_goals = bg


def refresh_interrupted(plan: AgentPlan, emu, memory=None) -> bool:
    """At each review: drop the paused focus if its own criterion is now met. A verify:/criterion-less
    one never auto-clears (L1 returns to it or sends "interrupted": ""). Returns True if cleared."""
    if plan.interrupted.text and goal_status(plan.interrupted, emu, memory) == "met":
        plan.interrupted = Goal()
        return True
    return False
