from __future__ import annotations


def validate_step(step: dict) -> tuple[bool, str | None]:
    """Deterministic step-5 VALIDATE for L1-emitted quest steps. Returns (ok, error_message).
    Mirrors the reconciler's kind contract so a bad criterion is caught (and repaired) BEFORE it
    reaches compile_steps_to_directives (which would raise)."""
    from .planner_llm import Planner
    kind = str(step.get("kind") or "action")
    try:
        map_id = int(step.get("map"))
    except (TypeError, ValueError):
        return False, f"step map is missing/invalid: {step.get('map')!r}"
    dw = step.get("done_when")
    if kind == "travel":
        if step.get("talk") or step.get("who"):
            return False, "travel step cannot talk; use kind:action"
        parsed = Planner._parse_done_when(dw, map_id) if dw else {"on_map": map_id}
        if parsed is not None and parsed != {"on_map": map_id}:
            return False, "travel step criterion must be on_map"
        return True, None
    # kind == "action"
    parsed = Planner._parse_done_when(dw, map_id)
    if parsed is None:
        return False, f"action step needs a checkable done_when; {dw!r} did not parse"
    if parsed == {"on_map": map_id}:
        return False, "action step cannot complete on arrival (on_map)"
    return True, None


def run_l1_pipeline(emu, context: dict, planner, *, hard_event: bool, on_trace=None) -> dict | None:
    """Orchestrate the L1 reasoning pipeline: triage (gated unless hard_event) -> brainstorm ->
    decide -> validate/repair-once-then-break for each proposed step. Returns a proposal dict
    ({"add", "remove", "assessment", "catch", + any goals/notepad/interrupted/legacy keys}) or None if there's nothing to do
    (triage said no change) or the decide output can't be salvaged (a step fails validation even
    after repair -- keep the standing plan rather than partially apply a broken decision).

    Decoupled from any recorder: callers that want to record the trace pass ``on_trace``."""
    if not hard_event:
        t = planner.l1_triage(context)
        if on_trace:
            on_trace({"stage": "triage", **t})
        if not t.get("change"):
            return None

    b = planner.l1_brainstorm(emu, context)
    if on_trace:
        on_trace({"stage": "brainstorm", "assessment": b.get("assessment", "")})

    d = planner.l1_decide(context, b)
    if on_trace:
        on_trace({"stage": "decide", "add": len(d.get("add", [])), "remove": d.get("remove", []),
                  "anchors": [s.get("after") for s in d.get("add", []) if isinstance(s, dict)]})

    validated_add = []
    for step in d.get("add", []):
        if not isinstance(step, dict):   # defensive (l1_decide already filters non-dicts)
            continue
        ok, err = validate_step(step)
        if ok:
            validated_add.append(step)
            continue
        fixed = planner.l1_repair(context, step, err)
        ok2, err2 = validate_step(fixed)
        if ok2:
            # placement is not repair's job: the ORIGINAL anchor always wins (REPAIR may drop or
            # echo a mangled `after`)
            fixed = {k: v for k, v in fixed.items() if k != "after"}
            if "after" in step:
                fixed["after"] = step["after"]
            validated_add.append(fixed)
        else:
            if on_trace:
                on_trace({"stage": "invalid_criterion", "step": step, "error": err2 or err})
            return None

    remove = d.get("remove") or []
    # tiered goals (spec §3.3): goals / notepad / interrupted / catch ride along with a step edit,
    # and a DECIDE that carries one of them WITHOUT a step edit is returned too — the loop's
    # goals.detect_change decides whether it really changes anything (a reworded echo does not).
    extras = {k: d[k] for k in ("goals", "notepad", "interrupted", "mission", "milestone") if k in d}
    carries = (isinstance(d.get("goals"), dict) and bool(d.get("goals"))
               or isinstance(d.get("notepad"), str)
               or d.get("interrupted") == ""
               or d.get("catch") == "clear"
               or (isinstance(d.get("catch"), list) and bool(d.get("catch"))))
    # NO-OP DECIDE == NO CHANGE. If decide neither added nor removed a step and carries no goals/
    # notepad/catch edit, re-applying it would only churn (legacy mission/milestone rewording, an
    # echoed "catch": []) and re-fire an l1_review, which reads as L1 "restating" the task instead
    # of continuing it. Keep the standing plan (and its active step) and return None. This is what
    # lets a wedge-triggered deep review say "keep going".
    if not validated_add and not remove and not carries:
        if on_trace:
            on_trace({"stage": "no_change", "why": "decide made no add/remove; continue active step"})
        return None
    if on_trace and not validated_add and not remove:
        on_trace({"stage": "goals_candidate",
                  "keys": [k for k in ("goals", "notepad", "interrupted", "catch") if k in d]})

    return {
        "add": validated_add,
        "remove": remove,
        "assessment": d.get("assessment"),
        # optional standing battle goal (design §7.1): species list / "clear" / None (unchanged)
        "catch": d.get("catch"),
        **extras,
    }
