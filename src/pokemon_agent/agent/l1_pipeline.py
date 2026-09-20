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
