from __future__ import annotations
from dataclasses import dataclass
from .plan import Directive, Intent
from ..games.pokemon_red.maps import map_name


@dataclass
class QuestStep:
    id: str
    map: int
    talk: bool = False
    who: str | None = None
    done_when: str | None = None
    why: str = ""
    status: str = "pending"          # pending | active | done | wedged
    provisional: bool = False        # a synthesized bootstrap default; superseded once L1 adds real steps
    kind: str = "action"             # "travel" = reach a map (on_map completion ok);
                                      # "action" = done by a state change (needs a real criterion)


def _criterion(done_when: str | None, map_id: int, kind: str) -> dict:
    from .planner_llm import Planner
    parsed = Planner._parse_done_when(done_when, map_id)
    if kind == "travel":
        # a travel leg completes on ARRIVAL by definition — always on_map, ignoring any (bogus)
        # criterion a directly-constructed step might carry (validate_step gates L1-emitted ones).
        return {"on_map": map_id}
    # kind == "action": a real, non-on_map criterion is REQUIRED (never complete-on-arrival)
    if parsed is None or parsed == {"on_map": map_id}:
        raise ValueError(
            f"action quest step (map {map_id}) needs a checkable done_when, got {done_when!r}")
    return parsed


def compile_steps_to_directives(steps: list[QuestStep]) -> list[Directive]:
    """Expand each QuestStep into its 1-2 Directives (TRAVEL + optional TALK_TO), tagged quest_id —
    following strategize()'s expansion shape. The model-authored done_when attaches to the talk step
    when there is one, else to the travel step (deliberate improvement over strategize, which forces
    on_map on the travel directive)."""
    out: list[Directive] = []
    for s in steps:
        crit = _criterion(s.done_when, s.map, s.kind)
        nm = map_name(s.map)
        if s.talk:
            out.append(Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": s.map},
                                 success={"on_map": s.map}, quest_id=s.id,
                                 reason=f"quest: go to {nm} — {s.why}"[:120]))
            tgt = {"kind": "npc", "map": s.map}
            if s.who:
                tgt["sprite"] = s.who
            out.append(Directive(intent=Intent.TALK_TO, target=tgt, success=crit, quest_id=s.id,
                                 reason=f"quest: talk to {s.who or 'someone'} in {nm} — {s.why}"[:120]))
        else:
            out.append(Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": s.map},
                                 success=crit, quest_id=s.id, reason=f"quest: go to {nm} — {s.why}"[:120]))
    return out


def reconcile_quests(current, proposal, *, next_id, on_event=None):
    """Deterministically merge L1's proposal into the canonical plan, preserving progress.
    Keeps done + active steps; removes only named pending steps; ALWAYS drops wedged steps (they are
    replaced by adds); dedups adds by (map, done_when).

    PLACEMENT: an add may carry ``"after"``: the id of a step that survives into the result as active
    or pending (place it after that step, and after adds already placed there), or ``"end"`` (append at
    the tail). Omitted/null -> the default slot right after done + active (today's behavior; with no
    anchors the result is exactly ``done + active + new + pending``). An anchor equal to the active id
    means "next" (default). Any other anchor falls back to the default slot and reports
    ``on_event("l1_anchor_fallback", {"after", "reason"})`` — a fallback changes placement, never drops."""
    add = proposal.get("add") or []
    remove = set(proposal.get("remove") or [])
    done = [s for s in current if s.status == "done"]
    active = [s for s in current if s.status == "active"]
    pending = [s for s in current if s.status == "pending" and s.id not in remove]
    have = {(s.map, s.done_when or "on_map") for s in done + active + pending}
    status_of = {s.id: s.status for s in current}
    live = {s.id for s in active + pending}
    active_ids = {s.id for s in active}

    def anchor_of(a):
        after = a.get("after")
        if after is None or after == "end":
            return after
        if not isinstance(after, str):
            reason = "not_string"
        elif after in active_ids:
            return None                      # "after the active step" == next == the default slot
        elif after in live:
            return after
        elif after not in status_of:
            reason = "unknown"
        elif status_of[after] == "wedged":
            reason = "wedged"
        elif status_of[after] == "done":
            reason = "done"
        else:
            reason = "removed"
        if on_event is not None:
            on_event("l1_anchor_fallback", {"after": after, "reason": reason})
        return None

    placed = []                              # (new step, anchor) in emitted order
    for a in add:
        try:
            mp = int(a["map"])
        except (KeyError, TypeError, ValueError):
            continue
        dw = a.get("done_when")
        key = (mp, dw or "on_map")
        if key in have:
            continue
        have.add(key)
        step = QuestStep(id=next_id(), map=mp, talk=bool(a.get("talk")),
                         who=(a.get("who") or None), done_when=dw, why=str(a.get("why") or "")[:80],
                         kind=str(a.get("kind") or "action"))
        placed.append((step, anchor_of(a)))

    out = done + active + pending
    default_slot = len(done) + len(active)   # head of pending (well-defined with no active step)
    last_on = {}                             # anchor id -> the step last inserted after it
    for step, anchor in placed:
        if anchor is None:
            out.insert(default_slot, step)
            default_slot += 1
        elif anchor == "end":
            out.append(step)
        else:
            ref = last_on.get(anchor, anchor)
            idx = next(i for i, s in enumerate(out) if s.id == ref)
            out.insert(idx + 1, step)
            last_on[anchor] = step.id
            if idx < default_slot:
                default_slot += 1
    return out
