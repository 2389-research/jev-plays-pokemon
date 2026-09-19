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


def _criterion(done_when: str | None, map_id: int) -> dict:
    from .planner_llm import Planner
    parsed = Planner._parse_done_when(done_when, map_id)
    return parsed if parsed is not None else {"on_map": map_id}


def compile_steps_to_directives(steps: list[QuestStep]) -> list[Directive]:
    """Expand each QuestStep into its 1-2 Directives (TRAVEL + optional TALK_TO), tagged quest_id —
    following strategize()'s expansion shape. The model-authored done_when attaches to the talk step
    when there is one, else to the travel step (deliberate improvement over strategize, which forces
    on_map on the travel directive)."""
    out: list[Directive] = []
    for s in steps:
        crit = _criterion(s.done_when, s.map)
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


def reconcile_quests(current, proposal, *, next_id):
    """Deterministically merge L1's proposal into the canonical plan, preserving progress.
    Keeps done + active steps; removes only named pending steps; ALWAYS drops wedged steps (they are
    replaced by adds); inserts adds after the active step, dedup by (map, done_when)."""
    add = proposal.get("add") or []
    remove = set(proposal.get("remove") or [])
    done = [s for s in current if s.status == "done"]
    active = [s for s in current if s.status == "active"]
    pending = [s for s in current if s.status == "pending" and s.id not in remove]
    have = {(s.map, s.done_when or "on_map") for s in done + active + pending}
    new_steps = []
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
        new_steps.append(QuestStep(id=next_id(), map=mp, talk=bool(a.get("talk")),
                                   who=(a.get("who") or None), done_when=dw, why=str(a.get("why") or "")[:80]))
    return done + active + new_steps + pending
