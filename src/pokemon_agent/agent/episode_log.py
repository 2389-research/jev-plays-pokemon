"""What happened since L1's last review, and every attempt L1 has made — including removed steps.

runs/ss-anne-20260923: L1 re-issued "travel to Vermilion" ~10 times and flip-flopped on the same fact
4 times, because each review saw a snapshot: a wedged step was deleted from the plan when replaced
(its failure vanished with it), nothing said which maps the agent had bounced between, and the
notepad (L1's own summary) never mentioned the failures. This log is harness-written ground truth:

  * events  — maps entered, steps done / wedged / removed (with the real reason), battles,
              discoveries, portals found blocked; ``since(step)`` summarizes them for L1;
  * attempts — a ledger keyed by what the step asked for (kind, map, who, done_when): how often it
              was tried and how each try ended. Identical re-issues aggregate, so "tried 9x, wedged 9x"
              is visible even after the steps themselves are gone.
"""
from __future__ import annotations

from ..games.pokemon_red.maps import map_name


def step_desc(s) -> str:
    """A short human label for a QuestStep."""
    kind = getattr(s, "kind", "action")
    where = map_name(s.map)
    if kind == "travel":
        return f"travel to {where}"
    if kind == "explore":
        return f"explore {where}" + (f" (prefer {s.who})" if s.who else "")
    who = f"talk to {s.who} in " if s.talk and s.who else ("talk in " if s.talk else "")
    return f"{who}{where} until {s.done_when}"


class EpisodeLog:
    def __init__(self, max_events: int = 500):
        self.max_events = max_events
        self.events: list[dict] = []
        self.attempts: dict[str, dict] = {}
        self.last_review_step = 0

    # ---- recording ------------------------------------------------------------------------------
    def record(self, step: int, type_: str, **data) -> None:
        self.events.append({"step": step, "type": type_, **data})
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    @staticmethod
    def _key(s) -> str:
        return f"{getattr(s, 'kind', 'action')}|{s.map}|{(s.who or '').lower()}|{s.done_when or ''}"

    def attempt(self, step: int, s, outcome: str, reason: str = "") -> None:
        """A plan step started ("started") or ended ("done" / "wedged" / "removed")."""
        a = self.attempts.setdefault(self._key(s), {"what": step_desc(s), "tries": 0, "done": 0, "wedged": 0,
                                                     "removed": 0, "first_step": step, "last_step": step,
                                                     "last_reason": ""})
        a["last_step"] = step
        if outcome == "started":
            a["tries"] += 1
        elif outcome in ("done", "wedged", "removed"):
            a[outcome] += 1
            if reason:
                a["last_reason"] = reason[:200]
        self.record(step, f"step_{outcome}", what=step_desc(s), id=s.id,
                    **({"reason": reason[:200]} if reason else {}))

    def mark_review(self, step: int) -> None:
        self.last_review_step = step

    # ---- views ----------------------------------------------------------------------------------
    def since(self, step: int | None = None, *, now: int) -> dict:
        """A compact account of what happened after ``step`` (default: the last review)."""
        step = self.last_review_step if step is None else step
        evs = [e for e in self.events if e["step"] > step]
        out: dict = {"steps": now - step}
        maps = [e["map_name"] for e in evs if e["type"] == "map_enter"]
        if maps:
            out["maps_entered"] = self._runs(maps)
            counts: dict[str, int] = {}
            for m in maps:
                counts[m] = counts.get(m, 0) + 1
            repeats = {m: c for m, c in counts.items() if c >= 3}
            if repeats:
                out["entered_repeatedly"] = repeats
        changes = [f"{e['type'][5:]}: {e['what']}" + (f" — {e['reason']}" if e.get("reason") else "")
                   for e in evs if e["type"] in ("step_done", "step_wedged", "step_removed")]
        if changes:
            out["plan_changes"] = changes[-10:]
        for typ, key in (("battle", "battles"), ("discovery", "discoveries"),
                         ("portal_blocked", "blocked"), ("explore_end", "explored"), ("action", "actions")):
            items = [e.get("text") for e in evs if e["type"] == typ and e.get("text")]
            if items:
                out[key] = items[-6:]
        return out

    @staticmethod
    def _runs(seq: list[str]) -> str:
        """'A, B, A, B, A' -> 'A -> B -> A -> B -> A' collapsed to run-lengths when it repeats a lot."""
        if len(seq) <= 8:
            return " -> ".join(seq)
        return " -> ".join(seq[:3]) + f" -> ... ({len(seq) - 6} more) ... -> " + " -> ".join(seq[-3:])

    def attempts_view(self, limit: int = 8) -> list[str]:
        """Attempts that FAILED at least once, most recent first — so a retry loop is visible."""
        rows = [a for a in self.attempts.values() if a["wedged"] or a["removed"]]
        rows.sort(key=lambda a: a["last_step"], reverse=True)
        out = []
        for a in rows[:limit]:
            tail = f" — last: {a['last_reason']}" if a["last_reason"] else ""
            out.append(f"{a['what']}: tried {a['tries']}x (done {a['done']}, wedged {a['wedged']}, "
                       f"removed {a['removed']}){tail}")
        return out

    # ---- persistence ----------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"events": self.events[-200:], "attempts": self.attempts, "last_review_step": self.last_review_step}

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeLog":
        e = cls()
        e.events = list(d.get("events") or [])
        e.attempts = dict(d.get("attempts") or {})
        e.last_review_step = int(d.get("last_review_step") or 0)
        return e
