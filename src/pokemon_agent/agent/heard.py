"""What the agent has HEARD: dialogue parsed into complete messages, with who said it and where.

runs/ss-anne-20260923: an NPC told the agent "That bush in front of the shop is in the way. There
might be a way around." four times. The old log stored every typing frame as its own entry in a
30-entry ring ("That" / "That bush in front of t" / ...), so ~3 real messages survived, and L1 never
saw dialogue at all.

The game draws a message two lines at a time in the bottom box (tile rows 14 and 16): a line grows as
it types, then scrolls up. ``DialogueParser`` stitches those frames into one message. ``HeardLog``
closes a message when the box closes, tags the speaker (the sprite / object the player faced when it
opened) and keeps three horizons:

  * ``recent``  — verbatim messages from the last ``window`` steps (what happened lately);
  * ``by_map``  — verbatim distinct messages per map with repeat counts; over budget, the oldest
                   are folded into that map's summary;
  * ``digest``  — a long-term summary of leads and ideas across everything heard, re-summarized as
                   messages age out of ``recent``.

Summaries use an optional ``summarize(kind, previous, messages) -> str | None`` callable (an LLM);
without one — or when it fails — a deterministic compaction keeps the gist.
"""
from __future__ import annotations

import json
import re

from ..providers.parsing import strip_fences


class DialogueParser:
    """Stitch successive text-box frames into complete lines."""

    def __init__(self):
        self.lines: list[str] = []

    def restarted(self, visible: list[str]) -> bool:
        """The same box started over (A reopened the conversation on the frame it closed): the top
        line is the START of this message again, after the message had moved on."""
        top = " ".join((visible[0] if visible else "").split())
        return len(self.lines) >= 3 and bool(top) and self.lines[0].startswith(top)

    def feed(self, visible: list[str]) -> None:
        for v in visible:
            v = " ".join((v or "").split())
            if not v:
                continue
            last = self.lines[-1] if self.lines else None
            if last is not None and v != last and v.startswith(last):
                self.lines[-1] = v                     # the line is still typing
            elif any(v == x or x.startswith(v) for x in self.lines[-2:]):
                continue                               # already have it (scrolled up, or a partial)
            else:
                self.lines.append(v)

    def close(self) -> str:
        text = " ".join(self.lines)
        self.lines = []
        return re.sub(r"\s+", " ", text).strip()

    @property
    def open(self) -> bool:
        return bool(self.lines)


def _fmt(m: dict, map_names: bool = True) -> str:
    who = m.get("speaker") or "?"
    at = f" at ({m['at'][0]},{m['at'][1]})" if m.get("at") else ""
    where = f" in {m.get('map_name')}" if map_names and m.get("map_name") else ""
    rep = f" (heard {m['count']}x)" if m.get("count", 1) > 1 else ""
    return f'{who}{at}{where}: "{m["text"]}"{rep}'


class HeardLog:
    def __init__(self, *, window: int = 400, recent_max: int = 150, per_map: int = 15,
                 digest_chars: int = 1500, map_summary_chars: int = 600):
        self.window, self.recent_max, self.per_map = window, recent_max, per_map
        self.digest_chars, self.map_summary_chars = digest_chars, map_summary_chars
        self.recent: list[dict] = []
        self.by_map: dict[int, dict] = {}              # map -> {"messages": [...], "summary": str}
        self.digest: str = ""
        self._aged: list[dict] = []                    # left `recent`, not yet in the digest
        self._parser = DialogueParser()
        self._open: dict | None = None                 # the message being read (speaker, map, start)
        self._seq = 0

    # ---- intake ---------------------------------------------------------------------------------
    def observe(self, step: int, *, map_id, map_name: str | None, active: bool, lines: list[str],
                speaker: str | None = None, speaker_xy=None, in_battle: bool = False) -> dict | None:
        """One step. Returns the message that just CLOSED (if any)."""
        if in_battle or not active:
            return self._close(step)
        if not any((ln or "").strip() for ln in lines):
            return None
        if self._open is None or self._open["map"] != map_id:
            closed = self._close(step)
            self._open = {"map": map_id, "map_name": map_name, "speaker": speaker,
                          "at": list(speaker_xy) if speaker_xy else None, "step": step}
            self._parser.feed(lines)
            return closed
        if self._parser.restarted(lines):             # a new conversation with no idle frame between
            info = dict(self._open)
            closed = self._close(step)
            self._open = {**info, "step": step}
            self._parser.feed(lines)
            return closed
        if self._open.get("speaker") is None and speaker:
            self._open["speaker"], self._open["at"] = speaker, (list(speaker_xy) if speaker_xy else None)
        self._parser.feed(lines)
        return None

    def _close(self, step: int) -> dict | None:
        if self._open is None:
            return None
        text, info = self._parser.close(), self._open
        self._open = None
        if len(text) < 4:
            return None
        mid = info["map"]
        bucket = self.by_map.setdefault(mid, {"messages": [], "summary": ""})
        same = next((m for m in bucket["messages"]
                     if m["text"] == text and m.get("speaker") == info.get("speaker")), None)
        if same is not None:                           # a repeat: count it, surface it again
            same["count"] += 1
            same["last_step"] = step
            msg = same
        else:
            self._seq += 1
            msg = {"id": self._seq, "step": info["step"], "last_step": step, "map": mid,
                   "map_name": info.get("map_name"), "speaker": info.get("speaker"), "at": info.get("at"),
                   "text": text, "count": 1}
            bucket["messages"].append(msg)
        self.recent = [m for m in self.recent if m is not msg] + [msg]
        self._trim(step)
        return msg

    def _trim(self, step: int) -> None:
        keep = [m for m in self.recent if m["last_step"] >= step - self.window][-self.recent_max:]
        kept = {id(m) for m in keep}
        self._aged += [m for m in self.recent if id(m) not in kept]
        self.recent = keep

    # ---- summarization (bounded memory) ---------------------------------------------------------
    def compact(self, summarize=None, *, min_aged: int = 6) -> bool:
        """Fold overflow into summaries. Per map: beyond ``per_map`` distinct messages, the oldest go
        into that map's summary. Globally: messages that aged out of ``recent`` go into the digest.
        Returns True when anything was summarized."""
        changed = False
        for mid, b in self.by_map.items():
            over = len(b["messages"]) - self.per_map
            if over <= 0:
                continue
            old = sorted(b["messages"], key=lambda m: m["last_step"])[:over]
            ids = {id(m) for m in old}
            b["messages"] = [m for m in b["messages"] if id(m) not in ids]
            b["summary"] = self._summarize(summarize, "map", b["summary"], old, self.map_summary_chars)
            changed = True
        if len(self._aged) >= min_aged:
            self.digest = self._summarize(summarize, "digest", self.digest, self._aged, self.digest_chars)
            self._aged = []
            changed = True
        return changed

    @staticmethod
    def _summarize(summarize, kind: str, previous: str, msgs: list[dict], limit: int) -> str:
        out = None
        if summarize is not None:
            try:
                out = summarize(kind, previous, [_fmt(m) for m in msgs])
            except Exception:
                out = None
        if not out:                                    # deterministic fallback: keep the gist
            add = "; ".join(f"{m.get('speaker') or '?'} ({m.get('map_name')}): {m['text'][:90]}" for m in msgs)
            out = (previous + " | " if previous else "") + add
        out = " ".join(str(out).split())
        return out if len(out) <= limit else "…" + out[-limit:]

    # ---- views ----------------------------------------------------------------------------------
    def since(self, step: int, limit: int = 12) -> list[str]:
        """Messages first heard or heard AGAIN after ``step`` (newest last)."""
        return [_fmt(m) for m in self.recent if m["last_step"] > step][-limit:]

    def here(self, map_id, limit: int = 10) -> dict:
        b = self.by_map.get(map_id) or {}
        msgs = sorted(b.get("messages", []), key=lambda m: m["last_step"])[-limit:]
        return {"summary": b.get("summary", ""), "messages": [_fmt(m, map_names=False) for m in msgs]}

    def distinct_count(self) -> int:
        return sum(len(b["messages"]) for b in self.by_map.values())

    # ---- persistence ----------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"recent_ids": [m["id"] for m in self.recent], "digest": self.digest,
                "aged_ids": [m["id"] for m in self._aged], "seq": self._seq,
                "by_map": {str(k): v for k, v in self.by_map.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "HeardLog":
        h = cls()
        h.by_map = {int(k): v for k, v in (d.get("by_map") or {}).items()}
        h.digest = d.get("digest") or ""
        h._seq = int(d.get("seq") or 0)
        by_id = {m["id"]: m for b in h.by_map.values() for m in b["messages"]}
        h.recent = [by_id[i] for i in d.get("recent_ids") or [] if i in by_id]
        h._aged = [by_id[i] for i in d.get("aged_ids") or [] if i in by_id]
        return h


SUMMARIZE_SYSTEM = """You keep the long-term memory of an agent playing Pokémon Red. You are given
PREVIOUS (the summary so far) and NEW (things people, signs and objects said, with who and where).
Write an updated summary of what's worth remembering: hints, directions, requests, warnings, where
things are, who wants what, and anything that suggests a way forward (e.g. "a bush blocks the path
by the Cerulean shop; someone said there might be a way around"). Keep facts and leads; drop small
talk. Name places and people. Stay under {limit} characters.
Return ONLY JSON: {{"summary": "<text>"}}"""


def llm_summarizer(provider):
    """A ``summarize`` callable backed by a chat provider (None when there's no provider)."""
    if provider is None:
        return None

    def summarize(kind: str, previous: str, messages: list[str]) -> str | None:
        limit = 600 if kind == "map" else 1500
        raw, _lat, _usage = provider.chat_json(SUMMARIZE_SYSTEM.format(limit=limit),
                                               {"previous": previous, "new": messages})
        data = json.loads(strip_fences(raw))
        return str(data.get("summary") or "") or None
    return summarize
