"""Outcome labeler (design §4) — a PURE post-processor over a finished record-dir. Reads
`log.jsonl`, computes a per-step progress vector and a per-episode `outcome.json`. Runs
offline on any past run; no emulator, no network.

Grounding note (verified against RunRecorder): the base `log.jsonl` does NOT carry a
`caught` event, a `battle_end.result`, or a `deliver` intent. So catches are derived from
party-length growth (the recorder writes `party` every step); `battle_result`/`delivered`
are best-effort (parse `detail`, prefer structured `extra` on capture runs)."""
from __future__ import annotations

import json
import re
from pathlib import Path

_LEVEL = re.compile(r"L(\d+)")
_HP = re.compile(r"(\d+)/(\d+)")


def _party_level_hp(party: list[str]) -> tuple[int, int]:
    """Sum of levels and current HP across a party of `"SPECIES L# hp/max"` strings."""
    lvl = hp = 0
    for s in party or []:
        m = _LEVEL.search(s or "")
        if m:
            lvl += int(m.group(1))
        h = _HP.search(s or "")
        if h:
            hp += int(h.group(1))
    return lvl, hp


def step_progress(prev: dict | None, cur: dict) -> dict:
    """The per-step progress delta (spec §4). `map_hop_delta` needs PortalGraph + component,
    so it is left out here (added by the export join when available) — this is the log-only
    directional signal."""
    prev = prev or {}
    p_party, c_party = prev.get("party") or [], cur.get("party") or []
    plvl, php = _party_level_hp(p_party)
    clvl, chp = _party_level_hp(c_party)
    events = cur.get("events") or []
    kinds = {e.get("kind") for e in events}
    # battle_result is best-effort: the base log's battle_end event carries no result key, so this
    # is None on real logs unless a structured flag is added later. Kept for forward-compat.
    battle = next((e.get("result") for e in events if e.get("kind") == "battle_end"), None)
    return {
        "map_changed": prev.get("map_id") != cur.get("map_id") and prev != {},
        "level_delta": clvl - plvl,
        "hp_delta": chp - php,
        "items_delta": len(cur.get("items") or []) - len(prev.get("items") or []),
        "battle_result": battle,
        "caught": len(c_party) > len(p_party),   # party grew => a catch (log-only signal)
        "wedged": any(k in kinds for k in ("step_wedged", "quest_step_wedged")),
    }


def iter_rows(record_dir: str | Path):
    p = Path(record_dir) / "log.jsonl"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def label_run(record_dir: str | Path, *, goal_map: int | None = None,
              terminal_reason: str = "end") -> dict:
    """Compute + write `outcome.json` for a finished run; returns the outcome dict."""
    record_dir = Path(record_dir)
    rows = list(iter_rows(record_dir))
    caught = 0
    arrivals: dict[str, int] = {}
    reached = False
    prev_party = None
    for r in rows:
        party_n = len(r.get("party") or [])
        if prev_party is not None and party_n > prev_party:
            caught += party_n - prev_party      # a catch => the party grew (log-only signal)
        prev_party = party_n
        mid = r.get("map_id")
        if mid is not None and str(mid) not in arrivals:
            arrivals[str(mid)] = r.get("step")
        if goal_map is not None and mid == goal_map:
            reached = True
    out = {
        "reached_goal_map": reached,
        "goal_map": goal_map,
        "caught_count": caught,
        # best-effort in Phase 1 (no clean persisted signal): parse detail, else structured extra.
        "delivered": any("deliver" in str(r.get("detail") or "").lower() for r in rows),
        "steps": len(rows),
        "terminal_reason": terminal_reason,
        "map_arrival_steps": arrivals,
    }
    try:
        (record_dir / "outcome.json").write_text(json.dumps(out, indent=2))
    except Exception:
        pass
    return out
