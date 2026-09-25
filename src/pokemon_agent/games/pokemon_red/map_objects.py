"""Interactable background objects per map (signs, PCs, machines, trash cans ...), ripped from pokered
by ``scripts/gen_map_objects.py``. RAM has no sprite for these, so without this table the agent can't
aim at "the PC" at all."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_STOP = {"the", "a", "an", "to", "of", "in", "on", "at", "use", "runs", "and", "with"}


@lru_cache(maxsize=1)
def _table() -> dict[int, list[dict]]:
    raw = json.loads((Path(__file__).with_name("map_objects.json")).read_text())
    return {int(k): v for k, v in raw.items()}


def objects_on(map_id) -> list[dict]:
    try:
        return list(_table().get(int(map_id), []))
    except (TypeError, ValueError):
        return []


def _tokens(s: str) -> set[str]:
    """Content words, possessives dropped ("Bill's PC" -> {pc}): a request for the PERSON "Bill" must
    never resolve to his PC."""
    words = re.findall(r"[a-z0-9.'é]+", str(s).lower())
    return {w for w in words if w not in _STOP and not w.endswith("'s") and len(w) >= 2}


def match_objects(objects: list[dict], name: str | None) -> list[dict]:
    """Objects whose name shares a content word with ``name`` ("the PC" -> Bill's PC)."""
    want = _tokens(name or "")
    return [o for o in objects if want & _tokens(o.get("name", ""))] if want else []
