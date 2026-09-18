"""Turn the objects/NPCs/exits we already extract into navigable GO-TO targets.

Each target is a plain dict the decider can offer as one `Choice` option and the
navigator can path to:  {key, label, desc, x, y, interact}.  `key` is a safe
criteria key; `interact` is True for things you press A on (people, Poké Balls,
signs) and False for exits (you just step onto them).
"""
from __future__ import annotations

import re


def _key(prefix: str, x: int, y: int) -> str:
    return f"goto_{re.sub(r'[^a-z0-9]+', '_', prefix.lower()).strip('_')}_{x}_{y}"


def build_targets(game_state: dict | None, exits: list[dict] | None) -> list[dict]:
    """Concrete places worth going, most-actionable first (NPCs/objects, then exits)."""
    targets: list[dict] = []
    seen: set[tuple[int, int]] = set()

    for n in (game_state or {}).get("npcs") or []:
        try:
            x, y = int(n["x"]), int(n["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if (x, y) in seen:
            continue
        seen.add((x, y))
        name = str(n.get("sprite") or "person")
        flags = []
        if n.get("talked_to"):
            flags.append("already talked to")
        if n.get("interact_did_nothing"):
            flags.append("A did nothing here before")
        note = f" ({'; '.join(flags)})" if flags else ""
        targets.append({
            "key": _key(name, x, y),
            "label": f"{name} ({x},{y})",
            "desc": f"Walk to {name} at ({x},{y}), face it, and press A{note}.",
            "x": x, "y": y, "interact": True,
        })

    for e in exits or []:
        try:
            x, y = int(e["x"]), int(e["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if (x, y) in seen:
            continue
        seen.add((x, y))
        dest = e.get("dest_name") or e.get("dest_map")
        targets.append({
            "key": _key("exit", x, y),
            "label": f"exit to {dest} ({x},{y})",
            "desc": f"Walk onto the exit at ({x},{y}) to leave toward {dest}.",
            "x": x, "y": y, "interact": False, "dest_map": e.get("dest_map"),
        })
    return targets
