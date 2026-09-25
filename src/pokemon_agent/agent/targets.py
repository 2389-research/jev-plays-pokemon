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


def _name_hit(npc: dict, name: str) -> bool:
    nm = str(npc.get("sprite") or "").lower()
    s = str(name).lower()
    return bool(nm) and (nm in s or s in nm)


def name_matches(npcs: list[dict], name: str | None) -> list[dict]:
    """NPCs whose sprite name matches `name` (case-insensitive substring either way)."""
    return [n for n in npcs if name and _name_hit(n, name)] if name else []


def npc_key(npc: dict, map_id) -> tuple:
    """Identity of an NPC on a map: its sprite slot when known (survives wandering), else its tile."""
    if npc.get("slot") is not None:
        return (map_id, int(npc["slot"]))
    return (map_id, int(npc["x"]), int(npc["y"]))


def select_npc(npcs: list[dict], *, sprite: str | None, picked, player, want_kind: str = "person",
               chooser=None, tried=None) -> dict | None:
    """Choose which sprite an ``approach_npc`` target means. Pure, so it's testable without a loop.

    1. **Candidate pool**: the sprites matching the requested name if any do; otherwise the sprites
       of the wanted kind (``"item"`` for a GRAB_ITEM errand, ``"person"`` otherwise — a sprite with
       no ``kind`` counts as a person); all sprites only if that set is empty. Everything below
       works WITHIN the pool, so a person request can never land on an item ball.
    2. **Track the leg's pick** by locality (robust to moving / duplicate-named NPCs) — but only a
       pick made on THIS map (``picked = [x, y, map_id]``); a pick from another map (e.g. cached on
       a torn warp frame) or a legacy ``[x, y]`` pick is ignored.
    3. ``chooser(pool)`` (the calibrated Jev pick) when there are several candidates.
    4. Nearest not-yet-talked candidate.

    ``tried`` (``npc_key``s of NPCs whose conversations didn't achieve the step) are removed from the
    pool AFTER the kind fallback; None when every candidate has been tried (the caller wedges).
    """
    if not npcs:
        return None
    import re
    at = re.search(r"\((\d+)\s*,\s*(\d+)\)", str(sprite or ""))
    if at:
        # a request naming a TILE ("Rocket at (11,2)", "Lift Key item ball at (11,2)"): the matching sprite
        # nearest that tile — runs/sleeves-hideout picked the Rocket at (23,12), in another area, for
        # "Rocket at (11,2)" (right next to the player), and Giovanni for the key ball ~700 steps running
        ax, ay = int(at.group(1)), int(at.group(2))
        name = re.sub(r"\s*(at|near|on)?\s*\(\d+\s*,\s*\d+\).*$", "", str(sprite), flags=re.I).strip()
        pool = name_matches(npcs, name)
        if not pool:
            itemish = re.search(r"\b(ball|item|key|pickup)\b", str(sprite), re.I)
            pool = [n for n in npcs if (n.get("kind") == "item") == bool(itemish)] or list(npcs)
        if tried:
            pmap0 = getattr(player, "map_id", None)
            pool = [n for n in pool if npc_key(n, pmap0) not in {tuple(t) for t in tried}] or []
            if not pool:
                return None
        return min(pool, key=lambda n: (abs(int(n["x"]) - ax) + abs(int(n["y"]) - ay),
                                        abs(int(n["x"]) - player.x) + abs(int(n["y"]) - player.y)))
    pool = name_matches(npcs, sprite)
    if not pool:
        if want_kind == "item":
            pool = [n for n in npcs if n.get("kind") == "item"]
        else:
            pool = [n for n in npcs if n.get("kind") != "item"]
        pool = pool or list(npcs)
    pmap = getattr(player, "map_id", None)
    if tried:
        tried = {tuple(t) for t in tried}
        pool = [n for n in pool if npc_key(n, pmap) not in tried]
        if not pool:
            return None
    if isinstance(picked, (list, tuple)) and len(picked) >= 3 and picked[2] == pmap:
        px, py = int(picked[0]), int(picked[1])
        return min(pool, key=lambda n: abs(int(n["x"]) - px) + abs(int(n["y"]) - py))
    if chooser is not None and len(pool) > 1:
        npc = chooser(pool)
        if npc is not None:
            return npc
    fresh = [n for n in pool if not n.get("talked_to")] or pool
    return min(fresh, key=lambda n: abs(int(n["x"]) - player.x) + abs(int(n["y"]) - player.y))
