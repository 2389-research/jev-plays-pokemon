"""Which TM/HM moves a species can learn (scripts/gen_tmhm.py rips pokered's learnsets).

Player knowledge: before the agent can choose to use Cut it has to know who can learn it
(runs/sleeves-misty: none of Wartortle, Weedle, Magikarp can — but Weedle's Beedrill can)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

HMS = ("Cut", "Fly", "Surf", "Strength", "Flash")
FIELD_MOVES = HMS + ("Dig", "Teleport", "Softboiled")   # moves with a use from the party menu


@lru_cache(maxsize=1)
def _data() -> dict:
    try:
        return json.loads((Path(__file__).with_name("tmhm.json")).read_text())
    except Exception:
        return {"machines": {}, "learnsets": {}}


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def machine_move(item: str) -> str | None:
    """'HM01' / 'HM01 Cut' / 'TM28 Dig' -> the move it teaches."""
    m = re.match(r"\s*((?:HM|TM)\s*0*\d+)", str(item or ""), re.I)
    if not m:
        return None
    code = re.sub(r"\s+", "", m.group(1)).upper()
    code = code[:2] + f"{int(code[2:]):02d}"
    return _data()["machines"].get(code)


def learnable(species) -> list[str]:
    table = {_norm(k): v for k, v in _data()["learnsets"].items()}
    return list(table.get(_norm(species), []))


def can_learn(species, move) -> bool:
    return _norm(move) in {_norm(m) for m in learnable(species)}


def learners(move) -> list[str]:
    return sorted(s for s, ms in _data()["learnsets"].items() if _norm(move) in {_norm(m) for m in ms})


def hm_line(species) -> str | None:
    """HMs this species can learn now, plus what an evolution adds: 'Surf, Strength' or
    'none; as Beedrill: Cut'. None when neither it nor its evolutions learn any."""
    from .evolution import evolution_line
    now = [h for h in HMS if can_learn(species, h)]
    seen, later = set(now), []
    for step in (evolution_line(species) or "").split(" -> "):
        evo = re.split(r" at | with | by ", step.split(" / ")[0])[0].strip()
        extra = [h for h in HMS if evo and can_learn(evo, h) and h not in seen]
        if extra:
            later.append(f"as {evo}: {', '.join(extra)}")
            seen |= set(extra)
    if not now and not later:
        return None
    return "; ".join([", ".join(now) or "none"] + later)
