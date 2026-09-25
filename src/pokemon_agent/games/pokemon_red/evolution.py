"""What a Pokémon will become (scripts/gen_evolutions.py rips the next step per species from pokered).

Player knowledge, shown next to each party member so L1 weighs a Pokémon by its future too
(runs/sleeves-mtmoon: a Magikarp benched as "dead weight" — Gyarados at L20)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _next() -> dict[str, list[str]]:
    try:
        return json.loads((Path(__file__).with_name("evolutions.json")).read_text())
    except Exception:
        return {}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def evolution_line(species: str | None) -> str | None:
    """'Kakuna at L7 -> Beedrill at L10' / 'Flareon with a Fire Stone / Jolteon with a ...'; None if it
    doesn't evolve (or is fully evolved)."""
    table = {_norm(k): v for k, v in _next().items()}
    steps = table.get(_norm(species or ""))
    if not steps:
        return None
    if len(steps) > 1:
        return " / ".join(steps)
    nxt = steps[0]
    rest = evolution_line(nxt.split(" at ")[0].split(" with ")[0].split(" by ")[0])
    return nxt + (f" -> {rest}" if rest else "")
