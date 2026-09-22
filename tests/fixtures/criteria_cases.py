"""Fixture situations for the Layer 2.5 criteria-quality eval (scripts/eval_criteria.py).

Each case mirrors the ``context`` shape `Planner.l1_decide` expects — the same keys
`ReasonLoop._run_l1` builds: current_map, party, items, badges, plan, signals, mission,
milestone (see src/pokemon_agent/agent/reason_loop.py:_run_l1 and
src/pokemon_agent/agent/planner_llm.py:l1_decide). ``party`` uses the real `read_party` shape
(species/level/hp/max_hp/...); ``items`` is a list of item name strings, matching
`game_signals`.

Each case has either:
  {"expect_exact": "<the exact done_when a correct step should emit>"}, or
  {"expect_family": "<the _parse_done_when predicate family a correct step should emit>"}
"""
from __future__ import annotations


def _party(species: str, level: int, hp: int, max_hp: int) -> list[dict]:
    return [{
        "species": species, "nickname": species.upper(), "level": level,
        "hp": hp, "max_hp": max_hp, "status": "OK", "moves": [],
    }]


def _signals(party: list[dict], items: list[str], badges: int, *,
             blocked_for_n: int = 0, emergency_heal: bool = False) -> dict:
    total = sum(p["max_hp"] for p in party) or 1
    cur = sum(p["hp"] for p in party)
    return {
        "party": party,
        "hp_frac": cur / total,
        "min_level": min((p["level"] for p in party), default=None),
        "badges": badges,
        "items": list(items),
        "blocked_for_n": blocked_for_n,
        "emergency_heal": emergency_heal,
    }


CASES = [
    {
        "name": "deliver_parcel",
        "expect_exact": "no_item:oaks_parcel",
        "context": {
            "current_map": {"id": 1, "name": "Viridian City"},
            "party": _party("Squirtle", 7, 14, 23),
            "items": ["Oak's Parcel"],
            "badges": 0,
            "plan": [],
            "signals": _signals(_party("Squirtle", 7, 14, 23), ["Oak's Parcel"], 0),
            "mission": "Reach Pewter City and beat Brock",
            "milestone": "You are carrying Oak's Parcel — deliver it to Professor Oak in his "
                         "lab in Pallet Town before the Viridian gatekeeper will let you pass "
                         "north toward Pewter City.",
        },
    },
    {
        "name": "pickup_parcel",
        "expect_exact": "has_item:oaks_parcel",
        "context": {
            "current_map": {"id": 1, "name": "Viridian City"},
            "party": _party("Squirtle", 7, 20, 23),
            "items": [],
            "badges": 0,
            "plan": [],
            "signals": _signals(_party("Squirtle", 7, 20, 23), [], 0),
            "mission": "Reach Pewter City and beat Brock",
            "milestone": "The Viridian Mart clerk is holding Oak's Parcel for Professor Oak; "
                         "you need to obtain it before you can deliver it and be allowed north.",
        },
    },
    {
        "name": "heal_low_hp",
        "expect_family": "hp_frac",
        "context": {
            "current_map": {"id": 1, "name": "Viridian City"},
            "party": _party("Squirtle", 7, 2, 23),
            "items": [],
            "badges": 0,
            "plan": [],
            "signals": _signals(_party("Squirtle", 7, 2, 23), [], 0, emergency_heal=True),
            "mission": "Reach Pewter City and beat Brock",
            "milestone": "Squirtle is nearly fainted (2/23 HP); heal the party at the Viridian "
                         "Poké Center before doing anything else.",
        },
    },
    {
        "name": "reach_pewter",
        "expect_family": "on_map",
        "context": {
            "current_map": {"id": 1, "name": "Viridian City"},
            "party": _party("Squirtle", 12, 40, 40),
            "items": [],
            "badges": 0,
            "plan": [],
            "signals": _signals(_party("Squirtle", 12, 40, 40), [], 0),
            "mission": "Reach Pewter City and beat Brock",
            "milestone": "You are healthy and ready for Brock — travel north from Viridian "
                         "City to Pewter City (map id 2).",
        },
    },
    {
        "name": "grind_underleveled",
        "expect_family": "level",
        "context": {
            "current_map": {"id": 1, "name": "Viridian City"},
            "party": _party("Squirtle", 7, 23, 23),
            "items": [],
            "badges": 0,
            "plan": [],
            "signals": _signals(_party("Squirtle", 7, 23, 23), [], 0),
            "mission": "Reach Pewter City and beat Brock",
            "milestone": "Squirtle is too weak (level 7) for Brock; grind in the grass near "
                         "Viridian City to reach about level 12 before challenging the gym.",
        },
    },
]
