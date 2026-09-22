"""Isolate the L1 DECIDE call and measure reliability of its structured output.

The run logs show L1 repeatedly emitting quest steps with a MISSING/invalid `kind`, which the
reconciler then rejects as malformed -> wedge -> re-plan -> thrash. This harness calls l1_decide N
times on a fixed, representative (Route 2 -> forest -> Pewter) situation and scores how many emitted
steps are well-formed (valid kind + a done_when consistent with that kind). Run before/after adding
json_schema structured output to see if the malformation is prompt-compliance (fixable by a schema).

Live (spends strategist credits): uv run python scripts/eval_l1_decide.py [N]
"""
import sys

sys.path.insert(0, "src")

VALID_DONE = ("on_map", "has_item:", "no_item:", "level>=", "badges>=", "hp_frac>=", "talked", "verify:")


def score_step(s: dict) -> tuple[bool, str]:
    kind = s.get("kind")
    if kind not in ("travel", "action"):
        return False, f"kind missing/invalid: {kind!r}"
    dw = s.get("done_when")
    if not isinstance(dw, str) or not any(dw == v or dw.startswith(v) for v in VALID_DONE):
        return False, f"done_when invalid: {dw!r}"
    if kind == "travel" and dw != "on_map":
        return False, f"travel step done_when must be on_map, got {dw!r}"
    if kind == "action" and dw == "on_map":
        return False, "action step cannot use on_map"
    if kind == "travel" and s.get("talk"):
        return False, "travel step must not talk"
    return True, "ok"


# The REAL provoking situation from the thrashing run: the plan has WEDGED/malformed steps and gaps,
# and brainstorm asks to re-emit them properly. This is where L1 re-emits steps -> and drops `kind`.
CONTEXT = {
    "current_map": 50,   # Viridian Forest South Gate (mid-corridor, as in the thrash)
    "party": ["Squirtle L8 20/25"],
    "items": [],
    "badges": 0,
    "plan": [
        # malformed/wedged steps exactly as they appeared in the run (missing kind, out of order)
        {"id": "q7", "map": 51, "done_when": "on_map", "status": "wedged", "why": "enter Viridian Forest"},
        {"id": "q8", "map": 2, "done_when": "on_map", "status": "wedged", "why": "reach Pewter City"},
        {"id": "q12", "map": 41, "done_when": "hp_frac>=0.95", "status": "wedged",
         "why": "heal at Pewter Pokecenter"},
    ],
    "signals": {"hp_frac": 0.8, "needs_emergency_heal": False},
    "mission": "Beat Brock at the Pewter City Gym for the first badge.",
    "milestone": "Cross Viridian Forest to Pewter City, grinding Squirtle to ~L11 on the way.",
}
BRAINSTORM = {"assessment": (
    "The forest and Pewter travel steps (q7/q8) and the heal step (q12) are marked wedged and appear "
    "malformed (missing the kind field), so they are unusable. Re-emit the corridor properly: a step "
    "to enter Viridian Forest, grind Squirtle to ~L11, travel on to Pewter City, and heal at the "
    "Pewter Poke Center — anchored to the existing plan, minimal edits.")}


def run(n, model=None, use_schema=False):
    from pokemon_agent.agent.planner_llm import Planner
    from pokemon_agent.providers.lunaroute import LunaRouteProvider
    strat = LunaRouteProvider(model=model)
    planner = Planner(goal_map=2, level_target=13, strategist=strat)
    tag = f"{model or 'account-default'}  schema={use_schema}"
    print(f"\n################ DECIDE x{n}  ({tag}) ################")
    total_steps = good_steps = empty = 0
    reasons = {}
    for i in range(n):
        # (schema wiring added in the next step; for now this measures the json_object baseline)
        out = planner.l1_decide(CONTEXT, BRAINSTORM)
        add = out.get("add") or []
        if not add:
            empty += 1
        for s in add:
            total_steps += 1
            ok, why = score_step(s)
            good_steps += ok
            if not ok:
                reasons[why] = reasons.get(why, 0) + 1
    print(f"  calls: {n}  |  empty add (no change): {empty}")
    print(f"  emitted steps: {total_steps}  |  well-formed: {good_steps}"
          f"  ({(good_steps/total_steps*100 if total_steps else 100):.0f}%)")
    if reasons:
        print("  malformations:")
        for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {c}x  {r}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for m in ["glm-5.3", "deepseek-4.1-flash"]:
        try:
            run(n, model=m)
        except Exception as e:
            print(f"model {m}: {type(e).__name__}: {e}")
