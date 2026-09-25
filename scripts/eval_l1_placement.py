"""Live isolation eval for L1 step placement (spec docs/superpowers/specs/2026-09-22-l1-step-placement-design.md §4).

Replays CAPTURED DECIDE inputs (runs/fix-accept-20260922-2259/decisions.jsonl) through the current
Planner.l1_decide + reconcile_quests, plus two synthetic negative controls. Each case runs N times:

  held-out  step-25 incident: the Pewter travel / gym adds must reconcile AFTER the pending delivery q2
  (a)       step-138 wedge replacement: the replacement must go to the FRONT (no after -> q2 / "end")
  (b)       emergency heal with a pending plan: the heal must be default-placed (NEXT)
  (c)       steady state (correct, ordered plan): no edits at all

Replay mapping: a captured record's `input` is the DECIDE `state` dict ->
l1_decide(context=input, brainstorm={"assessment": input["brainstorm"]}).

Live (spends strategist credits): uv run python scripts/eval_l1_placement.py [N] [model]
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from pokemon_agent.agent.l1_pipeline import validate_step  # noqa: E402
from pokemon_agent.agent.quest_reconciler import QuestStep, reconcile_quests  # noqa: E402

DECISIONS = Path("runs/fix-accept-20260922-2259/decisions.jsonl")
ON_PATH = {0, 40, 12}   # Pallet Town, Oak's Lab, Route 1 — the way to the delivery


def captured(step: int) -> dict:
    for line in DECISIONS.read_text().splitlines():
        r = json.loads(line)
        if r["layer"] == "l1_decide" and r["step"] == step:
            return r["input"]
    raise SystemExit(f"no captured l1_decide input at step {step}")


def plan_steps(ctx: dict) -> list[QuestStep]:
    return [QuestStep(id=p["id"], map=p["map"], kind=p.get("kind", "action"), talk=bool(p.get("talk")),
                      done_when=p.get("done_when"), status=p.get("status", "pending")) for p in ctx["plan"]]


def ids():
    n = [100]

    def nxt():
        n[0] += 1
        return f"n{n[0]}"
    return nxt


def heal_case(base: dict) -> dict:
    ctx = copy.deepcopy(base)
    ctx["party"] = [{**ctx["party"][0], "hp": 3}]
    ctx["signals"] = {**ctx["signals"], "party": ctx["party"], "hp_frac": 0.12, "emergency_heal": True}
    ctx["brainstorm"] = ("SQUIRTLE is at 3/25 HP (emergency). Heal at the Viridian City Pokémon Center "
                         "first, then continue south to Pallet Town and deliver Oak's Parcel.")
    return ctx


def steady_case(base: dict) -> dict:
    ctx = copy.deepcopy(base)
    ctx["current_map"] = {"id": 12, "name": "Route 1"}
    ctx["plan"] = [
        {"id": "q1", "map": 0, "kind": "travel", "talk": False, "done_when": "on_map", "status": "active"},
        {"id": "q2", "map": 40, "kind": "action", "talk": True, "done_when": "no_item:Oaks Parcel", "status": "pending"},
        {"id": "q3", "map": 2, "kind": "travel", "talk": False, "done_when": "on_map", "status": "pending"},
        {"id": "q4", "map": 54, "kind": "action", "talk": True, "done_when": "badges>=1", "status": "pending"},
    ]
    ctx["brainstorm"] = ("The plan is correct and in order: finish walking south to Pallet Town, deliver "
                         "Oak's Parcel to Professor Oak, then head north to Pewter City and challenge Brock. "
                         "We're making progress on Route 1 — just continue the active step.")
    return ctx


def call_failed(out: dict) -> bool:
    """l1_decide never raises: a failed call (transport/parse error) returns {"add": [], "remove": []}
    with NO assessment. That's not a model decision, so it's reported separately and excluded from the
    pass rate (in production the pipeline treats it as "no change" and keeps the standing plan)."""
    return not (out or {}).get("assessment") and not (out or {}).get("add") and not (out or {}).get("remove")


def judge(case: str, ctx: dict, out: dict) -> tuple[bool, str]:
    adds, remove = out.get("add") or [], out.get("remove") or []
    bad = [validate_step(a)[1] for a in adds if not validate_step(a)[0]]
    if bad:
        return False, f"malformed step(s): {bad}"
    events = []
    result = reconcile_quests(plan_steps(ctx), {"add": adds, "remove": remove}, next_id=ids(),
                              on_event=lambda k, p: events.append(p))
    order = [(s.id, s.map, s.status) for s in result]
    anchors = [a.get("after") for a in adds]
    tag = f"anchors={anchors} order={[(i, m) for i, m, st in order if st != 'done']}" + (f" fallbacks={events}" if events else "")
    if case == "held-out":
        # Outcomes: MISORDERED (any new step OFF the delivery path lands before the pending delivery —
        # the incident) = fail; ANCHORED (off-path steps added, all after the delivery) = pass;
        # DEFERRED (no off-path steps added) = safe, placement not exercised; DELIVERY_REMOVED = L1
        # dropped/replaced the delivery step (its own outcome, never a crash).
        live = [s for s in result if s.status != "done"]
        deliv = [i for i, s in enumerate(live) if (s.done_when or "").lower().startswith("no_item:oak")]
        if not deliv:
            return False, "DELIVERY_REMOVED " + tag
        d = deliv[0]
        off_path = [i for i, s in enumerate(live) if s.id.startswith("n") and s.map not in ON_PATH
                    and not (s.done_when or "").startswith("hp_frac")]
        if any(i < d for i in off_path):
            return False, "MISORDERED " + tag
        if off_path:
            return True, "ANCHORED " + tag
        return False, "DEFERRED — no off-path step added (" + tag + ")"
    if case == "(a) wedge-replacement":
        # the REPLACEMENT (the re-added travel to Pallet, map 0) must be default-placed at the front;
        # other adds anchored after q2 (e.g. a post-delivery heal the brainstorm asks for) are correct use
        repl = [a for a in adds if a.get("map") == 0]
        live = [s for s in result if s.status != "done"]
        first_is_repl = bool(live) and live[0].id.startswith("n") and live[0].map == 0
        repl_unanchored = bool(repl) and all(a.get("after") in (None,) for a in repl)
        return first_is_repl and repl_unanchored, tag
    if case == "(b) emergency-heal":
        heal = [a for a in adds if str(a.get("done_when", "")).startswith("hp_frac")]
        if not heal:
            return False, "no heal step (" + tag + ")"
        live = [s for s in result if s.status != "done"]
        heal_ids = [s.id for s in result if s.done_when and s.done_when.startswith("hp_frac")]
        pos = [i for i, s in enumerate(live) if s.id in heal_ids]
        return (bool(pos) and pos[0] <= 1), tag     # right after the active step (or first if none)
    if case == "(c) steady-state":
        return (not adds and not remove), tag
    raise ValueError(case)


def use_prompt_from(rev: str) -> None:
    """Swap in DECIDE_SYSTEM from a git revision (baseline comparison: the old prompt never emits
    `after`, so the new reconciler reproduces the old placement exactly)."""
    import ast
    import subprocess
    from pokemon_agent.agent import planner_llm
    src = subprocess.check_output(["git", "show", f"{rev}:src/pokemon_agent/agent/planner_llm.py"], text=True)
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DECIDE_SYSTEM" for t in node.targets):
            planner_llm.DECIDE_SYSTEM = ast.literal_eval(node.value)
            return
    raise SystemExit(f"DECIDE_SYSTEM not found at {rev}")


def main(n: int, model: str, prompt_rev: str | None = None, only: str | None = None) -> int:
    from pokemon_agent.agent.planner_llm import Planner
    from pokemon_agent.providers.lunaroute import LunaRouteProvider

    if prompt_rev:
        use_prompt_from(prompt_rev)
        print(f"(DECIDE_SYSTEM from {prompt_rev})")
    planner = Planner(goal_map=2, level_target=12, strategist=LunaRouteProvider(model=model))
    s25 = captured(25)
    cases = [("held-out", s25), ("(a) wedge-replacement", captured(138)),
             ("(b) emergency-heal", heal_case(s25)), ("(c) steady-state", steady_case(s25))]
    if only:
        cases = [c for c in cases if c[0] == only]
    all_ok = True
    for name, ctx in cases:
        passes = failed = 0
        outcomes: dict[str, int] = {}
        print(f"== {name}  (model={model}, N={n})")
        for i in range(n):
            out = planner.l1_decide(ctx, {"assessment": ctx["brainstorm"]})
            if call_failed(out):
                failed += 1
                print(f"   run {i + 1}: CALL_FAILED (empty fallback, excluded)")
                continue
            ok, why = judge(name, ctx, out)
            passes += ok
            key = why.split()[0] if why and why.split()[0].isupper() else ("PASS" if ok else "FAIL")
            outcomes[key] = outcomes.get(key, 0) + 1
            print(f"   run {i + 1}: {'PASS' if ok else 'FAIL'}  {why}")
        done = n - failed
        need = max(1, round(0.8 * done)) if done else 1
        verdict = done > 0 and passes >= need
        all_ok &= verdict
        print(f"   -> {passes}/{done} successful calls ({failed} failed calls excluded) {outcomes} "
              f"{'OK' if verdict else 'BELOW THRESHOLD'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=5)
    ap.add_argument("model", nargs="?", default="glm-5.3")
    ap.add_argument("--prompt-rev", default=None, help="use DECIDE_SYSTEM from this git revision (baseline)")
    ap.add_argument("--only", default=None, help="run a single case, e.g. held-out")
    a = ap.parse_args()
    raise SystemExit(main(a.n, a.model, a.prompt_rev, a.only))
