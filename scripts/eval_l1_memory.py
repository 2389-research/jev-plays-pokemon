"""Live evals: does L1 make good decisions from the new memory (spec 2026-09-24)?

Each scenario starts from a REAL captured L1 input from a run that got stuck, with the memory fields
the loop now builds (since_last_review / attempts / heard / unexplored_here / stall) filled in from
what actually happened in that run. Brainstorm and decide run LIVE; graders are structural.

  stuck-retry   runs/ss-anne step 886: "travel to Vermilion" failed 9x, bounced Cerulean <-> Route 4,
                heard "the bush ... is in the way. There might be a way around." 4x, the trashed house
                door is unexplored. PASS = no plain re-issue of the failed travel as the fix, and a step
                that investigates Cerulean (explore on map 3/62, or any step on map 62).
  heard-hint    runs/vermilion-team step 1116: Bill went into his teleporter, "talk to Bill" wedged
                (nobody here), Bill said "go to my PC and run the Cell Separation System!".
                PASS = a step on map 88 aimed at the PC (who matches it) — or an explore step there.
  no-overreact  the ss-anne context mid-journey with progress and nothing failing (only unexplored
                doors listed). PASS = triage says no change, or no explore / no removal of the active step.

Live (spends credits): uv run python scripts/eval_l1_memory.py [--n 5] [--only a,b] [--kb]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

sys.path.insert(0, "src")

from pokemon_agent.agent.l1_pipeline import run_l1_pipeline  # noqa: E402
from pokemon_agent.games.pokemon_red.map_objects import match_objects, objects_on  # noqa: E402


def captured(run: str, step: int) -> dict:
    for line in open(f"runs/{run}/decisions.jsonl"):
        r = json.loads(line)
        if r["layer"] == "l1_decide" and r["step"] == step:
            inp = r["input"]
            inp = inp.get("state", inp)
            return {k: v for k, v in inp.items() if k not in ("brainstorm", "maps")}
    raise SystemExit(f"no captured l1_decide at {run} step {step}")


BUSH = ('Cooltrainer M at (15,18) in Cerulean City: "That bush in front of the shop is in the way. '
        'There might be a way around." (heard 4x)')


def stuck_retry():
    ctx = captured("ss-anne-20260923", 886)
    for s in ctx["plan"]:
        if s["id"] == "q24":
            s["status"] = "wedged"
            s["why_wedged"] = ("no known way from this part of Cerulean City to Vermilion City; kept moving between "
                               "Cerulean City (14x), Route 4 (13x)")
        if s["id"] == "q23":
            s["status"] = "removed_earlier"
    ctx["plan"] = [s for s in ctx["plan"] if s["status"] != "removed_earlier"]
    ctx.update({
        "since_last_review": {"steps": 46, "maps_entered": "Route 4 -> Cerulean City -> Route 4 -> ... (20 more) ... -> "
                              "Route 4 -> Cerulean City -> Route 4",
                              "entered_repeatedly": {"Cerulean City": 14, "Route 4": 13},
                              "plan_changes": ["wedged: travel to Vermilion City — no known way from this part of "
                                               "Cerulean City to Vermilion City; kept moving between Cerulean City "
                                               "(14x), Route 4 (13x)"],
                              "heard": [BUSH]},
        "attempts": ["travel to Vermilion City: tried 9x (done 0, wedged 9, removed 0) — last: no known way from "
                     "this part of Cerulean City to Vermilion City",
                     "travel to Route 5: tried 5x (done 0, wedged 5, removed 0) — last: no known way from this part "
                     "of Cerulean City to Route 5",
                     "talk to the Rocket in Cerulean City until verify:is the Rocket gone and the path south open?: "
                     "tried 2x (done 0, wedged 2, removed 1) — last: couldn't reach or talk to the Rocket"],
        "heard": {"digest": "Cerulean: a Cooltrainer says the bush by the shop blocks the path and there might be a "
                            "way around. Bill (Route 25) gave the S.S. Ticket for the S.S. Anne in Vermilion.",
                  "here": {"summary": "", "messages": [BUSH.replace(" in Cerulean City", "")]}},
        "unexplored_here": ["door to Cerulean Trashed House (never visited) at (27,11)",
                            "door to Cerulean Trade House (never visited) at (13,15)",
                            "door to Bike Shop (never visited) at (13,25)",
                            "door to Cerulean Mart (never visited) at (25,25)",
                            "Super Nerd at (9,21) (not talked to)", "Guard at (28,12) (not talked to)"],
        "stall": {"steps_without_progress": 310,
                  "critique": {"diagnosis": "The agent has re-issued travel to Vermilion 9 times and keeps crossing "
                                            "the west edge to Route 4; the router reports no known way south from "
                                            "this part of Cerulean.",
                               "suggestion": "Stop retrying the travel step; explore Cerulean's unvisited doors "
                                             "and people — the Cooltrainer hinted there is a way around the bush."}},
    })

    def grade(prop):
        adds = (prop or {}).get("add") or []
        if not adds:
            return False, "no steps added"
        first = adds[0]
        investigates = any((a.get("kind") == "explore" and int(a.get("map", -1)) in (3, 62))
                           or int(a.get("map", -1)) == 62 for a in adds)
        plain_retry = first.get("kind") == "travel" and int(first.get("map", -1)) in (5, 16)
        return investigates and not plain_retry, f"first={first.get('kind')}:{first.get('map')} who={first.get('who')}"
    return ctx, True, grade


BILL = ('Monster at (6,5) in Bills House: "Hiya! I\'m a POKéMON... ...No I\'m not! Call me BILL! I\'m a true blue '
        'POKéMANIAC! ... I screwed up an experiment and got combined with a POKéMON! So, how about it? Help me out '
        'here! When I\'m in the TELEPORTER, go to my PC and run the Cell Separation System!"')


def heard_hint():
    ctx = captured("vermilion-team-20260923", 1116)
    for s in ctx["plan"]:
        if s["status"] == "wedged":
            s["why_wedged"] = "nobody to talk to here now (they left or went somewhere); objects here: " \
                              "Bill's PC (runs the Cell Separator teleporter machine)"
    ctx["current_map"] = {"id": 88, "name": "Bills House", "people": [],
                          "objects": ["Bill's PC (runs the Cell Separator teleporter machine) at (1,4)"]}
    ctx.update({
        "since_last_review": {"steps": 60, "plan_changes": ["wedged: talk to Bill in Bills House until has_item:S.S. "
                                                            "Ticket — nobody to talk to here now"],
                              "heard": [BILL]},
        "attempts": ["talk to Bill in Bills House until has_item:S.S. Ticket: tried 3x (done 0, wedged 3, removed 0)"
                     " — last: nobody to talk to here now (they left or went somewhere)"],
        "heard": {"digest": "", "here": {"summary": "", "messages": [BILL.replace(" in Bills House", "")]}},
        "unexplored_here": ["Bill's PC (runs the Cell Separator teleporter machine) at (1,4) (not checked)"],
        "stall": {"steps_without_progress": 60},
    })
    pc = objects_on(88)

    def grade(prop):
        adds = (prop or {}).get("add") or []
        ok = any(int(a.get("map", -1)) == 88 and ((a.get("kind") == "explore")
                                                  or (a.get("who") and match_objects(pc, a.get("who"))))
                 for a in adds)
        return ok, "; ".join(f"{a.get('kind')}:{a.get('map')} who={a.get('who')}" for a in adds) or "no adds"
    return ctx, True, grade


def no_overreact():
    ctx = captured("ss-anne-20260923", 886)
    for s in ctx["plan"]:
        if s["status"] == "wedged":
            s["status"] = "pending"
            s.pop("why_wedged", None)
    ctx.update({
        "since_last_review": {"steps": 25, "maps_entered": "Cerulean Trashed House -> Cerulean City -> Route 5",
                              "plan_changes": ["done: talk to the Nurse in Cerulean Pokecenter until hp_frac>=1.0"]},
        "unexplored_here": ["door to Route 5 Gate (never visited) at (8,9)", "Youngster at (5,20) (not talked to)",
                            "Lass at (12,7) (not talked to)"],
        "stall": {"steps_without_progress": 3},
    })
    ctx["current_map"] = {"id": 16, "name": "Route 5"}

    def grade(prop):
        if prop is None:
            return True, "triage: no change"
        adds = prop.get("add") or []
        active = [s["id"] for s in ctx["plan"] if s["status"] == "active"]
        bad = any(a.get("kind") == "explore" for a in adds) or any(r in active for r in prop.get("remove") or [])
        return not bad, "; ".join(f"{a.get('kind')}:{a.get('map')}" for a in adds) or "no adds"
    return ctx, False, grade


def travel_phase():
    """runs/fresh-squirtle step 55: an action step at the Viridian Mart is ACTIVE while the agent walks
    Route 1 (its travel half). L1 read that as a bug and re-issued remove+add 15x (silently ignored).
    Now it sees "doing" + what happened to its last edit. PASS = q2 is left alone."""
    ctx = captured("fresh-squirtle-20260924", 55)
    for s in ctx["plan"]:
        if s["status"] == "active":
            s["doing"] = "travelling to Viridian Mart"
    ctx.update({"since_last_review": {"steps": 5, "your_last_edits": [
        "remove q2 IGNORED: it is the ACTIVE step (travelling to Viridian Mart); active steps can't be removed",
        "add (map 42, has_item:Poke Ball>=5) IGNORED: step q2 already covers it"]},
        "stall": {"steps_without_progress": 4}})

    def grade(prop):
        if prop is None:
            return True, "triage: no change"
        bad = "q2" in (prop.get("remove") or []) or any(
            int(a.get("map", -1)) == 42 and "Poke Ball" in str(a.get("done_when")) for a in prop.get("add") or [])
        return not bad, f"remove={prop.get('remove')} adds={[(a.get('kind'), a.get('map')) for a in prop.get('add') or []]}"
    return ctx, False, grade


REVIEWER = None          # set in main(): the critic that reviews interrupts (fast model, like the loop)


def review(ctx, prop):
    """What the loop does with an interrupt request: ask the reviewer. Returns (requested, approved, why)."""
    active = [s for s in ctx["plan"] if s["status"] == "active"]
    aid = active[0]["id"] if active else None
    if not prop or aid not in (prop.get("remove") or []):
        return False, False, "no interrupt"
    why = ((prop.get("interrupt_active") or {}).get("why") or "").strip()
    if not why:
        return True, False, "removed the active step with no interrupt reason (ignored)"
    state = {"active": f"{active[0].get('who')} on map {active[0].get('map')} until {active[0].get('done_when')}",
             "doing": active[0].get("doing"), "reason": why,
             "proposed": [{k: a.get(k) for k in ("kind", "map", "who", "done_when", "why")} for a in prop.get("add") or []],
             "party": ctx.get("party"), "current_map": (ctx.get("current_map") or {}).get("name"), "recent_feedback": []}
    ok, verdict = REVIEWER.judge_interrupt(0, state)
    return True, ok, f"reason={why[:90]!r} -> {'APPROVED' if ok else 'REJECTED'}: {verdict[:110]}"


def urgent_interrupt():
    """runs/fresh-squirtle2 step 147: Squirtle 4/23 HP on Route 1, the active step is still travelling to
    the Viridian Mart. PASS = L1 interrupts the active step for a heal AND the reviewer approves."""
    ctx = captured("fresh-squirtle2-20260924", 147)
    ctx.update({"since_last_review": {"steps": 5}, "stall": {"steps_without_progress": 2}})

    def grade(prop):
        req, ok, detail = review(ctx, prop)
        removed = set((prop or {}).get("remove") or [])
        heal = any("hp_frac" in str(a.get("done_when")) for a in (prop or {}).get("add") or []) or any(
            s["status"] == "pending" and "hp_frac" in str(s.get("done_when")) and s["id"] not in removed
            for s in ctx["plan"])                    # an existing pending heal step becomes next
        return req and ok and heal, detail
    return ctx, True, grade


def _travel_phase_graded():
    ctx, hard, _ = travel_phase()

    def grade(prop):
        req, ok, detail = review(ctx, prop)
        return (not req) or (not ok), detail if req else "left the active step alone"
    return ctx, hard, grade


SCENARIOS = {"stuck-retry": stuck_retry, "heard-hint": heard_hint, "no-overreact": no_overreact,
             "travel-phase": _travel_phase_graded, "urgent-interrupt": urgent_interrupt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--fast-model", default="deepseek-4.1-flash")
    ap.add_argument("--only", default=",".join(SCENARIOS))
    ap.add_argument("--kb", action="store_true")
    a = ap.parse_args()
    from pokemon_agent.agent.knowledge import KnowledgeBase
    from pokemon_agent.agent.planner_llm import Planner
    from pokemon_agent.providers.lunaroute import LunaRouteProvider
    kb = KnowledgeBase.from_env() if a.kb else None
    planner = Planner(goal_map=2, level_target=12, provider=LunaRouteProvider(model=a.fast_model),
                      strategist=LunaRouteProvider(model=a.model, max_tokens=900), knowledge=kb)
    global REVIEWER
    from pokemon_agent.agent.critic import Critic
    REVIEWER = Critic(LunaRouteProvider(model=a.fast_model))
    ok_all = True
    for name in a.only.split(","):
        build = SCENARIOS[name]
        passes = 0
        print(f"== {name} (model={a.model}, N={a.n}, kb={'on' if kb else 'off'})")
        for i in range(a.n):
            ctx, hard, grade = build()
            trace = []
            prop = run_l1_pipeline(None, copy.deepcopy(ctx), planner, hard_event=hard, on_trace=trace.append)
            ok, detail = grade(prop)
            passes += ok
            assess = next((t.get("assessment") for t in trace if t.get("stage") == "brainstorm"), "") or ""
            print(f"   run {i + 1}: {'PASS' if ok else 'FAIL'}  {detail}")
            print(f"            brainstorm: {assess[:260]}")
            if prop and prop.get("notepad"):
                print(f"            notepad: {prop['notepad'][:200]}")
        print(f"   -> {passes}/{a.n}")
        ok_all &= passes >= max(1, int(0.8 * a.n))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
