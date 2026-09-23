"""Live scenario evals for L1 tiered goals (spec docs/superpowers/specs/2026-09-23-l1-tiered-goals-design.md §4).

Brainstorm runs LIVE (never a hand-written brainstorm that states the answer); graders are structural.
Failed calls (empty fallback) are excluded from pass rates and counted.

  1 diversion   captured step-25 input + injected emergency: an hp_frac step placed NEXT (pos <= 1) and
                primary/secondary unchanged
  2 return      states/pokecenter_lowhp healed in RAM; tertiary heal now met, interrupted = forest, plan =
                only the done heal step: no new heal step and a step toward the forest (maps 13/51/2)
  3 story-gate  captured step-25 input with secondary "Get to Pewter City": 0 off-path steps before the
                pending delivery, delivery kept
  4 churn       every captured no-op DECIDE input, goals derived + tertiary seeded from the active step:
                goals/notepad edit rate <= 10% (overall + per run); `--triage` also replays the captured
                triage inputs live with and without goals
  5 chapter     states/lab_deliver with the parcel removed from the bag in RAM; secondary (deliver) met,
                notepad with the next chapter, plan = only the done delivery: first added step goes forward
  6 notepad     sizes + echo rate (report only)

Live (spends credits): uv run python scripts/eval_l1_goals.py [--n 10] [--only diversion,return,...]
                       [--churn-n 2] [--triage] [--kb]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from eval_l1_placement import ON_PATH, call_failed, captured, heal_case, ids, plan_steps  # noqa: E402

from pokemon_agent.agent import goals as G  # noqa: E402
from pokemon_agent.agent.l1_pipeline import validate_step  # noqa: E402
from pokemon_agent.agent.plan import AgentPlan, Goal, Goals  # noqa: E402
from pokemon_agent.agent.quest_reconciler import reconcile_quests  # noqa: E402
from pokemon_agent.games.pokemon_red.maps import map_name  # noqa: E402

ROM = "roms/pokemon_red.gb"
PARCEL = 70
NOTEPAD_CH5 = "After the parcel: heal if needed, then Route 2 -> Viridian Forest -> Pewter City (Brock)."


# ---- helpers -----------------------------------------------------------------------------------
def plan_for(ctx: dict, tertiary: Goal | None = None, notepad: str = "", interrupted: Goal | None = None) -> AgentPlan:
    p = AgentPlan(mission=ctx.get("mission") or "", milestone=ctx.get("milestone") or "")
    if tertiary is not None:
        p.apply_goals({"tertiary": tertiary})
    p.notepad = notepad
    if interrupted is not None:
        p.interrupted = interrupted
    return p


def with_goals(ctx: dict, plan: AgentPlan, emu=None, status: dict | None = None) -> dict:
    """Turn a (legacy) captured context into a goals context seen by the new prompts."""
    ctx = {k: v for k, v in ctx.items() if k not in ("mission", "milestone")}
    ctx.update(G.goals_view(plan, emu))
    if status:
        ctx["goal_status"] = {**ctx["goal_status"], **status}
    return ctx


def decide(planner, ctx: dict) -> tuple[dict, str]:
    b = planner.l1_brainstorm(None, ctx)
    return planner.l1_decide(ctx, b), b.get("assessment", "")


def emu_from(state: str):
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    emu = PyBoyEmulator(ROM, window="null")
    emu.load_state(Path(f"states/{state}.state"))
    emu.tick(2)
    return emu


def remove_bag_item(emu, iid: int) -> None:
    n = emu.read_memory(0xD31D)
    pairs = [(emu.read_memory(0xD31E + 2 * i), emu.read_memory(0xD31F + 2 * i)) for i in range(n)]
    pairs = [p for p in pairs if p[0] != iid]
    emu.write_memory(0xD31D, len(pairs))
    for i, (a, q) in enumerate(pairs):
        emu.write_memory(0xD31E + 2 * i, a)
        emu.write_memory(0xD31F + 2 * i, q)
    emu.write_memory(0xD31E + 2 * len(pairs), 0xFF)


def heal_party_in_ram(emu) -> None:
    base, size = 0xD16B, 0x2C
    for i in range(emu.read_memory(0xD163)):
        b = base + i * size
        emu.write_memory(b + 1, emu.read_memory(b + 0x22))
        emu.write_memory(b + 2, emu.read_memory(b + 0x23))


def base_ctx(emu, plan_rows: list[dict]) -> dict:
    from pokemon_agent.agent.signals import game_signals
    sig = game_signals(emu)
    sig.update({"blocked_for_n": 0, "emergency_heal": False})
    mid = emu.read_memory(0xD35E)
    return {"current_map": {"id": mid, "name": map_name(mid)}, "party": sig["party"], "items": sig["items"],
            "badges": sig["badges"], "plan": plan_rows, "signals": sig}


def live_order(ctx: dict, out: dict):
    adds = [a for a in (out.get("add") or []) if validate_step(a)[0]]
    res = reconcile_quests(plan_steps(ctx), {"add": adds, "remove": out.get("remove") or []}, next_id=ids())
    return [s for s in res if s.status != "done"]


def fmt(out: dict) -> str:
    adds = [(a.get("map"), a.get("done_when"), a.get("after")) for a in out.get("add") or []]
    g = out.get("goals") or {}
    return (f"adds={adds} rm={out.get('remove')} goals={ {k: (v or {}).get('text') if isinstance(v, dict) else v for k, v in g.items()} }"
            + (f" intr={out['interrupted']!r}" if "interrupted" in out else "")
            + (f" notepad={len(out['notepad'])}ch" if isinstance(out.get("notepad"), str) else ""))


# ---- scenarios -----------------------------------------------------------------------------------
def sc_diversion():
    base = heal_case(captured(25))
    plan = plan_for(base, tertiary=Goal(text="Deliver Oak's Parcel to Professor Oak", done_when="no_item:Oak's Parcel"))
    ctx = with_goals(base, plan, status={"tertiary": "unmet"})   # the parcel is in the bag
    ctx.pop("brainstorm", None)        # heal_case's answer-stating brainstorm is NOT used (R2-#8)

    def grade(out):
        ch = G.detect_change(plan, out, step_edit=bool(out.get("add") or out.get("remove")))
        touched = sorted(set((ch.goals if ch else {})) & {"primary", "secondary"})
        live = live_order(ctx, out)
        pos = [i for i, s in enumerate(live) if (s.done_when or "").startswith("hp_frac")]
        ok = bool(pos) and pos[0] <= 1 and not touched
        tert = "tertiary" in (ch.goals if ch else {})
        return ok, f"heal_pos={pos[:1]} touched={touched} tertiary_rewritten={tert}"
    return ctx, grade


def sc_return():
    emu = emu_from("pokecenter_lowhp")
    heal_party_in_ram(emu)
    mid = emu.read_memory(0xD35E)
    rows = [{"id": "q7", "map": mid, "kind": "action", "talk": True, "done_when": "hp_frac>=1.0", "status": "done"}]
    ctx0 = base_ctx(emu, rows)
    ctx0.update(mission="Earn the Boulder Badge from Brock in Pewter City",
                milestone="Get through Viridian Forest to Pewter City with a team that can win")
    plan = plan_for(ctx0, tertiary=Goal(text="Heal the party at the Viridian Pokémon Center", done_when="hp_frac>=1.0"),
                    interrupted=Goal(text="Cross Viridian Forest to Pewter City"))
    ctx = with_goals(ctx0, plan, emu)
    emu.close()
    assert ctx["goal_status"]["tertiary"] == "met", ctx["goal_status"]

    def grade(out):
        adds = [a for a in out.get("add") or [] if validate_step(a)[0]]
        heal = [a for a in adds if str(a.get("done_when", "")).startswith("hp_frac")]
        fwd = [a for a in adds if a.get("map") in (13, 51, 2)]
        return (not heal and bool(fwd)), f"heal_adds={len(heal)} forward={[a.get('map') for a in fwd]}"
    return ctx, grade


def sc_story_gate():
    base = captured(25)
    plan = plan_for(base)
    plan.apply_goals({"secondary": Goal(text="Get to Pewter City")})
    ctx = with_goals(base, plan)
    ctx.pop("brainstorm", None)

    def grade(out):
        live = live_order(ctx, out)
        deliv = [i for i, s in enumerate(live) if (s.done_when or "").lower().startswith("no_item:oak")]
        if not deliv:
            return False, "DELIVERY_REMOVED"
        off = [i for i, s in enumerate(live) if s.id.startswith("n") and s.map not in ON_PATH
               and not (s.done_when or "").startswith("hp_frac")]
        bad = [i for i in off if i < deliv[0]]
        label = "MISORDERED" if bad else ("ANCHORED" if off else "DEFERRED")
        return not bad, f"{label} order={[(s.id, s.map) for s in live]}"
    return ctx, grade


def sc_chapter():
    emu = emu_from("lab_deliver")
    remove_bag_item(emu, PARCEL)
    mid = emu.read_memory(0xD35E)
    rows = [{"id": "q9", "map": mid, "kind": "action", "talk": True, "done_when": "no_item:Oak's Parcel", "status": "done"}]
    ctx0 = base_ctx(emu, rows)
    ctx0.update(mission="Earn the Boulder Badge from Brock in Pewter City", milestone="")
    plan = plan_for(ctx0, notepad=NOTEPAD_CH5)
    plan.apply_goals({"secondary": Goal(text="Deliver Oak's Parcel to Professor Oak", done_when="no_item:Oak's Parcel")})
    ctx = with_goals(ctx0, plan, emu)
    emu.close()
    assert ctx["goal_status"]["secondary"] == "met", ctx["goal_status"]

    def grade(out):
        adds = [a for a in out.get("add") or [] if validate_step(a)[0]]
        if not adds:
            return False, "no step added"
        first = adds[0]
        # forward from Oak's Lab: Pallet (0) is just leaving the lab, Route 1 (12) / Viridian (1) / Route 2 (13) /
        # the forest (51) are the chapter; a heal first is also fine
        ok = first.get("map") in (0, 12, 1, 13, 51) or str(first.get("done_when", "")).startswith("hp_frac")
        return ok, f"first={first.get('map')}:{first.get('done_when')}"
    return ctx, grade


SCENARIOS = {"diversion": sc_diversion, "return": sc_return, "story-gate": sc_story_gate, "chapter": sc_chapter}


# ---- churn ----------------------------------------------------------------------------------------
def noop_inputs() -> list[tuple[str, int, dict]]:
    out = []
    for f in sorted(glob.glob("runs/*/decisions.jsonl")):
        for line in open(f):
            r = json.loads(line)
            o = r.get("output_parsed")
            if r["layer"] == "l1_decide" and isinstance(o, dict) and not o.get("add") and not o.get("remove"):
                out.append((Path(f).parent.name, r["step"], r["input"]))
    return out


def seed_tertiary(ctx: dict) -> Goal:
    act = next((p for p in ctx.get("plan") or [] if p.get("status") == "active"), None)
    if act is None:
        return Goal()
    where = map_name(act["map"])
    dw = G.clean_criterion(act.get("done_when"))
    if act.get("kind") == "travel" or not dw:
        return Goal(text=f"Get to {where}")
    return Goal(text=f"{act['done_when']} at {where}", done_when=dw)


def run_churn(planner, n: int) -> bool:
    rows = noop_inputs()
    per_run = defaultdict(Counter)
    total = Counter()
    notes_sizes = []
    print(f"== churn: {len(rows)} captured no-op DECIDE inputs x N={n}")
    for run, step, inp in rows:
        base = {k: v for k, v in inp.items() if k not in ("brainstorm", "maps")}
        plan = plan_for(base, tertiary=seed_tertiary(base))
        ctx = with_goals(base, plan)
        for _ in range(n):
            out = planner.l1_decide(ctx, {"assessment": inp.get("brainstorm", "")})
            c = per_run[run]
            if call_failed(out):
                c["failed"] += 1
                total["failed"] += 1
                continue
            c["calls"] += 1
            total["calls"] += 1
            step_edit = bool(out.get("add") or out.get("remove"))
            ch = G.detect_change(plan, out, step_edit=step_edit)
            if step_edit:
                c["step_edit"] += 1
                total["step_edit"] += 1
            if ch and (ch.goals or ch.notepad is not None):
                filled = ch.goals and all(not getattr(plan.goals, t).text for t in ch.goals) and ch.notepad is None
                key = "filled_empty" if filled else "goal_edit"
                c[key] += 1
                total[key] += 1
                if ch.notepad is not None:
                    notes_sizes.append(len(ch.notepad))
                print(f"   {run} s{step}: {key} {fmt(out)}")
    calls = max(1, total["calls"])
    rate = total["goal_edit"] / calls
    for run, c in per_run.items():
        print(f"   {run}: calls={c['calls']} goal_edit={c['goal_edit']} filled_empty={c['filled_empty']} "
              f"step_edit={c['step_edit']} failed={c['failed']}")
    print(f"   -> goals/notepad edit rate {total['goal_edit']}/{total['calls']} = {rate:.1%} "
          f"(filled-empty {total['filled_empty']}, step edits {total['step_edit']}, failed {total['failed']}) "
          f"notepad sizes={notes_sizes} {'OK' if rate <= 0.10 else 'ABOVE 10%'}")
    return rate <= 0.10


def run_triage(planner, old_rev: str) -> bool:
    import subprocess
    import ast
    from pokemon_agent.providers.parsing import strip_fences
    src = subprocess.check_output(["git", "show", f"{old_rev}:src/pokemon_agent/agent/planner_llm.py"], text=True)
    old = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "TRIAGE_SYSTEM" for t in n.targets))
    rows = []
    for f in sorted(glob.glob("runs/*/decisions.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["layer"] == "l1_triage":
                rows.append(r["input"])
    new_c = old_c = new_n = old_n = 0
    for inp in rows:
        t = planner.l1_triage(inp)                  # new prompt; goals derived from mission/milestone
        new_n += 1
        new_c += bool(t.get("change"))
        try:
            content, _l, _u = planner.provider.chat_json(old, inp)
            old_n += 1
            old_c += bool(json.loads(strip_fences(content)).get("change"))
        except Exception:
            pass
    nr, orr = new_c / max(1, new_n), old_c / max(1, old_n)
    print(f"== triage replay ({len(rows)} captured inputs): with goals {new_c}/{new_n} = {nr:.1%} change; "
          f"without goals (prompt @{old_rev}) {old_c}/{old_n} = {orr:.1%}  {'OK' if nr <= orr + 0.05 else 'ABOVE +5pts'}")
    return nr <= orr + 0.05


# ---- main -----------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--fast-model", default="deepseek-4.1-flash")
    ap.add_argument("--only", default="diversion,return,story-gate,chapter,churn")
    ap.add_argument("--churn-n", type=int, default=2)
    ap.add_argument("--triage", action="store_true")
    ap.add_argument("--triage-baseline-rev", default="33ab266")
    ap.add_argument("--kb", action="store_true", help="let brainstorm search the Orrery KB (ORRERY_WORKSPACE_ID)")
    a = ap.parse_args()

    from pokemon_agent.agent.knowledge import KnowledgeBase
    from pokemon_agent.agent.planner_llm import Planner
    from pokemon_agent.providers.lunaroute import LunaRouteProvider
    kb = KnowledgeBase.from_env() if a.kb else None
    planner = Planner(goal_map=2, level_target=12, provider=LunaRouteProvider(model=a.fast_model),
                      strategist=LunaRouteProvider(model=a.model, max_tokens=900), knowledge=kb)
    only = set(a.only.split(","))
    all_ok = True
    notepads = []
    for name, build in SCENARIOS.items():
        if name not in only:
            continue
        ctx, grade = build()
        passes = failed = 0
        print(f"== {name} (model={a.model}, N={a.n}, kb={'on' if kb else 'off'})")
        for i in range(a.n):
            out, _assess = decide(planner, ctx)
            if call_failed(out):
                failed += 1
                print(f"   run {i + 1}: CALL_FAILED")
                continue
            ok, why = grade(out)
            passes += ok
            if isinstance(out.get("notepad"), str):
                notepads.append((name, len(out["notepad"]), G.norm(out["notepad"]) == G.norm(ctx.get("notepad"))))
            print(f"   run {i + 1}: {'PASS' if ok else 'FAIL'}  {why}  | {fmt(out)}")
        done = a.n - failed
        verdict = done > 0 and passes >= max(1, round(0.8 * done))
        all_ok &= verdict
        print(f"   -> {passes}/{done} ({failed} failed calls excluded) {'OK' if verdict else 'BELOW THRESHOLD'}")
    if "churn" in only:
        all_ok &= run_churn(planner, a.churn_n)
    if a.triage:
        all_ok &= run_triage(planner, a.triage_baseline_rev)
    if notepads:
        print(f"== notepad hygiene: {len(notepads)} notepads sent; sizes={[s for _, s, _ in notepads]}; "
              f"echoes={sum(e for _, _, e in notepads)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
