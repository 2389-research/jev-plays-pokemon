"""L2 under real conditions: does the proposer's target actually get the agent to the person/thing? (2026-09-24)

Uses the repo's own path end to end — no hand-written prompt or context:
  * a ReasoningLoop built the way scripts/run_agent.py builds it (TypeSafe decider, reflection +
    L2 proposer on LunaRouteProvider("deepseek-4.1-flash", max_tokens=900), glm-5.3 strategist),
    loaded from the moment's save + the run's memory;
  * the run's own directive, objective, focus and stuck state at that step (from its decisions.jsonl);
    the loop's own _default_target / _propose_target
    (PROPOSER_SYSTEM + the real context: map_view, npcs, objects, candidate_exits, trail, ...);
  * the target is then FOLLOWED through the loop's own executor (_resolve_target) on the emulator.

L2 names WHO/WHAT; the router decides where to stand (spec 2026-09-24-interaction-routing).
Success = within 40 steps a conversation opens that the intended target started: the sprite faced when
the text opens is the target person, or for an object, the faced tile is the object's tile.

uv run python scripts/eval_l2_live.py [--n 5] [--only misty-stuck,bills-pc]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

SCENARIOS = [
    {"name": "misty-stuck", "run": "sleeves-cerulean-20260924", "step": 1610, "who": (4, 2)},  # trainer on her front
    {"name": "misty-far", "run": "sleeves-cerulean-20260924", "step": 1430, "who": (4, 2)},
    {"name": "clerk", "run": "sleeves-explore-20260924", "step": 127, "who": "clerk"},          # across a counter
    {"name": "nurse", "run": "sleeves-explore-20260924", "step": 830, "who": "nurse"},          # across a counter
    {"name": "bills-pc", "run": "bill-live-20260924", "step": 41, "object": (1, 4)},            # used facing north
    # a wanderer (RAM movement byte $FE). The run never had a step about him, so this one directive is
    # the logged one with its target renamed — everything else (context, prompt, executor) is live.
    {"name": "youngster", "run": "sleeves-explore-20260924", "step": 127, "who": "youngster",
     "retarget": {"sprite": "Youngster", "reason": "quest: talk to the Youngster in the Viridian Mart"}},
]


def row_at(run, step):
    for line in open(f"runs/{run}/log.jsonl"):
        r = json.loads(line)
        if r["step"] == step:
            return r
    raise SystemExit(f"no step {step} in {run}")


def recorded_l2(run, step):
    """The real L2 proposer input the run recorded at (or just before) this step: its objective, focus,
    milestone and whether it was a stuck re-ask. The logged directive and the run's final memory don't
    carry these (the directive's reason isn't logged; latest.mem.json holds the END-of-run plan)."""
    last = None
    for line in open(f"runs/{run}/decisions.jsonl"):
        r = json.loads(line)
        if r["layer"] == "l2_propose_target" and int(r["step"]) <= step:
            last = r
    return (last or {}).get("input") or {}


def make_loop(emu, run):
    """ReasoningLoop exactly as scripts/run_agent.py builds it for --decider typesafe."""
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.knowledge import KnowledgeBase
    from pokemon_agent.agent.memory import AgentMemory
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import Reasoner
    from pokemon_agent.agent.session import Session
    from pokemon_agent.agent.typesafe_reasoner import TypeSafeReasoner
    from pokemon_agent.core.models import GoalState
    from pokemon_agent.observations.builder import ObservationBuilder
    from pokemon_agent.providers.lunaroute import LunaRouteProvider
    vprov = LunaRouteProvider(model="deepseek-4.1-flash", max_tokens=900)
    reasoner = TypeSafeReasoner(model=None, reflector=Reasoner(vprov), min_confidence=0.45, wait_on_low_confidence=False)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu), reasoner=reasoner,
                         session=Session(GoalState(primary="play", current="play")), vision=False, reflect_every=8,
                         low_conf_reflect=0.35, memory=AgentMemory.load(Path(f"runs/{run}/latest.mem.json")),
                         strategist_provider=LunaRouteProvider(model="glm-5.3", max_tokens=700),
                         knowledge=KnowledgeBase.from_env(), autonomous=True)
    loop._l1_due = lambda: False
    return loop


def observe(loop):
    """What step_once does before deciding: build obs + ingest the RAM collision + annotate npcs."""
    from pokemon_agent.games.pokemon_red.map_reader import read_collision_map
    obs, _ = loop.builder.build(capture_screenshot=False)
    cm = read_collision_map(loop.controller.emu)
    if cm and obs.player and cm["map_id"] == obs.player.map_id:
        loop.world.ingest_collision(cm["map_id"], cm["width"], cm["height"], cm["walkable"], cm.get("counters"),
                                    cm.get("terrain"))
    npc_cells = {(int(n["x"]), int(n["y"])) for n in (obs.game_state or {}).get("npcs") or []}
    obs.map_view = loop.world.render_semantic(obs.player, obs.exits, npc_cells) or obs.map_view
    obs.exits = loop._resolve_exits(obs.exits, obs.player.map_id)
    obs.game_state["npcs"] = loop.interactions.annotate_npcs(obs.player, obs.game_state["npcs"])
    return obs


def run_one(sc):
    from pokemon_agent.agent.plan import Directive
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.games.pokemon_red.game_state import read_facing, read_npcs, read_screen_text
    r = row_at(sc["run"], sc["step"])
    emu = PyBoyEmulator("roms/pokemon_red.gb", window="null")
    emu.load_state(Path(f"runs/{sc['run']}/{r['state']}"))
    emu.tick(4)
    loop = make_loop(emu, sc["run"])
    dd = dict(r["directive"])
    if sc.get("retarget"):
        dd["target"] = {**(dd.get("target") or {}), "sprite": sc["retarget"]["sprite"]}
        dd["reason"] = sc["retarget"]["reason"]
        dd["intent"] = "talk_to"
    rec = recorded_l2(sc["run"], sc["step"])
    if not sc.get("retarget"):
        dd["reason"] = rec.get("objective") or dd.get("reason")
    d = Directive.model_validate(dd)
    loop._directive = d
    if loop._plan is not None:                        # the plan as it was at this moment, not at run end
        from pokemon_agent.agent.plan import Goal
        loop._plan.goals.tertiary = Goal(text=rec.get("focus") or "")
        loop._plan.milestone = rec.get("milestone") if rec.get("milestone") not in (None, "None") else ""
    stuck = str(rec.get("why") or "").startswith("the last target")
    # who had been talked to AT THIS MOMENT on this map (the final memory knows the future)
    mid = emu.read_memory(0xD35E)
    talked = loop.interactions.talked
    talked -= {k for k in talked if k[0] == mid}
    talked |= {(mid, n["x"], n["y"]) for n in rec.get("npcs") or [] if n.get("talked_to")}
    who = sc.get("who")
    if isinstance(who, str):                          # by name (not "nearest the top")
        who = next((n["x"], n["y"]) for n in read_npcs(emu) if who in str(n.get("sprite")).lower())
    want = (sc["retarget"]["sprite"] if sc.get("retarget") else (d.target or {}).get("sprite") or "").lower()
    obs = observe(loop)
    default = loop._default_target(d, obs) or {"kind": "exit"}
    target = loop._propose_target(obs, d, default, stuck=stuck)
    first = {k: target.get(k) for k in ("kind", "x", "y", "sprite", "object")}
    nm = str(target.get("sprite") or target.get("object") or "").lower().removeprefix("the ")
    first["named"] = bool(nm) and (nm in want or want.split(" at ")[0].split(" (")[0].removeprefix("the ") in nm)
    if "--notes" in sys.argv:
        print(f"      {first} :: {str(target.get('note'))[:220]}")
    start_map = obs.player.map_id
    for i in range(40):
        obs = observe(loop)
        if obs.player.map_id != start_map:
            return {"ok": False, "why": f"left the map (to {obs.player.map_id})", "target": first, "steps": i}
        occ = {(int(n["x"]), int(n["y"])) for n in obs.game_state.get("npcs") or []}
        move = loop._resolve_target(target, d, obs, loop._blocked_dirs(obs), occ)
        if move is None:
            why = loop._approach_block or f"no move at {(obs.player.x, obs.player.y)}"
            return {"ok": False, "why": why, "target": first, "steps": i}
        loop.controller.execute(move)
        txt, active = read_screen_text(emu)
        if active:
            f = read_facing(emu)
            fs = f.get("facing_sprite") or {}
            if sc.get("object"):
                ok = tuple(f.get("front_tile") or ()) == tuple(sc["object"])
                return {"ok": ok, "why": f"used the tile {f.get('front_tile')}", "target": first, "steps": i + 1}
            ok = (fs.get("x"), fs.get("y")) == tuple(who)
            why = f"talked to {fs.get('sprite')} at ({fs.get('x')},{fs.get('y')})"
            if not fs:                                # text we didn't start (a trainer spotted us, a sign...)
                why = f"interrupted at {(obs.player.x, obs.player.y)}: \"{str(txt)[:40]}\""
            return {"ok": ok, "why": why, "target": first, "steps": i + 1}
    return {"ok": False, "why": "40 steps, no conversation", "target": first, "steps": 40}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--notes", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    tot = ok = named = reached = 0
    for sc in [x for x in SCENARIOS if not a.only or x["name"] in a.only.split(",")]:
        res = [run_one(sc) for _ in range(a.n)]
        k = sum(x["ok"] for x in res)
        nk = sum(x["target"]["named"] for x in res)
        rk = sum(x["ok"] for x in res if x["target"]["named"])
        tot += len(res)
        ok += k
        named += nk
        reached += rk
        print(f"  {sc['name']:12} {k}/{len(res)}  L2 named it {nk}/{len(res)}, router reached it {rk}/{nk}  " + " | ".join(
            f"{x['target'].get('kind')}:{x['target'].get('sprite') or x['target'].get('object') or (x['target'].get('x'), x['target'].get('y'))}"
            f" -> {x['why'][:60]} ({x['steps']})" for x in res), flush=True)
    print(f"  -> {ok}/{tot} conversations with the intended target; L2 named it {named}/{tot}; "
          f"router reached a named target {reached}/{named}")


if __name__ == "__main__":
    main()
