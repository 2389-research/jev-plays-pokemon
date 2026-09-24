"""L2 under real conditions: does the proposer's target actually get the agent to the person? (2026-09-24)

Uses the repo's own path end to end — no hand-written prompt or context:
  * a ReasoningLoop built the way scripts/run_agent.py builds it (TypeSafe decider, reflection +
    L2 proposer on LunaRouteProvider("deepseek-4.1-flash", max_tokens=900), glm-5.3 strategist),
    loaded from the moment's save + the run's memory;
  * the run's own directive at that step; the loop's own _default_target / _propose_target(stuck=True)
    (PROPOSER_SYSTEM + the real context: map_view, npcs, reachable, candidate_exits, trail, ...);
  * the target is then FOLLOWED through the loop's own executor (_resolve_target) on the emulator.

Modes (same prompt + context):
  current  what the proposer does today (typed target)
  coords   the proposal under test: when stuck, ask for a TILE; once the agent reaches it, normal
           routing (approach_npc) resumes

Success = within 40 steps the agent is in a conversation that the intended person started (the sprite
it faces when the text opens is the target person).

uv run python scripts/eval_l2_live.py [--n 5] [--modes current,coords]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

SCENARIOS = [
    {"name": "misty-stuck", "run": "sleeves-cerulean-20260924", "step": 1610, "who": (4, 2)},
    {"name": "misty-far", "run": "sleeves-cerulean-20260924", "step": 1430, "who": (4, 2)},
    {"name": "clerk", "run": "sleeves-explore-20260924", "step": 127, "who": None},
    {"name": "nurse", "run": "sleeves-explore-20260924", "step": 830, "who": None},
]
COORDS_WHY = ("you've been stuck on this for a while: answer with a TILE target {\"kind\": \"tile\", \"x\": <int>, "
              "\"y\": <int>} — the exact tile to walk to and stand on (it must be in REACHABLE and not a person) — "
              "so that from there you can do the objective; normal routing resumes once you're there")


def row_at(run, step):
    for line in open(f"runs/{run}/log.jsonl"):
        r = json.loads(line)
        if r["step"] == step:
            return r
    raise SystemExit(f"no step {step} in {run}")


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


def run_one(sc, mode):
    from pokemon_agent.agent.plan import Directive
    from pokemon_agent.core.models import InteractAction
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    from pokemon_agent.games.pokemon_red.game_state import read_facing, read_npcs, read_screen_text
    r = row_at(sc["run"], sc["step"])
    emu = PyBoyEmulator("roms/pokemon_red.gb", window="null")
    emu.load_state(Path(f"runs/{sc['run']}/{r['state']}"))
    emu.tick(4)
    loop = make_loop(emu, sc["run"])
    d = Directive.model_validate(r["directive"])
    loop._directive = d
    who = sc["who"]
    if who is None:                                   # the clerk / nurse by name (not "nearest the top")
        who = next((n["x"], n["y"]) for n in read_npcs(emu)
                   if any(k in str(n.get("sprite")).lower() for k in ("clerk", "nurse")))
    seen = {}
    if mode == "coords":                              # same prompt + context; only the stuck reason changes
        orig = loop.planner.propose_target

        def ask(emu_, ctx):
            ctx = copy.deepcopy(ctx)
            ctx["why"] = COORDS_WHY
            seen["ctx_keys"] = sorted(ctx)
            return orig(emu_, ctx)
        loop.planner.propose_target = ask
    obs = observe(loop)
    default = loop._default_target(d, obs) or {"kind": "exit"}
    target = loop._propose_target(obs, d, default, stuck=True)
    first = {k: target.get(k) for k in ("kind", "x", "y", "sprite")}
    if "--notes" in sys.argv:
        print(f"      [{mode}] {first} :: {str(target.get('note'))[:220]}")
    start_map = obs.player.map_id
    for i in range(40):
        obs = observe(loop)
        if obs.player.map_id != start_map:
            return {"ok": False, "why": f"left the map (to {obs.player.map_id})", "target": first, "steps": i}
        occ = {(int(n["x"]), int(n["y"])) for n in obs.game_state.get("npcs") or []}
        if mode == "coords" and target.get("kind") == "tile" and (obs.player.x, obs.player.y) == (target["x"], target["y"]):
            target = default                          # arrived: normal routing resumes
        move = loop._resolve_target(target, d, obs, loop._blocked_dirs(obs), occ)
        if move is None:
            return {"ok": False, "why": f"no move at {(obs.player.x, obs.player.y)}", "target": first, "steps": i}
        loop.controller.execute(move)
        txt, active = read_screen_text(emu)
        if active:
            f = read_facing(emu).get("facing_sprite") or {}
            ok = (f.get("x"), f.get("y")) == tuple(who)
            return {"ok": ok, "why": f"talked to {f.get('sprite')} at ({f.get('x')},{f.get('y')})", "target": first,
                    "steps": i + 1}
    return {"ok": False, "why": "40 steps, no conversation", "target": first, "steps": 40}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--modes", default="current,coords")
    ap.add_argument("--notes", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    for mode in a.modes.split(","):
        tot = ok = 0
        print(f"==== mode={mode}")
        for sc in [x for x in SCENARIOS if not a.only or x["name"] in a.only.split(",")]:
            res = [run_one(sc, mode) for _ in range(a.n)]
            k = sum(x["ok"] for x in res)
            tot += len(res); ok += k
            print(f"  {sc['name']:12} {k}/{len(res)}  " + " | ".join(
                f"{x['target'].get('kind')}{'@'+str((x['target'].get('x'), x['target'].get('y'))) if x['target'].get('kind') == 'tile' else ':'+str(x['target'].get('sprite'))}"
                f" -> {x['why'][:34]} ({x['steps']})" for x in res))
        print(f"  -> {ok}/{tot}")


if __name__ == "__main__":
    main()
