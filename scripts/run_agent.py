# /// script
# requires-python = ">=3.12"
# ///
"""Run the Pokémon agent loop.

Examples:
  # No ROM needed — drive the FakeEmulator maze with LunaRoute (proves the loop):
  uv run python scripts/run_agent.py --demo --provider lunaroute --model deepseek-4.1-flash

  # Real game, visible window:
  uv run python scripts/run_agent.py --rom roms/pokemon-red.gb --provider lunaroute \
      --model glm-5.3-vision --vision --goal "Leave the room"

  # Fully offline (mock decisions):
  uv run python scripts/run_agent.py --demo --provider mock
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_agent.actions.controller import ActionController
from pokemon_agent.agent.loop import AgentLoop
from pokemon_agent.agent.session import Session
from pokemon_agent.core.models import AgentDecision, GoalState, MoveAction, Direction
from pokemon_agent.observations.builder import ObservationBuilder
from pokemon_agent.providers.interface import ProviderError


def build_provider(name: str, model: str | None, vision: bool):
    if name == "lunaroute":
        from pokemon_agent.providers.lunaroute import LunaRouteProvider
        return LunaRouteProvider(model=model)
    if name == "mock":
        from pokemon_agent.providers.mock import MockProvider
        # cycle simple south-ward taps
        return MockProvider(
            [AgentDecision(action=MoveAction(direction=Direction.SOUTH), decision_note="mock: go south")],
            cycle=True,
        )
    raise SystemExit(f"unknown provider {name!r}")


def build_emulator(args):
    if args.demo:
        from pokemon_agent.emulator.fake_emulator import FakeEmulator
        return FakeEmulator()
    if not args.rom:
        raise SystemExit("provide --rom PATH or use --demo")
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
    # speed 0 = unbounded (PyBoy's per-tick frame limiter off). The controller polls
    # movement with many small tick() calls, and each call sleeps ~16ms under speed 1
    # (~200ms/tile) — so the default agent speed is unbounded. Watching a window? pass
    # --speed 2/3 for a real-time-ish multiplier you can actually see.
    speed = args.speed
    if speed is None:
        speed = 1 if (not args.headless) else 0
    emu = PyBoyEmulator(args.rom, window="null" if args.headless else "SDL2", speed=speed)
    print(f"emulator: window={'null' if args.headless else 'SDL2'} speed={speed}"
          f"{' (unbounded)' if speed == 0 else ''}")
    if args.load_state:
        emu.load_state(Path(args.load_state))
        print(f"loaded save state: {args.load_state}")
    return emu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom")
    ap.add_argument("--demo", action="store_true", help="use FakeEmulator maze, no ROM")
    ap.add_argument("--headless", action="store_true", help="no SDL window (real emulator)")
    ap.add_argument("--load-state", help="path to a .state to start from (e.g. an overworld save)")
    ap.add_argument("--save-state-out", help="save emulator state to this path on exit")
    ap.add_argument("--state", help="load a named fixture from states/<NAME>.state")
    ap.add_argument("--save-state", help="save to states/<NAME>.state on exit (a reusable fixture)")
    ap.add_argument("--provider", default="lunaroute", choices=["lunaroute", "mock"])
    ap.add_argument("--mode", default="plan-act", choices=["plan-act", "reason"],
                    help="plan-act = vision planner + text actor; reason = integrated reason-and-act every step")
    ap.add_argument("--decider", default="llm", choices=["llm", "typesafe"],
                    help="reason mode: llm = generative LunaRoute action; typesafe = calibrated Choice classifier")
    ap.add_argument("--typesafe-model", default=None, help="TypeSafe model id (default: account default)")
    ap.add_argument("--min-confidence", type=float, default=0.45,
                    help="typesafe: opt-in wait gate threshold (only if --wait-gate is set)")
    ap.add_argument("--wait-gate", action="store_true",
                    help="typesafe: turn a low-confidence move into a wait (off by default)")
    ap.add_argument("--low-conf-reflect", type=float, default=None,
                    help="re-plan when decider confidence drops below this (typesafe default 0.35; 0 disables)")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="reason mode: save (emulator state + memory) every N steps")
    ap.add_argument("--checkpoint-dir", default=None, help="checkpoint dir (default runs/ckpt)")
    ap.add_argument("--resume", action="store_true",
                    help="resume memory + emulator state from --checkpoint-dir/latest")
    ap.add_argument("--goal-map", type=int, default=None,
                    help="travel-target map id for the route hint (e.g. 2 = Pewter City)")
    ap.add_argument("--level-target", type=int, default=0,
                    help="grind the party to >= this level before pushing to the goal (readiness need)")
    ap.add_argument("--no-vision", action="store_true",
                    help="reason mode: navigate from TEXT state only (fast text model, no screenshot)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--vision", action="store_true", help="send screenshot to the ACTOR every step")
    ap.add_argument("--reflect-every", type=int, default=0,
                    help="run a vision planner every N steps (0 = off). Try 10.")
    ap.add_argument("--planner-model", default="glm-5.3-vision", help="vision model for reflection")
    ap.add_argument("--pather", default="bfs", choices=["bfs", "jev", "policy"],
                    help="overworld pathing: 'bfs' (deterministic shortest), 'jev' (calibrated per-step direction), "
                         "or 'policy' (Jev picks a routing objective per leg — shortest/dodge-grass/farm-exp — a weighted router executes)")
    ap.add_argument("--strategist-model", default="glm-5.3",
                    help="strong text model for tier-2 quest planning when the path is story-gated")
    ap.add_argument("--orrery-workspace", default=None,
                    help="Orrery noosphere workspace id for the knowledge base (or env ORRERY_WORKSPACE_ID)")
    ap.add_argument("--orrery-url", default=None, help="Orrery base URL (default env or http://localhost:8100)")
    ap.add_argument("--record-dir", default=None, help="run-recorder output dir (default runs/rec-<ts>); full per-step state + screenshots + new-area save states + viewer.html")
    ap.add_argument("--no-record", action="store_true", help="disable the full run recorder")
    ap.add_argument("--goal", default="Leave the current area")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--speed", type=int, default=None,
                    help="emulation speed; 0=unbounded. Default: unbounded when --headless, 1 (real time) with a window")
    ap.add_argument("--log", default=None, help="JSONL episode log path")
    ap.add_argument("--screenshot-logging", default="every_step", choices=["none", "errors_only", "every_step"],
                    help="save per-step frames next to the log so steps can be reconstructed")
    args = ap.parse_args()

    # named fixture shorthands -> states/<NAME>.state
    states_dir = Path(__file__).resolve().parents[1] / "states"
    if args.state:
        args.load_state = str(states_dir / f"{args.state}.state")
    if args.save_state:
        states_dir.mkdir(exist_ok=True)
        args.save_state_out = str(states_dir / f"{args.save_state}.state")

    emu = build_emulator(args)
    session = Session(GoalState(primary=args.goal, current=args.goal))
    logger = None
    if args.log:
        from pokemon_agent.logging.episode_logger import EpisodeLogger
        logger = EpisodeLogger(args.log, provider=args.provider, model=args.model or "?",
                               screenshot_mode=args.screenshot_logging)

    # --- integrated reason-and-act mode ---
    if args.mode == "reason":
        if args.provider != "lunaroute":
            raise SystemExit("--mode reason needs --provider lunaroute (vision)")
        from pokemon_agent.providers.lunaroute import LunaRouteProvider
        from pokemon_agent.agent.reasoner import Reasoner
        from pokemon_agent.agent.reason_loop import ReasoningLoop
        use_vision = not args.no_vision
        if args.decider == "typesafe":
            # per-step decisions go to TypeSafe (state, no image); reflection is
            # text-only too, so run it on a FAST text model and never send an image.
            # (Reflection on glm-5.3-vision was the ~13s stall — it wasn't the image,
            # just a slow vision-class model on a big text prompt.)
            use_vision = False
            rmodel = args.planner_model if args.planner_model != "glm-5.3-vision" else "deepseek-4.1-flash"
        else:
            rmodel = (args.model or "deepseek-4.1-flash") if args.no_vision else args.planner_model
        vprov = LunaRouteProvider(model=rmodel, max_tokens=900)
        try:
            print(f"warming up reasoner ({rmodel}, vision={use_vision})...", flush=True)
            vprov.chat_json("Reply with {\"ok\": true}.", {"ping": 1},
                            image=emu.screenshot() if use_vision else None)
        except Exception as e:
            print(f"  (warmup skipped: {e})")

        def on_event_r(kind, payload):
            if kind == "reflect":
                p = payload["plan"]
                print(f"  == REFLECT ({payload['latency_ms']}ms) progress: {p['progress'][:80]}")
                print(f"     objective: {p['next_objective'][:90]}")
                if p.get("explore_note"):
                    print(f"     explore: {p['explore_note'][:80]}")
            elif kind == "reason":
                a = payload["action"]
                print(f"  [{payload['step']}] {payload['latency_ms']}ms @ {payload['location'][:55]}")
                print(f"       -> {a} :: {payload['reasoning'][:75]}")
            elif kind == "stuck":
                print(f"  [!] step {payload['step']} stuck kind={payload['kind']} setback={payload['setback']}")
            elif kind == "checkpoint":
                print(f"  [ckpt] step {payload['step']} -> {payload['dir']}")
            elif kind == "directive":
                print(f"  [DIRECTIVE] step {payload['step']} intent={payload['intent']} "
                      f"target={payload['target']} success={payload['success']}")
            elif kind == "directive_done":
                print(f"  [directive ✓] step {payload['step']} {payload['intent']} — {payload['reason']}")
            elif kind == "directive_suspend":
                print(f"  [directive ⏸] step {payload['step']} suspended {payload['intent']} (higher need)")
            elif kind == "directive_impossible":
                print(f"  [directive ✗] step {payload['step']} {payload['intent']} impossible -> replan")
            elif kind == "escalate":
                print(f"  [escalate?] step {payload['step']} story-gate score={payload['score']}")
            elif kind == "quest":
                print(f"  [QUEST] step {payload['step']} ({payload['len']} steps):")
                for s in payload["plan"]:
                    print(f"       - {s}")
            elif kind == "quest_done":
                print(f"  [quest ✓] step {payload['step']} quest complete")
            elif kind == "kb_search":
                print(f"  [kb search] step {payload['step']} q={payload['query']!r} -> {payload['results']} hits")

        low_conf_reflect = args.low_conf_reflect
        strategist_provider = None
        if args.decider == "typesafe":
            from pokemon_agent.agent.typesafe_reasoner import TypeSafeReasoner
            # generative reflection stays on LunaRoute; per-step decisions go to TypeSafe.
            reasoner = TypeSafeReasoner(model=args.typesafe_model, reflector=Reasoner(vprov),
                                        min_confidence=args.min_confidence,
                                        wait_on_low_confidence=args.wait_gate)
            use_vision = False  # TypeSafe reads structured state, not screenshots
            if low_conf_reflect is None:
                low_conf_reflect = 0.35  # unsure actor -> re-plan instead of thrash
            # tier-2 strategist: a strong text model for quest planning when story-gated
            strategist_provider = LunaRouteProvider(model=args.strategist_model, max_tokens=700)
            print(f"decider=typesafe (model={args.typesafe_model or 'default'}), reflection via {rmodel}, "
                  f"strategist={args.strategist_model}, wait_gate={args.wait_gate}, low_conf_reflect={low_conf_reflect}")
        else:
            reasoner = Reasoner(vprov)
        low_conf_reflect = None if (low_conf_reflect is not None and low_conf_reflect <= 0) else low_conf_reflect

        # Orrery knowledge base (optional): retrieval-grounded quest planning
        from pokemon_agent.agent.knowledge import KnowledgeBase
        knowledge = KnowledgeBase.from_env(args.orrery_url, args.orrery_workspace)
        if knowledge is not None:
            print(f"knowledge base: Orrery {knowledge.base_url} workspace={knowledge.workspace_id}")

        # --- persistent memory + checkpointing (P0) ---
        from pokemon_agent.agent.memory import AgentMemory
        ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (states_dir.parent / "runs" / "ckpt")
        memory = AgentMemory()
        if args.resume:
            memfile = ckpt_dir / "latest.mem.json"
            statefile = ckpt_dir / "latest.state"
            if memfile.exists() and statefile.exists():
                memory = AgentMemory.load(memfile)
                emu.load_state(statefile)
                print(f"resumed memory+state from {ckpt_dir} (map_history={memory.map_history[-6:]})")
            else:
                print(f"--resume: no checkpoint at {ckpt_dir}, starting fresh")
        recorder = None
        if not args.no_record and args.mode == "reason":
            from pokemon_agent.logging.run_recorder import RunRecorder, unique_run_dir
            rec_dir = unique_run_dir(Path(args.record_dir) if args.record_dir else (
                states_dir.parent / "runs" / f"rec-{time.strftime('%Y%m%d-%H%M%S')}"))
            recorder = RunRecorder(emu, rec_dir)
            print(f"recording run -> {rec_dir}  (view: serve runs/ and open _viewer.html, or open {rec_dir}/viewer.html)")

        loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                             reasoner=reasoner, session=session, logger=logger, recorder=recorder,
                             vision=use_vision, reflect_every=(args.reflect_every or 8),
                             low_conf_reflect=low_conf_reflect, memory=memory,
                             checkpoint_every=args.checkpoint_every,
                             checkpoint_dir=(ckpt_dir if args.checkpoint_every else None),
                             goal_map=args.goal_map, level_target=args.level_target,
                             strategist_provider=strategist_provider, knowledge=knowledge,
                             pather=args.pather, on_event=on_event_r)
        print(f"running REASON mode decider={args.decider} model={rmodel} vision={use_vision} goal={args.goal!r}")
        try:
            loop.run(max_steps=args.steps)
        finally:
            if args.save_state_out:
                emu.save_state(Path(args.save_state_out))
                print(f"saved state -> {args.save_state_out}")
            if logger:
                logger.close()
            if recorder:
                recorder.close()
            emu.close()
        print(f"done. final player state read: {emu.read_memory(0xD362)},{emu.read_memory(0xD361)} map={emu.read_memory(0xD35E)}")
        return

    provider = build_provider(args.provider, args.model, args.vision)

    # optional vision planner (reflection cadence)
    planner = None
    if args.reflect_every > 0:
        if args.provider != "lunaroute":
            raise SystemExit("--reflect-every needs --provider lunaroute (vision planner)")
        from pokemon_agent.providers.lunaroute import LunaRouteProvider
        from pokemon_agent.agent.planner import Planner
        planner_provider = LunaRouteProvider(model=args.planner_model, max_tokens=800)
        # Absorb the vision model's cold start now (WITH an image, so the multimodal
        # path is warm), so the FIRST real reflection isn't empty.
        try:
            print("warming up vision planner...", flush=True)
            planner_provider.chat_json(
                "Reply with a JSON object {\"ok\": true}.",
                {"ping": 1},
                image=emu.screenshot(),
            )
        except Exception as e:
            print(f"  (warmup skipped: {e})")
        planner = Planner(planner_provider)

    def on_event(kind, payload):
        if kind == "plan":
            p = payload["plan"]
            print(f"  == PLAN ({payload['latency_ms']}ms) [{p['progress']}] reason={payload.get('reason')}")
            print(f"     scene: {p['scene']}")
            for sg in p.get("subgoals", []):
                mark = "x" if sg["status"] == "done" else " "
                print(f"       [{mark}] {sg['text']}")
            if p.get("landmarks"):
                print(f"     landmarks: {p['landmarks']}")
            print(f"     -> checkpoint: {p['next_checkpoint']}")
            print(f"     memory: {p['memory']}")
        elif kind == "decision":
            a = payload["action"]
            print(f"  [{session.step}] think {payload['latency_ms']}ms -> {a} :: {payload['decision_note']}")
        elif kind == "result":
            r = payload["result"]
            stuck = payload["stuck"]
            tag = " STUCK" if stuck.get("stuck") else ""
            print(f"        result={r['result']} moved={r.get('player_moved')} events={r['events']}{tag}")

    loop = AgentLoop(
        builder=ObservationBuilder(emu), provider=provider,
        controller=ActionController(emu), session=session,
        vision=args.vision, planner=planner, reflect_every=args.reflect_every or 10,
        logger=logger, on_event=on_event,
    )
    print(f"running {args.provider} model={args.model or 'default'} demo={args.demo} "
          f"vision={args.vision} reflect_every={args.reflect_every}")
    try:
        loop.run(max_steps=args.steps)
    except ProviderError as e:
        print(f"PROVIDER ERROR: {e}", file=sys.stderr)
    finally:
        if args.save_state_out:
            emu.save_state(Path(args.save_state_out))
            print(f"saved state -> {args.save_state_out}")
        if logger:
            logger.close()
        emu.close()
    print(f"done. final player state read: {emu.read_memory(0xD362)},{emu.read_memory(0xD361)} map={emu.read_memory(0xD35E)}")


if __name__ == "__main__":
    main()
