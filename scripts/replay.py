# /// script
# requires-python = ">=3.12"
# ///
"""Replay / eval harness (design §5) over captured runs.

DECISION replay (no emulator) — diff a candidate model against captured decisions:
  uv run python scripts/replay.py decisions runs/rec-A --layer jev_policy \
      --candidate my_pkg.candidates:policy_predict

BEHAVIORAL replay (ROM-guarded) — reload a step anchor and run forward with a swapped layer:
  uv run python scripts/replay.py behavioral runs/rec-A --from-step 341 --rom roms/pokemon_red.gb \
      --swap l2=my_pkg.candidates:LocalL2 --steps 30

A candidate is loaded from ``module.path:attr``. If ``attr`` is a class it is instantiated;
for DECISION replay it must be (or expose) a ``predict(input)->parsed`` callable; for a SWAP it
is an object implementing the layer's method (e.g. ``propose_target``) or a bare callable.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_agent.logging.replay import (  # noqa: E402
    behavioral_replay, load_layer_rows, replay_decisions,
)


def _load(spec: str):
    """Import ``module.path:attr``; instantiate it if it's a class."""
    mod_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"candidate spec must be 'module:attr', got {spec!r}")
    obj = getattr(importlib.import_module(mod_name), attr)
    return obj() if isinstance(obj, type) else obj


def _as_predict(cand):
    if hasattr(cand, "predict"):
        return cand.predict
    if callable(cand):
        return cand
    raise SystemExit("decision candidate must be callable or expose .predict(input)->parsed")


def cmd_decisions(args) -> None:
    predict = _as_predict(_load(args.candidate))
    rows = load_layer_rows(Path(args.source), args.layer)
    if not rows:
        raise SystemExit(f"no rows for layer {args.layer!r} in {args.source}")
    report = replay_decisions(rows, predict)
    summary = report.summary()
    print(f"decision replay: layer={args.layer} rows={summary['n']}")
    print(f"  agreement            {summary['agree']}/{summary['n']}  ({summary['agreement_rate']:.1%})")
    print(f"  agreement (progress) {summary['agree_on_progress']}/{summary['n_progress']} "
          f" ({summary['progress_agreement_rate']:.1%})")
    print(f"  disagreements        {summary['disagreements']}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"summary": summary, "disagreements": report.disagreements}, indent=2))
        print(f"  wrote {args.out}")


def _reason_loop_builder(args):
    """Default behavioral-replay loop builder — a reason-mode agent matching run_agent.py's
    typesafe path. ``build_loop(emu) -> (loop, recorder)``."""
    def build(emu):
        import time as _t

        from pokemon_agent.actions.controller import ActionController
        from pokemon_agent.agent.reason_loop import ReasoningLoop
        from pokemon_agent.agent.reasoner import Reasoner
        from pokemon_agent.agent.session import Session
        from pokemon_agent.agent.typesafe_reasoner import TypeSafeReasoner
        from pokemon_agent.core.models import GoalState
        from pokemon_agent.logging.run_recorder import RunRecorder, unique_run_dir
        from pokemon_agent.observations.builder import ObservationBuilder
        from pokemon_agent.providers.lunaroute import LunaRouteProvider

        meta = {}
        mp = Path(args.source) / "run_meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        vprov = LunaRouteProvider(model=args.model or "deepseek-4.1-flash", max_tokens=900)
        reasoner = TypeSafeReasoner(model=args.typesafe_model, reflector=Reasoner(vprov))
        rec_dir = unique_run_dir(Path("runs") / f"replay-{_t.strftime('%Y%m%d-%H%M%S')}")
        recorder = RunRecorder(emu, rec_dir, state_every=1)
        loop = ReasoningLoop(
            builder=ObservationBuilder(emu), controller=ActionController(emu), reasoner=reasoner,
            session=Session(GoalState(primary=args.goal, current=args.goal)), vision=False,
            recorder=recorder, goal_map=args.goal_map or meta.get("goal_map"),
            level_target=meta.get("level_target", 0), pather=meta.get("pather", "bfs"),
            l1_every=meta.get("l1_every", 5), capture_mode="distill")
        return loop, recorder
    return build


def cmd_behavioral(args) -> None:
    swaps = {}
    for spec in (args.swap or []):
        layer, _, cand_spec = spec.partition("=")
        if not cand_spec:
            raise SystemExit(f"--swap must be 'layer=module:attr', got {spec!r}")
        swaps[layer] = _load(cand_spec)
    out_dir = behavioral_replay(
        Path(args.source), args.from_step, rom_path=args.rom,
        build_loop=_reason_loop_builder(args), swaps=swaps, steps=args.steps)
    print(f"behavioral replay from step {args.from_step}"
          f"{' with swaps ' + ','.join(swaps) if swaps else ''} -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay/eval harness over captured runs (design §5).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decisions", help="diff a candidate against captured decisions (no emulator)")
    d.add_argument("source", help="a record-dir (decisions.jsonl) or an exported <layer>.jsonl")
    d.add_argument("--layer", required=True, help="the layer to replay (e.g. l2_propose_target, jev_policy)")
    d.add_argument("--candidate", required=True, help="module:attr -> predict(input)->parsed")
    d.add_argument("--out", default=None, help="write the full report JSON here")
    d.set_defaults(func=cmd_decisions)

    b = sub.add_parser("behavioral", help="reload a step anchor and run forward with a swapped layer")
    b.add_argument("source", help="the original record-dir (has states/*_step<N>.state anchors)")
    b.add_argument("--from-step", type=int, required=True, help="step whose anchor to reload")
    b.add_argument("--rom", required=True, help="path to the ROM")
    b.add_argument("--swap", action="append", default=None, help="layer=module:attr (repeatable)")
    b.add_argument("--steps", type=int, default=20, help="steps to run forward")
    b.add_argument("--goal", default="continue")
    b.add_argument("--goal-map", type=int, default=None)
    b.add_argument("--model", default=None, help="reflection model")
    b.add_argument("--typesafe-model", default=None)
    b.set_defaults(func=cmd_behavioral)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
