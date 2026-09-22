# /// script
# requires-python = ">=3.12"
# ///
"""Export captured runs into per-layer supervised datasets (design §7).

Examples:
  # every layer from two runs -> datasets/<layer>.jsonl
  uv run python scripts/distill_export.py runs/rec-A runs/rec-B --out datasets

  # only the L2 proposer, only from successful episodes, only progress-making steps, deduped
  uv run python scripts/distill_export.py runs/* --layer l2_propose_target \
      --only-successful-episodes --min-progress 1 --dedup --out datasets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_agent.logging.distill_export import export  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Export per-layer distillation datasets from record-dirs.")
    ap.add_argument("record_dirs", nargs="+", help="one or more capture record-dirs (with decisions.jsonl)")
    ap.add_argument("--out", default="datasets", help="output directory for <layer>.jsonl files")
    ap.add_argument("--layer", action="append", default=None,
                    help="restrict to this layer (repeatable); default = all layers")
    ap.add_argument("--only-successful-episodes", action="store_true",
                    help="keep only rows from runs whose outcome.reached_goal_map is true")
    ap.add_argument("--min-progress", type=int, default=None,
                    help="keep only rows whose step made >= N progress (map change / level / item / catch)")
    ap.add_argument("--dedup", action="store_true", help="drop duplicate (layer, input) rows")
    ap.add_argument("--include-rounds", action="store_true",
                    help="also export intermediate KB-search 'model_round' records (off by default)")
    args = ap.parse_args()

    counts = export(
        [Path(d) for d in args.record_dirs], Path(args.out),
        layers=args.layer, only_successful=args.only_successful_episodes,
        min_progress=args.min_progress, dedup=args.dedup, include_rounds=args.include_rounds,
    )
    total = sum(counts.values())
    print(f"exported {total} rows across {len(counts)} layer(s) -> {args.out}/")
    for layer, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {layer:24s} {n}")


if __name__ == "__main__":
    main()
