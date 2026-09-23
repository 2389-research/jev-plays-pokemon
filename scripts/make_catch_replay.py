"""Build a resume source that exercises catching through the FULL agent loop (watchable in the viewer).

Starts from `states/capture_wild.state` (a wild battle in Viridian Forest, 5 Poké Balls in the bag —
built by scripts/make_capture_fixture.py) and a memory whose plan carries a standing catch goal
(`battle_goals.catch = ["any"]`) plus team-building goals. Everything after the first frame is the real
loop: battle_L2 picks CAPTURE, the ball macro throws, and L1 then manages the team (e.g. clears the catch
goal at low HP, later sets a species wishlist).

  uv run python scripts/make_catch_replay.py [--mem runs/<run>/latest.mem.json] [--out runs/verify-src/catch]
  uv run python scripts/run_agent.py --mode reason --decider typesafe --rom roms/pokemon_red.gb \
      --goal-map 2 --level-target 13 --steps 80 --resume-from runs/verify-src/catch --headless \
      --orrery-workspace 6d677a16 --l1-every 5 --capture distill --record-dir runs/catch-replay
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")

from pokemon_agent.agent.memory import AgentMemory  # noqa: E402
from pokemon_agent.agent.plan import AgentPlan, Goal, Goals  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="states/capture_wild.state")
    ap.add_argument("--mem", default=None, help="a learned memory to reuse (world/graph); fresh if omitted")
    ap.add_argument("--out", default="runs/verify-src/catch")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.state, out / "latest.state")
    mem = AgentMemory.load(Path(a.mem)) if a.mem else AgentMemory()
    mem.plan = AgentPlan(
        goals=Goals(primary=Goal(text="Earn the Boulder Badge by beating Brock in Pewter Gym", done_when="badges>=1"),
                    secondary=Goal(text="Build a team: catch a second Pokémon before Brock"),
                    tertiary=Goal(text="Catch the wild Pokémon in this battle")),
        battle_goals={"catch": ["any"]})
    mem.map_history = [50, 51]
    mem.save(out / "latest.mem.json")
    print(f"wrote {out}/latest.state + latest.mem.json (catch goal: any)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
