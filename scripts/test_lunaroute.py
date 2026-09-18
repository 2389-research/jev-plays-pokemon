#!/usr/bin/env python
"""LunaRoute diagnostics: list models / smoke-test a decision.

  uv run python scripts/test_lunaroute.py --list-models
  uv run python scripts/test_lunaroute.py --smoke --model deepseek-4.1-flash
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_agent.core.models import GameMode, GoalState, Observation, PlayerState
from pokemon_agent.providers.interface import AgentRequest
from pokemon_agent.providers.lunaroute import LunaRouteProvider
from pokemon_agent.agent.prompts import SYSTEM_PROMPT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    prov = LunaRouteProvider(model=args.model)
    if args.list_models:
        for m in sorted(prov.list_models()):
            print(m)
    if args.smoke:
        obs = Observation(
            mode=GameMode.OVERWORLD,
            player=PlayerState(x=3, y=2, map_id=40),
            walkability=["###", "#@.", "#.#"],
            available_actions=["move_east", "move_south", "press_a"],
        )
        req = AgentRequest(system_prompt=SYSTEM_PROMPT, observation=obs, goal=GoalState())
        resp = prov.complete(req)
        print(f"model={resp.model} latency={resp.latency_ms}ms usage={resp.usage}")
        print("decision:", resp.decision.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
