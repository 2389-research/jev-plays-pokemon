#!/usr/bin/env python3
"""Rip every arrow (spinner) tile: standing on it, the game moves the player along a fixed path.

pokered: each map script with a `...ArrowTilePlayerMovement` table lists `map_coord_movement x, y, Label`;
`Label` is an RLE list of (direction, count) that DecodeArrowMovementRLE feeds to the simulated joypad
(played back in reverse; the landing tile is the trigger plus the sum of the moves either way).
runs/sleeves-next: Rocket Hideout B3F (18,16) pushes the player UP 1; the router kept stepping back onto it
toward the B4F door and was pushed back ~750 times. Output:
src/pokemon_agent/games/pokemon_red/spinners.json  {map_id: {"x,y": [land_x, land_y]}}
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rip_portals import POKERED, REPO  # noqa: E402
from pokemon_agent.agent.portal_graph import PortalGraph  # noqa: E402

OUT = REPO / "src" / "pokemon_agent" / "games" / "pokemon_red" / "spinners.json"
STEP = {"PAD_UP": (0, -1), "PAD_DOWN": (0, 1), "PAD_LEFT": (-1, 0), "PAD_RIGHT": (1, 0)}


def main() -> None:
    pg = PortalGraph.load()
    by_name = {v["name"]: mid for mid, v in pg.maps.items()}
    out: dict[str, dict] = {}
    for f in sorted((POKERED / "scripts").glob("*.asm")):
        text = f.read_text()
        table = re.search(r"^(\w*ArrowTilePlayerMovement):\n(.*?)\n\s*db -1", text, re.M | re.S)
        if not table or f.stem not in by_name:
            continue
        lists = {m.group(1): m.group(2) for m in re.finditer(r"^(\w+):\n((?:\s*db PAD_\w+, \d+\n)+)", text, re.M)}
        spins = {}
        for x, y, label in re.findall(r"map_coord_movement\s+(\d+),\s*(\d+),\s*(\w+)", table.group(2)):
            dx = dy = 0
            for d, n in re.findall(r"db (PAD_\w+), (\d+)", lists[label]):
                dx += STEP[d][0] * int(n)
                dy += STEP[d][1] * int(n)
            spins[f"{x},{y}"] = [int(x) + dx, int(y) + dy]
        out[str(by_name[f.stem])] = spins
        print(f"{f.stem} (map {by_name[f.stem]}): {len(spins)} arrow tiles")
    OUT.write_text(json.dumps(out, sort_keys=True))
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
