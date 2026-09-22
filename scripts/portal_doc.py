"""Render the ripped portal graph as a MARKDOWN map document for Orrery ingestion.

Orrery is a document -> knowledge-graph product: we hand it a readable, natural-language document of
the directed map connectivity and it extracts the graph. So we emit the corridor as markdown — one
section per area, each stating its exits as directed relations ("Viridian Forest connects NORTH to
Viridian Forest North Gate"), plus the coordinate of each exit as a plain fact.

Run: uv run python scripts/portal_doc.py  -> writes docs/orrery/kanto_corridor_map.md
"""
import json
import re
from pathlib import Path

GRAPH = json.load(open("scripts/out/portals_corridor.json"))
MAPS, PORTALS = GRAPH["maps"], GRAPH["portals"]
CARD = {"north": "north", "south": "south", "east": "east", "west": "west"}


def friendly(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", name).replace("1 F", "1F")


def mname(mid) -> str:
    m = MAPS.get(str(mid))
    return friendly(m["name"]) if m else f"Map {mid}"


def main():
    out = ["# Kanto Overworld Map — Pallet Town to Pewter City Corridor",
           "",
           "This document is the navigation map for the Pallet Town to Pewter City route in Pokémon "
           "Red. Each area lists the adjacent areas you can travel to and the compass direction of "
           "travel. Directions are ground truth (map-edge connections from the game data; gate/door "
           "exits from the exit tile's position). Coordinates are the exit tile on the current map.",
           ""]
    for mid_str, m in sorted(MAPS.items(), key=lambda kv: int(kv[0])):
        mid = int(mid_str)
        name = friendly(m["name"])
        out.append(f"## {name}")
        # directed cardinal exits, deduped by (direction, dest)
        seen = {}
        for p in PORTALS.values():
            if p["map"] == mid and p["dest_map"] is not None and p.get("direction") in CARD:
                key = (p["direction"], p["dest_map"])
                seen.setdefault(key, p)
        if seen:
            for (d, dm), p in sorted(seen.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                out.append(f"- {name} connects **{d}** to **{mname(dm)}** "
                           f"(exit tile at x={p['coord'][0]}, y={p['coord'][1]}).")
        else:
            out.append(f"- {name} has no cardinal-direction exits recorded.")
        out.append("")
    doc = Path("docs/orrery/kanto_corridor_map.md")
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(out))
    print(f"wrote {doc} ({len(out)} lines, {len(MAPS)} areas)")


if __name__ == "__main__":
    main()
