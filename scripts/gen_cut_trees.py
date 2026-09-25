#!/usr/bin/env python3
"""Every small tree Cut removes, per map, with the static component on each side of it.

The portal graph's components treat these trees as walls, so a way that only needs Cut looked like
"no route" and the router sent the agent the long way around (runs/sleeves-east: on Route 9, four
tiles from the tree at (5,8), "go to Route 10" walked back through Cerulean toward Saffron's guards).
PortalGraph turns each tree into crossings that need Cut. Tree tiles: engine/overworld/cut.asm ($3D in
the OVERWORLD tileset, $50 in GYM). Components come from the shipped graph's own grid, so they match.
Output: src/pokemon_agent/games/pokemon_red/cut_trees.json  {map_id: [{"x", "y", "sides": {comp: [ax, ay]}}]}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import rip_portals as rp  # noqa: E402
from pokemon_agent.agent.portal_graph import PortalGraph  # noqa: E402

OUT = rp.REPO / "src" / "pokemon_agent" / "games" / "pokemon_red" / "cut_trees.json"
TREE = {"OVERWORLD": 0x3D, "GYM": 0x50}


def main() -> None:
    pg = PortalGraph.load()
    consts, tilesets = rp.parse_map_constants(), rp.parse_tilesets()
    out: dict[str, list] = {}
    for name in rp.all_map_names():
        try:
            md = rp.load_map(name, consts, tilesets)
        except Exception:
            continue
        tid = TREE.get(md.tileset)
        if tid is None or md.id not in pg.maps:
            continue
        grid = pg._grid(md.id)
        for (x, y), t in sorted(md.tiles.items()):
            if t != tid:
                continue
            sides: dict[str, list[int]] = {}
            for nb in ((x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)):
                c = grid.get(nb)
                if c is not None:
                    sides.setdefault(str(c), list(nb))
            out.setdefault(str(md.id), []).append({"x": x, "y": y, "sides": sides})
    OUT.write_text(json.dumps(out, sort_keys=True))
    joins = sum(1 for ts in out.values() for t in ts if len(t["sides"]) >= 2)
    print(f"wrote {OUT.relative_to(rp.REPO)}: {sum(len(v) for v in out.values())} trees on {len(out)} maps, "
          f"{joins} join two areas; Route 9: {out.get(str(next(m for m, v in pg.maps.items() if v['name'] == 'Route9')))}")


if __name__ == "__main__":
    main()
