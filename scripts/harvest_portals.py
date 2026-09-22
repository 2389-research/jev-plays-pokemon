"""Portal-graph harvester SPIKE (deterministic, from RAM save states).

For each map we have a save state for: load it, read the RAM warp table + full-map collision
(ground-truth readers), split the walkable set into connected components, and place each warp /
map-edge as a PORTAL tagged with the walkable component it sits in. The crux test: on Route 2 (13),
does the north edge (-> Pewter) land in a DIFFERENT component than the South-Gate warp (-> forest)?
If so, the coarse-node bug is fixed by geometry alone.

Run: uv run python scripts/harvest_portals.py
"""
import sys
from collections import deque, defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator
from pokemon_agent.games.pokemon_red.state import read_exits
from pokemon_agent.games.pokemon_red.map_reader import read_collision_map
from pokemon_agent.games.pokemon_red.maps import map_name
from pokemon_agent.games.pokemon_red.map_graph_data import CONNECTIONS

ROM = "roms/pokemon_red.gb"
WWARPENTRIES = 0xD3AF  # y,x,destWarp,destMap (4 bytes each)
WNUMWARPS = 0xD3AE


def latest_state_per_map():
    """Newest save state file for each map id across all runs."""
    best: dict[int, Path] = {}
    for p in sorted(Path("runs").glob("*/states/map*.state")):
        mid = int(p.name.split("_")[0][3:])
        # prefer the most recently modified
        if mid not in best or p.stat().st_mtime > best[mid].stat().st_mtime:
            best[mid] = p
    return best


def components(walkable: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """4-connected flood fill -> {cell: component_id}."""
    comp: dict[tuple[int, int], int] = {}
    cid = 0
    for start in walkable:
        if start in comp:
            continue
        q = deque([start])
        comp[start] = cid
        while q:
            x, y = q.popleft()
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in walkable and nb not in comp:
                    comp[nb] = cid
                    q.append(nb)
        cid += 1
    return comp


def harvest(emu, path: Path) -> dict:
    emu.load_state(path)
    emu.tick(2)
    coll = read_collision_map(emu)
    if coll is None:
        return {}
    mid = coll["map_id"]
    W, H = coll["width"], coll["height"]
    walk = coll["walkable"]
    comp = components(walk)
    ncomp = len(set(comp.values()))
    sizes = defaultdict(int)
    for c in comp.values():
        sizes[c] += 1

    # warps (add the destWarp index the base reader drops)
    warps = []
    n = emu.read_memory(WNUMWARPS)
    for i in range(min(n, 32)):
        y = emu.read_memory(WWARPENTRIES + i * 4 + 0)
        x = emu.read_memory(WWARPENTRIES + i * 4 + 1)
        dwarp = emu.read_memory(WWARPENTRIES + i * 4 + 2)
        dest = emu.read_memory(WWARPENTRIES + i * 4 + 3)
        warps.append({"x": x, "y": y, "dest_map": dest, "dest_warp": dwarp,
                      "dest_name": map_name(dest), "comp": comp.get((x, y), "off-walk")})

    # edge connections for this map, grouped by the component their border tiles land in
    edges = []
    for (fm, direction, to) in CONNECTIONS:
        if fm != mid:
            continue
        if direction == "north":
            border = [(x, 0) for x in range(W)]
        elif direction == "south":
            border = [(x, H - 1) for x in range(W)]
        elif direction == "west":
            border = [(0, y) for y in range(H)]
        else:  # east
            border = [(W - 1, y) for y in range(H)]
        bw = [c for c in border if c in walk]
        by_comp = defaultdict(list)
        for c in bw:
            by_comp[comp[c]].append(c)
        for cslot, cells in by_comp.items():
            cells.sort()
            rep = cells[len(cells) // 2]
            edges.append({"dir": direction, "dest_map": to, "dest_name": map_name(to),
                          "comp": cslot, "rep": rep, "nwalk": len(cells)})

    return {"map_id": mid, "name": map_name(mid), "W": W, "H": H,
            "ncomp": ncomp, "sizes": dict(sizes), "warps": warps, "edges": edges}


def main():
    states = latest_state_per_map()
    focus = [13, 50, 51, 1, 0]  # Route 2, South Gate, Forest, Viridian, Pallet
    emu = PyBoyEmulator(ROM, window="null")
    try:
        for mid in focus:
            if mid not in states:
                print(f"\n=== map {mid}: NO SAVE STATE ===")
                continue
            r = harvest(emu, states[mid])
            if not r:
                print(f"\n=== map {mid}: collision unreadable ===")
                continue
            print(f"\n=== map {r['map_id']} {r['name']}  ({r['W']}x{r['H']})  "
                  f"components={r['ncomp']} sizes={r['sizes']} ===")
            print("  WARPS:")
            for w in r["warps"]:
                print(f"    ({w['x']:2},{w['y']:2}) comp={w['comp']}  -> map {w['dest_map']} "
                      f"{w['dest_name']} (warp#{w['dest_warp']})")
            print("  EDGES (map-border connections):")
            for e in r["edges"]:
                print(f"    {e['dir']:5} comp={e['comp']} rep={e['rep']} nwalk={e['nwalk']} "
                      f"-> map {e['dest_map']} {e['dest_name']}")
    finally:
        emu.close()


if __name__ == "__main__":
    main()
