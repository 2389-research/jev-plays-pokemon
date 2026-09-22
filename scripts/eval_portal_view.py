"""Agent-reads-it eval: render the portal graph as the scoped natural-language view and check that
LunaRoute models DECIDE the right next portal by name (no coordinates, no route handed to them).

For each scenario we render what the agent would see standing on a map/component, state a goal, and ask
the model which neighbouring area to head to. Ground truth = a BFS first-hop over the portal graph. We
score the model's pick against it. This tests the REPRESENTATION (can a model route from it), not BFS.

Live (spends LunaRoute credits): uv run python scripts/eval_portal_view.py
"""
import json
import re
import sys
from collections import deque

sys.path.insert(0, "src")

GRAPH = json.load(open("scripts/out/portals_corridor.json"))
MAPS, PORTALS = GRAPH["maps"], GRAPH["portals"]


def friendly(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", name).replace("1 F", "1F")


def map_name(mid) -> str:
    m = MAPS.get(str(mid))
    return friendly(m["name"]) if m else f"map {mid}"


def same_comp_portals(mid, comp):
    return [p for p in PORTALS.values() if p["map"] == mid and p["component"] == comp]


def neighbors(mid, comp):
    """Distinct neighbouring maps reachable ON FOOT from this component, with the portal to use."""
    out = {}
    for p in same_comp_portals(mid, comp):
        dm = p["dest_map"]
        if dm is not None and dm not in out:
            out[dm] = p
    return out


def _comps_of(mid):
    return {p["component"] for p in PORTALS.values() if p["map"] == mid}


def first_hop(mid, comp, goal_map):
    """Ground-truth: BFS over the portal graph (nodes = (map, component)); return the MAP of the
    first hop on the shortest path to the goal map."""
    start = (mid, comp)
    prev = {start: None}
    q = deque([start])
    goal_node = None
    while q:
        node = q.popleft()
        cm, cc = node
        if cm == goal_map:
            goal_node = node
            break
        for p in same_comp_portals(cm, cc):
            dp = PORTALS.get(p.get("dest_portal") or "")
            dm = p["dest_map"]
            if dm is None:
                continue
            dests = [(dm, dp["component"])] if dp else [(dm, c) for c in _comps_of(dm)]
            for t in dests:
                if t not in prev:
                    prev[t] = node
                    q.append(t)
    if goal_node is None:
        return None
    path = [goal_node]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    path.reverse()
    return path[1][0] if len(path) > 1 else goal_map


CARDINAL = {"north": "NORTH", "south": "SOUTH", "east": "EAST", "west": "WEST"}


def render(mid, comp, goal_map):
    """The scoped natural-language view (names + compass direction of travel, no coordinates)."""
    lines = [f"YOU ARE: {map_name(mid)}."]
    lines.append("\nEXITS FROM HERE (the compass direction you travel to take each):")
    for dm, p in neighbors(mid, comp).items():
        d = p.get("direction", "interior")
        way = f"go {CARDINAL[d]}" if d in CARDINAL else "go inside"
        lines.append(f"  - {way}  ->  {map_name(dm)}")
    # directed nearby connectivity: which way you travel between areas (cardinal links only)
    lines.append("\nHOW AREAS CONNECT (direction of travel between them):")
    seen = set()
    frontier = {mid}
    for _ in range(3):
        nxt = set()
        for m in sorted(frontier):
            if m in seen:
                continue
            links = {}
            for p in PORTALS.values():
                if p["map"] == m and p["dest_map"] is not None and p.get("direction") in CARDINAL:
                    links[(p["direction"], p["dest_map"])] = None
            if links:
                lines.append(f"  {map_name(m)}:  " +
                             ";  ".join(f"{CARDINAL[d]} -> {map_name(dm)}" for (d, dm) in links))
                seen.add(m)
            nxt.update(dm for (_, dm) in links)
        frontier = nxt - seen
    lines.append(f"\nGOAL: reach {map_name(goal_map)}.")
    return "\n".join(lines)


SYS = ("You are navigating a game world. You are given where you are, the places you can walk to from "
       "here, and how areas connect. Choose the SINGLE adjacent place to go to next that makes progress "
       "toward the goal. Reply ONLY as JSON: {\"go_to\": \"<exact place name from the list>\", \"why\": \"<short>\"}.")

SCENARIOS = [
    ("Forest -> Pewter (must pick NORTH gate)", 51, 0, 2, 47),
    ("Route2 SOUTH -> Pewter (must detour via South Gate/forest)", 13, 10, 2, 50),
    ("Route2 NORTH -> Pewter (direct north edge)", 13, 0, 2, 2),
    ("Viridian -> Pewter (go north to Route 2)", 1, 0, 2, 13),
    ("Forest -> Viridian (reverse: pick SOUTH gate)", 51, 0, 1, 50),
]


def run(model):
    from pokemon_agent.providers.lunaroute import LunaRouteProvider
    prov = LunaRouteProvider(model=model)
    print(f"\n################ MODEL: {model} ################")
    passed = 0
    for title, mid, comp, goal, expect in SCENARIOS:
        view = render(mid, comp, goal)
        gt = first_hop(mid, comp, goal)
        try:
            content, _, _ = prov.chat_json(SYS, {"situation": view})
            data = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
            pick = data.get("go_to", "")
            why = data.get("why", "")
        except Exception as e:
            print(f"  [{title}] CALL FAILED: {e}")
            continue
        ok = friendly(MAPS[str(expect)]["name"]).lower() in pick.lower()
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {title}")
        print(f"        model -> {pick!r}  (expect {map_name(expect)!r}; bfs first-hop map {gt})")
        if not ok:
            print(f"        why: {why}")
    print(f"  SCORE: {passed}/{len(SCENARIOS)}")


if __name__ == "__main__":
    print("=== sample rendered view (Forest, goal Pewter) ===")
    print(render(51, 0, 2))
    for m in ["deepseek-4.1-flash", "glm-5.3"]:
        try:
            run(m)
        except Exception as e:
            print(f"model {m}: {type(e).__name__}: {e}")
