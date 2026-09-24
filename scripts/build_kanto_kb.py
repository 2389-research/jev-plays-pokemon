"""Build searchable Kanto knowledge for L1 and ingest it into Orrery.

The harness keeps MECHANICS (the PortalGraph routes step by step); the knowledge L1 *reasons* with —
what's where, what a Mart sells, which Pokémon live on a route, what blocks a path — belongs in the
Orrery knowledge base, where L1's brainstorm searches it. Every document here except the curated
story-order overview is GENERATED from ground truth: the pokered disassembly (marts, prices, wild
encounters, guard drinks) and the shipped Kanto portal graph (connections, buildings, one-way ledges,
dungeon floors, story gates).

  uv run python scripts/build_kanto_kb.py                 # write runs/kb/kanto/*.md (inspect)
  uv run python scripts/build_kanto_kb.py --ingest        # + replace the "Kanto: ..." docs in Orrery
      [--workspace 6d677a16] [--url http://localhost:8100]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pokemon_agent.agent.portal_graph import PortalGraph, _friendly  # noqa: E402
from pokemon_agent.games.pokemon_red.constants import ITEMS  # noqa: E402
from pokemon_agent.games.pokemon_red.game_state import resolve_item_id  # noqa: E402

POKERED = Path(os.environ.get("POKERED", "/tmp/pokered"))
OUT = REPO / "runs" / "kb" / "kanto"
PREFIX = "Kanto: "
SLOT_CHANCE = [51, 51, 39, 25, 25, 25, 13, 13, 11, 3]          # data/wild/probabilities.asm (/256)
OUTSIDE = {"OVERWORLD", "PLATEAU"}
DUNGEON_TILESETS = {"CAVERN", "FOREST", "CEMETERY", "FACILITY", "SHIP", "SHIP_PORT", "UNDERGROUND"}
FLOOR = re.compile(r"(B?\d+F|Entrance|Exit)$")


def name(n: str) -> str:
    return _friendly(n)


def item_name(const: str) -> str:
    guess = const.replace("_", " ").title()
    iid = resolve_item_id(guess)
    return ITEMS.get(iid, guess) if iid is not None else guess


def species(const: str) -> str:
    return const.replace("_M", " M").replace("_F", " F").replace("_", " ").title()


# ---------------------------------------------------------------- pokered data
def prices() -> dict[str, int]:
    out = {}
    for m in re.finditer(r"bcd3\s+(\d+)\s*;\s*(\w+)", (POKERED / "data/items/prices.asm").read_text()):
        out[m.group(2)] = int(m.group(1))
    return out


def marts() -> list[tuple[str, list[str]]]:
    out = []
    text = (POKERED / "data/items/marts.asm").read_text()
    for m in re.finditer(r"(\w+?)(?:Mart)?(\w*?)ClerkText::\s*(?:;[^\n]*)?\n\s*script_mart\s+([^\n]+)", text):
        label = m.group(1) + m.group(2)
        if label.startswith("Unused"):
            continue
        out.append((label, [t.strip() for t in m.group(3).split(",")]))
    return out


def wild_tables() -> dict[str, dict]:
    """{map file name: {"grass": (rate, [(lvl, species)]), "water": (...)}} — Red version."""
    out = {}
    for f in sorted((POKERED / "data/wild/maps").glob("*.asm")):
        text = f.read_text()
        text = re.sub(r"IF DEF\(_BLUE\).*?ENDC", "", text, flags=re.S)
        text = re.sub(r"IF DEF\(_RED\)(.*?)ENDC", r"\1", text, flags=re.S)
        tab = {}
        for kind in ("grass", "water"):
            m = re.search(rf"def_{kind}_wildmons\s+(\d+)(.*?)end_{kind}_wildmons", text, re.S)
            if m and int(m.group(1)) > 0:
                slots = [(int(a), b) for a, b in re.findall(r"db\s+(\d+)\s*,\s*(\w+)", m.group(2))]
                tab[kind] = (int(m.group(1)), slots)
        if tab:
            out[f.stem] = tab
    return out


def wild_summary(slots: list[tuple[int, str]]) -> list[tuple[str, int, int, float]]:
    agg: dict[str, list] = {}
    for i, (lvl, sp) in enumerate(slots):
        a = agg.setdefault(sp, [lvl, lvl, 0.0])
        a[0], a[1] = min(a[0], lvl), max(a[1], lvl)
        a[2] += SLOT_CHANCE[i] / 256 * 100 if i < len(SLOT_CHANCE) else 0
    return sorted(((species(sp), lo, hi, pct) for sp, (lo, hi, pct) in agg.items()), key=lambda t: -t[3])


# ---------------------------------------------------------------- graph-derived docs
def area_lines(pg: PortalGraph, mid: int) -> list[str]:
    """Describe each walkable area of a map by its exits (the part of the map you are in matters:
    e.g. Route 4's Mt. Moon exit shelf is only reachable from inside the cave)."""
    by_comp: dict[int, list[str]] = defaultdict(list)
    for p in pg.portals_on(mid):
        if p["dest_map"] is None or p["component"] is None:
            continue
        dest = pg.map_name(p["dest_map"])
        if p["kind"] == "edge":
            desc = f"walk off the {p['direction']} edge to {dest}"
        elif p["kind"] == "ledge":
            desc = (f"hop DOWN a one-way ledge ({p['hop']}) into area {p['dest_component']}"
                    if p["dest_map"] == mid else f"hop down a one-way ledge ({p['hop']}) onto {dest}")
        elif p["kind"] == "elevator":
            desc = "elevator"
        else:
            desc = f"door/entrance to {dest}"
        if desc not in by_comp[p["component"]]:
            by_comp[p["component"]].append(desc)
    if not by_comp:
        return []
    if len(by_comp) == 1:
        return ["Exits: " + "; ".join(next(iter(by_comp.values()))) + "."]
    lines = [f"This map has {len(by_comp)} separate walkable areas (you can't walk between them directly):"]
    for c, exits in sorted(by_comp.items()):
        lines.append(f"  - area {c}: " + "; ".join(exits) + ".")
    return lines


def group_key(map_name_: str) -> str:
    for pre in ("SSAnne", "SafariZone", "DiglettsCave", "SilphCo", "PokemonMansion", "RocketHideout",
                "PokemonTower", "SeafoamIslands", "VictoryRoad", "CeruleanCave", "RockTunnel", "MtMoon"):
        if map_name_.startswith(pre):
            return pre
    return FLOOR.sub("", map_name_)


def build_docs() -> dict[str, str]:
    pg = PortalGraph.load()
    wild = wild_tables()
    docs: dict[str, str] = {}
    names = {mid: m["name"] for mid, m in pg.maps.items()}
    tiles = {mid: m.get("tileset") for mid, m in pg.maps.items()}

    # --- areas: every outdoor map
    wild_where: dict[str, list[str]] = defaultdict(list)
    for wname, tab in wild.items():
        for kind, (rate, slots) in tab.items():
            for sp, lo, hi, pct in wild_summary(slots):
                wild_where[sp].append(f"{name(wname)} ({kind}, L{lo}-{hi}, {pct:.0f}%)")

    def wild_block(map_file: str) -> list[str]:
        tab = wild.get(map_file)
        if not tab:
            return []
        out = []
        for kind, (rate, slots) in tab.items():
            mons = ", ".join(f"{sp} L{lo}{'' if lo == hi else f'-{hi}'} ({pct:.0f}%)"
                             for sp, lo, hi, pct in wild_summary(slots))
            where = "in the tall grass / cave floor" if kind == "grass" else "while surfing or fishing on water"
            out.append(f"Wild Pokémon {where} (encounter rate {rate}): {mons}.")
        return out

    for mid, nm in sorted(names.items()):
        if tiles[mid] not in OUTSIDE:
            continue
        inside = sorted({pg.map_name(p["dest_map"]) for p in pg.portals_on(mid)
                         if p["kind"] in ("warp", "elevator") and p["dest_map"] is not None
                         and tiles.get(p["dest_map"]) not in OUTSIDE})
        body = [f"{name(nm)} is an outdoor area of Kanto (map id {mid})."]
        body += area_lines(pg, mid)
        if inside:
            body.append("Buildings / entrances here: " + ", ".join(inside) + ".")
        body += wild_block(nm)
        docs[f"{PREFIX}{name(nm)} (area guide)"] = "\n".join(body)

    # --- dungeons / multi-floor places
    groups: dict[str, list[int]] = defaultdict(list)
    for mid, nm in names.items():
        if tiles[mid] in DUNGEON_TILESETS:
            groups[group_key(nm)].append(mid)
    # a real dungeon: a cave/forest, or a multi-floor place — not a single gym / house / ship cabin
    groups = {g: m for g, m in groups.items()
              if len(m) > 1 or any(tiles[x] in ("CAVERN", "FOREST") for x in m)}
    for g, mids in sorted(groups.items()):
        mids.sort()
        body = [f"{name(g)} is a dungeon / multi-floor place with {len(mids)} map(s): "
                + ", ".join(pg.map_name(m) for m in mids) + "."]
        outs = sorted({(p["dest_map"], p["map"]) for m in mids for p in pg.portals_on(m)
                       if p["dest_map"] is not None and p["dest_map"] not in mids})
        if outs:
            body.append("Ways in/out: " + "; ".join(f"{pg.map_name(src)} <-> {pg.map_name(dm)}" for dm, src in outs) + ".")
        # through-routes: from each outside area next to the dungeon, to the place the OTHER exits lead
        # (e.g. Mt. Moon: Route 4 west -> ... -> Cerulean City, since both exits sit on Route 4)
        exit_areas = []
        for m in mids:
            for p in pg.portals_on(m):
                dm = p["dest_map"]
                if dm is None or dm in mids:
                    continue
                dp = pg.portals.get(p.get("dest_portal") or "")
                area = (dm, dp["component"] if dp else None)
                if area not in exit_areas and area[1] is not None:
                    exit_areas.append(area)

        def beyond(area, avoid_map):
            """First other map reachable on foot from an outside area (following ledges/edges)."""
            seen, q = {area}, [area]
            while q:
                mp, c = q.pop(0)
                for p in pg.exits_from(mp, c):
                    if p["dest_map"] in mids:
                        continue
                    if p["dest_map"] not in (mp, avoid_map):
                        return p["dest_map"]
                    for t in pg._dest_nodes(p):
                        if t not in seen:
                            seen.add(t)
                            q.append(t)
            return None

        for a in exit_areas:
            for b in exit_areas:
                if a == b:
                    continue
                target = beyond(b, a[0])
                if target is None:
                    continue
                r = pg.route(a[0], a[1], target)
                if not r or not any(p["map"] in mids for p in r):
                    continue
                steps = " -> ".join([pg.map_name(r[0]["map"])] + [pg.map_name(p["dest_map"]) for p in r])
                line = f"To cross from {pg.map_name(a[0])} to {pg.map_name(target)}: {steps}."
                if line not in body:
                    body.append(line)
        for m in mids:
            lines = area_lines(pg, m)
            if lines:
                body.append(f"{pg.map_name(m)}: " + " ".join(lines))
            body += [f"{pg.map_name(m)} — {w}" for w in wild_block(names[m])]
        docs[f"{PREFIX}{name(g)} (dungeon guide)"] = "\n".join(body)

    # --- marts
    pr = prices()
    lines = ["What each Poké Mart sells in Pokémon Red (price in Pokédollars). Only these items can be",
             "bought at that Mart — plan purchases for the Mart that stocks the item."]
    for label, items in marts():
        where = re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", label.replace("Clerk", "").replace("Mart", " Mart")).strip()
        where = re.sub(r"\s+", " ", where)
        stock = [f"{item_name(i)} (${pr.get(i, '?')})" if not i.startswith("TM_")
                 else f"{i.replace('TM_', 'TM ').replace('_', ' ').title()} (${pr.get(i, '?')})" for i in items]
        lines.append(f"- {where}: " + ", ".join(stock))
        names_only = ", ".join(x.split(" ($")[0] for x in stock)
        docs[f"{PREFIX}{where} sells: {names_only}"] = (
            f"The {where} (a Poké Mart in Pokémon Red) sells exactly these items — anything else is not "
            f"available there: " + ", ".join(stock) + ". Talk to the clerk at the counter to buy.")
    docs[f"{PREFIX}Poké Mart inventories and prices (all Marts)"] = "\n".join(lines)

    # --- where to catch each Pokémon
    spp = sorted(wild_where)
    for i in range(0, len(spp), 12):
        chunk = spp[i:i + 12]
        lines = ["Where to catch these wild Pokémon in Pokémon Red (area, grass or water, level range, share",
                 "of encounters there) — for catching and team building:"]
        lines += [f"- {sp}: " + "; ".join(wild_where[sp]) for sp in chunk]
        docs[f"{PREFIX}where to catch {', '.join(chunk)}"] = "\n".join(lines)

    # --- story gates / blockers (graph gates + data + well-known blockers)
    drinks = [item_name(c) for c in re.findall(r"db\s+([A-Z_]+)", (POKERED / "data/items/guard_drink_items.asm").read_text())]
    lines = ["Story gates and blockers in Pokémon Red — places you can't pass until something is done:",
             f"- Saffron City gates (Routes 5/6/7/8): the guards are thirsty; give one a drink ({', '.join(drinks)} —"
             " sold in the vending machines on the Celadon Department Store roof). After that all four gates open.",
             "- Viridian City north exit: an old man blocks it until you deliver Oak's Parcel.",
             "- Mt. Moon B2F: the Dome Fossil and Helix Fossil sit in the two-wide corridor that leads to the exit",
             "  ladder — walk up to one and press A to take it (you only get one; the Super Nerd takes the other);",
             "  once they're gone the corridor opens toward the Route 4 exit and Cerulean.",
             "- Cerulean City: the ONLY way from the city to its south exit (Route 5 -> Vermilion) is THROUGH the",
             "  trashed house: in its front door, out the hole in its back wall into the backyard (a Team Rocket",
             "  grunt there gives TM28 Dig), then south. A police officer stands at the front door until Bill gives",
             "  you the S.S. Ticket. The city's west edge only leads back to Route 4 / Mt. Moon.",
             "- Snorlax sleeps across Route 12 and Route 16: wake it with the Poké Flute (from Mr. Fuji in Lavender after",
             "  clearing Pokémon Tower with the Silph Scope).",
             "- Cycling Road (Route 17, via the Route 16/18 gates): requires a Bicycle (Cerulean Bike Shop, with the",
             "  Bike Voucher from the Pokémon Fan Club chairman in Vermilion).",
             "- Route 22 gate / Route 23: guards check your badges on the way to Victory Road.",
             "- HM obstacles: small trees need CUT (HM01, S.S. Anne captain); water needs SURF (HM03, Safari Zone);",
             "  boulders need STRENGTH (HM04, Safari Zone warden after returning his Gold Teeth); dark caves are",
             "  easier with FLASH (HM05, Oak's aide on Route 2 once you've caught 10 Pokémon).",
             ]
    docs[f"{PREFIX}story gates and blockers"] = "\n".join(lines)

    # --- curated story order (NOT derived from data)
    docs[f"{PREFIX}story order and main path (curated overview)"] = CURATED_STORY
    return docs


CURATED_STORY = """Curated overview (hand-written, not derived from game data) of the main path through Pokémon Red.
1. Pallet Town -> Route 1 -> Viridian City: get Oak's Parcel at the Viridian Mart, deliver it to Oak (Pokédex).
2. Route 2 -> Viridian Forest -> Pewter City: BROCK (Rock/Ground; Water and Grass are super-effective). Boulder Badge.
3. Route 3 (east of Pewter; trainers) -> Route 4 west (Mt. Moon Pokémon Center) -> MT. MOON (1F, B1F, B2F; Team Rocket,
   choose the Dome or Helix fossil) -> exit to Route 4 east -> Cerulean City.
4. Cerulean: MISTY (Water; Grass and Electric are super-effective). Cascade Badge. North: Nugget Bridge (Route 24, rival)
   and Route 25 to Bill's house (S.S. Ticket). Then through the trashed house (Rocket, TM28 Dig) south to Route 5.
5. Route 5 -> Underground Path -> Route 6 -> Vermilion City: board the S.S. Anne with the ticket, get HM01 CUT from the
   captain. LT. SURGE (Electric; Ground is immune/super-effective). Thunder Badge (Cut usable outside battle).
6. Route 11 / Diglett's Cave back to Route 2; Route 9 -> Route 10 -> ROCK TUNNEL (Flash helps) -> Lavender Town.
7. Route 8 -> Underground Path -> Route 7 -> Celadon City: ERIKA (Grass; Fire/Ice/Flying/Poison/Bug super-effective).
   Rainbow Badge. Game Corner -> Rocket Hideout -> Silph Scope. Buy drinks on the Department Store roof.
8. Lavender: Pokémon Tower with the Silph Scope, rescue Mr. Fuji -> Poké Flute (wakes Snorlax).
9. To Fuchsia City (Cycling Road with a Bicycle, or Routes 12-15): KOGA (Poison; Ground/Psychic super-effective).
   Soul Badge (Surf usable). Safari Zone: HM03 SURF, and the Gold Teeth -> warden gives HM04 STRENGTH.
10. Give a drink to a Saffron gate guard -> Saffron City: Silph Co. (Giovanni; Master Ball), then SABRINA (Psychic; Bug
    is super-effective, Ghost in Gen 1 is bugged). Marsh Badge.
11. Surf south (Routes 19/20, Seafoam Islands) -> Cinnabar Island: Pokémon Mansion (Secret Key) -> BLAINE (Fire; Water/
    Ground/Rock super-effective). Volcano Badge.
12. Viridian City gym: GIOVANNI (Ground; Water/Grass/Ice super-effective). Earth Badge.
13. Route 22 -> Route 23 (badge checks) -> VICTORY ROAD (Strength boulders) -> Indigo Plateau: Elite Four (Lorelei Ice,
    Bruno Fighting, Agatha Ghost, Lance Dragon) and the Champion (rival)."""


# ---------------------------------------------------------------- Orrery
def _req(url, ws, method="GET", body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"X-Workspace-Id": ws, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def ingest(docs: dict[str, str], url: str, ws: str, *, only_missing: bool = False, prune: bool = True) -> None:
    existing, offset = [], 0
    while True:
        page = _req(f"{url}/documents?limit=200&offset={offset}", ws)
        items = page if isinstance(page, list) else (page.get("documents") or page.get("items") or [])
        existing += items
        if len(items) < 200:
            break
        offset += 200
    have = {d.get("title") for d in existing}
    # prune generated docs that no longer exist; replace (or, with only_missing, keep) the rest
    # replace docs being (re)ingested; prune stale generated docs only on a FULL run
    drop = [d for d in existing if (d.get("title") or "").startswith(PREFIX)
            and ((d["title"] in docs and not only_missing) or (prune and d["title"] not in docs))]
    for d in drop:
        _req(f"{url}/documents/{urllib.parse.quote(str(d['id']))}", ws, method="DELETE")
    todo = {t: c for t, c in docs.items() if not (only_missing and t in have)}
    failed = []
    for i, (title, content) in enumerate(todo.items(), 1):
        for attempt in range(3):
            try:
                _req(f"{url}/ingest/text", ws, method="POST", body={"title": title, "content": content}, timeout=300)
                break
            except Exception as e:  # noqa: BLE001 — keep going; report at the end
                if attempt == 2:
                    failed.append((title, repr(e)[:80]))
        print(f"  [{i}/{len(todo)}] {title[:70]}", flush=True)
    print(f"ingested {len(todo) - len(failed)}/{len(todo)} docs into workspace {ws} (removed {len(drop)}); "
          f"failed: {failed}")
    # the search index doesn't pick up newly ingested chunks by itself: refresh it so L1 can find them
    res = _req(f"{url}/search/rebuild", ws, method="POST", timeout=600)
    print(f"search index rebuilt: {res}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--only-missing", action="store_true", help="ingest only docs whose title isn't there yet")
    ap.add_argument("--only", default=None, help="replace just the docs whose title contains this text (no pruning)")
    ap.add_argument("--workspace", default=os.environ.get("ORRERY_WORKSPACE_ID", "6d677a16"))
    ap.add_argument("--url", default=os.environ.get("ORRERY_BASE_URL", "http://localhost:8100"))
    a = ap.parse_args()
    docs = build_docs()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.md"):
        old.unlink()
    for title, content in docs.items():
        fn = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + ".md"
        (OUT / fn).write_text(f"# {title}\n\n{content}\n")
    print(f"wrote {len(docs)} docs to {OUT} ({sum(len(c) for c in docs.values()) // 1024} KB)")
    if a.ingest:
        sel = {t: c for t, c in docs.items() if a.only.lower() in t.lower()} if a.only else docs
        ingest(sel, a.url, a.workspace, only_missing=a.only_missing, prune=a.only is None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
