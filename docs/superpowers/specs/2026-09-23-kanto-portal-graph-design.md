# Kanto Portal Graph — Design

**Goal:** extend the ground-truth PortalGraph (ripped from the pokered disassembly) from the 8-map Pallet→Pewter corridor to **all of Kanto**, so L2 routing works for every chapter (Mt. Moon, Cerulean, …) without per-corridor work. "Pull Kanto, enrich as needed" (user, 2026-09-23).

## 1. Current state (verified)
- `scripts/rip_portals.py` is generic over a map list; only `CORRIDOR_IDS/CORRIDOR_NAMES` (8 maps) limit it. Output is written to `scripts/out/portals_corridor.json`; the shipped `src/pokemon_agent/games/pokemon_red/portal_graph.json` (8 maps, 26 KB) was copied from it once (afc1d3d).
- Runtime (`agent/portal_graph.py`, `reason_loop._portal_next`) is map-agnostic: routes over `(map, component)` nodes, locates the player by flood-filling live collision to the nearest portal tile. It only engages when **both** maps are in the graph — which is why the Route 3 → Mt. Moon legs wedged (`verify-team`, 4× "go to Mt Moon 1F" wedges).
- The WorldGraph fallback knows all outdoor map-edge connections (direction only) and building warps only once seen.

**Experiment — rip all 223 headers with the current code:**
- 221 load; 2 fail: `UndergroundPathNorthSouth` (.blk 92 B, expected 96) and `UndergroundPathRoute7Copy` (no data file).
- 946 portals; 4 warps with no destination.
- **Wrong:** Pewter → Cerulean = "Route 3 → Route 4 east edge → Cerulean" (skips Mt. Moon; impossible in-game).
- **Unreachable:** Pewter → Mt. Moon 1F; Cerulean → Vermilion.

## 2. Fixes

**K1 — rip all maps.** `rip_portals.py --maps all` (new default) enumerates every `data/maps/headers/*.asm`; per-map load failures are reported and skipped, never fatal. The existing corridor checks stay. Write the shipped JSON directly (`--out src/.../portal_graph.json`), plus a `version` field (pokered commit + ripper hash).

**K2 — one-way ledges.** Ledge tiles (from `data/tilesets/ledge_tiles.asm`: player direction, standing tile, ledge tile) must not join components in both directions. Remove ledge cells from the undirected walkable set; for each ledge (standing cell A, ledge cell L, landing cell B = L + dir) add a **directed** `kind: "ledge"` portal on the same map: `coord = A`, `hop = dir`, `dest_map = same map`, `dest_component = comp(B)`. `PortalGraph.route` follows it one way. Executor: a portal target of kind `ledge` is reached at A, then **steps in `hop`** (bypassing the ledge veto in `_blocked_dirs` for that one move).

**K3 — doors on non-walkable tiles.** A warp whose tile isn't walkable (building doors, cave mouths) gets the component of an adjacent walkable approach tile (the side you enter from), not None. Fixes Pewter → Mt. Moon 1F.

**K4 — load robustness.** Tolerate a short `.blk` (pad with the border block and report it); skip "Copy" maps with no data. Unresolved `LAST_MAP` warps are left `dest_map: null` with `note`.

**K5 — story/HM gates are out of scope (enrich as needed).** Static walkability ignores story gates (Saffron guards, Route 22/23 badge checks, Snorlax), HM obstacles (Cut trees and boulders are non-walkable → unreachable; Surf water → unreachable), elevators (dynamic destinations) and Silph teleporters. Routes through a story gate will wedge and L1 learns from `why_wedged`. A small `enrichment` table in the ripper (map/portal → note, e.g. "requires a drink for the Saffron guard") can be added per incident.

**K6 — L1 visibility (optional, small).** `reachable_maps` could be offered to L1 later; not in this change.

## 3. Acceptance (golden routes + validation)
- **Collision validation:** static decode == RAM `read_collision_map` for **every** map with a save state in `states/` or `runs/*/states` (≥15 maps today, incl. Route 3, Pewter buildings); report which maps are validated vs. static-only.
- **Golden routes** (asserted in `tests/unit/test_kanto_portal_graph.py`, on the shipped JSON):
  - Viridian Forest → Pewter via the north gate (the existing corridor proof);
  - **Pewter → Cerulean passes Mt. Moon 1F, B1F and B2F** (never the Route 4 east edge from the west half);
  - Pewter → Mt. Moon 1F reachable;
  - Cerulean → Pewter is **not** reachable via Route 4 east→west (the ledge is one-way): it must go through Mt. Moon or report unreachable;
  - Route 1 south → Pallet OK, and Pallet → Viridian OK (Route 1 ledges are one-way south).
- **No regressions:** the 8 corridor maps' portals/components unchanged (same ids/components) except where K2/K3 intentionally change them; full suite green.
- **Live:** resume `verify-team` (Route 3): `_portal_next` returns a portal toward Mt. Moon 1F; the agent enters Mt. Moon within ~150 steps (no "go to Mt Moon 1F" wedge loop).

## 4. Out of scope
Story/HM gates (K5), elevators/teleporters, Safari Zone, L1 route visibility (K6).
