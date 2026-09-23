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

## 1b. Review round 1 — corrected diagnosis (verified)
The ledge/door diagnoses above were WRONG. Ledge tiles ($36/$37/$27/$0D/$1D) are already non-walkable
(not in `Overworld_Coll`), and door warps already get the smallest adjacent component. The real cause of all
three experiment failures is **edge pairing that ignores the connection offset**: pokered's `connection`
macro shifts the crossing coordinate by `_x = offset * -2` (verified `macros/scripts/maps.asm:189`), but
`build_portal_graph` pairs edges by nearest raw coordinate — `route3:edge_north_c0` (x=60) should land at
Route 4 x=10 (Mt. Moon side, comp 2) but is paired to `route4:edge_south_c4` (x=87, beyond Route 3's width).
The **shipped 8-map JSON has the same bug** (9 edges land in the wrong component, e.g. `route1:edge_north_c2`
→ Viridian comp 2 instead of 0). With offset-aware pairing + directed ledges, Pewter → Cerulean correctly
goes Mt. Moon 1F → B1F → B2F → B1F → Route 4 ledge → Cerulean. Revised fixes below (K0–K9 supersede K2/K3).

## 2. Fixes (revised after review round 1)

**K0 — offset-aware edge pairing (the root fix).** Pair at cell level: for each walkable source border cell,
the target cell = the cell shifted by `-2*offset` on the seam axis on the neighbour's facing border; keep only
crossings whose target cell is walkable. Emit one edge portal per (source component, target component) with
`dest_component` set and a representative chosen only from crossing cells.

**K1 — rip all maps** (as below), compact JSON (~290 KB), `version` field.

**K2' — directed ledge portals (OVERWORLD tileset only).** Standing cell A (tile $2C/$39 per `ledge_tiles.asm`),
ledge cell L = A+dir with the ledge tile, landing B = A+2·dir (may cross a map edge → resolve through the
connection offset; e.g. Route 4 → Route 3, Route 17 → Route 18). One `kind:"ledge"` portal per
(comp A, comp B) with `hop`, `dest_map`, `dest_component`. Runtime: `_dest_nodes` honours `dest_component`
(ledges must never fall back to "any component").

**K3' — door approach cells.** Store each warp's `approach` cell (the neighbour you step from — opposite the
trigger step; inward for border warps) and index exactly that in `component_at` (no `setdefault` collisions,
no approach indexing for ledges). `_portal_next`'s sibling swap only among portals in the same component.

**K4 — load robustness:** pad short `.blk`; drop `*Copy` headers from the map list AND the name index; hard
check: no dangling `dest_portal`.

**K5 — gated portals (default table now, extend as needed):** Saffron's four gates, the Cerulean trashed house
(police early), the Route 22/23 badge gates, Snorlax (Routes 12/16), the Cycling Road bike checks. `route()`
skips `gated` portals by default. HM obstacles stay non-walkable (cut trees / water → unreachable); boulders,
spinners, Mansion switches noted.

**K6 — LAST_MAP resolution like the game:** `wLastMap` is only set when leaving an OUTSIDE map (OVERWORLD/
PLATEAU tileset), so resolve an indoor LAST_MAP warp to the outside map that reaches it through chains of
explicit warps. Elevators get `kind:"elevator"` and are excluded from routing.

**K7 — tile-pair (elevation) collisions.** Cut `pair_collision_tile_ids.asm` pairs (CAVERN/FOREST) when
computing static components (Mt. Moon, Rock Tunnel, Victory Road, Seafoam split heavily). The runtime flood
fill has the same blind spot, so ship each map's static component grid (compact) and have `component_at`
look up the player's cell (live flood fill only as a fallback for cells not in the grid, e.g. a cut tree).

**K8 — executor ledge hop.** A ledge target is not "reached" at A; at A emit `MoveAction(hop)` bypassing only
the ledge veto for that direction; wait until `wMovementFlags` bit 6 (ledge) clears; then drop the held
target (same map, so `_current_target` wouldn't); never record a blocked hop in the dead-end ledger.

**K9 — render_view** skips ledge portals and only lists reachable nodes (not used at runtime today).


### Original fixes (superseded by review round 1 — kept for history)

**K1 — rip all maps.** `rip_portals.py --maps all` (new default) enumerates every `data/maps/headers/*.asm`; per-map load failures are reported and skipped, never fatal. The existing corridor checks stay. Write the shipped JSON directly (`--out src/.../portal_graph.json`), plus a `version` field (pokered commit + ripper hash).

**K2 — one-way ledges.** Ledge tiles (from `data/tilesets/ledge_tiles.asm`: player direction, standing tile, ledge tile) must not join components in both directions. Remove ledge cells from the undirected walkable set; for each ledge (standing cell A, ledge cell L, landing cell B = L + dir) add a **directed** `kind: "ledge"` portal on the same map: `coord = A`, `hop = dir`, `dest_map = same map`, `dest_component = comp(B)`. `PortalGraph.route` follows it one way. Executor: a portal target of kind `ledge` is reached at A, then **steps in `hop`** (bypassing the ledge veto in `_blocked_dirs` for that one move).

**K3 — doors on non-walkable tiles.** A warp whose tile isn't walkable (building doors, cave mouths) gets the component of an adjacent walkable approach tile (the side you enter from), not None. Fixes Pewter → Mt. Moon 1F.

**K4 — load robustness.** Tolerate a short `.blk` (pad with the border block and report it); skip "Copy" maps with no data. Unresolved `LAST_MAP` warps are left `dest_map: null` with `note`.

**K5 — story/HM gates are out of scope (enrich as needed).** Static walkability ignores story gates (Saffron guards, Route 22/23 badge checks, Snorlax), HM obstacles (Cut trees and boulders are non-walkable → unreachable; Surf water → unreachable), elevators (dynamic destinations) and Silph teleporters. Routes through a story gate will wedge and L1 learns from `why_wedged`. A small `enrichment` table in the ripper (map/portal → note, e.g. "requires a drink for the Saffron guard") can be added per incident.

**K6 — L1 visibility (optional, small).** `reachable_maps` could be offered to L1 later; not in this change.

## 3. Acceptance (revised)
- Golden routes (on the shipped JSON): Forest → Pewter via the north gate; Pewter → Mt. Moon 1F; **Pewter →
  Cerulean via Mt. Moon 1F/B1F/B2F then the Route 4 ledge**; **Cerulean → Pewter = None** without Cut (Mt. Moon's
  east exit is a shelf reachable only from inside; Diglett's Cave ends in Route 2's closed pocket);
  **Cerulean → Vermilion via the Underground Path** (gates skipped); a Rock Tunnel traversal; Mt. Moon B2F's
  internal split.
- Ledges: no ledge portal hops north; Route 1's pocket components connect only via ledges; the cross-map
  Route 4 → Route 3 ledge exists. No dangling `dest_portal`.
- Expected corridor diffs (K0): the 9 edge `dest_portal`s change to the offset-correct components; tests that
  hard-code edge ids are updated; `test_loop_portal_next_none_off_graph` uses an unused map id.
- Collision validation (static vs RAM) for every map with a save state incl. Mt. Moon (59/60/61) — noting it
  cannot detect offset pairing, ledges or pair collisions (the goldens cover those).
- Live: from Route 3 the agent enters Mt. Moon; a ledge hop executes in one press (verify on a Route 3/4 state).

## 3-old. Acceptance (superseded)
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
