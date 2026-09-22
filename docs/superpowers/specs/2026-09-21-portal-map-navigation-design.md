# Portal-Map Navigation Representation — Design

**Goal:** Give the agent a complete, ground-truth, *readable* map of how areas connect — a "portal graph" — so it can interpret cross-map connectivity and **decide** where to go, while a deterministic pathfinder executes the walk. Fix the bug where the agent reaches Viridian Forest and bounces in/out unable to cross to Pewter.

**Architecture:** Author a portal graph offline from ground truth (pokered warp tables + map connections + per-map collision). The agent never builds or infers the map; it reads a scoped, natural-language connectivity view (validated encoding), decides which portal/point-of-interest to head for **by name**, and the existing servo/pathfinder walks to that tile. The agent is never told which portal it *must* take.

**Tech stack:** Python; existing `WorldGraph`, `map_graph_data.CONNECTIONS`, `state.read_exits`, `map_reader.read_collision_map`, and the L1/L2/servo split in `reason_loop.py`.

---

## 1. Background & root cause (why this is needed)

In run `runs/full-run-20260921-195847` the agent delivered Oak's Parcel and navigated all the way to the Viridian Forest gate, then bounced between the South Gate (map 50), Route 2 (13), and the forest (51) for ~110 steps, never crossing to Pewter. Diagnosis (evidence in that run's log):

- **L1's strategy was correct and consistent** — it repeatedly and correctly concluded "cross Viridian Forest to Pewter." Not a knowledge/KB problem.
- The **routing graph is incomplete**. `AgentMemory` defaults to `full_kanto_graph()`, built only from `map_graph_data.CONNECTIONS` (map-**edge** connections). Warp/gate/building connections are learned lazily and only for visited maps. So `route(Forest 51 → Pewter 2)` returns `None` → the quest step wedges → L1 re-plans every few steps (270 reviews / 91 changes in that run = the visible thrash).
- Even the (unused) `seeded_kanto_graph()`, which *does* list the forest corridor, routes **wrong**: `route(51 → 2) = [51, 50, 13, 2]` — exit the forest back out the **South** gate. Cause: **map-node granularity is too coarse.** Route 2 (13) is a single node spanning two physically-disconnected regions (south and north of the forest); BFS treats them as the same place, so Pewter looks "directly north of Route 2" from anywhere and the forest looks like a pointless detour.

**Conclusion:** the fix is a *complete, correctly-grained, readable* world model — not a plan patch and not a scripted itinerary. The requirement "you must cross the forest" must **emerge** from the map's geometry, never be authored as a route.

## 2. Locked design decisions (from brainstorming)

1. **Complete map from step 0** — rip the full ground-truth world offline; the agent is never blind. Exploration is not the variable; *decision-making* is.
2. **Portal-graph structure** — nodes = portals (doors/edges/warps); links = warp-links (portal→dest portal) + within-map walk-reachability (from collision). The coarse-node bug is fixed because portals on the same map are only linked if collision says they're mutually walkable.
3. **Agent decides, pathfinder executes** — the validated architecture from Gemini/Claude Plays Pokémon. The map is read-only; the agent picks a portal *by name*; the deterministic servo walks to its tile.
4. **Representation = scoped natural-language adjacency, named not coordinate-addressed** — see §5. Grounded in "Talk like a Graph" (ICLR 2024: adjacency-list/"incident" NL phrasing with meaningful names is the robust best encoder) and the two Pokémon runs (externalize connectivity as text; never make the LLM infer reachability; self-drawn maps and coordinate *reasoning* were the documented failure modes).
5. **Coordinates are ground-truth handles, not identity** — points of interest are shown *with* coordinates as a neutral factual menu the harness re-provides every turn; the agent selects by name and the pathfinder consumes the coord. The agent never remembers or does geometry on coordinates. Nothing is ranked or forced.
6. **Orrery is the knowledge backend** — the portal graph + coordinates + semantic metadata are stored in Orrery (entities + relations); this is the system of record, an intentional dogfood of Orrery, and the KG L1 queries. Routing loads a snapshot into an in-memory portal graph at run start (cache) so the per-step hot path is fast and deterministic — routing never depends on a live HTTP query.
7. **The map is extracted deterministically, never by LLM** — warps, coordinates, connections and walk-reachability are ripped from ground truth (RAM readers / pokered). GLM is used only for the fuzzy semantic layer (labels, POI descriptions, region tags) and never authors a coordinate or an edge. (LLM coordinate reasoning is the documented CPP hallucination failure.)
8. **Build region-by-region** — do the Pallet→Pewter (Brock) corridor first, validate each harvested map against live RAM, then expand map-by-map.

Out of scope for this spec (separate follow-on): **Pokémon capture** (no `CATCH` intent / ball-throw battle policy today). Also out of scope: re-planning cadence/hysteresis tuning (tracked separately); this spec removes the *cause* of most re-plans (unroutable steps) but does not retune the detector.

## 3. Data model — the portal graph

A **Portal** is a named connection point on a map:

```
Portal:
  id:        stable string, e.g. "route2:southgate_door" (NAME-based identity)
  map:       int map id
  coord:     (x, y)          # the tile; a HANDLE for the pathfinder, not identity
  kind:      "warp" | "edge"  # door/stairs/gate warp, or a map-border edge
  dest_map:  int             # map you arrive on
  dest_portal_id: str | None # the paired portal on dest_map (warps pair up)
  label:     human name, e.g. "South Gate", "north edge"
```

Two link types:

- **Inter-map link** (`Portal.dest_map`/`dest_portal_id`): stepping on this portal takes you to `dest_map`. From pokered `warp_events` (warps) and `CONNECTIONS` (edges). Ground truth, static.
- **Intra-map reachability**: for each map, the sets of portals that are **mutually walkable on foot**, computed as connected components of the walkable collision grid. This is the piece that fixes the coarse-node bug: on Route 2, `{southgate_door, south_edge}` is one component and `{north_edge_to_pewter, northgate_door}` is another — they are NOT linked, so BFS can't shortcut south-Route-2 → Pewter.

**Routing:** BFS over portals — from the player's current position, the reachable portals (same component) are the frontier; each has a `dest_map`; recurse. `route(from_pos → target_map)` yields an ordered list of portals to traverse. The forest crossing emerges: from south Route 2 the only forward portal is the South Gate; inside the forest the forward portal is the North Gate; etc. No itinerary is authored.

This supersedes the map-node `WorldGraph` for cross-map routing. `WorldGraph`'s existing `add_warp`/`observe_exits` (live RAM warp ingestion) remain as the runtime refinement/validation layer; the portal graph is the authored ground-truth spine.

## 4. Build pipeline (static pull from pokered/ROM → validate with RAM → Orrery)

**Primary source = the pokered disassembly (static, complete, no traversal required).** This is the only source consistent with "complete map from step 0": it has every map's data whether or not the agent has ever been there. A parser (`scripts/rip_portals.py`) reads, per map:

1. `data/maps/objects/*.asm` → **warp portals** `(map, x, y, dest_map, dest_warp_id)`; pair each to its destination portal via `dest_warp_id`. (`dest_map` of `0xFF` = dynamic "return to last map" — resolved contextually, excluded from static edges.)
2. map headers / `constants/map_constants.asm` → **connections** (N/S/E/W → adjacent map). Same source as the existing `CONNECTIONS`.
3. `maps/*.blk` + the tileset's blockset + `data/tilesets/*_collision.asm` → the **walkable grid** → connected components → the "mutually walkable on foot" sets. This is the same decode `map_reader.py` already performs from RAM, sourced from static files instead.
4. **Edge portals + representative-tile rule (crux):** for a border connection, the edge portal's `coord` is a *walkable* tile on that border; if the border's walkable tiles span more than one component, emit **one edge portal per component**. This is what makes Route 2 correct — the north-edge→Pewter portal lands only in the north component, unreachable from the south component (empirically confirmed on the harvested map 13: south = comp 5, north edge/gate = comp 4).
5. Emit portal records `{id, map, coord, kind, dest_map, dest_portal, label, component}`.

**RAM is the validation oracle, not the source.** `map_reader.read_collision_map` / `state.read_exits` (100%-validated on Pallet) check the parser's output for maps we happen to have save states for (`0,1,12,13,37..44,50,51`). This catches parser bugs; it does **not** gate coverage — Pewter (2), the forest North Gate (47), and every other map come straight from the static rip without visiting.

**No coordinate or edge is produced by an LLM.** GLM is optional and only enriches the semantic layer (POI descriptions, region tags, friendly labels), reviewed before load.

**Alternative source (self-contained fallback):** parse the ROM's own map-header/blockdata/tileset tables directly (same decode as the RAM reader, ROM-addressed) if we prefer not to depend on the pokered checkout.

**Load into Orrery.** The ripped records are written to Orrery as entities (`Map`, `Portal`, `POI`) and relations (`warps_to`, `edge_connects`, `walk_reachable`, `has_coord`), workspace `6d677a16`. At run start the router loads a snapshot into the in-memory portal graph (cache); Orrery is the system of record and the KG L1 queries.

## 5. Representation surfaced to the agent

Read-only, **scoped to the current map + its 1-hop neighbors** (never dump the whole world — context bloat and edge-ordering degrade graph reasoning). Named identity; coords only in the POI block. Rendered as structured natural language (the ICLR-best "incident/adjacency" encoding). Example, standing in Viridian Forest:

```
YOU ARE: Viridian Forest  (large map; two gatehouse exits)

ON FOOT FROM HERE you can reach these exits:
  • North Gate  → Viridian Forest North Gate
  • South Gate  → Viridian Forest South Gate

NEARBY MAP CONNECTIONS:
  Viridian City —(north edge)→ Route 2
  Route 2 —(South Gate)→ Viridian Forest —(North Gate)→ Route 2 —(north edge)→ Pewter City

POINTS OF INTEREST HERE:
  • North Gate (door)      at (x, y)
  • item ball              at (x, y)
```

Plus a **fact-only lookup tool** for distant goals (never returns a route the agent must take — returns connectivity facts):

```
reaches("Pewter City")
 → from Viridian Forest, via North Gate → Route 2 (north) → Pewter City
```

The agent reads this, decides ("I want Pewter → head for the North Gate"), names the portal, and the pathfinder walks to its tile. This replaces/enriches the `candidate_exits` and `map_view` context currently passed into `_propose_target` / `_pick_waypoint`.

## 6. Decide → pathfinder handoff (framework integration)

Minimal change to the existing L1/L2/servo split:

- **L2 proposer** (`planner_llm._propose_target`) is given the §5 view and returns a chosen portal *by name* (or a POI by name). Fixes the current `unknown kind 'door'/'edge'` failures — the proposer's accepted vocabulary is unified with what the map offers (named portals), instead of only `kind:"tile"`.
- **Resolver/servo** (`reason_loop._resolve_target`/`_servo_step`) maps the chosen portal name → its `coord` → routes to that tile (existing BFS/servo, warp-aware) → steps onto it to cross.
- **Cross-map routing** uses the portal graph's `route(current_pos → target_map)` to know the *next portal* to head for, replacing the coarse `WorldGraph.next_hop` for building/gate routes.
- **L1** may call `reaches(target)` when composing quests, but still emits objectives, not routes.

## 7. Testing / validation

- **Unit — portal graph correctness:** `route(Forest 51 → Pewter 2)` yields South/North-gate-correct portals (via **North** Gate, not South); `route(Route2-south → Pewter)` goes through the forest, not the phantom direct edge; intra-map components split Route 2 into two.
- **Unit — representation:** snapshot tests of the §5 render for a few maps (named, scoped, coords only in POI block, no whole-world dump).
- **Unit — handoff:** L2 returns a named portal; resolver maps name→coord→move; `unknown kind` path is gone.
- **Parser validation:** ripped collision matches `read_collision_map` on known maps.
- **Live:** a headless run from `lab_deliver` crosses the forest to Pewter (the current failure) and the L1 re-plan count on that leg drops sharply (no unroutable-step thrash).

## 8. Decisions & remaining questions

Resolved in brainstorming:
- **Backend = Orrery** (system of record + L1 KG), with an in-memory routing cache loaded at run start.
- **Extraction = deterministic RAM harvest** (not LLM); GLM only for semantic metadata.
- **Rollout = region-by-region**, Brock corridor first, validated against live RAM.

Remaining:
1. **Orrery schema shape:** exact entity/relation types and how a portal's `walk_reachable` component is represented (per-map component id vs pairwise). Settle when wiring the loader.
2. **POI phase:** portals-only first (all the forest needs); POIs (NPCs/items with coords, incl. any GLM-authored labels) as phase 2 once portals + routing are proven.
3. **Missing-map capture:** one targeted run/save-state pass to grab Pewter (2), forest North Gate, Route 22 before the corridor is complete.
