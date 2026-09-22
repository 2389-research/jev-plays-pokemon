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

## 4. Offline build pipeline (what's required)

Produce a static data module `map_portals.py` (mirrors how `map_graph_data.CONNECTIONS` was ripped), consumed by the portal graph builder. Inputs, all ground truth:

1. **`warp_events`** per map → warp portals `(map, x, y, dest_map, dest_warp_id)`; pair each warp to its destination portal via `dest_warp_id`. Source: pokered `data/maps/objects/*.asm`.
2. **`connections`** per map → edge portals (which border → which adjacent map). Already ripped as `CONNECTIONS`; convert border directions to representative edge portals.
3. **Blockdata + tileset collision** per map → the walkable grid → connected components → the intra-map reachability sets. Source: pokered `.blk` map files + tileset collision lists (the same collision semantics `map_reader.py` already decodes from RAM).
4. **Points of interest** (optional, phase 2): `object_events` (NPCs, item balls, signs) → named POIs `(map, x, y, label, kind)`.

Build dependency: obtain pokered (clone `pret/pokered`, or parse the ROM's map headers directly — the ROM is self-contained). The parser is a one-time offline script under `scripts/`; its output is committed data (`do not edit by hand`, like `map_graph_data.py`). Validate the parser's collision output against `read_collision_map` on several already-reachable maps (Pallet is 100%-validated today) before trusting it.

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

## 8. Open questions

1. **pokered source:** clone `pret/pokered` for the rip, or parse the ROM headers directly (self-contained)? Recommend clone for readability; ROM-parse as fallback.
2. **Orrery mirror:** also publish the portal graph into the KG as queryable facts for L1's brainstorm, or keep it a static module the router reads? Recommend static-first (deterministic routing), KG mirror as a later, additive step.
3. **POIs in phase 1 or 2?** The forest crossing needs only portals; POIs (NPCs/items with coords) can follow once portals are proven.
