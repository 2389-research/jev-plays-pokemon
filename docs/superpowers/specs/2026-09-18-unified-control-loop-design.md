# Unified control loop — reflection as the mid-level target proposer

## Problem

The reflection loop (`_maybe_reflect` → `self._plan`) and the navigation/directive loop are
**separate and don't communicate**. The reflection produces good short-term objectives ("step west
to (4,6), then south to the exit at (4,11)"), but in the main overworld path that output is only
consumed by Jev in the menu/legacy branches — it never drives movement. Movement is driven by the
directive→router chain, which **returns None (does nothing)** when it can't compute a route (e.g. a
fresh load where a `0xFF` "return" door hasn't resolved, so the graph doesn't know the building's
exit leads anywhere). Result: the agent freezes with a planner literally telling it where to go.

## Design: one connected loop

```
L1  arbiter/strategist  → goal map + active quest step + acceptance criterion
MID proposer            → ONE typed short-term target toward the goal (+ short rolling note),
   (merged reflection      and the mechanism that fires to get UNSTUCK
    + L2 waypoint)
JEV                     → routing policy (shortest / dodge-grass / farm-exp)
L3  weighted router     → resolve the typed target to a tile, step toward it under the policy
```

**Cadence (option B):** the proposer chooses the target per *leg*; Jev+policy+router navigate the
tiles *within* the leg. The target is held across frames until **reached / stuck / map changed**,
then the proposer fires again. The LLM decides *where*; the router handles the tiles between — never
a per-frame LLM call, but the proposer is always the one choosing the destination.

## Proposer interface

Replaces the orphaned `_maybe_reflect` target role AND `_pick_waypoint`/`next_waypoint`.

- **In:** goal (target map + quest reason), current-map semantic ASCII + player pos, exits, npcs,
  recent trail, its own recent targets, and `stuck` (bool + reason).
- **Out:** a typed target + one-line note:
  - `tile(x, y)` — head to a coordinate on this map
  - `exit` — leave the current building/area toward the goal (nearest exit door / boundary edge)
  - `approach_npc(sprite | "nearest")` — reach and talk to a person
  - `enter(map)` — step through the door leading to that map
- **Resolution (deterministic, in the loop → router):**
  - `tile` → policy-route to (x,y)
  - `exit` → nearest exit door / boundary edge toward goal → route + step through
  - `approach_npc` → find the sprite → route adjacent → `InteractAction`
  - `enter` → that map's door → route + step through
- **Never bails to None:** if it can't decide, default to `exit` (leave) so the loop always moves.

## Stuck → unstick wire

If the router can't get closer to the current target for `K` steps → set `stuck=True` and re-invoke
the proposer *with the reason* so it proposes something different (its whole purpose). If it still
can't after a couple of tries → escalate to L1 (re-strategize the quest).

## Integration / what changes

- `_navigate_leg` collapses to: hold-or-propose a typed target → Jev policy → router step toward it;
  re-propose on reached/stuck/map-change. The `hop is None → return None` bail is removed.
- The directive/quest layer (L1) still sets the map-level goal + acceptance criteria; the proposer
  turns that into concrete on-map targets each leg.
- `TALK_TO`/`GRAB_ITEM` become `approach_npc` targets (with the sprite named by the strategist where
  possible), fixing the "talk to a specific NPC in a crowd" gap.

## Testing

- Unit: each target kind resolves to the right route/interact; typed-output parsing; the
  stuck→re-propose→escalate ladder; proposer never yields None.
- Fixtures driving the unified loop: leave Oak's Lab (route around the rival to the exit), Viridian
  south exit, Poké Center exit.

## Findings that motivated this (2026-09-18)

Traced live, from a run started at the natural save `roms/pokemon_red.gb.state`:

1. **Orphaned reflection.** `_maybe_reflect` → `self._plan.next_objective` produces correct spatial
   guidance ("step west to (4,6), south to the exit at (4,11)") but nothing in the overworld path
   reads it; movement is the directive→router chain, which `return None`s when it can't route. The
   planner literally knew the answer and the agent stood still. → this whole spec.
2. **The natural save is INSIDE the scripted rival battle** (`wIsInBattle` @0xD057 = 2,
   `battle_kind: trainer`). `ObservationBuilder` sets `exits = read_exits(emu) if mode == OVERWORLD
   else []`, so `obs.exits` is `[]` in battle — correct, but it misled debugging. In a real run the
   loop's `_battle_turn` must WIN the rival battle first; then it's overworld-lab with exits present.
   `read_exits(emu)` itself is fine (returns the 2 lab warps at (4,11)/(5,11), destMap 0xFF).
3. **`0xFF` "return" doors don't resolve on a fresh load** (`_prev_map is None`), so the graph never
   learns `lab→Pallet` and `next_hop` is None → the bail. The unified loop's "never bail; default to
   `exit`" plus `_leave_via_nearest_exit` (route to the nearest exit door, step through) fixes this:
   we don't need to know a door's destination to USE it to leave a building.
4. **`talk_to` needs a SPECIFIC npc.** Generic `{kind: npc}` + "interact with nearest person" mashes
   A on the wrong NPC (the rival, not Oak) → the `approach_npc(sprite)` target kind; the strategist
   should name the target sprite ("Oak") in talk steps.

## Already done this session (uncommitted, 188 tests green)

- Viridian **south-edge exit**: map-edge crossings BFS to the boundary (funnels through the x=19-20
  gap) instead of following L2 band-waypoints. Validated on `states/viridian_stuck.state` (23 steps).
- **Building-door step-through** no longer gates on `blocked_dirs` (the tile past a doormat reads as
  WALL but the game warps you). Validated on `states/pc_stuck.state`.
- `hop is None → _leave_via_nearest_exit` fallback added in `_navigate_leg`.
- **Reverted** the `talk_to`-interacts-nearest-person regression (it A-spammed the rival).

## Concrete implementation plan

1. `Planner.propose_target(emu, context) -> dict` — one LunaRoute call returning a typed target
   `{kind: tile|exit|approach_npc|enter, x?, y?, sprite?, map?, note}` (extends `next_waypoint`).
2. `ReasoningLoop`: hold `self._target`; `_navigate_leg` = propose-or-hold → resolve → Jev policy →
   router step; re-propose on reached/stuck/map-change; delete the `return None` bail (default `exit`).
   Resolvers reuse existing pieces: tile→`_policy_route`/`_bfs_move`; exit→`_leave_via_nearest_exit`
   + `_edge_step`; approach_npc→find sprite + route adjacent + `InteractAction`; enter→door route +
   `_warp_exit_dir`.
3. Fold the reflection's rolling note into the proposer output (one mid-level call).
4. Tests + fixtures (lab exit post-battle, Viridian south, PC exit).

## Status / next step

Design approved by the user. Implementation NOT started (only the pre-req fixes above are in).
Next: build `propose_target` + the `_navigate_leg` collapse, then run from
`roms/pokemon_red.gb.state` (it will fight the rival, then must leave the lab and navigate onward).
