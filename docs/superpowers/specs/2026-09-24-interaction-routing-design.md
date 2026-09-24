# Interaction routing — design (2026-09-24)

## Problem
Reaching a person or object to interact with is geometry with one right answer. It is not a judgement
call. It should never depend on a model reading coordinates off a grid, and it should never fail because
a pathfinder treated an occupied tile as free.

Evidence:
- **runs/sleeves-cerulean.** Misty (4,2) had a beaten trainer standing on her front tile (4,3). The
  executor chose (4,3) as the side to stand on and walked into the trainer 15+ times, across two wedges
  and an explore. Her open side (5,2) was one step north. Root cause: every pathfinder removes the GOAL
  tile from the obstacle set (`navigator.step_toward`: "never treat the target tile itself as an
  obstacle"; `_bfs_full_collision`: `blocked = occupied - {goal}`; `_leave_via_nearest_exit`: likewise).
  An occupied destination therefore looks one step away. `cd7de43` patched one caller (stand-tile choice).
  The general hole remains.
- **Asking the model for the tile doesn't fix it.** On the real L2 path with the live context
  (scripts/eval_l2_live.py; captured request in the session notes), deepseek-4.1-flash
  (`reasoning_effort="none"`) picked the correct stand tile 2 times out of 18 when forced to answer with a
  tile. Its other picks were the occupied tile, walls, and Misty's own tile. Asked directly, it knows a
  person's tile isn't standable (8/10, and 10/10 once the legend says so). Knowing that doesn't carry over
  to choosing where to stand.

Conclusion: the model's job is WHO/WHAT and WHY. The router's job is WHERE TO STAND and HOW TO GET THERE.
Model-picked coordinates stay only for open-ended movement ("go look over there"). A coordinate-picking
mode for interactions is explicitly out of scope.

## Design

### 1. Interaction targets (what L1/L2 name)
`{kind, target, action}`:
- `approach_npc`: a person, by sprite name (optionally with coordinates). Action: talk.
- `use_object`: a background object from `map_objects.json` (PC, sign, machine). Action: press A.
- `enter`: a door or warp. Action: step onto it or through it.
- `tile`: an area, for exploration only (no interaction).

L2's prompt makes the first three the standard answer for anything involving a person, object or door.

### 2. Interaction tiles (router)
For a target, compute the set of tiles an interaction can be done from:
- **person:** the 4 orthogonal neighbours; across a counter tile, the tile 2 away in a straight line.
- **object:** its required side (`face`) if the table sets one, else the 4 neighbours.
- **door/warp:** the tile itself, plus the step direction; a warp that was arrived through needs a step off
  and back on (already handled: `6adde73`).

A tile qualifies only if it is **standable**:
- walkable terrain (RAM collision);
- not occupied by a sprite;
- not a warp tile, unless the warp is the target;
- not across an elevation cut from the target.

One shared predicate, `standable(map, tile, sprites)`, is used everywhere a destination is chosen.

### 3. Occupancy-aware pathfinding (all pathfinders)
- A destination may be exempt from TERRAIN (door tiles are often off the walkable set). It is never exempt
  from SPRITE OCCUPANCY.
- If the goal is occupied, the pathfinder returns `blocked(by=<sprite>, at=(x,y))`, not a move.
- Applies to the navigator BFS, `_bfs_full_collision`, `_leave_via_nearest_exit` and the grind/explore
  sub-targets.
- Ledges stay one-way edges; elevation cuts stay respected.

### 4. Selection and re-evaluation
- Choose the nearest reachable interaction tile by walking distance (BFS, sprites as obstacles).
- Re-evaluate every step, so a wandering person or one stepping into the path is handled.
- If the only side is blocked by someone who moves (not a stationary sprite such as a beaten trainer),
  wait a few steps before reporting.

### 5. Precise failure reporting
When no interaction tile is reachable, report instead of trying:
- to L2 on the next proposal;
- to L1 as the step's `why_wedged`.

The report carries a structured reason, e.g. "Misty (4,2): front (4,3) occupied by Cooltrainer F; east
(5,2) open but unreachable from (x,y); north/west walls". This reuses the existing blocker mechanism
(`_report_blockers`, `_mark_step(..., "wedged", reason)`).

### 6. Verify the interaction
After pressing A, check who answered: the facing sprite, or the map object at the faced tile, when the
text opens.
- If it was someone else (the agent bumped into another person, or the pick resolved to the wrong
  sprite): don't count it as the target conversation, record it, and retry once. A repeat goes to L1
  with the reason.
- The heard log already records the speaker (`HeardLog`), so the check has a ground-truth source.

## Context clean-up (done with this spec)
- `_candidate_exits`: map-edge exits only in directions the map really connects (from the portal graph).
  The captured Cerulean Gym request listed 32 fake edge exits along its walls; it now lists only the 2
  real doors.
- `PROPOSER_SYSTEM`: no reference to a `REACHABLE` field. `reachable` is only used to validate answers and
  was never sent to the model.
- `propose_target`: a rejected pick is re-asked WITH the reason, e.g. `rejected: "(4,3) is occupied by
  Cooltrainer F — a person's tile can't be stood on"`. Previously the identical prompt was repeated blind.
- Map legend: `N = a person/NPC (NOT walkable; …)`. Labelling only some tiles as walkable implied N was
  walkable (model belief 8/10 → 10/10).

## Testing
- **Unit:**
  - `standable()` covers wall, water, sprite, warp and cut cases;
  - interaction tiles for a plain person, a counter person, an object with and without `face`, a door;
  - every pathfinder returns `blocked` for an occupied goal and still reaches an unoccupied door that is
    off the walkable set;
  - the failure report names the occupant;
  - the who-answered check rejects a different speaker.
- **Live** (real saves, real loop via scripts/eval_l2_live.py; success = a conversation opened by the
  intended target):
  - Misty with a trainer on her front tile (sleeves-cerulean 1610);
  - Viridian Mart clerk and Pewter Pokécenter nurse (across counters);
  - Bill's PC (object with a required side);
  - a wandering NPC.
- **Regression:** the full suite, plus one fresh-from-Squirtle run to confirm nothing breaks in the
  opening.

## Out of scope
- A model coordinate-picking mode for interactions (dropped; see Problem).
- Field moves (Cut, Surf, Strength) and bag items. Separate specs.
