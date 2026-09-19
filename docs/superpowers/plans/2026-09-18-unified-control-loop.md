# Unified Control Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the mid-level layer (reflection, merged with the L2 waypoint call) propose ONE machine-usable typed target every leg that ALWAYS drives the router — so the agent never freezes with a planner that "knows the answer" but stands still, and L2 can force navigation and get unstuck.

**Architecture:** `Planner.propose_target(emu, context)` returns a typed target (`tile` / `exit` / `enter` / `approach_npc`) with a one-line note; it never returns None. `ReasoningLoop` holds one typed target across frames (cadence B — re-propose on reached / stuck / map-change), resolves it deterministically to a move via existing resolvers (BFS / weighted-policy / Jev-path / door-step-through / edge-crossing / approach-NPC), Jev still picks the routing policy inside a leg, and the reflection note is folded into the proposer output. The old `_navigate_leg` `return None` "await-plan" freeze is replaced by: re-propose to unstick, then hand up to L1 (the existing quest/strategize/heal escalation) only after that fails.

**PRIORITY — the model's proposal drives movement, and a model failure BREAKS LOUDLY (does not silently wander).** The bug we are fixing is NOT "the agent didn't know the way out" — in the frozen run the reflection model explicitly said *"step west to (4,6), then south to the exit at (4,11)."* It knew. The failure was that its answer was **orphaned** — nothing turned it into movement. So the ordering is strict, and it treats a genuine model failure as a bug to surface, not to paper over:

1. **The proposer (the reflection model) chooses the target every leg** — a tile, an exit, an enter, or an approach-npc — and that choice is what the router enacts. This is the primary, load-bearing path. When the model gives an answer, we enact it, full stop (never orphan it).
2. **A configured model that FAILS to produce a usable target is an ERROR, and the agent breaks visibly.** If a provider *is* wired (a live run) but the call errors, returns empty/garbage, or names an unreachable tile after the retry, `propose_target` returns an **`unresolved`** marker (never a deterministic guess). The loop then emits a loud `proposer_failed` event carrying the raw response + reason, and the agent **stalls in place (a visible WaitAction) rather than deterministically wandering toward some exit.** That way the break shows up in the logs and the viewer — debuggable — instead of being masked by aimless deterministic movement. (This is the behavior the user explicitly asked for: "flag the issues and allow us to debug in logs, rather than wandering around deterministically with no model guidance… I would rather it break.")
3. **The deterministic `_default_target` is used ONLY in the no-provider mode** (tests / offline — where running with no LLM is intentional, not an error). It is never a live-run fallback for a failed model call.

`_default_target` (no-provider mode only) yields `exit` in one narrow case (cross-map with no *known* route out yet). In a live run that case is exactly where the model's guidance is required, so a model failure there is flagged and broken per rule 2 — we do not silently substitute the nearest-door guess.

**Tech Stack:** Python 3.13, PyBoy, LunaRoute (OpenAI-compatible, `chat_json`), TypeSafe/Jev (`system_one` Choice), pytest, `uv`.

**Working copy:** `~/Documents/GitHub/jev-plays-pokemon` (all commands below run there). Run tests with `uv run python -m pytest -q`.

---

## Reference: current code (already read)

- `src/pokemon_agent/agent/reason_loop.py`
  - `step_once` (planner path lines ~244-272): `_maybe_reflect` (orphaned) → `_manage_directive` → dispatch to `_servo_step` (TALK_TO/GRAB_ITEM) or `_navigate_leg` (everything else) → on `None`, `_servo_fail += 1`, `_replan_next = True`, and a `WaitAction(6)` "await-plan" (**the freeze**).
  - `_navigate_leg` (~521-608): explicit-tile BFS; cross-map hop → door step-through / edge crossing / `_leave_via_nearest_exit` when hop is None; picks `_leg_wp` via `_pick_waypoint`; routes via `_policy_route` / `_jev_path` / `_bfs_move`; returns `None` when it can't get closer.
  - Resolvers to REUSE: `_policy_route`, `_jev_path`, `_bfs_move`, `_leave_via_nearest_exit`, `_edge_step`, `_warp_exit_dir`, `_on_goal_edge`, `_reachable_cells`.
  - `_commit_directive` (~432-443): resets `_leg_wp`, `_policy`, etc. — add target-state resets here.
- `src/pokemon_agent/agent/planner_llm.py`: `next_waypoint` + `WAYPOINT_SYSTEM` (keep — offline fallback + a test uses `_pick_waypoint`); `strategize` + `STRATEGIST_SYSTEM` (talk step → `Directive(TALK_TO, target={"kind":"npc","map":mp})`, **no sprite name**); `_llm_with_search`; `provider` = LunaRoute-fast.
- `src/pokemon_agent/agent/plan.py`: `Directive` with `target_xy`, `target_map`, `target_bearing`; `Intent`.
- `src/pokemon_agent/agent/typesafe_reasoner.py`: `choose_policy`, `path_step`, `judge` (unchanged here).
- `src/pokemon_agent/agent/targets.py`: NPC dict shape `{x, y, sprite, talked_to, interact_did_nothing}`.
- Tests to keep green: `tests/unit/test_executive.py` (esp. `test_navigate_leg_steps_through_door_on_arrival`, `test_navigate_leg_falls_back_to_exit_tile_without_provider`, `test_servo_*`), `tests/unit/test_executor_directive.py`, `tests/unit/test_routing.py`. Baseline: **188 pass**.

## File structure

- **Modify** `src/pokemon_agent/agent/planner_llm.py` — add `PROPOSER_SYSTEM` + `propose_target`.
- **Modify** `src/pokemon_agent/agent/reason_loop.py` — typed-target state, resolvers extraction, `_approach_npc`, `_default_target`, `_resolve_target`, `_current_target`, `_propose_target`, rewritten `_navigate_leg`, `step_once` dispatch + reflection fold.
- **Modify** `src/pokemon_agent/agent/planner_llm.py` — strategist names the talk sprite.
- **Create** `tests/unit/test_unified_loop.py` — all new unit tests.

Deterministic-first principle: every resolver works with no LLM provider (tests + offline). The proposer only *overrides* the deterministic default; when there's no provider it returns the default unchanged.

---

## Task 1: `Planner.propose_target` — the mid-level typed-target proposer

**Files:**
- Modify: `src/pokemon_agent/agent/planner_llm.py`
- Test: `tests/unit/test_unified_loop.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_unified_loop.py
from pokemon_agent.agent.planner_llm import Planner


class FakeProvider:
    def __init__(self, content): self.content = content
    def chat_json(self, system, state, image=None): return self.content, 0, {}


def _ctx(**kw):
    base = {"player": {"x": 3, "y": 3, "map_id": 1}, "reachable": {(3, 3), (3, 4), (4, 3)},
            "exit_tile": (4, 3), "map_view": ["..."], "objective": "go", "destination": "map 0",
            "goal_dir": "south", "npcs": [], "recent_trail": [], "recent_targets": [],
            "default": {"kind": "exit"}, "stuck": False, "why": "pick"}
    base.update(kw); return base


def test_propose_target_no_provider_returns_deterministic_default():
    # OFFLINE / tests: running with no LLM is intentional, not an error -> deterministic default.
    p = Planner(goal_map=0, provider=None)
    assert p.propose_target(emu=None, context=_ctx()) == {"kind": "exit"}


def test_propose_target_parses_tile():
    p = Planner(goal_map=0, provider=FakeProvider('{"kind":"tile","x":3,"y":4,"note":"south"}'))
    t = p.propose_target(emu=None, context=_ctx())
    assert t["kind"] == "tile" and (t["x"], t["y"]) == (3, 4) and t["note"] == "south"


def test_propose_target_configured_but_unreachable_tile_returns_unresolved():
    # a provider IS wired (live run) but it named an unreachable tile -> this is a model failure;
    # BREAK LOUDLY, do NOT silently substitute the deterministic default.
    p = Planner(goal_map=0, provider=FakeProvider('{"kind":"tile","x":9,"y":9,"note":"bad"}'))
    t = p.propose_target(emu=None, context=_ctx())
    assert t["kind"] == "unresolved" and "note" in t


def test_propose_target_passes_through_exit_and_approach_and_enter():
    for content, kind in [('{"kind":"exit","note":"leave"}', "exit"),
                          ('{"kind":"approach_npc","sprite":"Oak","note":"talk"}', "approach_npc"),
                          ('{"kind":"enter","map":1,"note":"door"}', "enter")]:
        p = Planner(goal_map=0, provider=FakeProvider(content))
        assert p.propose_target(emu=None, context=_ctx())["kind"] == kind


def test_propose_target_configured_but_bad_json_returns_unresolved():
    # provider wired but returns garbage -> model failure -> unresolved (flagged), not a guess.
    p = Planner(goal_map=0, provider=FakeProvider("not json"))
    assert p.propose_target(emu=None, context=_ctx())["kind"] == "unresolved"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run python -m pytest tests/unit/test_unified_loop.py -q`
Expected: FAIL (`AttributeError: 'Planner' object has no attribute 'propose_target'`).

- [ ] **Step 3: Implement `PROPOSER_SYSTEM` + `propose_target`**

Add after `WAYPOINT_SYSTEM` in `planner_llm.py`:

```python
PROPOSER_SYSTEM = """You are the MID-LEVEL PROPOSER for a Pokémon Red agent. The strategic layer
picked WHERE to go (a target map / errand). Each leg you look at the grid and propose ONE concrete,
machine-usable SHORT-TERM TARGET toward that goal — a deterministic router then enacts it and asks
you again when you REACH it, get STUCK, or the map changes. You are also the get-unstuck mechanism:
when WHY says the last target was unreachable, propose something DIFFERENT (a new tile, or leave).

Return ONLY ONE JSON object, one of these kinds:
  {"kind":"tile","x":<int>,"y":<int>,"note":"..."}   head to a walkable coordinate on THIS map
  {"kind":"exit","note":"..."}                        leave this building/area toward the goal
  {"kind":"enter","map":<int>,"note":"..."}           step through the door leading to that map
  {"kind":"approach_npc","sprite":"<name>","note":"..."}  reach and talk to that person

COORDINATES: (x,y); x = column (increases EAST), y = row (increases SOUTH, y=0 north). The MAP_VIEW
starts with a LEGEND naming every symbol (path, grass 'G' is walkable, '#'/water NOT walkable, one-way
ledges, doors, counters, NPCs) — READ IT, never guess a tile. Pick a 'tile' that is in REACHABLE and a
real step toward DESTINATION/GOAL_DIR, not one in RECENT_TARGETS you keep revisiting. Prefer EXIT_TILE
(or {"kind":"exit"}) when the way forward is out a door. Use DEFAULT as a strong hint — it is what the
deterministic layer would do; accept it unless you can do better or must get unstuck. NOTE is one short
sentence of your reasoning (this becomes the agent's visible short-term objective)."""


def propose_target(self, emu, context: dict) -> dict:
    """The unified mid-level proposer: ONE typed short-term target toward the goal, and the
    get-unstuck mechanism.

    Failure policy (the important part): a CONFIGURED provider that can't produce a usable target
    is a BUG we surface, not one we hide. So:
      * no provider wired (offline / tests): return context['default'] — the deterministic target,
        because running with no LLM is intentional there, not an error.
      * a provider IS wired but the call errors / returns empty|garbage / names an unreachable tile
        after the retry: return {"kind":"unresolved","reason":...} so the loop can flag it loudly
        (a 'proposer_failed' event) and BREAK VISIBLY, instead of silently wandering deterministically.
    Never returns None."""
    default = context.get("default") or {"kind": "exit"}
    if self.provider is None:
        return default
    reachable = context.get("reachable")
    exit_tile = context.get("exit_tile")
    state = {
        "objective": context.get("objective"),
        "destination": context.get("destination"),
        "goal_dir": context.get("goal_dir"),
        "player": context.get("player"),
        "map_view": context.get("map_view"),
        "exit_tile": list(exit_tile) if exit_tile else None,
        "npcs": context.get("npcs"),
        "recent_trail": context.get("recent_trail"),
        "recent_targets": context.get("recent_targets"),
        "default": default,
        "why": context.get("why", "pick the next target toward the goal"),
    }
    reason = "no response"
    last_raw = None
    for _ in range(2):  # one retry on an invalid pick
        try:
            content, _, _ = self.provider.chat_json(PROPOSER_SYSTEM, state)
            last_raw = content
            data = json.loads(strip_fences(content))
            kind = data.get("kind")
        except Exception as e:
            reason = f"call/parse failed: {e}"
            continue
        note = str(data.get("note") or "").strip()
        if kind == "tile":
            try:
                x, y = int(data["x"]), int(data["y"])
            except (KeyError, TypeError, ValueError):
                reason = "tile missing x/y"
                continue
            px = (context.get("player") or {}).get("x")
            py = (context.get("player") or {}).get("y")
            is_exit = exit_tile is not None and (x, y) == tuple(exit_tile)
            progresses = (px is None) or (x, y) != (px, py)
            if progresses and (is_exit or reachable is None or (x, y) in reachable):
                return {"kind": "tile", "x": x, "y": y, "note": note}
            reason = f"tile ({x},{y}) unreachable / no-op"
            continue
        if kind == "enter":
            try:
                return {"kind": "enter", "map": int(data["map"]), "note": note}
            except (KeyError, TypeError, ValueError):
                reason = "enter missing map"
                continue
        if kind == "approach_npc":
            return {"kind": "approach_npc", "sprite": (data.get("sprite") or None), "note": note}
        if kind == "exit":
            return {"kind": "exit", "note": note}
        reason = f"unknown kind {kind!r}"
    # a wired provider failed to produce a usable target -> surface it; do NOT guess deterministically
    return {"kind": "unresolved", "reason": reason, "raw": (str(last_raw)[:300] if last_raw else None),
            "note": f"proposer failed: {reason}"}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run python -m pytest tests/unit/test_unified_loop.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pokemon_agent/agent/planner_llm.py tests/unit/test_unified_loop.py
git commit -m "planner: propose_target — mid-level typed-target proposer (never None)"
```

---

## Task 2: Resolvers — turn a typed target into a move (deterministic)

**Files:**
- Modify: `src/pokemon_agent/agent/reason_loop.py`
- Test: `tests/unit/test_unified_loop.py`

Extract the routing-to-a-tile selection and add `enter` / `edge` / `approach_npc` resolvers + the `_resolve_target` dispatcher. All deterministic (no provider).

- [ ] **Step 1: Write failing tests**

```python
# add to tests/unit/test_unified_loop.py
from types import SimpleNamespace
from pokemon_agent.core.models import Direction, InteractAction, MoveAction
from pokemon_agent.agent.plan import Directive, Intent


def _nav_loop(map_id=42):
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.agent.reason_loop import ReasoningLoop
    from pokemon_agent.agent.reasoner import ReasonStep
    from pokemon_agent.agent.session import Session
    from pokemon_agent.core.models import GoalState, WaitAction
    from pokemon_agent.emulator.fake_emulator import FakeEmulator
    from pokemon_agent.observations.builder import ObservationBuilder

    class Stub:
        def reflect(self, **k): from pokemon_agent.agent.reasoner import ReflectionPlan; return ReflectionPlan(), 0, {}
        def step(self, **k): return ReasonStep(location="", objective="", reasoning="", action=WaitAction(frames=1)), 0, {}
    emu = FakeEmulator(map_id=map_id)
    loop = ReasoningLoop(builder=ObservationBuilder(emu), controller=ActionController(emu),
                         reasoner=Stub(), session=Session(GoalState(primary="p", current="p")),
                         vision=False, reflect_every=100, goal_map=99)
    return loop, emu


def test_resolve_enter_steps_through_door_when_on_it():
    loop, _ = _nav_loop(42)
    player = SimpleNamespace(x=3, y=7, map_id=42, facing="east")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    move = loop._resolve_target({"kind": "enter", "map": 1}, _d(), obs, set(), set())
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH


def test_resolve_exit_routes_to_nearest_exit_door():
    loop, _ = _nav_loop(42)
    # ingest a tiny walkable map so BFS can route
    cells = {(x, y) for x in range(4) for y in range(8)}
    loop.world.ingest_collision(42, 4, 8, cells, None, None)
    player = SimpleNamespace(x=1, y=1, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    move = loop._resolve_target({"kind": "exit"}, _d(), obs, set(), set())
    assert isinstance(move, MoveAction)  # a step toward the door, not None


def test_resolve_approach_npc_interacts_when_adjacent_and_facing():
    loop, _ = _nav_loop(0)
    player = SimpleNamespace(x=3, y=3, map_id=0, facing="north")
    obs = SimpleNamespace(player=player, map_dims=(6, 6),
                          game_state={"npcs": [{"x": 3, "y": 2, "sprite": "Oak"}]}, exits=[])
    move = loop._resolve_target({"kind": "approach_npc", "sprite": "Oak"}, _d(Intent.TALK_TO),
                                obs, set(), set())
    assert isinstance(move, InteractAction)


def test_resolve_tile_interacts_on_arrival_when_flagged():
    loop, _ = _nav_loop(0)
    player = SimpleNamespace(x=2, y=2, map_id=0, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(6, 6), game_state={}, exits=[])
    move = loop._resolve_target({"kind": "tile", "x": 2, "y": 2, "interact": True}, _d(Intent.TALK_TO),
                                obs, set(), set())
    assert isinstance(move, InteractAction)


def test_resolve_tile_that_is_an_exit_door_steps_through():
    # the model named the exit as a coordinate that IS a warp door -> step THROUGH it, don't stop.
    loop, _ = _nav_loop(42)
    player = SimpleNamespace(x=3, y=7, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    move = loop._resolve_target({"kind": "tile", "x": 3, "y": 7}, _d(), obs, set(), set())
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH


def _d(intent=Intent.TRAVEL):
    return Directive(intent=intent, target={"kind": "map", "map": 40}, success={"on_map": 40})
```

Note: `_resolve_target` needs the graph next-hop for `enter`; the door is found from `obs.exits` by `dest_map == map`, so `enter` doesn't need the graph. `_d()` targets map 40 only to satisfy `_default_target` callers elsewhere.

- [ ] **Step 2: Run to verify fail**

Run: `uv run python -m pytest tests/unit/test_unified_loop.py -q`
Expected: FAIL (`_resolve_target` missing).

- [ ] **Step 3: Implement resolvers**

In `reason_loop.py` add near `_leave_via_nearest_exit`:

```python
def _route_to_tile(self, obs, xy, blocked_dirs, avoid):
    """Route one step toward tile ``xy`` under the active pather (policy / jev / bfs)."""
    if self.pather == "policy" and getattr(self.reasoner, "choose_policy", None) is not None:
        return self._policy_route(obs, xy, avoid)
    if self.pather == "jev" and getattr(self.reasoner, "path_step", None) is not None:
        return self._jev_path(obs, xy, blocked_dirs, avoid)
    return self._bfs_move(obs.player, xy, interact=False, blocked_dirs=blocked_dirs, occupied=avoid)


def _enter_map(self, next_map, obs, blocked_dirs, occupied):
    """Resolve an ``enter(next_map)`` target: route to the door for that map and step through it.
    Falls back to leaving via the nearest exit when the specific door isn't in view."""
    player = obs.player
    door = next((e for e in (obs.exits or []) if e.get("dest_map") == next_map), None)
    if door is None:
        return self._leave_via_nearest_exit(player, obs, blocked_dirs, occupied)
    dxy = (int(door["x"]), int(door["y"]))
    if (player.x, player.y) == dxy:  # step THROUGH (don't gate on blocked: the game warps you)
        d = self._warp_exit_dir(dxy, obs.map_dims)
        return MoveAction(direction=d) if d is not None else None
    return self._bfs_move(player, dxy, interact=False, blocked_dirs=blocked_dirs, occupied=occupied)


def _cross_edge(self, goal_dir, next_map, obs, blocked_dirs, occupied):
    """Resolve a map-EDGE crossing toward ``goal_dir``: BFS to the boundary edge (funnels through
    the gap), then step off it."""
    player = obs.player
    if goal_dir is not None and self._on_goal_edge(player, obs.map_dims, goal_dir):
        if goal_dir not in blocked_dirs:
            return MoveAction(direction=Direction(goal_dir))
    return self._edge_step(player, obs, goal_dir, next_map, blocked_dirs, occupied)


def _approach_npc(self, sprite, obs, blocked_dirs, occupied):
    """Resolve an ``approach_npc`` target: reach the named sprite (or the nearest not-yet-talked
    person) and interact when adjacent + facing it. Routes around other NPCs."""
    player = obs.player
    npcs = [n for n in ((obs.game_state or {}).get("npcs") or []) if "x" in n and "y" in n]
    if not npcs:
        return self._leave_via_nearest_exit(player, obs, blocked_dirs, occupied)
    def pick():
        if sprite:
            named = [n for n in npcs if str(n.get("sprite") or "").lower() == str(sprite).lower()]
            if named:
                return named[0]
        fresh = [n for n in npcs if not n.get("talked_to")] or npcs
        return min(fresh, key=lambda n: abs(int(n["x"]) - player.x) + abs(int(n["y"]) - player.y))
    npc = pick()
    nx, ny = int(npc["x"]), int(npc["y"])
    adj = {Direction.NORTH: (nx, ny + 1), Direction.SOUTH: (nx, ny - 1),
           Direction.EAST: (nx - 1, ny), Direction.WEST: (nx + 1, ny)}  # tile you stand on to face npc
    facing_map = {"north": Direction.NORTH, "south": Direction.SOUTH,
                  "east": Direction.EAST, "west": Direction.WEST}
    for d, stand in adj.items():
        if (player.x, player.y) == stand:
            if facing_map.get(getattr(player, "facing", None)) == d:
                return InteractAction()          # adjacent AND facing -> talk
            return MoveAction(direction=d)        # adjacent, turn to face (a blocked step turns you)
    others = occupied - {(nx, ny)}
    # route to the nearest stand-tile adjacent to the npc
    reachable_stand = min(adj.values(), key=lambda c: abs(c[0] - player.x) + abs(c[1] - player.y))
    return self._bfs_move(player, reachable_stand, interact=False,
                          blocked_dirs=blocked_dirs, occupied=others)


def _resolve_target(self, target, directive, obs, blocked_dirs, occupied):
    """Turn a typed target ({tile|exit|enter|approach_npc}) into ONE move. Returns a MoveAction /
    InteractAction, or None when even this target can't make progress (caller then unsticks)."""
    if not target:
        return None
    kind = target.get("kind")
    player = obs.player
    if kind == "tile":
        xy = (int(target["x"]), int(target["y"]))
        door = next((e for e in (obs.exits or []) if (int(e["x"]), int(e["y"])) == xy), None)
        if (player.x, player.y) == xy:
            # a model-named tile that IS an exit door: step THROUGH the warp (the model said "leave
            # via (4,11)" — honor it), don't just stop on the doormat.
            if door is not None:
                d = self._warp_exit_dir(xy, obs.map_dims)
                return MoveAction(direction=d) if d is not None else None
            return InteractAction() if target.get("interact") else None
        avoid = set(occupied)
        return self._route_to_tile(obs, xy, blocked_dirs, avoid)
    if kind == "enter":
        return self._enter_map(int(target["map"]), obs, blocked_dirs, occupied)
    if kind == "approach_npc":
        return self._approach_npc(target.get("sprite"), obs, blocked_dirs, occupied)
    if kind == "edge":
        return self._cross_edge(target.get("dir"), target.get("next_map"), obs, blocked_dirs, occupied)
    if kind != "exit":
        return None  # unknown / unresolved kind -> no move (never silently wander)
    # exit: leave via nearest door; if none, try a boundary-edge crossing toward the goal
    move = self._leave_via_nearest_exit(player, obs, blocked_dirs, occupied)
    if move is not None:
        return move
    goal_dir = target.get("dir")
    if goal_dir is None and directive is not None and directive.target_map is not None:
        goal_dir = next_direction(self.memory.graph, player.map_id, directive.target_map)
    return self._cross_edge(goal_dir, None, obs, blocked_dirs, occupied)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run python -m pytest tests/unit/test_unified_loop.py -q`
Expected: PASS. Then full suite still green:
Run: `uv run python -m pytest -q` → 188 + new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pokemon_agent/agent/reason_loop.py tests/unit/test_unified_loop.py
git commit -m "reason_loop: typed-target resolvers (tile/exit/enter/approach_npc)"
```

---

## Task 3: `_default_target` + unified `_navigate_leg` (hold / propose / unstick / never-freeze)

**Files:**
- Modify: `src/pokemon_agent/agent/reason_loop.py`
- Test: `tests/unit/test_unified_loop.py`

- [ ] **Step 1: Write failing tests**

```python
# add to tests/unit/test_unified_loop.py

def test_default_target_cross_map_no_hop_is_exit():
    loop, _ = _nav_loop(42)
    loop.memory.graph.next_hop = lambda a, b: None
    player = SimpleNamespace(x=1, y=1, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={}, exits=[])
    assert loop._default_target(_d(), obs)["kind"] == "exit"


def test_default_target_cross_map_with_door_is_enter():
    loop, _ = _nav_loop(42)
    loop.memory.graph.next_hop = lambda a, b: (1, (3, 7))
    player = SimpleNamespace(x=1, y=1, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    t = loop._default_target(_d(), obs)
    assert t["kind"] == "enter" and t["map"] == 1


def test_navigate_leg_never_freezes_without_hop_uses_exit():
    # the lab-freeze case: no known route out (0xFF door unresolved) -> must MOVE, not return None
    loop, _ = _nav_loop(42)
    loop.memory.graph.next_hop = lambda a, b: None
    cells = {(x, y) for x in range(4) for y in range(8)}
    loop.world.ingest_collision(42, 4, 8, cells, None, None)
    player = SimpleNamespace(x=1, y=1, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    move = loop._navigate_leg(_d(), obs, set())
    assert isinstance(move, MoveAction)  # routed toward the exit door, not frozen


def test_navigate_leg_door_step_through_still_works():
    # regression: keep the existing door step-through behavior (was test_executive)
    loop, _ = _nav_loop(42)
    loop.memory.graph.next_hop = lambda a, b: (1, (3, 7))
    player = SimpleNamespace(x=3, y=7, map_id=42, facing="east")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={},
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    move = loop._navigate_leg(_d(), obs, set())
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH


class FailProvider:
    def chat_json(self, system, state, image=None):
        raise RuntimeError("boom")


def test_navigate_leg_configured_failure_ungrounded_stalls_and_flags():
    # a wired model FAILS and there's NO known route out -> deterministic default would be an
    # ungrounded 'exit'. Policy: do NOT wander; STALL (None) and flag proposer_failed for debugging.
    loop, _ = _nav_loop(42)
    loop.planner.provider = FailProvider()
    loop.memory.graph.next_hop = lambda a, b: None
    cells = {(x, y) for x in range(4) for y in range(8)}
    loop.world.ingest_collision(42, 4, 8, cells, None, None)
    events = []
    loop.on_event = lambda k, p: events.append((k, p))
    player = SimpleNamespace(x=1, y=1, map_id=42, facing="south")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={}, map_view=["x"],
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    move = loop._navigate_leg(_d(), obs, set())
    assert move is None
    assert any(k == "proposer_failed" for k, _ in events)


def test_navigate_leg_configured_failure_grounded_proceeds_and_flags():
    # a wired model FAILS but a KNOWN route exists (enter a door) -> proceed on the grounded route
    # (correct navigation, not a wander) AND still flag proposer_failed.
    loop, _ = _nav_loop(42)
    loop.planner.provider = FailProvider()
    loop.memory.graph.next_hop = lambda a, b: (1, (3, 7))
    player = SimpleNamespace(x=3, y=7, map_id=42, facing="east")
    obs = SimpleNamespace(player=player, map_dims=(4, 8), game_state={}, map_view=["x"],
                          exits=[{"x": 3, "y": 7, "dest_map": 1}])
    events = []
    loop.on_event = lambda k, p: events.append((k, p))
    move = loop._navigate_leg(_d(), obs, set())
    assert isinstance(move, MoveAction) and move.direction == Direction.SOUTH
    assert any(k == "proposer_failed" for k, _ in events)


# --- THE REGRESSION TESTS: a reflect/proposer model's stated target is actually navigated to ---
# In the frozen run the model literally said "step west to (4,6), then south to the exit at (4,11)"
# and the agent never moved (the proposal was orphaned). These lock in that a proposer's output
# TRANSLATES INTO MOVEMENT toward/through the named target.

import json as _json

_DELTA = {Direction.NORTH: (0, -1), Direction.SOUTH: (0, 1), Direction.EAST: (1, 0), Direction.WEST: (-1, 0)}


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _applied(player, move):
    dx, dy = _DELTA[move.direction]
    return (player.x + dx, player.y + dy)


class ProposerProvider:
    """A provider whose chat_json emits the exact target a reflect/proposer model would name."""
    def __init__(self, obj): self._obj = obj
    def chat_json(self, system, state, image=None): return _json.dumps(self._obj), 0, {}


def _lab_like_obs(px, py):
    # a 4x8 room with the only exit door at (3,7) — stand-in for the lab / a building interior
    player = SimpleNamespace(x=px, y=py, map_id=42, facing="south")
    return SimpleNamespace(player=player, map_dims=(4, 8), game_state={}, map_view=["room"],
                           exits=[{"x": 3, "y": 7, "dest_map": 1}])


def test_reflect_style_exit_proposal_is_navigated_toward_the_door():
    # model says "head to the exit and leave" -> the agent MOVES toward the door (not None/stall).
    loop, _ = _nav_loop(42)
    loop.planner.provider = ProposerProvider({"kind": "exit", "note": "head south to the exit and leave"})
    loop.memory.graph.next_hop = lambda a, b: None       # fresh-load lab: no known route out
    loop.world.ingest_collision(42, 4, 8, {(x, y) for x in range(4) for y in range(8)}, None, None)
    obs = _lab_like_obs(1, 1)
    move = loop._navigate_leg(_d(), obs, set())
    assert isinstance(move, MoveAction)
    assert _manhattan(_applied(obs.player, move), (3, 7)) < _manhattan((1, 1), (3, 7))  # got CLOSER


def test_reflect_style_named_tile_proposal_is_navigated_toward():
    # model names a concrete stepping-stone tile (like "go to (4,6)") -> the agent steps toward it.
    loop, _ = _nav_loop(42)
    loop.planner.provider = ProposerProvider({"kind": "tile", "x": 3, "y": 7, "note": "go to the exit at (3,7)"})
    loop.memory.graph.next_hop = lambda a, b: None
    loop.world.ingest_collision(42, 4, 8, {(x, y) for x in range(4) for y in range(8)}, None, None)
    obs = _lab_like_obs(1, 1)
    move = loop._navigate_leg(_d(), obs, set())
    assert isinstance(move, MoveAction)
    assert _manhattan(_applied(obs.player, move), (3, 7)) < _manhattan((1, 1), (3, 7))


def test_reflect_style_proposal_emits_target_event_and_folds_note():
    # the proposer's note becomes the visible short-term objective (reflection folded in).
    loop, _ = _nav_loop(42)
    loop.planner.provider = ProposerProvider({"kind": "exit", "note": "leave the lab, gym is north"})
    loop.memory.graph.next_hop = lambda a, b: None
    loop.world.ingest_collision(42, 4, 8, {(x, y) for x in range(4) for y in range(8)}, None, None)
    events = []
    loop.on_event = lambda k, p: events.append((k, p))
    loop._navigate_leg(_d(), _lab_like_obs(1, 1), set())
    assert any(k == "target" for k, _ in events)                       # a target event per leg
    assert loop._plan is not None and "leave the lab" in (loop._plan.next_objective or "")
```

Note: `_nav_loop`'s Stub reasoner has no `_plan` populated by default; `ReasoningLoop.__init__` sets `self._plan = self.memory.plan`. For the note-fold test, ensure `loop._plan` is a `ReflectionPlan` instance (the plan wrapper) — if `self.memory.plan` is None in the fake setup, initialize `loop._plan = ReflectionPlan()` at the top of that test (import it from `pokemon_agent.agent.reasoner`). The implementer should adjust the test setup minimally so `next_objective` is assignable, WITHOUT weakening the assertion.

- [ ] **Step 2: Run to verify fail**

Run: `uv run python -m pytest tests/unit/test_unified_loop.py -q`
Expected: FAIL (`_default_target` missing / freeze test returns None).

- [ ] **Step 3: Implement**

Add target-state init in `__init__` (near the `_leg_wp` block):

```python
        # unified mid-level target (the proposer's typed choice, held across frames — cadence B)
        self._target: dict | None = None
        self._target_map: int | None = None
        self._target_stuck = 0          # consecutive legs the current target made no progress
        self._recent_targets: deque = deque(maxlen=6)
```

Add reset in `_commit_directive` (alongside the `_leg_wp = None` resets):

```python
        self._target = None
        self._target_stuck = 0
        self._recent_targets.clear()
```

Add the constant near the other budgets:

```python
TARGET_REPROPOSE_LIMIT = 2  # times the proposer may re-pick a DIFFERENT target to unstick before L1
```

Add `_default_target`, `_target_reached`, `_current_target`, `_propose_target`, and rewrite `_navigate_leg`:

```python
def _default_target(self, directive, obs) -> dict | None:
    """The deterministic typed target for this leg (what the router would do with no LLM). None
    means 'arrived on the target map' (the success predicate ends the directive)."""
    player = obs.player
    txy = directive.target_xy
    tmap = directive.target_map
    if directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM) and txy is None:
        return {"kind": "approach_npc", "sprite": (directive.target or {}).get("sprite")}
    if txy is not None and (tmap is None or tmap == player.map_id):
        # NOTE: in production a talk/grab WITH a concrete tile is dispatched to _servo_step (see
        # _dispatch_servo), so this interact=True case is only exercised by the resolver's unit test;
        # kept for completeness (a tile target that should press A on arrival).
        interact = directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM)
        return {"kind": "tile", "x": txy[0], "y": txy[1], "interact": interact}
    if tmap is None or tmap == player.map_id:
        return None
    hop = self.memory.graph.next_hop(player.map_id, tmap)
    if hop is None:
        return {"kind": "exit"}          # no known route out -> leave; the graph learns the edge
    next_map, _tile = hop
    door = next((e for e in (obs.exits or []) if e.get("dest_map") == next_map), None)
    if door is not None:
        return {"kind": "enter", "map": next_map}
    goal_dir = next_direction(self.memory.graph, player.map_id, tmap)
    return {"kind": "edge", "dir": goal_dir, "next_map": next_map}


@staticmethod
def _target_reached(target, obs) -> bool:
    """Only a 'tile' target can be 'reached' in place (others resolve to a move each frame and
    end when the map changes / an interaction fires)."""
    if target.get("kind") != "tile":
        return False
    return (obs.player.x, obs.player.y) == (int(target["x"]), int(target["y"]))


def _current_target(self, directive, obs) -> dict | None:
    """Hold the current typed target across frames; (re)pick via the proposer on
    reached / stuck / map-change. Returns None only when we've arrived on the target map."""
    player = obs.player
    default = self._default_target(directive, obs)
    if default is None:
        self._target = None
        return None
    held = self._target
    if (held is not None and self._target_map == player.map_id
            and not self._target_reached(held, obs)):
        return held
    proposed = self._propose_target(obs, directive, default, stuck=False)
    if proposed.get("kind") == "unresolved":
        # a configured model failed AND there's no grounded route (proposer already flagged it).
        # Don't cache the failure — retry the model next frame (a transient 429/cold-start recovers);
        # this leg stalls visibly. Persistent failures keep flagging + stalling = a visible break.
        self._target = None
        self._target_map = None
        return proposed
    self._target = proposed
    self._target_map = player.map_id
    self._target_stuck = 0
    return self._target


def _propose_target(self, obs, directive, default, *, stuck) -> dict:
    """Ask the mid-level proposer for the typed target (folding reflection: its note becomes the
    visible short-term objective). Deterministic default when offline. Never None."""
    prov = getattr(self.planner, "provider", None) if self.planner else None
    if prov is None:
        return default
    player = obs.player
    occupied = {(int(n["x"]), int(n["y"])) for n in ((obs.game_state or {}).get("npcs") or [])
                if "x" in n and "y" in n}
    tmap = directive.target_map
    goal_dir = next_direction(self.memory.graph, player.map_id, tmap) if tmap is not None else None
    exit_tile = None
    if tmap is not None:
        hop = self.memory.graph.next_hop(player.map_id, tmap)
        if hop is not None:
            door = next((e for e in (obs.exits or []) if e.get("dest_map") == hop[0]), None)
            if door is not None:
                exit_tile = (int(door["x"]), int(door["y"]))
    ctx = {
        "map_view": obs.map_view,
        "player": {"x": player.x, "y": player.y, "map_id": player.map_id},
        "objective": directive.reason,
        "destination": (f"{map_name(tmap)} (map {tmap})" if tmap is not None else "the goal"),
        "goal_dir": goal_dir,
        "exit_tile": exit_tile,
        "npcs": [{"x": int(n["x"]), "y": int(n["y"]), "sprite": n.get("sprite"),
                  "talked_to": n.get("talked_to")}
                 for n in ((obs.game_state or {}).get("npcs") or []) if "x" in n and "y" in n],
        "recent_trail": list(self._recent)[-8:],
        "recent_targets": [dict(t) for t in self._recent_targets],
        "reachable": self._reachable_cells(player, occupied),
        "default": default,
        "stuck": stuck,
        "why": ("the last target was unreachable or made no progress; propose a DIFFERENT one"
                if stuck else "pick the next target toward the goal"),
    }
    target = self.planner.propose_target(self.controller.emu, ctx)
    if not target:
        target = default
    if target.get("kind") == "unresolved":
        # a CONFIGURED model failed. ALWAYS flag it loudly (logs + viewer) so it's debuggable.
        self.on_event("proposer_failed", {"step": self.session.step, "reason": target.get("reason"),
                                          "raw": target.get("raw"), "default": default, "stuck": stuck})
        if default.get("kind") != "exit":
            target = default          # a GROUNDED route (enter/edge/tile/approach) -> proceed on it
                                      # (correct navigation, not a wander) — but the failure is flagged.
        else:
            if self._plan is not None:
                self._plan.next_objective = f"[proposer failed] {target.get('reason')}"
            return target             # UNGROUNDED (would be an exit/explore guess) -> break visibly
    note = target.get("note")
    if note and self._plan is not None:
        self._plan.next_objective = note      # fold the reflection note (mid-level -> visible objective)
    self._recent_targets.append({k: target.get(k) for k in ("kind", "x", "y", "map", "sprite")})
    self.on_event("target", {"step": self.session.step, "target": target, "stuck": stuck})
    return target


def _navigate_leg(self, directive: Directive, obs, blocked_dirs: set[str]):
    """UNIFIED within-map navigation: the mid-level proposer picks ONE typed target toward the
    goal (held across frames), Jev picks the routing policy, and the deterministic router enacts
    it. When a target can't make progress, RE-PROPOSE a different one (the get-unstuck job);
    only after that also fails do we return None so L1 (quest/strategize) escalates. The old
    'return None -> wait forever' freeze is gone: the default target is always 'exit' (leave)."""
    player = obs.player
    if player is None:
        return None
    occupied = {(int(n["x"]), int(n["y"])) for n in ((obs.game_state or {}).get("npcs") or [])
                if "x" in n and "y" in n}
    target = self._current_target(directive, obs)
    if target is None:
        return None  # arrived on the target map; the success predicate ends the directive
    if target.get("kind") == "unresolved":
        return None  # configured model failed with no grounded route -> already flagged; STALL
                     # visibly (step_once's await-plan wait) rather than wander deterministically
    move = self._resolve_target(target, directive, obs, blocked_dirs, occupied)
    if move is not None:
        self._target_stuck = 0
        return move
    # stuck: re-propose a DIFFERENT target (the proposer's whole purpose), a couple of times
    self._target_stuck += 1
    if self._target_stuck <= TARGET_REPROPOSE_LIMIT:
        self._target = self._propose_target(obs, directive, self._default_target(directive, obs) or {"kind": "exit"}, stuck=True)
        self._target_map = player.map_id
        if self._target.get("kind") == "unresolved":
            self._target = None
            return None  # flagged; stall (don't wander)
        move = self._resolve_target(self._target, directive, obs, blocked_dirs, occupied)
        if move is not None:
            return move
    return None  # genuinely wedged -> _manage_directive escalates (story gate / re-strategize / heal)
```

Note: delete the old `_navigate_leg` body (the ~90 lines it replaces). Keep `_pick_waypoint`, `_policy_route`, `_jev_path`, `_farm_step`, `_pick_policy`, `_edge_step`, `_leave_via_nearest_exit`, `_bfs_move`, `_warp_exit_dir`, `_on_goal_edge` — the resolvers call them.

- [ ] **Step 4: Run to verify pass + full suite**

Run: `uv run python -m pytest tests/unit/test_unified_loop.py tests/unit/test_executive.py -q`
Expected: PASS (incl. the two migrated door / exit-fallback executive tests).
Run: `uv run python -m pytest -q` → all green.

- [ ] **Step 5: Commit**

```bash
git add src/pokemon_agent/agent/reason_loop.py tests/unit/test_unified_loop.py
git commit -m "reason_loop: unify _navigate_leg around a held typed target + unstick (no freeze)"
```

---

## Task 4: Wire the proposer into `step_once` + fold reflection; route TALK_TO via approach_npc

**Files:**
- Modify: `src/pokemon_agent/agent/reason_loop.py`
- Test: `tests/unit/test_unified_loop.py`

Extract the dispatch into a small helper so it's directly unit-testable (avoids a tautological/brittle `step_once` spy test — reviewer advisory), and drop the orphaned `_maybe_reflect` on the planner path.

- [ ] **Step 1: Write failing test**

```python
# add to tests/unit/test_unified_loop.py

def test_dispatch_servo_talk_without_tile_uses_navigate_leg():
    loop, _ = _nav_loop(0)
    calls = []
    loop._navigate_leg = lambda d, o, b: calls.append("nav")
    loop._servo_step = lambda d, o, b: calls.append("servo")
    d_notile = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 0}, success={"talked_on_map": 0})
    d_tile = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 0, "x": 2, "y": 2},
                       success={"talked_on_map": 0})
    loop._dispatch_servo(d_notile, None, set())
    loop._dispatch_servo(d_tile, None, set())
    assert calls == ["nav", "servo"]  # no-tile talk -> approach_npc via nav; tile talk -> servo
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run python -m pytest tests/unit/test_unified_loop.py::test_dispatch_servo_talk_without_tile_uses_navigate_leg -q`
Expected: FAIL (`_dispatch_servo` missing).

- [ ] **Step 3: Add `_dispatch_servo`, call it from `step_once`, drop the orphaned reflect**

Add the helper method to `ReasoningLoop`:

```python
def _dispatch_servo(self, directive, obs, blocked_dirs):
    """Route a target-bearing directive to the right servo: a talk/grab with a concrete TILE ->
    the deterministic BFS+interact servo; anything else (incl. a talk/grab with NO tile yet) ->
    the unified nav leg, whose approach_npc finds and reaches the person."""
    if directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM) and directive.target_xy is not None:
        return self._servo_step(directive, obs, blocked_dirs)
    return self._navigate_leg(directive, obs, blocked_dirs)
```

In `step_once`, replace this exact current block (note the multi-line comment must be matched too):

```python
        blocked_dirs = self._blocked_dirs(obs, self._next_hop_map(obs, directive))
        self._maybe_reflect(obs, player_desc)
        if directive is not None and directive.target_bearing:
            # TALK_TO / GRAB_ITEM head to a concrete NPC/item tile (no navigation reasoning) ->
            # deterministic servo. Everything else (TRAVEL / GRIND: traverse this map toward an
            # exit) is driven by L2: LunaRoute picks the next grid tile, BFS routes to it.
            if directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM):
                servo = self._servo_step(directive, obs, blocked_dirs)
            else:
                servo = self._navigate_leg(directive, obs, blocked_dirs)
```

with:

```python
        blocked_dirs = self._blocked_dirs(obs, self._next_hop_map(obs, directive))
        # NOTE: no separate _maybe_reflect on the planner path — reflection is now folded INTO the
        # mid-level proposer inside _navigate_leg (its note becomes self._plan.next_objective). The
        # legacy no-planner path above still calls _maybe_reflect.
        if directive is not None and directive.target_bearing:
            servo = self._dispatch_servo(directive, obs, blocked_dirs)
```

Verify the real surrounding text first with `grep -n "_maybe_reflect" src/pokemon_agent/agent/reason_loop.py` and read the block, so the Edit matches exactly (reviewer advisory: the current file has the multi-line comment between `_maybe_reflect` and the `if`).

- [ ] **Step 4: Run full suite**

Run: `uv run python -m pytest -q`
Expected: all green (the `_maybe_reflect` removal must not break `test_executive` — it doesn't assert a reflect event on the planner path; verify).

- [ ] **Step 5: Commit**

```bash
git add src/pokemon_agent/agent/reason_loop.py tests/unit/test_unified_loop.py
git commit -m "reason_loop: proposer drives the overworld leg; TALK_TO(no tile) -> approach_npc"
```

---

## Task 5: Strategist names the talk sprite (so approach_npc targets the right person)

**Files:**
- Modify: `src/pokemon_agent/agent/planner_llm.py`
- Test: `tests/unit/test_unified_loop.py`

- [ ] **Step 1: Write failing test**

```python
# add to tests/unit/test_unified_loop.py

def test_strategize_talk_step_carries_sprite_name():
    import json as _json
    plan = {"plan": "deliver", "steps": [{"map": 40, "talk": True, "who": "Oak",
                                          "done_when": "no_item:Oak's Parcel", "why": "hand it over"}]}
    class P:
        def chat_json(self, system, state, image=None): return _json.dumps(plan), 0, {}
    from pokemon_agent.agent.planner_llm import Planner

    class FakeEmu:
        def read_memory(self, a): return 0
    pl = Planner(goal_map=40, strategist=P())
    # stub the game_state reads the strategist makes
    import pokemon_agent.agent.planner_llm as m
    pl_party = m.read_party; pl_items = m.read_items; pl_badges = m.read_badges
    m.read_party = lambda e: []; m.read_items = lambda e: []; m.read_badges = lambda e: {"count": 0}
    try:
        quest = pl.strategize(FakeEmu(), memory=None, why="deliver the parcel")
    finally:
        m.read_party = pl_party; m.read_items = pl_items; m.read_badges = pl_badges
    talk = [d for d in quest if d.intent.name == "TALK_TO"]
    assert talk and (talk[0].target or {}).get("sprite") == "Oak"
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run python -m pytest tests/unit/test_unified_loop.py::test_strategize_talk_step_carries_sprite_name -q`
Expected: FAIL (sprite not set).

- [ ] **Step 3: Implement**

In `STRATEGIST_SYSTEM`, extend the steps schema line to include an optional `who`:

```
{"plan": "one-line summary",
 "steps": [{"map": <int map id>, "talk": <true|false>, "who": "<npc name to talk to, e.g. Oak>",
            "done_when": "<criterion>", "why": "<short>"}]}
```

and add one sentence: `When "talk" is true, set "who" to the NPC you must talk to (e.g. "Oak", "the Mart clerk") so the agent approaches the RIGHT person, not the nearest one.`

In `strategize`, when building the TALK_TO directive, attach the sprite:

```python
                if step.get("talk"):
                    who = str(step.get("who") or "").strip() or None
                    tgt = {"kind": "npc", "map": mp}
                    if who:
                        tgt["sprite"] = who
                    quest.append(Directive(intent=Intent.TALK_TO, target=tgt,
                                           success=(done or {"talked_on_map": mp}),
                                           reason=f"quest: talk to {who or 'someone'} in {map_name(mp)} "
                                                  f"[{step.get('done_when') or 'talked'}] — {why_s}"))
```

- [ ] **Step 4: Run to verify pass + full suite**

Run: `uv run python -m pytest -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/pokemon_agent/agent/planner_llm.py tests/unit/test_unified_loop.py
git commit -m "planner: strategist names the talk NPC (who) so approach_npc reaches the right person"
```

---

## Task 6: Live validation from the rival-battle save (manual, not a unit test)

**Files:**
- Use: `roms/pokemon_red.gb.state` (the natural save — paused INSIDE the scripted rival battle).
- Scratch script only (not committed).

- [ ] **Step 1: Sanity — full suite green**

Run: `uv run python -m pytest -q` → all pass.

- [ ] **Step 2: Run a bounded live game with the recorder + viewer**

Run (needs LunaRoute + TypeSafe keys, which are fish universal vars — run via fish):
```bash
fish -c 'python scripts/run_agent.py --load-state roms/pokemon_red.gb.state --goal-map <PEWTER> --pather policy --max-steps 400 --record'
```
(Confirm the exact `run_agent.py` flags first with `uv run python scripts/run_agent.py --help`.)

Expected sequence to watch in the run log / viewer (`http://localhost:8010/_viewer.html`):
1. `_battle_turn` wins the rival battle (`wIsInBattle` → 0).
2. Overworld lab: a `target` event appears (kind `exit` or `approach_npc`), the agent MOVES (no "await-plan" wait-loop), reaches a lab door, steps through → map changes to Pallet.
3. Leaves Pallet north → Route 1 → Viridian.

- [ ] **Step 3: If it wedges, read the `target` / `stuck` / `directive_impossible` events**

The unstick ladder is: `target(stuck=false)` → (no progress) → `target(stuck=true)` → (still none) → `_navigate_leg` returns None → `directive_impossible` / quest re-strategize. Confirm each rung fires. File any real bug as a new task; do NOT loosen a test to make it pass.

- [ ] **Step 4: Update memory + spec status**

Update `~/.claude/.../memory/pokemon-agent-emulator.md`: unified loop IMPLEMENTED; note the live result. Mark the spec `docs/superpowers/specs/2026-09-18-unified-control-loop-design.md` "Status" as implemented.

- [ ] **Step 5: Social update** (per CLAUDE.md — call the user "Nomzor").

---

## Definition of done

- **The model's proposal drives movement.** When the proposer returns a valid target, the router enacts it — it is never orphaned (the freeze bug is fixed at its root: the model's answer becomes movement).
- **A configured-model failure BREAKS LOUDLY, not silently.** Every failed proposal emits a `proposer_failed` event (reason + raw response) in the logs/viewer. With no grounded route it STALLS in place (visible break) instead of wandering deterministically; with a known route it proceeds on that route (still flagged). No silent deterministic wander when the model was supposed to guide.
- The deterministic `_default_target` is used only in the no-provider mode (tests/offline), never as a live-run fallback for a failed call.
- A model-named exit *tile* is stepped through (honored), not stopped on.
- One `target` event per leg (cadence B), not per frame; the note is the visible short-term objective.
- Reflection is folded into the proposer (no separate orphaned `_maybe_reflect` on the planner path).
- `approach_npc` reaches and interacts with the NAMED sprite (Oak), not the nearest (rival).
- Escalation to L1 (quest / re-strategize / heal) still fires — only after the proposer's unstick attempts fail.
- Full suite green (188 + new).
- Live: wins the rival battle, leaves the lab, reaches Viridian — and if the model errors, the run visibly stalls with `proposer_failed` in the log rather than drifting.
```
