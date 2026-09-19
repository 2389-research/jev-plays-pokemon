# Unified Control Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the mid-level layer (reflection, merged with the L2 waypoint call) propose ONE machine-usable typed target every leg that ALWAYS drives the router — so the agent never freezes with a planner that "knows the answer" but stands still, and L2 can force navigation and get unstuck.

**Architecture:** `Planner.propose_target(emu, context)` returns a typed target (`tile` / `exit` / `enter` / `approach_npc`) with a one-line note; it never returns None (defaults to `exit`). `ReasoningLoop` holds one typed target across frames (cadence B — re-propose on reached / stuck / map-change), resolves it deterministically to a move via existing resolvers (BFS / weighted-policy / Jev-path / door-step-through / edge-crossing / approach-NPC), Jev still picks the routing policy inside a leg, and the reflection note is folded into the proposer output. The old `_navigate_leg` `return None` "await-plan" freeze is replaced by: re-propose to unstick, then default to `exit`; only after the proposer also fails does it hand up to L1 (the existing quest/strategize/heal escalation).

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


def test_propose_target_offline_returns_default():
    p = Planner(goal_map=0, provider=None)
    assert p.propose_target(emu=None, context=_ctx()) == {"kind": "exit"}


def test_propose_target_parses_tile():
    p = Planner(goal_map=0, provider=FakeProvider('{"kind":"tile","x":3,"y":4,"note":"south"}'))
    t = p.propose_target(emu=None, context=_ctx())
    assert t["kind"] == "tile" and (t["x"], t["y"]) == (3, 4) and t["note"] == "south"


def test_propose_target_rejects_unreachable_tile_falls_back_to_default():
    p = Planner(goal_map=0, provider=FakeProvider('{"kind":"tile","x":9,"y":9,"note":"bad"}'))
    assert p.propose_target(emu=None, context=_ctx()) == {"kind": "exit"}


def test_propose_target_passes_through_exit_and_approach_and_enter():
    for content, kind in [('{"kind":"exit","note":"leave"}', "exit"),
                          ('{"kind":"approach_npc","sprite":"Oak","note":"talk"}', "approach_npc"),
                          ('{"kind":"enter","map":1,"note":"door"}', "enter")]:
        p = Planner(goal_map=0, provider=FakeProvider(content))
        assert p.propose_target(emu=None, context=_ctx())["kind"] == kind


def test_propose_target_bad_json_returns_default():
    p = Planner(goal_map=0, provider=FakeProvider("not json"))
    assert p.propose_target(emu=None, context=_ctx()) == {"kind": "exit"}
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
    get-unstuck mechanism. Never returns None — defaults to {"kind":"exit"} (leave toward the
    goal) so the loop always has something to enact. When no provider is wired (offline/tests)
    or the model's pick is invalid, returns context['default'] (the deterministic layer's choice,
    itself defaulting to exit)."""
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
    for _ in range(2):  # one retry on an invalid pick
        try:
            content, _, _ = self.provider.chat_json(PROPOSER_SYSTEM, state)
            data = json.loads(strip_fences(content))
            kind = data.get("kind")
        except Exception:
            continue
        note = str(data.get("note") or "").strip()
        if kind == "tile":
            try:
                x, y = int(data["x"]), int(data["y"])
            except (KeyError, TypeError, ValueError):
                continue
            px = (context.get("player") or {}).get("x")
            py = (context.get("player") or {}).get("y")
            is_exit = exit_tile is not None and (x, y) == tuple(exit_tile)
            progresses = (px is None) or (x, y) != (px, py)
            if progresses and (is_exit or reachable is None or (x, y) in reachable):
                return {"kind": "tile", "x": x, "y": y, "note": note}
            continue  # unreachable / no-op pick -> retry, then fall back
        if kind == "enter":
            try:
                return {"kind": "enter", "map": int(data["map"]), "note": note}
            except (KeyError, TypeError, ValueError):
                continue
        if kind == "approach_npc":
            return {"kind": "approach_npc", "sprite": (data.get("sprite") or None), "note": note}
        if kind == "exit":
            return {"kind": "exit", "note": note}
    return default
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
        if (player.x, player.y) == xy:
            return InteractAction() if target.get("interact") else None
        avoid = set(occupied)
        return self._route_to_tile(obs, xy, blocked_dirs, avoid)
    if kind == "enter":
        return self._enter_map(int(target["map"]), obs, blocked_dirs, occupied)
    if kind == "approach_npc":
        return self._approach_npc(target.get("sprite"), obs, blocked_dirs, occupied)
    if kind == "edge":
        return self._cross_edge(target.get("dir"), target.get("next_map"), obs, blocked_dirs, occupied)
    # exit (default): leave via nearest door; if none, try a boundary-edge crossing toward the goal
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
```

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
    self._target = self._propose_target(obs, directive, default, stuck=False)
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
    move = self._resolve_target(target, directive, obs, blocked_dirs, occupied)
    if move is not None:
        self._target_stuck = 0
        return move
    # stuck: re-propose a DIFFERENT target (the proposer's whole purpose), a couple of times
    self._target_stuck += 1
    if self._target_stuck <= TARGET_REPROPOSE_LIMIT:
        self._target = self._propose_target(obs, directive, {"kind": "exit"}, stuck=True)
        self._target_map = player.map_id
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

- [ ] **Step 1: Write failing test**

```python
# add to tests/unit/test_unified_loop.py

def test_talk_to_without_tile_routes_via_navigate_leg_approach_npc(monkeypatch):
    loop, _ = _nav_loop(0)
    calls = {"nav": 0, "servo": 0}
    loop._navigate_leg = lambda d, o, b: (calls.__setitem__("nav", calls["nav"] + 1) or None)
    loop._servo_step = lambda d, o, b: (calls.__setitem__("servo", calls["servo"] + 1) or None)
    from pokemon_agent.agent.plan import Directive, Intent
    d = Directive(intent=Intent.TALK_TO, target={"kind": "npc", "map": 0}, success={"talked_on_map": 0})
    # exercise just the dispatch branch used in step_once
    servo = (loop._servo_step(d, None, set()) if (d.intent in (Intent.TALK_TO, Intent.GRAB_ITEM)
             and d.target_xy is not None) else loop._navigate_leg(d, None, set()))
    assert calls["nav"] == 1 and calls["servo"] == 0
```

(This test documents the intended dispatch rule; Step 3 makes `step_once` follow it.)

- [ ] **Step 2: Run to verify fail**

Run: `uv run python -m pytest tests/unit/test_unified_loop.py::test_talk_to_without_tile_routes_via_navigate_leg_approach_npc -q`
Expected: PASS actually (it inlines the rule) — so instead assert the real `step_once` path. Simpler: skip a brittle monkeypatch test and rely on Task 3's `approach_npc` tests + the dispatch change being obvious. If keeping, verify it fails BEFORE editing `step_once` by asserting on a real `step_once` run with a TALK_TO directive and a stub `_navigate_leg` spy. Use judgment; do not force a flaky test.

- [ ] **Step 3: Edit `step_once` dispatch + drop the orphaned reflect on the planner path**

In `step_once`, replace:

```python
        self._maybe_reflect(obs, player_desc)
        if directive is not None and directive.target_bearing:
            if directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM):
                servo = self._servo_step(directive, obs, blocked_dirs)
            else:
                servo = self._navigate_leg(directive, obs, blocked_dirs)
```

with:

```python
        # NOTE: no separate _maybe_reflect here — reflection is now folded into the mid-level
        # proposer inside _navigate_leg (its note becomes self._plan.next_objective). The
        # legacy no-planner path above still calls _maybe_reflect.
        if directive is not None and directive.target_bearing:
            # an explicit NPC/item TILE -> deterministic servo (BFS + interact). A talk/grab with
            # no tile yet -> the unified nav's approach_npc finds and reaches the person.
            if directive.intent in (Intent.TALK_TO, Intent.GRAB_ITEM) and directive.target_xy is not None:
                servo = self._servo_step(directive, obs, blocked_dirs)
            else:
                servo = self._navigate_leg(directive, obs, blocked_dirs)
```

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

- `_navigate_leg` never returns None on a fresh-load cross-map leg with a reachable exit (freeze fixed).
- One `target` event per leg (cadence B), not per frame; note is the visible short-term objective.
- Reflection is folded into the proposer (no separate orphaned `_maybe_reflect` on the planner path).
- `approach_npc` reaches and interacts with the NAMED sprite (Oak), not the nearest (rival).
- Escalation to L1 (quest / re-strategize / heal) still fires — only after the proposer's unstick attempts fail.
- Full suite green (188 + new).
- Live: wins the rival battle, leaves the lab, reaches Viridian.
```
