# Running Jev Plays Pokémon — full setup & operations

The durable reference for how this project is wired and run. (Session context evaporates on
compaction; this file is the source of truth. Keep it updated when the setup changes.)

---

## 1. What this is

An LLM agent that autonomously plays Pokémon Red through PyBoy, aiming to reach the first gym
(Brock, Pewter City). Three control layers:

- **L1 — strategic planner** (`agent/planner_llm.py: Planner.revise_quests` + `agent/quest_reconciler.py`):
  "at this point in the game, what should we be doing?" Runs periodically + on events, reviews the
  quest plan against game state + the Orrery knowledge base, and revises it by emitting quests in the
  executor's `done_when` DSL (or "no change"). A deterministic reconciler applies the proposal,
  preserving progress and replacing wedged steps. Maintains a durable `AgentPlan`
  (mission / milestone / tried_failed).
- **L2 — tactical proposer** (`agent/reason_loop.py: _navigate_leg` → `Planner.propose_target`):
  turns the active quest step into ONE concrete short-term target (tile / exit / enter / approach_npc).
- **L3 — Jev + router** (`agent/typesafe_reasoner.py` + `agent/routing.py` + BFS): Jev picks the
  routing policy / battle move / which NPC; a weighted-Dijkstra/BFS router enacts the tiles.

The executive that ties them together is `agent/reason_loop.py: ReasoningLoop` (`--mode reason`).

---

## 2. Prerequisites

- **Python 3.13 + [uv](https://docs.astral.sh/uv/).** `uv sync` creates `.venv/` with everything
  (PyBoy bundles SDL2). Run everything with `uv run ...`.
- **The ROM (you supply your own, legally-owned):** `roms/pokemon_red.gb`. Gitignored; never committed.
- **A starting save state:** the agent boots from a `.state`, not a cold cartridge (it doesn't do the
  intro / name-entry). Play the intro yourself, name your character, and save. The default natural
  save is `roms/pokemon_red.gb.state` — **it is paused INSIDE the scripted rival battle**
  (`wIsInBattle`@0xD057 = 2); the loop's battle controller wins it, then it's in Oak's Lab overworld.
  Named fixtures live in `states/<name>.state` (use `--state <name>`). All `.state` files are gitignored.
- **API keys** (see §3) and **the Orrery KB** running (see §4).

---

## 3. API keys / environment

Two providers. **On this machine the keys are fish universal exported variables** — so they are
present in an interactive `fish` shell, but **NOT inherited by a plain `bash`/non-fish shell**. From
tooling that isn't fish, wrap commands as `fish -c '...'`.

| Env var | What | Notes |
|---|---|---|
| `LUMAROUTE_API_KEY` | LunaRoute (LLM gateway) key | **Misspelled `LUMA` on purpose** (that's the real var). `LUNAROUTE_API_KEY` also accepted. Keys start `lr_`. |
| `LUNAROUTE_BASE_URL` | LunaRoute base | default `https://gw.lunaroute.com/v1` |
| `LUNAROUTE_REASONING_EFFORT` | reasoning budget | default `"none"` — **essential**; these are reasoning models and the default burns the output budget → empty content / `finish_reason=length`. |
| `TYPESAFE_API_KEY` | TypeSafe/Jev (calibrated classifier) | pricing ~$0.042/M input tokens, output free. |
| `ORRERY_WORKSPACE_ID` | Orrery KB workspace | or pass `--orrery-workspace`. Ours: **`6d677a16`**. |
| `ORRERY_BASE_URL` | Orrery KB base | or `--orrery-url`. default `http://localhost:8100`. |

**LunaRoute gotchas:** streaming is required (non-streaming can hang); chain-of-thought comes back in
`message.reasoning`, not `content` (we only read `content`); 429 = concurrency cap (~8 in-flight) →
backoff; models: `deepseek-4.1-flash` (fast text, the L2/mid model), `glm-5.3` (strategist / L1),
`glm-5.3-vision` (vision). Anything else 404s.

---

## 4. Orrery knowledge base (REQUIRED for L1 to be smart)

L1's strategist queries an Orrery knowledge graph to recognize story gates and locations (e.g. "the
Viridian north gate is closed until Oak's Parcel is delivered → go to the Viridian Mart"). **Without
the KB attached, L1 cannot recognize the parcel errand and the agent grinds cluelessly at the gate.**
This is the #1 "it looks broken" gotcha — it's not broken, the KB just wasn't attached.

- Runs locally at **`http://localhost:8100`**, workspace **`6d677a16`** (`GET /health` → `{"status":"ok"}`;
  `GET /search?q=...` with header `X-Workspace-Id: <ws>`).
- Attach it with **`--orrery-workspace 6d677a16`** (or set `ORRERY_WORKSPACE_ID`).
- Consumed by `agent/knowledge.py: KnowledgeBase` → the planner's `_llm_with_search` tool-loop
  (L1 quest planning) and battle type-effectiveness lookups. If no workspace is configured the KB is
  simply disabled (degraded planning), not an error.
- Quick check: `curl -s http://localhost:8100/health` and
  `curl -s "http://localhost:8100/search?q=Oak+parcel+Viridian&limit=3" -H "X-Workspace-Id: 6d677a16"`.

---

## 5. How to run

Canonical run — win the rival battle, leave the lab, head toward Pewter, deliver the parcel if
blocked. **`--mode reason` is required** (the default `plan-act` is the legacy vision loop and does
NOT use the L1/L2/L3 executive):

```fish
uv run python scripts/run_agent.py \
  --rom roms/pokemon_red.gb \
  --load-state roms/pokemon_red.gb.state \
  --mode reason \
  --goal "Reach Pewter City and beat Brock; deliver Oak's Parcel first if the way north is blocked" \
  --goal-map 2 \
  --level-target 12 \
  --decider typesafe \
  --pather policy \
  --no-vision \
  --orrery-workspace 6d677a16 \
  --l1-every 5 \
  --steps 600 \
  --record-dir runs/<name>
```

- Add `--headless` for no SDL window (faster); omit it to watch the game live.
- The runner requires `--rom` even with `--load-state` (the save layers on the ROM).
- Keys are fish universal vars → run from `fish` (or wrap `fish -c '...'`).
- Runs never clobber: reusing a `--record-dir` name auto-appends a timestamp.

Smoke test: same command with `--steps 120`.

---

## 6. Full parameter reference (`scripts/run_agent.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--rom PATH` | — | ROM path (required unless `--demo`). |
| `--demo` | off | FakeEmulator maze, no ROM (loop smoke test). |
| `--load-state PATH` | — | start from a `.state` (layered on the ROM). |
| `--state NAME` | — | shorthand for `states/NAME.state`. |
| `--save-state NAME` / `--save-state-out PATH` | — | save state on exit (reusable fixture). |
| `--headless` | off | no SDL window (real emulator). |
| `--speed N` | unbounded headless / 1 windowed | emulation speed; 0 = unbounded. |
| `--mode {plan-act,reason}` | `plan-act` | **use `reason`** for the L1/L2/L3 executive. `plan-act` = legacy vision planner + text actor. |
| `--decider {llm,typesafe}` | `llm` | reason mode: `typesafe` = Jev (calibrated Choice; battle moves, policy, choose_npc, path_step). Use `typesafe`. |
| `--pather {bfs,jev,policy}` | `bfs` | overworld pathing. `policy` = Jev picks a routing objective per leg (shortest / dodge-grass / farm-exp), a weighted router executes. |
| `--goal-map N` | — | target map id (route hint + directive success). **2 = Pewter City.** (0=Pallet, 1=Viridian, 12=Route 1, 40=Oak's Lab, 41=Viridian PokéCenter, 42=Viridian Mart.) |
| `--level-target N` | 0 | grind the party to ≥N before pushing to the goal (readiness). |
| `--l1-every N` | 5 | run the L1 strategic review every N legs (also runs on events: stuck/wedge/level-up/low-HP/map milestone). |
| `--orrery-workspace ID` | env | Orrery KB workspace (**`6d677a16`**). Without it L1 has no story-gate knowledge. |
| `--orrery-url URL` | env / `:8100` | Orrery base URL. |
| `--strategist-model` | `glm-5.3` | strong text model for L1 quest planning. |
| `--planner-model` | `glm-5.3-vision` | vision model for reflection (auto-swapped to `deepseek-4.1-flash` under `--decider typesafe --no-vision`). |
| `--no-vision` | off | reason mode: navigate from TEXT state only (fast text model, no screenshots to the actor). Recommended with `--decider typesafe`. |
| `--reflect-every N` | 0 | (legacy) vision planner every N steps. |
| `--record-dir DIR` | `runs/rec-<ts>` | full run recorder (per-step JSONL + screenshots + new-area save states + viewer). Auto-uniquified on collision. |
| `--no-record` | off | disable the recorder. |
| `--steps N` | 30 | max agent steps. |
| `--checkpoint-every N` / `--checkpoint-dir` / `--resume` | — | periodic (state+memory) checkpoints; resume from them. |
| `--goal "..."` | "Leave the current area" | the text mission fed to LLM prompts. Set a real mission. |
| `--log PATH` / `--screenshot-logging` | — | JSONL episode log (separate from the recorder). |

---

## 7. Watching runs — the viewer / player

The run recorder (`logging/run_recorder.py`) writes, per run dir:
- `log.jsonl` — one line per step (player, npcs, context, directive, `plan_steps`, `mission`,
  `milestone`, `l1_last`, action, result, events).
- `shots/NNNNNN.png` — the screen each step.
- `states/map<M>_step<N>.state` — reloadable save on entering a new map.
- `viewer.html` — a **single-run** scrub viewer for that run.
- It also drops **`runs/_viewer.html`** — the **multi-run PLAYER** (source: `logging/player.html`):
  a run-picker dropdown (newest-first, with timestamps), play/pause (space), speed, live-follow,
  ←/→ keys, and objective / policy / quest panels.

Serve and open:
```fish
cd runs && python -m http.server 8010
# multi-run player (pick any run):  http://localhost:8010/_viewer.html
# a single run:                     http://localhost:8010/<run-dir>/viewer.html
```
Gotcha: if you get a 404, a **stale http.server from a previous session** may be shadowing port 8010
(esp. on IPv6 `::1`). Kill it (`lsof -nP -iTCP:8010 -sTCP:LISTEN`) and restart.

---

## 8. Tests

```fish
uv run python -m pytest -q          # full suite (must be green; ~246)
```
Always run via `uv run` (the venv has `typesafe_sdk` etc.; a bare `pytest` will `ModuleNotFoundError`).

---

## 9. Tuning constants (where the knobs live)

`agent/reason_loop.py`:
- `L1_EVERY_N_LEGS_DEFAULT = 5` — L1 review cadence (also `--l1-every`).
- `BLOCK_TRIGGER = 6` — consecutive no-progress legs that fire L1 early / wedge the active step.
- `SERVO_FAIL_LIMIT = 6` — consecutive no-route steps → directive impossible.
- `TARGET_REPROPOSE_LIMIT = 2` — L2 unstick re-proposals before handing up to L1.
- `POLICY_BUDGET = 15` — steps a Jev routing policy runs before re-selection.
- `JEV_OVERRIDE_CONF = 0.6` — Jev may override the BFS path suggestion only at/above this.
- `JEV_NPC_CONF = 0.4` — trust Jev's NPC pick only at/above this (else nearest).
- `WARP_BACK = 0xFF` — a warp dest of 0xFF = "return to last map" (0xFF doors resolved via prev-map).

`agent/signals.py`:
- `HEAL_EMERGENCY = 0.15` — party HP fraction at/below which a near-faint forces an L1 heal ping.

Key RAM addresses live in `games/pokemon_red/state.py` / `map_reader.py` (e.g. wIsInBattle 0xD057,
wCurMap 0xD35E, wXCoord 0xD362, wYCoord 0xD361, wCurMapWidth 0xD369, wCurMapHeight 0xD368 — both in
BLOCKS, tiles = ×2). The full RAM collision map is **re-read every step** (ground truth; self-heals
stale map-entry reads and mid-map changes like cut trees).

---

## 10. Current status & known limitations (2026-09-19)

- **Works:** rival battle (Jev picks moves) → leaves the lab → Pallet → Route 1. L1 plans the parcel
  errand up front (with the KB), runs periodically (mostly "no change"), keeps durable
  mission/milestone, heals via an L1 ping, and re-plans when stuck (wedge → L1 replaces the step).
- **Not yet end-to-end:** it does not physically complete the Oak's-Parcel round-trip. Two open,
  separate problems (L2/L3 + plan quality, tracked for a follow-up effort):
  1. **Cross-map / building navigation** — routing to a building on a *distant* map (e.g. "go to
     Viridian Mart (42)" from Pallet is unroutable via `next_hop`; needs decompose → reach the town,
     then enter), and selecting the right building entrance in a town (agent can bounce into wrong
     houses, e.g. Red's House 37↔38).
  2. **L1 plan-ordering quality** — occasionally sequences Route 2 / grinding before the parcel (LLM
     prompt tuning). Use `runs/l1-validate*/` as fixtures.
- **Branches / PRs:** `unified-control-loop` (PR #1: proposal-drives-movement, per-step collision
  re-read, Jev policies), `l1-strategic-planner` (PR #2, based on #1: the L1 redesign). `main` is
  branch-protected (PR + 1 approval required; admin can bypass for solo work).

## 11. Design docs
- `docs/superpowers/specs/2026-09-18-unified-control-loop-design.md`
- `docs/superpowers/specs/2026-09-19-l1-strategic-planner-design.md`
- `docs/superpowers/plans/2026-09-18-unified-control-loop.md`
- `docs/superpowers/plans/2026-09-19-l1-strategic-planner.md`
