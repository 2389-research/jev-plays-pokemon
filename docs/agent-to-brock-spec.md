# Spec (v2): autonomous agent that plays Pokémon Red to the first gym (Brock)

Status: revised after a 3-way review (architecture, prior-art, red-team). Goal: a
system that plays **autonomously for a prolonged period** — Pallet → starter →
Route 1 → Viridian (+ Oak's Parcel) → Route 2 → Viridian Forest → Pewter → **beat
Brock (Boulder Badge)** — recovering from problems on its own.

> **The load-bearing lesson from review:** survival hinges on real CAPABILITIES
> (menus, in-battle policy, cross-map routing), **not** on a robustness "recovery
> ladder." The ladder cannot make a required menu selection, cannot advance a forced
> cutscene, and reloading re-arms the exact battle/menu that wedged you. Robustness
> is a safety net for the *overworld-lost* case only. Build the capabilities.

## 0. Current-code reality (what actually exists vs. what the plan assumes)

- **Battle is NOT wired into the agent loop.** `ReasoningLoop.step_once` only
  dispatches `GoToAction` or `ActionController` (Move/Press/Interact/Wait/AdvanceDialog).
  `battle.use_move` is driven *only* by `scripts/run_battle.py`. In the real loop the
  agent can only mash A / push the D-pad in a battle.
- **Mode detection is 3-bucket** (`detect_mode`: BATTLE/OVERWORLD; `read_context` adds
  heuristic `dialog`). No menu / name-entry / shop / whiteout / tutorial / cutscene
  bucket. Every menu-shaped gate is mis-bucketed as dialog(mash A) or overworld(walk).
- **No cross-map graph.** `WorldMap` is per-`map_id`; `Navigator` BFS is single-map.
  "route to Pewter" does not exist.
- **No persistent memory / no recovery.** `StuckDetector` exists but is **not imported**
  by `ReasoningLoop`; no recovery ladder; WorldMap/InteractionMemory are in-RAM only,
  never serialized. Only `emu.save_state()` persists — so a reload drops all learned map.
- **Two parallel stacks** (`loop.py`+`reasoner.py` generative actor vs
  `reason_loop.py`+`typesafe_reasoner.py`). The plan is the second; the first must be retired.
- **The planner→executor interface is currently fictional:** `TypeSafeReasoner.step()`
  reads only `plan.next_objective` (one string) and drops subgoals/notes; there are two
  incompatible plan types (`planner.Plan` vs `ReflectionPlan`).

## 1. Architecture: nested planner / executor

```
STRATEGIST (LunaRoute, System 2, rare, generative)  → typed Plan
EXECUTOR   (Jev/TypeSafe, System 1, per-step)        → one action (Choice) + Noul/Score
CONTROLLERS (deterministic)                          → buttons
PERCEPTION (RAM extractors)                          → state
MILESTONE/EVENT DETECTOR (RAM, deterministic)        → truth for "done" & progress
```

- **One canonical loop/reasoner:** `reason_loop.ReasoningLoop` + `TypeSafeReasoner`.
  Retire `loop.py`/`reasoner.py` (dead-code confusion otherwise).
- **One typed Plan contract** (replaces the two plan types). The strategist emits it;
  the executor **consumes it structurally**, not as free text:
  ```
  Plan = {
    mission, milestone,                     # current milestone
    objective,                              # concrete next objective (string)
    mode_hint,                              # overworld | battle | menu | shop | route
    option_bias,                            # constrain/expand/bias the executor's Choice set
    target,                                 # optional concrete target (tile/npc/map/item)
    notes, hypotheses[], tried_failed[],    # strategy + option generation + ledger
  }
  ```
  The active subgoal maps to a **mode + a concrete option set** the executor reads —
  the strategist must be able to *steer the option set*, not just hope a string is honored.
- **Exercise the interface from P1** with a scripted/stub strategist so it's continuous,
  not first integrated at P5.
- **Executor = Jev `Choice`** over the mode's action set (+ `Noul`/`Score` for genuinely
  fuzzy judgments only). **Milestone/"done" truth is RAM, never a Noul** (see §5).

## 2. Perception — extractors to add (P0/P1 prerequisites, none exist yet)

Have: pos/map/facing/tileset, exits, walkability, party, battle structs, NPCs, badges,
money, items, context (overworld/dialog/battle + `battle_type` + `forced_movement`).

**Add (each is a prerequisite for a phase, not a nicety):**
1. **menu_open + kind** (yes/no · list · name-entry-keyboard · shop) with a **trusted**
   cursor read (current `menu_raw` persists stale — needs a real "menu is up" signal).
2. **name-entry / keyboard** screen (a grid input mode, its own controller — NOT a list cursor).
3. **whiteout/blackout** (map→a Poké Center + money dropped + party HP refilled).
4. **battle_type branch** actually used: wild vs trainer vs **old-man tutorial** (data
   already in `read_context`, currently unused).
5. **forced_movement/cutscene** as an explicit **"locked — do not reload"** state (bit
   already read, never acted on).
6. **faint / must-send-next**, **PP-exhausted/Struggle**, **low-HP threshold**.
7. **warp-settled gate**: stable `(map_id,x,y)` for K frames before committing WorldMap
   writes (a warp flips map_id while coords are stale → phantom tiles/`_map_history`).
8. **ledge / directional-passability** tiles (BFS treats tiles bidirectional; ledges are
   one-way → false re-path oscillation).
9. **progress = graph-distance-to-objective** (see §5; the real stuck invariant).

## 3. Memory / world model (the hardest, least-built piece)

Persistent, **checkpointed alongside** emulator state (freeze the schema in P0):

- **Observed vs inferred.** Every memory entry tagged `observed` (RAM ground truth) or
  `inferred` (LLM guess); executor trusts `observed`. *(Prior art's #1 finding: every
  Claude-Plays-Pokémon failure was bad/contradictory memory treated as fact.)*
- **Tried-and-failed ledger.** Approaches that didn't work, consulted by the executor and
  by recovery so they aren't retried. Recovery must *generate alternative hypotheses*.
- **World graph.** Maps = nodes, warps = edges. **Warps give dest map id + dest warp
  index, NOT arrival x/y** — so you need a **warp table** + intra-map routing to the
  correct exit tile. Seed from a known Kanto connectivity table, reconcile with observed
  warp indices. This gates the strategist's "go to Pewter" and is the largest new build.
- **Location memory** (extends `WorldMap`/`InteractionMemory`), **persisted**.
- **Episodic log**, periodically **summarized by LunaRoute**, with a cheap
  **consistency/critic pass** that reconciles claims against current RAM before trusting.
- **Static knowledge — MANDATORY, vendored** (not "optional"):
  - **Story-gate table**, encoded as world-graph **edge preconditions**. To-Brock gates:
    `get-starter → Oak's-Parcel(Viridian Mart) → deliver-to-Oak → Route 2 unlocked
    (Viridian north guard) → Viridian Forest → Pewter → Brock`. A blind router wedges at
    the Viridian guard exactly like CPP wedged "waiting to talk to Bill."
  - **Type chart** (cheap, directly improves battle Choice).

## 4. Milestone / event detector (new named component — RAM, deterministic)

The single source of truth for progress and "done": badge flag, map id, event flags,
party species/levels, money. Drives the progress vector, strategist triggering, and
milestone completion. **Not a Noul.** Reserve Jev `Noul`/`Score` for fuzzy judgments
only ("am I making progress toward this objective", "am I ready for Brock").

## 5. Robustness — reframed (capability-first)

**Redefine "stuck."** The current/proposed invariant "no progress-vector change" is
wrong: the three biggest silent killers all keep state *changing* —
- **whiteout loop** (lose → respawn at Center, money↓, re-attempt same losing fight),
- **goal-less cross-map wandering** (position changes, goal-distance doesn't),
- **slow HP/PP decay** (party degrades with no single stuck moment).

Correct signal: **objective graph-distance not decreasing over N steps** + **setback
events** (whiteout, faint, money drop). Keep the local oscillation/loop detector too.

**Mode-gate every recovery** (the ladder must never fire blindly):
- In a **required battle** or **forced-movement/cutscene**: reload and B-spam are
  **forbidden**; the only move is advance/execute-turn/wait.
- A **required menu selection**: recovery is "make the correct selection" (P1), never
  B-out-forever.
- **Whiteout**: treat as a *setback* → forced strategist re-plan (heal + grind + re-route),
  counted as negative progress, not silently resumed.
- **Reload-on-stuck** (last resort): must restore **memory too** (else the world model
  desyncs from the reloaded position and repeats the mistake) **and** change subgoal +
  write the failed approach to the ledger. Blocked entirely mid-battle/mid-cutscene.

**Checkpointing:** `(emulator state + serialized memory)` every N steps and per
milestone. Resumable + inspectable.

**Watchdog:** **mode-aware** budgets (a battle "step" is ~35–40 frames/press × many
messages — far longer than an overworld step; a naive frame budget would kill a legit turn).

## 6. Modes, action sets, action masking

- **Overworld**: move / goto(target) / interact / advance_dialog / wait / route-to(map).
- **Dialog**: advance / (choice) select_option(N) / cancel.
- **Menu**: select_option(N) / cancel / (name-entry: its own keyboard controller).
- **Battle**: use_move(slot) [have, must be WIRED IN] / switch(mon) / use_item / run /
  catch / send-next-on-faint / tutorial-advance.
- **Shop / Poké Center**: buy(item,qty) / heal.
- **Action masking** on `context.kind`/`battle_type`: **no `run` vs trainers**, no `catch`
  vs trainers (CPP burned huge time trying to flee un-fleeable fights).

## 7. Orchestration loop

```
load(checkpoint or start_fixture); memory.restore()          # memory schema real from P0
while running and badges < 1:
    obs = perceive()
    milestones = milestone_detector(obs)                     # RAM truth
    if forced_movement(obs) or required_battle(obs):
        controllers.handle_mode(obs); continue               # never strategize/reload here
    if strategist_due(milestones, stuck, cadence):
        plan = strategist.reason(obs, memory.summary())      # LunaRoute; incl hypotheses+ledger
        memory.set_plan(plan)
    action = executor.decide(obs, memory.plan, memory)       # Jev Choice on mode's masked set
    result = controllers.execute(action)                     # deterministic (warp-settle gated)
    memory.record(obs, action, result, tag=observed/inferred)
    if due_checkpoint(): save(state, memory)
    if wedged(objective_distance, setbacks): recover(mode_gated)
report(progress_vector, milestones, cost, where_it_ended)
```

## 8. Evaluation

Milestone checklist (RAM, with step reached): got-starter → has-parcel → parcel-delivered
→ route2-unlocked → entered-forest → reached-Pewter → **beat-Brock**. Progress vector:
badges, map (graph-distance to Pewter), party levels, money, steps, wall-clock, Jev calls +
LunaRoute tokens (cost). Replayable per-step log (+periodic screenshots).

## 9. Phased build plan (revised)

- **P0 — Harness + THREE frozen schemas.** Canonical loop; **typed Plan contract**;
  **memory/checkpoint format**; **world-graph warp/edge schema**. Milestone/event detector
  (RAM). Progress vector. Stuck-detector wired in with the *new* invariant +
  forced-movement suppressor. (Won't reach far — proves harness + metrics + interfaces.)
- **P1 — Menu layer + stub strategist.** `select_option(N)`, yes/no, **name-entry
  keyboard controller**, start menu; menu_open/kind perception. Wire the scripted
  strategist so the plan→executor contract is exercised. → **got-starter, leave the lab.**
- **P2 — Battle policy, WIRED INTO THE LOOP.** Bridge `use_move` into the executor;
  add switch/item/run(masked)/catch/faint→send-next/PP-out/tutorial-advance; heal-or-flee
  `Choice`. → survive the rival + trainer + old-man-tutorial battles.
- **P3 — World graph + cross-map routing + Poké Center/Mart + heal policy + ledges.**
  "route-to(map)"; whiteout/low-HP/PP guardrails; directional passability. → reach Pewter.
- **P4 — Strategist + memory (observed/inferred + ledger + summarization/critic) +
  mode-gated recovery ladder + whiteout handling.** → self-manages long-horizon.
- **P5 — Integrate & iterate to Brock** (level-gate before the gym; fix long-tail).

Ordering fixes baked in: schemas + milestone detector + battle-wiring pulled earlier;
stub strategist from P1; stuck-invariant fixed in P0.

## 10. Known wedge catalog (design targets, from the red-team)

- **Tier 0 (recovery CANNOT fix — need P1/P2):** starter YES/NO → **nickname→keyboard
  trap**; required battles (rival, trainers, Brock's Geodude L12 + Onix L14) with
  faint/multi-mon; healing menu; Mart buy menu.
- **Tier 1 (silent derails the old detector misses):** **whiteout loop**;
  under-leveling + no-heal + **PP→Struggle→faint**; **Viridian old-man catching tutorial**
  (`battle_type==1`, blocks the tile); **cross-map wandering** (motion ≠ progress).
- **Tier 2 (map corruption/cutscene):** **one-way ledges**; cutscene graphics + forced
  movement (panic-reload risk); **warp-frame map-keying off-by-one**.
- **Tier 3:** wrong-map wandering (Route 22 optional rival); mode-blind watchdog killing a
  legit battle turn; forest-maze BFS looping.

## 11. Prior art (leverage)

- **Claude Plays Pokémon** — plan+memory scaffold; dominant failures were *memory
  correctness* + *option generation*, not pathing; ~35k actions/~140h for 3 badges with
  the LLM in the per-button loop → validates keeping the LLM OUT of the per-step loop.
- **PokeRL (arXiv 2502.19920)** — reward-hacking: heal-loops, battle-avoidance, and a
  **Charmander bias** (reward-sooner) → **pin the starter**.
- **pokegym / PufferLib** — encode loop-detection, spam penalties, visited-coordinate
  spatial memory; curriculum of concrete sequences.
- **"On getting unstuck" (LessWrong)** — re-test strong beliefs; force option generation;
  ledger tried-failed approaches. Directly shapes §3 + §5 recovery.

## 12. Validation gates (de-risk before P5)

- **Jev calibration** on this game-state distribution (we use confidence as a control
  signal — if miscalibrated, low-conf-reflect and stuck-Noul both misfire). Check against
  logged fixtures.
- **Real per-step token cost** — `step()` ships the full structured state every step;
  validate the "$0.0002/call, fine every step" assumption with actual usage.

## 12b. Recovery & durable memory (P4) — grounded in prior art

Meta-lesson from CPP / Voyager / Reflexion / Generative Agents / PokéAI: **ground truth
in the environment (RAM), store what failed durably and append-only, and make that
memory MECHANICALLY constrain the next action.** CPP failed because its memory was
model-editable, its "done" was model-asserted, and its critic was *advisory* (it told
Claude it was looping and Claude kept looping — even blacking out its team 8× believing
Mt. Moon was cleared). Voyager/Reflexion win because retry is *state-aware and enforced*.

**Our current gap:** the tried-failed ledger, the stuck detector, and the compass
route-hint are all inert or un-overridable — nothing changes the next action.

1. **Dead-end ledger (spatial, deterministic, append-only).** Key on RAM:
   `(map_id, x, y, direction) → BLOCKED | NO_EFFECT`, written by the *controller* (never
   the strategist) by comparing coords before/after a move. O(1) tile-keyed lookup (our
   RAM analog of Reflexion's episodic buffer). The model may add a reflection string but
   may **never delete** a BLOCKED edge (the CPP anti-pattern). Persist blocked edges so
   BFS treats them as non-edges.
2. **Executor mask (makes memory change behavior).** A BLOCKED `(tile, direction)` is
   **removed from the executor's Choice** at that tile; and the **compass route-hint is
   vetoed** when its direction is blocked (the immediate Viridian fix). This is the
   mechanism CPP's advisory critic lacked.
3. **Escalating recovery ladder** (stuck detector → each rung *changes* behavior):
   (1) reflex — write blocked edge, mask it, veto the hint; (2) deterministic BFS to the
   nearest unexplored frontier/exit, ignoring the hint (the "novel action" generator CPP
   lacked at Bill); (3) Reflexion-style durable reflection + strategist re-plan against
   the avoid-list; (4) Voyager-style goal escalation (abandon a repeatedly-failing
   subgoal). **Blackout guard:** any action the ledger shows failed ≥K times is
   auto-down-ranked, so "repeat the failing plan" can never be top-scoring.
4. **RAM-verifiable checklist + event-driven replan.** Strategist emits one active
   subgoal + a checklist whose items are **RAM predicates** (badge_count≥N, event_flag
   set, map_id==, item in bag); **a RAM verifier ticks them, never the model** (guards
   CPP false-completion). Replan triggers: item verified, recovery reaches rung 3, path
   fully blocked, or a **per-subgoal step budget** exceeded (the hard cap that would have
   broken CPP's 8× blackout loop). Strategist may only *append* to the avoid-list and
   *rewrite the checklist* — never delete grounded memory.

Build order: **#2 executor-mask + route-hint veto and #1 ledger first** (fix the Viridian
dead-end class), then #3 recovery ladder, then #4 checklist + replan.

## 12c. Needs arbitration (built) — the "hierarchy of needs"

Chosen mechanism (from a 7-approach review — subsumption, behavior trees, HTN, options,
Voyager curriculum, PokéAI, utility/Maslow AI): a **bucketed priority stack (subsumption
for the tiers) + utility/hysteresis within/between buckets**, framed as **options**
(each need = ⟨initiation predicate, controller policy, RAM termination⟩), with **HTN
living inside PROGRESS** (the story-gate chain).

- **Stack:** SURVIVE(100) > BATTLE(90) > READINESS(50) > PROGRESS(10); the agent pursues
  the highest ACTIVE need each step (a higher need pre-empts a lower one). Battle/dialog
  mode-dispatch sits above and masks the stack (can't flee a trainer / a cutscene).
- **RAM owns truth, strategist owns parameters** (the key CPP guard): `needs.py` predicates
  decide activation/termination from RAM; the strategist only sets goal_map, level_target,
  thresholds, and re-plans on escalation — it can never assert a need done.
- **Anti-thrash:** SURVIVE uses **dual-threshold hysteresis** (latch at HP<low_hp, release
  only at HP≥heal_hp — "heal up, don't flip"); READINESS has a hard ceiling (level≥target).
- **Anti-starvation:** a READINESS **step budget** yields to PROGRESS so grinding can't loop
  forever (the CPP-blackout cap).
- **Ledger/recovery plug in below:** the dead-end ledger is the intra-option action mask;
  the recovery ladder is an escalating override above arbitration.

Status: arbiter built + wired (READINESS grind-in-place ↔ PROGRESS route ↔ BATTLE), CLI
`--level-target`. Remaining: SURVIVE's heal *execution* (find the city Poké Center door →
nurse → heal is a warp/menu sub-pipeline; today SURVIVE latches + halts pushing + notes),
recovery **rung 3** (strategist re-plan on persistent stuck), and flee-when-low in wild battles.

## 13. Open decisions (reduced)

- **Starter: PINNED** to Squirtle or Bulbasaur (both strong vs Brock; Charmander is a
  known trap). Pick one — recommend **Squirtle**.
- **Static knowledge:** inject fixed facts (story gates, type chart, gym/city locations),
  discover the rest. (Resolved: inject.)
- **Autonomy bar:** is checkpoint-reload allowed at all (given it re-arms wedges unless
  mode-gated + memory-restored)? Or no-reload autonomy as the bar?
- **Grinding:** allow forest grinding to a **level gate** before Brock, bounded by
  target-level + step cap + loop guard? (Recommend yes, bounded.)
- **Strategist cadence:** event-driven (milestone/stuck/whiteout) + a slow heartbeat?
