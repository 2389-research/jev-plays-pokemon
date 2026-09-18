# Spec: robust planner/executor architecture for the Pokémon agent

Status: design (written via superpowers brainstorming + 3 research passes). Successor to
the ad-hoc control flow in `agent/reason_loop.py`. Companion to
`docs/agent-to-brock-spec.md` (the to-Brock roadmap); this spec defines the *control
architecture* that roadmap runs on.

## 1. Problem

The two tiers exist but the link between them is broken, and hand-coded heuristics have
grown up as *rivals* to both:

- **The plan has no teeth.** `LunaRoute` (planner) emits an `AgentPlan` whose useful
  fields (`mode_hint`, `target`, `option_bias`) are **dead** — `TypeSafeReasoner.step()`
  dumps the plan into the prompt blob and never reads them to change what the executor
  (Jev) can do. The executor chooses from a fixed option set and treats the plan as a
  string it may ignore.
- **Deterministic controllers bypass both tiers.** `reason_loop.py` accreted competing
  controllers — travel autopilot (route-hint), needs arbiter routing, frontier-escape
  recovery, anti-seam-oscillation, target-filtering — that *steer the character directly*,
  outvoting the executor and the plan. Result: the classic "planner and executor
  disagree / executor ignores the plan" failure, in the form "hand-coded controllers
  seize the wheel." This is why the agent drifts, oscillates at ledges, and can't be
  told "go talk to that NPC" and have it stick.

**Goal:** one coherent hierarchy — the planner sets a concrete directive; everything
below (executor + deterministic controllers) *serves that directive*; the executor is
mechanically bound to it, not asked nicely.

## 2. Decision (from the user): planner computes targets, executor is a servo

The planner does the strategic + spatial reasoning and hands down a **concrete target**.
The base-level work is mostly **deterministic** (pathfinding to the target); the
calibrated executor (Jev) is invoked only for the base-level *decisions* the
deterministic layer genuinely can't resolve (which menu option, which battle move,
disambiguation), and even then its options are shaped by the directive. So:

- **LunaRoute (planner):** rare, deliberative. Chooses intent + computes the concrete
  target + the success predicate. Owns replanning.
- **Deterministic controllers (the servo):** BFS pathing to `target`, menu operation,
  battle sequencing, dialog advance. Do the mechanical execution of the directive.
- **Jev (executor):** the calibrated decision-maker for residual base-level choices
  under the directive, with its option set masked/biased by the directive. Not a free
  agent; a bounded chooser.

Theory backing (research): options framework (⟨initiation, policy, termination⟩), 3T
"executive" middle layer, BDI **bold commitment** (don't re-deliberate every step in a
mostly-static world), SayCan (LLM proposes × calibrated feasibility scores), LLM+P
(hand geometric search to a deterministic solver — our BFS), Inner Monologue (feed
*why* back on replan), Voyager (`tried_failed` feeds the next task).

## 3. The core contract: the `Directive` (an option with teeth)

One **active** `Directive` at a time (near-bold commitment), with a small **stack** only
for suspensions (a higher-priority need preempting a lower one). Fields:

```
Directive:
  intent:  Enum   # travel | talk_to | grab_item | enter | heal | grind | shop | battle
                  # (closed enum — selects the executor's action repertoire)
  target:  {kind, map, x, y, id} | None   # concrete + typed; required for target-bearing intents
  success: Predicate   # MACHINE-CHECKABLE termination (the option's β / BDI drop-condition)
                       # e.g. {on_map: 2} | {party_size: ">=1"} | {talked_to: npc_id}
                       #      | {flag: parcel_delivered} | {hp_frac: ">=0.8"}
  allowed_options: list[str] | None   # optional explicit whitelist for the executor's Choice
  option_bias:     list[str]          # options to prefer (SayCan-style prior)
  reason: str                          # provenance / why (for logs + replan feedback)
```

`success` is the single most important currently-missing field — a directive without a
checkable termination is why nothing ever *commits*. The strategic wrapper
(`AgentPlan`: mission / milestone / hypotheses / `tried_failed`) stays, but it now carries
**exactly one active `directive`** the executor obeys, plus the suspension stack.

## 4. Component design (isolated units, clear interfaces)

- **`Directive` / `AgentPlan`** (`agent/plan.py`): the typed contract above. Pure data.
- **RAM verifier** (`games/pokemon_red/needs.py` + a new `predicates.py`): evaluates a
  `Predicate` against RAM every step. Deterministic. Owns *termination detection*. Reuses
  existing needs predicates; adds event-flag / map / party / hp / talked-to checks.
- **NeedsArbiter** (`agent/needs_arbiter.py`, refactored): the subsumption/priority layer.
  It no longer steers navigation; it **sets the directive's `intent`** (SURVIVE→`heal`,
  READINESS→`grind`, PROGRESS→`travel`, in-battle→`battle`) and hands that to the planner.
  One source of truth for "what are we doing."
- **Planner** (`agent/planner_llm.py`, wraps LunaRoute): given the arbiter's intent +
  RAM state + memory (world graph, `tried_failed`, avoid-list), **computes the concrete
  `target` and `success`** and emits the `Directive`. Runs only on a replan trigger.
- **Executive / loop** (`agent/reason_loop.py`, rewritten as the 3T executive): holds the
  active directive + stack, checks `success` each step via the verifier, dispatches to the
  controllers or the executor, and fires replan triggers. **Single source of truth = the
  active `Directive`.** No controller may pick a destination the directive didn't set.
- **Deterministic controllers** (existing, re-parented): `Navigator`/BFS (now serves
  `directive.target`, ledge-aware directed edges), `menus`, `battle`, dialog-advance.
- **Executor** (`agent/typesafe_reasoner.py`, refactored): builds its `Choice` **from the
  directive** (see §5), returns the residual base-level decision.

## 5. How the executor/servo consumes the directive (teeth)

Per-step dispatch in the executive:
1. **Check `success`** (RAM). If met → directive done → pop/replan. If the active mode is
   `battle`/`dialog`/forced-movement → hand to that controller (mode mask, as today).
2. **Deterministic first (servo):** if `intent` is target-bearing (`travel`/`talk_to`/
   `grab_item`/`enter`/`heal`/`shop` — i.e. all intents *except* `grind` and `battle`,
   which carry no `target`) and BFS (ledge-aware, directed) yields a next step toward
   `directive.target`, **take it** — no model call. This is LLM+P: the classical solver
   does the geometry. (`battle` is handled by the mode controller in step 1; `grind` has
   no fixed target — it routes to a grind area via a `travel` sub-directive and otherwise
   falls to the executor for encounter/move choices.)
3. **Executor (Jev) only for residual decisions** — when the deterministic layer can't
   resolve it (a menu/battle choice, a disambiguation among candidate targets, BFS has no
   path): build the `Choice` **from the directive**:
   - **Mask (initiation set):** restrict `_KIND_CRITERIA` by `intent` (e.g. `talk_to`
     drops raw `move_*`, keeps `goto_<target>` + `interact`); honor `allowed_options`.
     Generalizes the proven `blocked_dirs` masking from walls to *directive-illegal actions*.
   - **Inject the target as a first-class option** whenever `directive.target` is set, so
     "pursue the directive" is a button the model can press, not a hope.
   - **Bias = SayCan product:** combine the model's per-option probability with a cheap
     affordance prior (`option_bias` + target-distance) → argmax of the product.
   - **Surface compact plan context** in the executor state: `{intent, target, success,
     tried_failed, steps_on_directive}` — not the whole plan dump.

## 6. Termination & replanning (split ownership)

- **RAM owns detection** (cheap, every step): the `success` predicate + impossibility
  checks. Keep the LLM out of this loop.
- **Planner owns the next directive**, invoked *only* on a trigger, and fed **why**
  (Inner Monologue) + `tried_failed` (Voyager) so it never re-issues the failed directive.
- **Replan trigger set (few, explicit — bold commitment):**
  1. `success` predicate met.
  2. Directive impossible (BFS exhausts / N confirmed blocks on the goal edge).
  3. `StuckDetector` fires → route to **replan**, not to a rival frontier heuristic.
  4. Sustained low executor confidence over K **executor invocations** (not loop steps —
     the executor is only called for residual decisions, so this counter advances only when
     Jev is actually consulted). The cautious safety valve.
  5. A higher-tier need preempts (SURVIVE latches) → suspend current directive on the stack.
  6. **Arbiter intent no longer matches the active directive's intent** (a non-preempting
     change, e.g. READINESS→PROGRESS when the grind budget expires) → replan for the new
     intent. This covers *downgrades* that #5 (preemption/suspension) does not.
  Everything else → **do not replan** (keeps the planner rare and cheap).

## 7. What gets removed / re-parented

The competing heuristics collapse into "serve the directive":
- **Travel autopilot / route-hint** → only the `travel` intent's deterministic step,
  toward `directive.target`; never fires under a `talk_to`/`grab_item` directive.
- **Frontier-escape recovery** → becomes replan trigger #3 (hand up to the planner) plus
  a bounded deterministic escape *only while consistent with the directive*.
- **Anti-seam-oscillation / target-filtering / `effective_goal_map`** → subsumed: the
  directive's `target` is the only destination; BFS routes to it.
- **`NeedsArbiter.effective_goal_map()`** → removed; the arbiter sets `intent`, the
  planner sets the target.
Keep (as controllers/data): `Navigator` (+ ledge directed edges), `menus`, `battle`,
`world_graph`, `AgentMemory` (world map, dead-end ledger, `tried_failed`), `StuckDetector`
(re-pointed at replan), the RAM predicates.

## 8. Loop pseudocode

```
directive = None; stack = []
each step:
    obs = perceive(); ram = read_state()
    intent = arbiter.intent(ram)                     # subsumption: survive>battle>grind>travel
    if directive is None or replan_triggered(directive, ram, intent):
        push/suspend as needed (higher need preempts)
        directive = planner.plan(intent, ram, memory) # LunaRoute: concrete target + success
    if verifier.met(directive.success, ram):
        directive = stack.pop() or None; continue
    if mode_controller_applies(ram):                  # battle/dialog/forced -> controller owns step
        act = controller.step(directive); 
    elif step := navigator.next_toward(directive.target):  # deterministic servo (BFS, ledge-aware)
        act = step
    else:
        act = executor.choose(directive, obs)         # Jev: masked+biased residual decision
    result = execute(act); memory.record(result); checkpoint_if_due()
```

## 9. Testing

- **Directive/predicate unit tests** (MemFake, no ROM): each `Predicate` evaluates
  correctly against RAM; `success` fires exactly when the RAM condition holds.
- **Executor teeth** (fake Jev client): given a `talk_to` directive, `move_*` are masked
  and the target option is injected; SayCan bias picks the directive-consistent option.
- **Executive dispatch** (FakeEmulator + stub planner): deterministic step taken toward a
  target; replan fires on success/stuck; higher need suspends via the stack.
- **Live, fixture-guarded** (ROM): `talk_to` reaches+talks; `travel` crosses Route 1
  (ledge-aware) without drift; `battle` still won.
- Preserve the existing 122 tests; refactor keeps behavior where already correct.

## 10. Phasing (each independently landable, tests green)

1. **Directive + `success` predicates + RAM verifier** (unlocks commitment). *(§3,§6)*
2. **Executor derives its Choice from the directive** (mask + inject target). *(§5)* — teeth.
3. **Re-parent controllers under the directive; single source of truth**; delete the
   rival heuristics. *(§7)* — **Land the `StuckDetector`→replan trigger (§6 #3) in this
   same phase** (not in phase 4), since this phase deletes frontier-escape recovery which
   currently consumes `StuckDetector`; otherwise stuck handling is a no-op between phases 3
   and 4.
4. **Remaining replan triggers → planner, with feedback + `tried_failed`.** *(§6)*
5. **SayCan Say×Can biasing** using the confidences already emitted. *(§5)*

## 11. Open decisions / risks

- Directive granularity: confirmed **one active directive + suspension stack** (not a full
  per-step goal tree). Revisit only if a case needs >1 concurrent commitment.
- `heal`/`shop` intents depend on Poké-Center/Mart door + nurse/clerk sub-pipelines (still
  unbuilt); spec covers the contract, execution is a follow-on controller.
- Risk: over-masking the executor's options could trap it; the low-confidence replan
  trigger (#4) is the safety valve, and BFS-impossible (#2) escalates rather than loops.
- Migration risk: `reason_loop.py` is heavily accreted; rewrite it as the executive behind
  the same `ReasoningLoop.step_once()` interface so `run_agent.py`/tests keep working.

## 12. Sources
3T/executive: Gat 1998. BDI/bold-commitment: Rao & Georgeff 1991, Kinny & Georgeff 1991.
HTN: Georgievski & Aiello 2015. Options/HRL: Sutton, Precup & Singh 1999. LangGraph
Plan-and-Execute. SayCan: Ahn et al. 2022. Inner Monologue: Huang et al. 2022. LLM+P:
Liu et al. 2023. Voyager: Wang et al. 2023.

---
Next steps (post-compaction): run the spec-document-reviewer loop, get user sign-off, then
invoke writing-plans to produce the phased implementation plan (§10).
