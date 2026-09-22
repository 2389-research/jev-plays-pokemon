# Battle Subsystem: layered objectives, actions, and item/catch macros — Design

**Goal:** Give the agent a real battle brain and the menu mechanics it lacks — capturing Pokémon, using items, running, and buying at a Mart — structured as layers that mirror the overworld (L1 sets goals → a battle-tactical layer sets the battle objective → Jev executes per turn → deterministic macros do the keypresses).

**Architecture:** On the battle-start edge, a `battle_L2` reads L1's standing goals + the encounter and sets ONE cached battle objective (CAPTURE / GRIND-EXP / ESCAPE / SURVIVE). Each turn, Jev chooses a concrete action toward that objective (fight-move / throw-ball / use-item / run). Each action is a deterministic, RAM-verified macro built on the existing `menus.py` primitives. SHOP gets a parallel `buy` macro. Decisions are LLM/Jev; mechanics are deterministic.

**Tech stack:** Python; existing `battle.py` (`in_battle`, `fight_menu_showing`, `use_move`), `menus.py` (`select_option`, `answer_yesno`), the `_battle_turn` dispatch in `reason_loop.py`, and RAM readers for party/items/money/HP.

---

## 1. Background & current state

- Battle today is **fight-only.** `reason_loop._battle_turn` (dispatched at `step_once` when `ctx["in_battle"]`) advances intro text, then when the FIGHT menu is up calls `battle_agent.choose_move` + `battle.use_move`. There is **no ITEM, no RUN, no catch.** A wild battle can only end by fainting the enemy (or the player).
- **No item/catch/shop task macros exist.** `menus.py` has only generic primitives (`select_option(index)`, `answer_yesno`, `advance`, `cancel`). The `SHOP` intent is target-bearing (routes to the Mart) but has **no buy executor** — it reaches the counter and flails.
- **Live menu handling is per-step Jev.** The flow router sends `menu.open → Jev`, which picks options one step at a time with no notion of a multi-step task ("buy 3 Potions"). Healing works reliably today because it is expressed as a single decision (a `HEAL`→talk-the-nurse directive with `done_when hp_frac>=0.95` + counter-bump/advance-dialog handling) rather than Jev flailing at the counter — the same *macro-like reliability from one decision* we want here. Note the nurse is a yes/no; the Mart `buy` flow (BUY→list→quantity→YES) is materially more complex and is genuinely new mechanics, not "already solved."
- L1 already reasons correctly about these needs ("catch a Pidgey for backup", "buy Potions before the gym", "grind to L12"); it just has no mechanism to carry them out.

**Design principle (validated on navigation):** separate the DECISION (what/when — L1/Jev) from the MECHANICS (the exact keypress sequence — deterministic macro). Per-step LLM choosing through a fixed menu flow is fragile; a macro parameterized by a single decision is reliable.

## 2. Layered architecture (mirrors the overworld)

| Overworld | Battle | Cadence |
|---|---|---|
| L1 standing objectives | **L1** standing goals ("acquire a Pidgey", "grind to L12", "conserve balls/potions") | per-plan, infrequent |
| L2 proposer (portal/tile) | **battle_L2** (the battle OBJECTIVE) | **once per battle** (+ re-eval on triggers) |
| Jev (per-step policy) | **Jev** (per-turn action toward the objective) | per turn |
| pathfinder / servo | **macros** (throw_ball / use_item / run / use_move / buy) | mechanics |

### 2.1 What fires `battle_L2` (control flow — the crux)
`battle_L2` runs **once on the battle-start transition** (`in_battle` false→true, detectable by caching the last `in_battle` on the loop — a new instance attr, absent today), NOT every turn. **Ordering matters:** the edge-detect + objective-set must sit ABOVE `_battle_turn`'s intro-text / `fight_menu_showing` early-return, so the objective is set at the start of the encounter (during intro text) — not only after the menu first appears, which a very short intro could skip. It reads L1's standing goals + the encounter (species, level, is-trainer-vs-wild, our party/HP/items) and sets ONE **battle objective**, cached for the fight. It re-evaluates ONLY on real triggers:
- our active mon drops to critical HP → re-eval (may flip to SURVIVE or ESCAPE);
- (CAPTURE) the target reaches a catchable HP band → Jev throws.

So "when do we catch" is: L1 set the standing goal → `battle_L2` recognizes the encounter matches (right species, slot free, wild, catchable) → objective = CAPTURE. No always-on poll.

### 2.2 Battle objectives (the closed set `battle_L2` chooses from)
- **GRIND-EXP** — faint the enemy for XP (the current default; a trainer battle is always this or SURVIVE).
- **CAPTURE** — catch a wild Pokémon (only when L1 wants this species/any, a party/box slot is free, and it's a wild battle).
- **ESCAPE** — run (wild only; when the fight isn't worth it / risk too high).
- **SURVIVE** — stay alive: heal/switch (when our mon is endangered but we must win, e.g. a trainer).

### 2.3 Jev per-turn action selection (given the objective)
Each turn Jev maps (objective, live state) → one action:
- CAPTURE → target too healthy? a weakening move (low-damage / status) : `throw_ball`.
- GRIND-EXP → best damaging `use_move`.
- ESCAPE → `run`.
- SURVIVE → `use_item(Potion)` or switch, else a defensive move.
This extends today's `battle_agent.choose_move` into a `choose_action` returning a typed battle action.

## 3. Menu mechanics (exact, from pokered) → macros

All macros are deterministic keypress sequences on top of `menus.py`, each with a **RAM-checkable outcome**.

- **`throw_ball(ball)`** — wild battle menu is a 2×2 `FIGHT PKMN / ITEM RUN`. Sequence: DOWN (FIGHT→ITEM) → A (open bag) → cursor to the ball → A (throws). Outcome: ball count −1; on success party count +1.
- **`run()`** — battle menu → cursor to RUN → A. Outcome: `in_battle` false (if it succeeds; may fail and cost the turn).
- **`use_item(item, target)`** — battle: ITEM → bag → item → target mon; overworld: START → ITEM → bag → item → target. Outcome: item count −1 and the effect (e.g. HP up for a Potion).
- **`use_move(slot)`** — exists today; becomes the GRIND-EXP / weaken action.
- **`buy(item, qty)`** (SHOP executor) — talk clerk → BUY → item list → item → quantity (UP ×qty) → YES → B → SEE YA. Outcome: item count +qty; money − qty×price.

**Fixed vs. looked-up indices (important):** a menu's *structural* indices are constant and encoded in the macro, pinned by a fixture test (§5) — the 2×2 `FIGHT PKMN / ITEM RUN`, the `BUY/SELL/SEE YA` order, the `YES/NO` order. But the index of a *specific item* (the ball in `throw_ball`, the Potion in `use_item`, the item in `buy`) is **data-dependent** — bag order changes and lists can scroll past one screen — so it MUST be computed at runtime from the live bag / shop contents (`read_items` and the on-screen shop list), never hardcoded. Macros therefore take an item *name* and resolve its current index each call.

## 4. Integration & new capabilities

- **`_battle_turn`** becomes: detect battle-start → run `battle_L2`, cache the objective; each turn → `Jev.choose_action(objective, state)` → dispatch the chosen macro. The current fight-only path is the GRIND-EXP branch.
- **New `CATCH` capability** — a battle objective, not a target-bearing overworld intent; L1 expresses the desire as a standing goal ("acquire species X / any"), `battle_L2` turns an encounter into a CAPTURE objective. (No `CATCH` overworld Intent needed; it lives in the battle layer.)
- **`SHOP` executor** — the `buy` macro, invoked when a SHOP directive has reached the Mart and opened the counter. L1 already emits SHOP steps ("buy Potions", `has_item:Potion`).
- **Decision inputs** — `battle_L2` and Jev read: encounter species/level, wild-vs-trainer, our party (species/level/HP/status/slots free), items (balls/potions), and L1's standing goals (a compact "battle_goals" field on the plan/context).

## 5. Testing (per layer, isolation — the discipline that worked)

Every macro has a RAM-checkable outcome, so tests are deterministic assertions, not fuzzy. Fixtures are save states captured once (runs already checkpoint): a wild battle, a Mart counter, a low-HP party.

1. **Macros (deterministic, ROM-guarded, no LLM — like `test_gate_crossing`):**
   - `throw_ball` from a wild-battle fixture → ball count −1; on catch party +1.
   - `buy(Potion, 3)` from a Mart fixture → bag +3 Potions; money − 3×price.
   - `use_item(Potion)` from a low-HP fixture → HP increased.
   - `run()` from a wild-battle fixture → leaves battle (or costs a turn deterministically).
2. **battle_L2 (objective selection) — calibrated eval:** fixtures of (encounter, L1 goals, party/items) → assert the objective. *Pidgey + want-Pidgey + slot free + wild → CAPTURE; trainer → GRIND/SURVIVE; near-faint → ESCAPE.*
3. **Jev (per-turn action) — calibrated eval:** (objective, state) → assert the action. *CAPTURE + target full HP → weaken; CAPTURE + target red HP → throw.*
4. **Regression:** a wild-battle fixture where CAPTURE is set → the loop throws a ball within N turns and the ball count drops (catches the "flails in the menu" failure).

## 6. Phasing (each a shippable, tested slice)

1. **Mart `buy` macro + SHOP executor** — unblocks the *current* run (L1 is already asking for Potions); simplest to fixture and test; no battle changes.
2. **`throw_ball` / `use_item` / `run` macros** — deterministic, RAM-checked in isolation.
3. **battle_L2 objective layer + Jev `choose_action`** — wire the layers into `_battle_turn`; GRIND-EXP preserves today's behavior.
4. **CAPTURE end-to-end** — L1 standing goal → battle_L2 CAPTURE → Jev weaken/throw → macro; live-verify a catch.

## 7. Open questions

1. **How L1 expresses battle goals** — a structured `battle_goals` field (e.g. `{"catch": ["Pidgey"], "conserve": ["Poke Ball"]}`) on the plan vs. free-text the battle_L2 parses. Recommend structured, small.
2. **Weaken policy for CAPTURE** — a fixed HP band + status preference, or a Jev calibrated choice. Start fixed; make it Jev if needed.
3. **Balls/items in RAM** — RESOLVED: the readers already exist and are sufficient. `game_state.read_items` (WBAGITEMS 0xD31E → ordered `{item, qty}`), `read_money` (BCD 0xD347), `read_party`, `read_battle` (enemy species/HP/level), and `predicates` already supports `has_item` / `money`. No new readers needed — planning should budget zero for them; the macros both *drive* menus via these (runtime item-index lookup, §3) and *verify* outcomes via them.
4. **Safari Zone** — a different battle menu (no FIGHT); out of scope for the Brock run, note for later.
