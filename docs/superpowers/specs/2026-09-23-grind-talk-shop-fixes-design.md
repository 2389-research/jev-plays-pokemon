# Grind-in-place, NPC rotation, verified purchases — Design

**Source run:** `runs/brock-goals2-20260923` (resume of shopfix-verify2; Viridian Forest → Pewter → gym → Route 2). User report: "unwilling to go back into the forest to grind", "stuck in a dialogue loop in the gym for a few hundred steps", "never saw it actually buy a Poké Ball".

## 1. Root causes (verified from the log + replaying saved states)

**R1 — grinding has no executor once you've arrived.** A grind step (`action map 13, level>=13`) compiles to a TRAVEL directive to map 13 with a level success predicate. On map 13 `_default_target` returns None ("arrived"), so the agent stands still: 120 steps at Route 2 (8,0), **0 battles**; the stuck detector wedges it every 6 steps and L1 re-adds the identical step (~25×, steps 504–999). Same pattern in the forest earlier (steps 66–96 of brock-goals). The only grass behavior (`farm-exp`) is a routing *policy* that weaves toward a destination, and only under `--pather policy`.

**R2 — an unproductive conversation is repeated forever.** Target "Brock"; Gen 1's sprite table names Brock "Super Nerd", so no name matched and `select_npc` fell back to the nearest person — the Gym Guide (7,10) — and cached that pick. It talked to the Guide ~30× over 280 steps (steps 612–893); the block budget is frozen during conversations (F5), so nothing wedged sooner. It never tried the Jr. Trainer or Brock.

**R3 — purchases are never verified; a satisfied buy is repeated.**
- `shop_buy` returns `ok: True` once it answered YES — it never checks the bag. At Pewter Mart money was **$177** (Potion $300): 9 × "ok", 0 Potions (replayed from `states/map56_step383.state`).
- In brock-goals the Antidote buy **succeeded 16×** ($1,600): after a buy the clerk's "anything else?" reopened the counter, and the menu path re-bought because the directive's `has_item` success is only checked in `_manage_directive`, which the menu/dialogue paths skip. (e2c0c11 only stopped re-buys for *wedged* steps.)
- L1 never sees money, so it can't plan within budget.

**R4 (strategy, not a harness bug) — no Poké Balls, ever.** Items across all four runs: Parcel, Antidote only. The notepad asserts "got Pokédex + 5 Poké Balls" (false — Oak's balls require talking to him again); L1 trusted its notepad over ITEMS and never set a catch goal. Money is now too low for balls ($200).

## 2. Fixes

**F1 — grind in place (R1).**
- `_default_target`: when the directive's success has a `level` clause, the player is on the target map (or it has none), and success is unmet → `{"kind": "grind"}` instead of None.
- Pure `routing.grind_step(tiles, terrain, pos, last_dir, avoid) -> Direction | None`: on grass → keep `last_dir` if the next tile is walkable grass, else turn to a walkable grass neighbor (prefer not reversing), else reverse; off grass → first step toward the nearest reachable grass tile (`policy_first_step(..., "shortest")`); no reachable grass on the map → None.
- A grind target is never "reached" (held until success ends the directive).
- **Stuck accounting while grinding:** a grind step's stuck verdict is ignored while a wild battle ended within `GRIND_ENCOUNTER_WINDOW = 60` steps, or grinding started < 60 steps ago; after that, normal accounting resumes → wedge with `wedge_reason = "grind: no wild encounter in 60 steps on <map>"` (or `"grind: no reachable grass on <map>"` when `grind_step` returns None) so L1 moves the grind elsewhere.

**F2 — rotate after an unproductive conversation (R2).**
- `_approach_npc` records the NPC it interacts with (`target["talking_to"] = [x, y, map]`, `target["talk_step"]`).
- On the next navigate step, if a dialogue happened after `talk_step` and the directive's success is still unmet → add that NPC to `target["tried"]`, clear `picked`, emit `npc_rotate`.
- `select_npc(..., tried=...)` removes tried NPCs from the pool (by position on this map); if every candidate has been tried → wedge the step with `wedge_reason = "talked to everyone here (<sprites>); none satisfied <criterion>"`.

**F3 — verified, non-repeating purchases (R3).**
- `_maybe_shop`: if the directive's success already holds → close the counter, don't buy (extends e2c0c11's done/wedged rule to "active but satisfied").
- After `shop_buy` returns ok, compare the item's bag quantity (and money) before/after; unchanged → treat as failure `"purchase did not go through (money $<m>)"` → wedge + L1 (same path as not-sold).
- `game_signals` gains `money` (L1 sees it every review).

**F4 — ground-truth guidance (R4), prompt only.** DECIDE/BRAINSTORM: "ITEMS and MONEY are ground truth — if your NOTEPAD disagrees, trust ITEMS and fix the notepad. Catching needs Poké Balls in ITEMS; buying needs MONEY." No harness catch logic (L1 decides whether to catch).

## 3. Tests (TDD) + live verification
- Unit: `grind_step` (on grass straight / turn / reverse, off-grass approach, no grass → None); `_default_target` → grind; stuck exemption window + wedge reason; NPC rotation (dialogue after interact + unmet → tried/rotate; exhausted → wedge); `select_npc` tried filter; shop: satisfied → close, no buy; ok-but-bag-unchanged → wedge with money; `game_signals.money`.
- Live (from saved states of brock-goals2): Route 2 grind (≥1 wild battle within 60 steps, no wedge while battling); Pewter Gym at step 612 (moves off the Gym Guide within one conversation); Pewter Mart step 383 ($177): one failed buy, wedged with the money reason, L1 re-plans.

## 4. Out of scope
Harness-driven catching; sprite-name aliases for gym leaders (rotation makes them reachable); `--pather policy` changes.
