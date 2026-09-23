# Interaction Reliability Fixes — Design

**Goal:** Make "walk up to someone and talk to them" reliable, so the agent's own plans (e.g. *deliver Oak's Parcel*) are executed instead of being defeated by the layers beneath them. Five root causes, found and reproduced from recorded runs, each fixed at its source with a failing test first.

**Context:** This project is a Claude/Gemini-Plays-Pokémon-style run: L1 plans its own goals (Orrery KG + portal graph), L2 proposes targets, the executive/servo/controller carry them out. In the incident below L1 planned correctly every time; the failure was entirely in the harness.

---

## 1. Incident

- **`runs/brock-run-20260922-1839`** (1200 steps): got Oak's Parcel at step 257, then spent ~700 steps oscillating Lab ↔ Route 1 ↔ Viridian and wedged 107 steps. Never delivered.
- **`runs/brock-continue-20260922-2217`** (resume of the above): delivered at step 235, ~130 steps after first reaching the lab; L1 re-planned six times *during* Oak's delivery cutscene (including "pick up Oak's Parcel from the Viridian Mart" after handing it over).
- The parcel gates Pewter (the old man blocks north Viridian until it's delivered), so this blocks the whole run.

## 2. Root causes (all reproduced; evidence cited)

### R1 — `interact` is a 1-frame A tap that can alias with the game's joypad sampling
`ActionController._press` calls `emu.press(A)` (PyBoy `button()`, auto-released next tick) then ticks 24. Gen 1 samples joypad once per overworld-loop iteration; at a fixed action spacing the 1-frame tap can land on a non-sampled frame **every time**.
- *Repro:* from `runs/brock-run-20260922-1839/states/map40_step586.state` (at (5,3) facing Oak, parcel in bag): 1-frame taps never open Oak's text at even frame offsets, open on the 2nd tap at odd offsets; a single **8-frame hold opens it at every offset** (6 frames fails there; a different state needed only 3).
- *Effect:* the 1839 run pressed A **107×** at (5,3) facing Oak with zero dialog.

### R2 — `read_npcs` reports sprites the game has hidden
`game_state.read_npcs` skips only empty slots (picture id 0). Gen 1 hides "missable" objects via a bitfield (`wMissableObjectFlags` @ `0xD5A6`) indexed through the per-map list `wMissableObjectList` @ `0xD5CE` (pairs `(sprite slot, missable index)`, `0xFF`-terminated) — the slot stays populated.
- *Verified phantoms (RAM ground truth):* Pallet intro Oak (8,5); Lab second Oak (5,10); Lab Poké Balls (7,3),(8,3) (taken); Lab Blue (4,3) (left); Viridian awake old man (17,5).
- *Effect:* phantoms become approach targets, `facing_sprite`/`can_interact` hits, and BFS obstacles.

### R3 — the executive observes a half-completed warp
Frame-by-frame through Pallet → Oak's Lab door: on frame 18 the coords update to the door tile (12,11) **and** `wCurMap` flips to 40 on the same frame; the coords then stay at (12,11) and the sprite table stays Pallet's for **33 frames**, settling to (5,11) + lab sprites at frame 51. `ActionController._move` returns as soon as `_pos` (x, y, map) changes — i.e. at frame 18 — so the next observation is torn (map = Lab, coords/NPCs = Pallet). The reviewer confirmed two other transition shapes: **leaving** the lab, the map flips at frame 3 with coords still (5,11) (= the pre-step tile), switching to Pallet at frame 36, then the game auto-walks the player one tile with `wJoyIgnore = 0xFC` until ~frame 66; **map-edge connections** (Pallet↔Route 1, Route 1→Viridian) change map, coords and sprites on one frame and are already final.
- *Effect (steps 102–104 of the continue run):* L2 proposed `approach_npc Oak` on the torn frame; `_approach_npc` cached `picked=[8,5]` (the hidden Pallet Oak). Because `wCurMap` had already flipped, `_current_target`'s "re-propose on map change" guard saw no change, the target was held, and in the settled lab the locality tracker chose the sprite nearest (8,5) — the **invisible taken Poké Ball at (8,3)**, which it walked to and poked 9× (32 such steps in the 1839 run).
- **Contributing:** `_approach_npc` tracks `picked` by locality *before* matching the requested name, stores no map id, and considers `kind == "item"` sprites when looking for a person.

### R4 — the stuck detector's objective state is never reset per plan step
`self.stuck = StuckDetector()` is created once; `_commit_directive` resets eight counters but not the detector. `_best_dist` is therefore the minimum map-hop distance **ever** seen across all steps, and `_wedge` carries over. Once any past step reached distance 0, later steps can't register "getting closer"; after 40 non-improving steps `no_objective_progress` fires **every step**, so each fresh step (which starts with `_blocked_for_n = 0`) wedges after exactly `BLOCK_TRIGGER = 6` steps.
- *Effect:* continue-run attempts at 155 and 168 wedged at (5,4) — one tile from Oak — exactly 6 steps after the step was committed.
- *Correction to an earlier claim:* `objective_distance` is already measured against the **directive's** `target_map`, not the CLI `--goal-map`; the defect is the missing per-step reset.

### R5 — the executive acts, plans, and wedges during scripted sequences
Dialogue steps return early via `_advance_dialog`, but `_finish` still updates the stuck detector, and its position-oscillation check (≤2 tiles over 6 steps) counts standing still in a conversation as `local_loop`. Between text boxes the screen briefly shows no text (steps 196, 200, 205, 212, 217, 227, …), the flow router reports "overworld", and the full nav path runs: `_manage_directive` wedges the step on the accumulated `_blocked_for_n`, L1 re-plans, and move actions are issued mid-cutscene. During the rival's scripted entrance `wJoyIgnore` (`0xCD6B`) is `0xFF`/`0xFC` (input ignored) yet the agent kept issuing moves.

## 3. Design

### F1 — hold buttons long enough to register (R1)
`actions/controller.py` `_press`: replace the 1-frame tap with `hold(button)` → tick `PRESS_HOLD_FRAMES` → `release(button)` → tick `PRESS_SETTLE_FRAMES` (keep 24). `PRESS_HOLD_FRAMES = 10` (8 is the measured minimum in the worst state; 2 frames margin). Applies to `InteractAction`, `AdvanceDialogAction`, `PressAction`.
- *Deviation (implemented):* **direction** presses stay 1-frame taps — in the overworld a held direction would start a walk. Menu cursor moves through `PressAction(direction)` could in principle alias like A did; none are emitted today (only START/B), so this is recorded as a follow-up rather than changed.
- *Safety:* Gen 1 text/menus advance on a *new* press (`hJoyPressed` edge), so holding does not skip extra boxes or double-select. Macros that call `emu.press` directly (`menus.py`, `shop.py`, `battle_actions.py`) are **out of scope/unchanged** — they are RAM-verified and green.

### F2 — drop hidden sprites (R2)
`games/pokemon_red/game_state.py`: add `hidden_sprite_slots(emu) -> set[int]` (parse the missable list, test the flag bit) and skip those slots in `read_npcs`. Pure RAM reads; no behavior change for visible sprites. Parsing is bounded: at most 16 pairs, accept only slots 1–15 and flag indexes < 256 (`0xD5A6` + 32 bytes); any read error → empty set (nothing hidden). Consequence (intended): tiles under hidden sprites stop being BFS obstacles.

### F3 — never observe (or plan on) a half-completed warp (R3)
1. `actions/controller.py` `_move` — classify the transition by the coords at the moment the map id flips (`flip_xy`):
   - **release the direction button first** (so the player doesn't walk an extra tile once a warp commits);
   - if `manhattan(flip_xy, before_xy) <= 1` it is a **warp** (door/mat/stairs: the flip happens on or next to the pre-step tile) → keep ticking (every `POLL` frames, cap `WARP_SETTLE_MAX = 120`) until the coords differ from `flip_xy` — the moment coords + sprites commit (frame 51 entering the lab; frame 36 leaving it);
   - otherwise it is a **map-edge connection** (coords jump across the map) → already settled, return immediately (no added frames);
   - a map change **ends a multi-tile move** (the next iteration must not count the warp's coord jump as a tile moved);
   - settle frames are included in `frames_elapsed`; the post-warp scripted auto-step (`wJoyIgnore != 0`) is handled by F5, not here;
   - **same-coordinate warps** (e.g. Red's House 1F/2F stairs, both (7,1)) never change coords: after `WARP_SAMEXY_SETTLE = 60` frames on the new map (past the observed 34–36-frame commit window) the warp is treated as settled (`warp_settled_samexy`) instead of burning the 120-frame cap.
2. `agent/reason_loop.py` `_approach_npc` (defense in depth) — extract a pure `select_npc(npcs, *, sprite, picked, player, want_kind) -> dict | None`:
   - `picked` is stored as `[x, y, map_id]`; a pick whose map id ≠ the player's current map is ignored;
   - **candidate pool:** if a sprite name is given and any NPC matches it, pool = the name matches; else pool = sprites of the wanted kind — `want_kind = "item"` for `GRAB_ITEM`, `"person"` otherwise (`TALK_TO` etc.) — falling back to all sprites only if that set is empty;
   - locality tracking (`picked`), the Jev pick, and the nearest-not-talked fallback all operate **within** the pool.
   `want_kind` comes from the active directive's intent.

### F4 — per-step objective budget (R4)
`actions/stuck_detector.py`: add `reset_objective()` clearing `_best_dist`, `_wedge`, `_recent_pos`, `_count`, `_last_key` (keeps `_best_sig` — cumulative exploration progress — and money/setback tracking). `reason_loop._commit_directive` calls `self.stuck.reset_objective()`. Each plan step gets a fresh `wedge_threshold` budget measured against its own target.

### F5 — defer to the game while a script/conversation is running (R5)
In `reason_loop`:
- track `self._last_dialog_step` (set whenever the flow router routes to dialogue);
- `script_active = emu.read_memory(0xCD6B) != 0` (wJoyIgnore);
- `in_conversation = script_active or (step - self._last_dialog_step) <= CONVO_GRACE_STEPS` with `CONVO_GRACE_STEPS = 2`. **Not** the raw text detector (`ctx_kind == "dialog"`): when the flow router confidently routes a text read to navigate, trust it — otherwise a persistent false-positive read that the router overrides would stall ~95% of steps (implementation review, issue 1);
- conversation-ness is decided **once at step start** (`self._step_convo`, before the action runs) and reused in `_finish`: a step whose *own* action triggers a script is not frozen (it is the one overworld step per dialogue-loop cycle that keeps the loop escapable); the next step sees the script and is a frozen forced wait (implementation review, issue 2).
- **Executive:** when the flow router would take the overworld/navigation path but `in_conversation` is true, emit a `WaitAction(frames=12)` step instead — no `_manage_directive`, no L1 gate, no moves.
- **Timeout (never stall forever):** `self._forced_waits` counts **consecutive F5-forced Waits** only; any other step (dialogue, menu, battle, normal nav) resets it to 0 in `_finish`. If it reaches `FORCED_WAIT_MAX_STEPS = 40`, emit `script_wait_timeout`, fall through to normal handling, and **disarm** F5-forced waiting until `wJoyIgnore` is 0 and no step has been *routed to the dialogue flow* (`_last_dialog_step`) within the grace window (then re-arm). This covers every cause of a forced wait (a stuck `wJoyIgnore` *or* a false-positive dialog read). Evidence it won't fire in normal play: across 1200 anchors of the 1839 run `wJoyIgnore` was non-zero only briefly (the scripted auto-step leaving the lab); in the Pokédex cutscene it stays `0xFC` for 27+ steps but those are mostly dialogue steps, which reset the counter.
- **Stuck detector — freeze, don't reset:** pass `forced_movement=True` (its existing suppression path) when `in_conversation`; in `_finish`, a conversation-suppressed step leaves `_blocked_for_n` **unchanged** (neither increments nor resets). Resetting would make a dialogue loop (re-talking a blocking NPC, bumping a sign) unescapable, because every cycle would zero the block budget; freezing keeps it accumulating across the overworld steps of the cycle. Accepted latency: only one overworld step per dialogue cycle feeds the position-oscillation check, so a same-NPC dialogue loop reaches `BLOCK_TRIGGER` after ~12 cycles (bounded at ≤15 in its test). The existing battle/forced-movement suppression semantics are left as-is. The budget update is factored into a small method so it is unit-testable. A step **routed to dialogue** stays frozen even while the forced-wait gate is disarmed (only the grace/script terms depend on the gate being armed).
- **Cost (accepted):** the grace window adds up to two 12-frame wait steps after each conversation (no model calls; applies on the legacy no-planner path too).

### Considered and rejected
- **Story event flags as a progress signal** (`wEventFlags` popcount @ `0xD747`): the count stayed at 8 across the entire delivery window in the captured run (the flag is set later in the script), so it would not have helped; the step's own success predicate (parcel no longer in bag) already captures completion.
- **A RAM "warp in progress" flag:** a per-frame scan across the torn window found no clean semantic flag (only scratch/stack bytes); the coord-change settle in F3.1 is structural and robust.

## 4. Testing (each failing first)

Fixtures (local, `states/*.state` is gitignored — ROM-guarded tests `pytest.skip` when absent, matching existing practice):
- `states/oak_tap_alias.state` ← `runs/brock-run-20260922-1839/states/map40_step586.state` (at (5,3) facing Oak, parcel in bag)
- `states/lab_door_warp.state` ← `runs/brock-continue-20260922-2217/states/map0_step100.state` (Pallet (12,12), one step south of the lab door)
- `states/lab_exit_mat.state` ← `runs/brock-continue-20260922-2217/states/map40_step104.state` (settled in the lab on the exit mat (5,11))
- `states/pallet_route1_edge.state` ← `runs/brock-run-20260922-1839/states/map0_step15.state` (Pallet, on the edge row y=0 below Route 1)
- existing `states/battle.state`, `states/lab_deliver.state`, `states/pallet_ready.state`

| Fix | Test | Fails today because |
|---|---|---|
| F1 | integration: from `oak_tap_alias`, for pre-offsets 0–7, one `ActionController.execute(InteractAction())` opens a dialog | 1-frame taps don't register there |
| F1 | integration (regression): from `battle.state`, `AdvanceDialogAction` still advances the battle text and does not auto-select a FIGHT-menu option | — (guards the held press in battle) |
| F2 | integration: `read_npcs(lab_deliver)` excludes Oak(5,10), Blue(4,3), Poké Balls (7,3),(8,3), keeps Oak(5,2) + Ball(6,3); `read_npcs(pallet_ready)` has no Oak | phantoms are returned |
| F3.1 | integration: from `lab_door_warp`, `execute(MoveAction(north))` returns on map 40 at (5,11) with lab sprites | returns mid-warp at (12,11) with Pallet sprites |
| F3.1 | integration: from `lab_exit_mat`, `execute(MoveAction(south))` returns on map 0 with Pallet sprites (coords (12,11) or (12,12)) | returns mid-warp at (5,11) on map 0 |
| F3.1 | integration: from `pallet_route1_edge`, crossing north returns on map 12 with `frames_elapsed` < 40 (no settle wait on a connection) | — (guards the edge-crossing path) |
| F3.2 | unit: `select_npc` over the **real** lab NPC list captured from `read_npcs(oak_tap_alias)` before F2 (phantoms included), sprite "Oak", `picked=[8,5,0]`, player on map 40 → Oak (5,2); unnamed `want_kind="person"` never returns an item when a person exists; unnamed `want_kind="item"` returns the item ball | picks the Poké Ball at (8,3) |
| F4 | unit: detector driven past `wedge_threshold`, `reset_objective()` → next non-improving update is not stuck; `_best_sig` preserved. loop: `_commit_directive` calls `reset_objective()` | no reset exists |
| F5 | unit: `in_conversation` (dialog / grace / script_active); forced-wait counter + timeout + disarm/re-arm; budget update freezes (not resets) on conversation-suppressed steps; a repeated same-NPC dialogue cycle still reaches `BLOCK_TRIGGER`. loop: a grace-window overworld step yields `WaitAction` without calling `_manage_directive` | overworld nav runs mid-cutscene; suppression would reset the budget |

**Regression:** full suite green (baseline 438 passed, 1 skipped). **Live acceptance:** resume `runs/brock-run-20260922-1839` (still carrying the parcel) for ≤250 steps with `--capture distill --headless`; expect the parcel delivered (bag no longer contains it), no `local_loop`/`step_wedged` during the delivery dialog, and the agent leaving Pallet northward afterwards.

## 5. Out of scope / follow-ups
- **Anchor/observation timing:** the step-N state anchor appeared to reflect a later frame than step N+1's logged observation at the Pallet→Lab warp. Likely a symptom of R3 (observing mid-warp); re-check behavioral-replay fidelity at warp boundaries after F3 lands.
- The starter-selection loop — **not needed** (decision 2026-09-22): runs start from the post-Squirtle save `states/pallet_ready.state` (`--state pallet_ready`).
- Spinner/arrow-tile rides that end on a warp flip the map >1 tile from the pre-step tile and would be classified as a connection (today's torn behavior — no regression). Harden later by also requiring `flip_xy` on the new map's boundary for a connection.
- Hold-to-register for the direct-`emu.press` macros (shop/menus/battle) — revisit only if a macro is observed missing inputs.
