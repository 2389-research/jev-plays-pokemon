# Distillation-Capture & Replay Layer — Design

**Goal:** Turn the strong-model agent runs into a clean research substrate: capture every decision's `(input → output + confidence)` at every level, label each with outcomes, and anchor it for replay — so we can (a) distill any model-backed layer into a small local model and swap it in, and (b) replay/eval any model's decision-making at real points from real runs. Pure instrumentation; **zero change to agent behavior.**

**Architecture:** A thin capture hook wraps every decision (model-backed and deterministic) and writes a per-decision record + a replay anchor into the run's record-dir, plus a per-episode outcome file at run-end. Two tools consume it: an **export** (per-layer supervised datasets with quality filters) and a **replay/eval harness** (re-run any candidate model against captured inputs, or rewind the agent from a save-state anchor).

**Tech stack:** Python; the existing `RunRecorder`/record-dir + `on_event`, the checkpoint machinery (`_write_resume`), and the model providers (`LunaRouteProvider.chat_json`, `TypeSafeReasoner.system_one`/`choose_flow`, `battle_agent.choose_move`).

---

## 1. Background & current state

Every recorded run already writes a rich record-dir: `log.jsonl` (per-step map/player/party/items/exits/context/objective/target/action/events), `shots/`, per-area `.state` saves + resumable `latest.state`/`latest.mem.json`. So **observations, resolved actions, and partial decision traces exist.** The gaps for distillation + replay:
1. No **uniform, self-contained `(input → output + confidence)` record per decision** — the exact serialized input each model call saw, and its exact response. `l1_last` has a partial trace; L2/Jev/battle don't uniformly.
2. No **outcome/reward labels** (per-step progress, per-episode success) to filter/weight training data.
3. No **export** to per-layer datasets, and no **replay** anchoring.

Validated direction: per-layer distillation is tractable because the agent is layered (each model-backed layer is a small, bounded I/O) — and both LunaRoute models already read the portal view (5/5) and query Orrery (3/3), so the inputs we'd capture are learnable targets.

## 2. Locked decisions (from brainstorming)

1. **Target = per-layer swap-in.** Capture `(input→output)` per layer so each model-backed layer can be distilled and swapped one at a time, measuring the quality delta.
2. **Quality = label + filter at export.** Capture ALL decisions; attach outcome labels; the export tool filters/weights (e.g. only-successful-episodes, min-progress).
3. **Capture everything at every level** — model-backed layers (distill + replay targets) AND deterministic layers (replay, debugging, outcome attribution).
4. **Replay is first-class** — decision-replay (feed a stored input to any model, compare) and behavioral-replay (rewind from a save-state anchor).

## 3. Per-decision capture

A capture record, one per decision, appended to `<record-dir>/decisions.jsonl`:
```
{ run_id, step, seq,                 # seq = decision index within the step (multiple per step)
  layer,                             # e.g. "l1_decide", "l2_propose_target", "jev_flow", "battle_move"
  kind: "model" | "deterministic",
  model,                             # model id for model-backed calls; null for deterministic
  input,                             # the EXACT serialized state/prompt the layer received (self-contained)
  output_raw, output_parsed,         # raw response + parsed decision (for model calls)
  confidence,                        # calibrated conf where available (Jev); null otherwise
  tokens, latency_ms,                # cost signals (model calls)
  anchor }                          # replay anchor: the state-save path for this step (see §5)
```
**Levels captured:**
- **Model-backed** (distill + decision-replay): `l1_brainstorm`, `l1_decide`, `l1_repair`, `l2_propose_target`, `l2_next_waypoint`, `jev_flow`, `jev_npc`, `jev_policy`, `battle_move`.
- **Deterministic** (replay/attribution only): `battle_l2_objective`, `battle_choose_action`, `portal_next`, `servo_move`.

**Hook placement (locked).** The hook goes INSIDE each layer method, at the point where `content`/`latency`/`usage` (and the input `state`) are still in scope — because these methods currently do `content, _, _ = prov.chat_json(...)` and return only a *parsed* dict, discarding tokens/latency/raw. So this is ~8 edits across `planner_llm.py` (`l1_brainstorm`, `l1_decide`, `l1_repair`, `l1_review`/`strategize`, `propose_target`, `next_waypoint`), `typesafe_reasoner.py` (`system_one`/`choose_flow` — expose `usage["confidence"]`/`probabilities`), and `battle_agent.choose_move` (capture the discarded raw `ans`) — NOT a wrap at return boundaries. Each calls `capture.record(layer, input=state, output_raw=content, parsed=…, confidence=…, tokens=…, latency=…)`. The `Capture` helper is held on the loop like `on_event` (a side-effect-only `_tee`), buffered per step, flushed to `decisions.jsonl`. **Correct layer sites:** `propose_target`/`next_waypoint` live in `planner_llm.py` (the loop's `_propose_target` delegates to them). Deterministic layers call `capture.record(layer, input=…, output=…, kind="deterministic")` at their decision point.

**Multi-round LLM loops (the one real subtlety).** `l1_brainstorm`/`l1_review`/`strategize` run through `_llm_with_search`, which can issue up to ~4 `chat_json` calls (KB-search rounds) per decision. Capture rule: emit **one record per decision** (the layer's final input → final parsed output) as the distillation/replay unit, with `tokens` = the sum across rounds and a `search_queries` list; ALSO emit the intermediate rounds as separate records sharing a `group_id` and their own `seq` (kind `"model_round"`) for analyzing the search behavior. Distillation/decision-replay use the decision record; the round records are optional detail.

**The input passed is exactly what the model/function consumed** — a faithful, replayable example. No behavior change: capture is side-effecting logging only.

## 4. Outcome labeling

- **Per step** — a progress vector attached to that step's decisions: `{map_hop_delta, level_delta, items_delta, hp_delta, map_changed, battle_result, caught, wedged}`. Note: `map_hop_delta` is the change in **`len(route(cur_map, cur_component, goal_map))`** — a MAP-HOP distance from PortalGraph (there is no scalar tile distance), so it's coarse (0 for moves within a map) and needs the current component; that's fine as a directional signal, but the labeler must know it's map-hop granularity.
- **Per episode** — computed at run-end into `<record-dir>/outcome.json`: `{reached_goal_map, badges, caught_count, delivered, steps, terminal_reason}`, plus per-map arrival steps. Each decision record is joined to its episode outcome at export time (by `run_id`).

**Where the signals come from (grounding):** `map_changed`, `items_delta`, `hp_delta`, `level_delta` derive from `log.jsonl` (level/hp require parsing the party strings, formatted `"SPECIES L# hp/max"` — see the parse-contract note below). `badges`, `caught`, and `delivered` are NOT in the base per-step record — they come from the `events` stream / per-caller `extra`, so the labeler reads events, not just base fields (or we record structured party/flags in `extra` on capture runs to avoid lossy re-parsing). The labeler is a pure post-processor over a record-dir (runs offline on past runs too).

## 5. Replay anchors & the two replay modes

- **Decision replay (no emulator):** the captured `input` is self-contained, so `replay --layer l2_propose_target --model <candidate>` feeds each stored input to a candidate model and diffs its output against the recorded one (and against the step's outcome). This is the eval harness for a distilled/candidate model — thousands of authentic decision points, ground-truthed.
- **Behavioral replay (from a save state):** each decision's `anchor` is a loadable emulator state for that step. To make *any* point replayable we add **step-anchored saves** via the recorder's existing periodic branch (`state_every=1` under `--capture distill`). NOTE the recorder writes `states/map{mid}_step{step}.state` (not `step<NN>.state`), so the `anchor` stores that exact path and the labeler/replay glob by step rather than assume a name; and the current `_maybe_state` guard skips a step whose `map_id` is null (rare, mid-warp) — those steps won't be behavior-replayable, which is acceptable. Then `replay --from-step 341 --swap l2=<local model>` reloads that state and runs the agent forward with the swapped layer, to observe downstream effects. (Storage: step saves are ~160 KB each; a capture run trades disk for full replayability — gated by the flag.)

## 6. Storage & delivery

- Extends the record-dir: `decisions.jsonl` (+ `states/step<NN>.state` when step-anchoring) beside `log.jsonl`; `outcome.json` at run-end.
- **Always-on capture of decisions** when recording (it's cheap logging). **Step-anchored saves are opt-in** via `--capture distill` (disk-heavy). Run metadata (models, config, git SHA, ROM hash, portal-graph version) written once to `<record-dir>/run_meta.json` for reproducibility.

## 7. Export & swap-in

- `scripts/distill_export.py`: record-dirs → per-layer JSONL/parquet of `(input, output_parsed, confidence, step_outcome, episode_outcome)`, with filters `--layer`, `--only-successful-episodes`, `--min-progress`, `--dedup`. Versioned schema.
- **Swap-in path:** a distilled local model implements the same call interface for a layer (e.g. an `L2Provider.propose_target(input)->output`); run with it via a `--swap layer=model` flag; the same capture + outcome machinery measures the delta (still crosses the forest / catches / reaches Brock). Clean per-layer ablation.

## 8. Testing

- **Capture wrapper (unit):** a recorded call produces a well-formed record with the exact input/output/confidence; a step with multiple decisions gets distinct `seq`s.
- **Outcome labeler (unit):** progress deltas + episode outcome computed correctly from a synthetic log.
- **Export (unit):** filters select the right subset; schema stable.
- **Decision replay (integration):** feed a captured input back to the same model → parses to the same decision shape; a stub "candidate" diffs correctly.
- **Behavioral replay (integration, ROM-guarded):** a step-anchored save reloads and the agent continues.
- A short live `--capture distill` run produces a real `decisions.jsonl` + `outcome.json` + step saves to inspect.

## 8a. Phasing (three deliverables — build in order)

This is really three plans; land the first standalone (it's the risky "zero behavior change" surface):
1. **Instrumentation core** — the `Capture` hook at the ~8 model sites + deterministic sites, the outcome labeler, and step-anchored saves. Ship + verify no behavior change before anything else.
2. **`distill_export.py`** — per-layer datasets + filters.
3. **Replay/eval harness** — decision-replay (`--layer/--model`) and behavioral-replay + the `--swap layer=model` interface (`L2Provider`-style). Meaningful chunk; does not block (1)/(2).

**YAGNI:** deterministic-layer capture (§3) has no distillation payoff yet — implement it behind the same `--capture distill` flag (off by default) so the core stays lean.

**Party parse contract (for the labeler):** party appears in the log as strings `"SPECIES L# hp/max"` (e.g. `"Squirtle L9 8/27"`). The labeler parses `L(\d+)` and `(\d+)/(\d+)` for level/hp; to avoid lossy re-parsing, capture runs SHOULD also write a structured party (`[{species, level, hp, max}]`) into the step `extra`.

## 9. Out of scope / open questions

- **Training the small models** themselves (this layer produces the data + harness; the actual distillation training is downstream).
- **Input size:** capturing the exact serialized input can be large (ASCII maps, KB results). Store raw for fidelity; add optional compression/dedup at export. (Decision recommended: store raw, compress the file.)
- **PII/secrets:** none in game state; run_meta excludes API keys.
- **Step-anchor density:** every step (full replay, more disk) vs every N (cheaper, coarser rewind). Default every-step under `--capture distill`; tune later.
