# Heard log, episode memory, explore, observed gates — design (2026-09-24)

## Why
Two runs (vermilion-team, ss-anne) lost ~3,400 steps to loops. In both, the information L1 needed
already existed but never reached it:
- The game told the agent "That bush in front of the shop is in the way. There might be a way
  around." four times. The dialogue log kept scrolling fragments in a 30-entry buffer, and L1 never
  saw dialogue at all.
- Every wedge reached L1 with `why_wedged` = the step's own description. Removed steps vanished from
  L1's view, so it re-issued "travel to Vermilion" about 10 times and flip-flopped on the same fact
  4 times.
- A hand-written story-gate list in the map graph ("officer blocks the back door early on") was wrong
  and silently vetoed the only route south. The harness was writing a walkthrough, and L1 could not
  overrule it.

Principle: the harness supplies mechanics and information; L1 decides. Following Gemini Plays
Pokémon, the harness gives better tools (history summaries, explored/unexplored map, a critique on
stalls), not solutions.

## Components
1. **HeardLog** (`agent/heard.py`). Dialogue parsed into COMPLETE messages: typing frames are merged
   and scrolls de-duplicated; a message closes when the box closes. Each message records speaker
   (sprite / object name), the speaker's tile, map and step. Battle text is excluded.
   - `recent`: verbatim messages from the last ~400 steps.
   - `by_map`: verbatim distinct messages per map, with repeat counts. Over budget → the oldest are
     folded into that map's summary (LLM; deterministic fallback).
   - `digest`: long-term "leads and ideas" summary across everything heard, re-summarized as messages
     age out of `recent`.
2. **EpisodeLog** (`agent/episode_log.py`). What happened since L1's last review: maps entered (with
   counts), steps done / wedged / removed with REAL reasons, messages heard, discoveries. Plus an
   attempt ledger that survives step removal (identical attempts are aggregated with counts).
3. **Wedge reasons.** The generic wedge now explains itself: the router's view (no known route from
   this area / the portal it was heading for) and the maps it cycled through.
4. **Observed gates replace the GATED list.** No story gates in the map graph. When the agent heads
   for a portal and can't get through (a sprite on the path, or pushed back), that portal is recorded
   as blocked-by-observation, with what was seen and heard. It expires when progress changes (item,
   badge, party) or after a TTL, so it gets retried later.
5. **Exploration.**
   - `unexplored_here`: doors / edges on this map leading to never-visited maps, people not talked to,
     and objects not used.
   - An `explore` step kind: L1 adds `{"kind": "explore", "map": M, "who": "<optional preference>"}`.
     A deterministic executor visits the unexplored things (preference first, else nearest). The step
     completes on a discovery: a new map entered, a new message heard, or a route to a pending step's
     map appearing. It also completes when nothing is left to explore.
6. **Stall monitor + critic** (`agent/critic.py`). A deterministic monitor tracks progress (goal
   status, items / badges / levels, new maps, new messages, explored tiles). After ~150 steps without
   progress it raises `stalled`, and a fast-model critic reads the episode log, attempt ledger, heard
   digest and unexplored list and writes a short diagnosis for L1. L1 still decides; the critic never
   edits the plan.

## L1 sees (new fields)
`since_last_review`, `attempts`, `heard` (the digest plus this map's messages), `unexplored_here`,
`stall` (with `critique` when present). Triage sees `since_last_review` and `stall` too.

## Removed
`GATED` / `open_when` / `refresh_gates` and the Orrery "BLOCKED" wording. The object table (PCs,
signs) stays: it is information, not a solution.
