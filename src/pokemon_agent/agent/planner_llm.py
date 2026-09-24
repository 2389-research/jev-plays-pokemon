"""The Planner: turns the arbiter's intent into a concrete `Directive` (spec §4).

Division of labor (spec §2): the NeedsArbiter owns the coarse INTENT (deterministic,
RAM-driven, stable); this planner owns the concrete TARGET and the replan reasoning.
For a ``travel`` intent that means *choosing which map to head toward* — normally the next
hop on the graph route, but on a stuck/impossible replan the LLM can reroute to a different
reachable subgoal instead of re-issuing the identical failing directive (the loop the old
deterministic planner fell into). The base-level geometry is handed to BFS (LLM+P), and
``success`` predicates are computed deterministically here so they stay machine-checkable.

The LLM (LunaRoute) is consulted only for the travel target and only on a replan trigger —
rare and cheap (bold commitment). If no provider is wired, or the call fails / returns an
unreachable map, the planner falls back to the deterministic graph route. So the agent
still navigates with the LLM offline; the LLM makes it *smarter about rerouting*, not
load-bearing for the happy path.
"""
from __future__ import annotations

import json
import time

from ..games.pokemon_red import needs
from ..games.pokemon_red.constants import MAP_NAMES_RAW
from ..games.pokemon_red.game_state import read_badges, read_items, read_party, resolve_item_id
from ..games.pokemon_red.maps import map_name
from ..games.pokemon_red.needs import WCURMAP
from ..providers.parsing import strip_fences
from .plan import Directive, Intent

# arbiter need name -> executor intent
_NEED_TO_INTENT = {
    "survive": Intent.HEAL,
    "battle": Intent.BATTLE,
    "readiness": Intent.GRIND,
    "progress": Intent.TRAVEL,
    "idle": Intent.TRAVEL,
}


def intent_for_need(need_name: str) -> Intent:
    return _NEED_TO_INTENT.get(need_name, Intent.TRAVEL)


PLANNER_SYSTEM = """You are the STRATEGIC PLANNER for an agent playing Pokémon Red. You are
called only when a new plan is needed (a directive finished, failed, got stuck, or the
situation changed) — so think, don't rubber-stamp.

Your ONE job right now: pick the next MAP to travel toward, given the mission and the world
graph. Usually that is the next hop on the route to the goal. BUT if WHY_REPLAN says the
last attempt got stuck or was impossible, do NOT pick the same map again — reroute: choose a
different reachable map that makes progress (an intermediate area, or backtrack to try
another edge), using TRIED_FAILED and the ROUTE/NEIGHBORS to reason about it.

You are given: MISSION, GOAL_MAP, CURRENT_MAP, ROUTE (map ids from here to the goal),
NEIGHBORS (adjacent maps you can walk to now, with direction), MAP_HISTORY (recent maps —
if it alternates between two ids you are oscillating), TRIED_FAILED, and WHY_REPLAN.

TOOL — knowledge base: you may look things up in a Pokémon Red guide before choosing. To
search, reply with ONLY {"search": ["query1", "query2"]} (1-3 queries); results come back in
KNOWLEDGE_GATHERED and you can search again or finalize. SEARCH_ROUNDS_LEFT limits searches.

Return ONLY a JSON object (when ready):
{"target_map": <int map id to head toward next>, "reason": "one sentence why"}
target_map MUST be a real map id from ROUTE or NEIGHBORS (a reachable map), never the
current map."""


WAYPOINT_SYSTEM = """You are the NAVIGATOR (tier 2, tactical) for a Pokémon Red agent. The
strategic layer has chosen where to go; your job each leg is to look at the grid and pick the
next TILE to head to that makes progress toward that goal. A deterministic pathfinder (BFS) then
walks the agent to the tile you choose, and you are asked again once it arrives or can't get
closer — so pick the best next stepping-stone, not the whole path.

WHAT YOU MUST DO — OBJECTIVE and DESTINATION tell you where you are going and why (e.g. "reach
<a building> — next hop is <the town it's in> to the north"). Always move toward it.

MEMORY — you are NOT memoryless. RECENT_TRAIL is your last several frames as
"(x,y)mMAP action -> result" (watch for bouncing between the same tiles). RECENT_WAYPOINTS is
the tiles you already picked — do NOT pick the same ones again or reverse course; if you keep
ending up in the same place, commit to the EXIT_TILE / GOAL_DIR and push through.

COORDINATE SYSTEM: the grid uses (x, y). x = COLUMN (two header rows: tens then units, left→right).
y = ROW (labeled 'y<n>' at left, increasing DOWNWARD/south; y=0 is north). The MAP_VIEW begins with
a LEGEND that tells you exactly what every symbol is and its properties (path, grass, water, wall,
ledges with their one-way hop direction, doors, counters, NPCs) — READ IT; you never have to guess
what a tile is. Note grass ('G') IS walkable (wild battles there); '#' and water are NOT walkable;
ledges are one-way.

GOAL_DIR says which way the next area is. EXIT_TILE, when given, is the door/edge tile that
leaves toward the goal — heading to it (or onto it) is usually the right move; you MAY pick it
directly, and SHOULD once you're close. WHY says what happened last.

Pick a WALKABLE tile (path 'G' grass or a door) that is: (a) reachable from '@' WITHOUT crossing a
'#'/water/NPC or going the wrong way up a ledge; (b) a real step toward GOAL_DIR / EXIT_TILE and NOT
one you keep revisiting; (c) as far along a clear path toward the destination as you can see. TRACE
the path tile-by-tile in your head first and make sure every step is walkable.

Return ONLY JSON: {"path": "(x,y)->(x,y)->...", "x": <int>, "y": <int>, "reason": "..."}"""


PROPOSER_SYSTEM = """You are the MID-LEVEL PROPOSER for a Pokémon Red agent. The strategic layer
picked WHERE to go (a target map / errand). Your job each leg: look at the grid and pick ONE
COORDINATE to walk toward. A deterministic router walks you there and handles what happens on
arrival (stepping through a door, or off a map edge) — you just pick the tile. It asks you again
when you REACH it, get STUCK, or the map changes. You are also the get-unstuck mechanism: when WHY
says the last target was unreachable or made no progress, pick a DIFFERENT tile.

Return ONLY ONE JSON object. Normally a coordinate:
  {"x":<int>,"y":<int>,"why":"<one short sentence: why this tile>"}
To talk to a person instead of move:
  {"kind":"approach_npc","sprite":"<name>","why":"..."}

COORDINATES: (x,y); x = column (increases EAST), y = row (increases SOUTH, y=0 is the north edge).
The MAP_VIEW starts with a LEGEND naming every symbol (path, grass 'G' walkable, '#'/water NOT
walkable, one-way ledges, doors 'D', counters, NPCs) — READ IT, never guess a tile.

CANDIDATE_EXITS is the list of ways OFF this map, each an exact coordinate: kind "door" (a warp,
with its destination map) or "edge" (a walkable map-boundary opening, with its direction N/S/E/W).
To LEAVE toward the goal, pick the coordinate of the candidate that advances toward
DESTINATION/GOAL_DIR — e.g. a north 'edge' when the goal is north, or a 'door' whose dest is on the
way. IGNORE candidates that lead BACKWARD (e.g. a door back into a building you just left, or whose
dest is where you came from). If no candidate helps yet, walk toward the goal side of the map and
you'll be asked again. You may also pick any other walkable tile — candidates are hints, not a menu.

Pick a tile that is in REACHABLE and a real step toward the goal, not one in RECENT_TARGETS you keep
revisiting. WHY is one short sentence of your reasoning (it becomes the agent's visible short-term
objective and helps debugging — always include it)."""


STRATEGIST_SYSTEM = """You are the STRATEGIC planner (tier 2) for an agent playing Pokémon Red,
working through the game's gyms and story. You are called when the agent is
BLOCKED or UNSURE and needs a plan: a STORY GATE (an NPC who won't move, a locked path, a required
item/errand), OR a NEED it doesn't know how to satisfy — most commonly it must HEAL (party HP is
low) but doesn't know WHERE the nearest Poké Center is or how to get there. Work out the SEQUENCE
of steps that resolves the situation and routes the agent to the right place.

TOOL — knowledge base: you can look things up in a Pokémon Red guide/knowledge base before you
commit. To search, reply with ONLY {"search": ["query1", "query2"]} (1-3 queries); you'll get
the results back in KNOWLEDGE_GATHERED and can search again or finalize. SEARCH FIRST to ground
your plan in the guides rather than guessing — especially WHERE things are (which map has the
nearest Poké Center / Mart / the next objective) and story gates. SEARCH_ROUNDS_LEFT tells you how
many more searches you may do; when it hits 0 you must output the final plan.

HEALING: if WHY_BLOCKED says HP is low / needs to heal, plan to go to the nearest Poké Center map
and talk to the nurse, with done_when "hp_frac>=0.95" on the talk step; then a final step to
resume the main objective. Use the MAPS table to pick the correct Poké Center map id.

You are given: WHY_BLOCKED, CURRENT_MAP (id + name), PARTY, ITEMS, BADGES, MAPS (an id→name
table — use these exact ids), and KNOWLEDGE_GATHERED (results of your searches so far — TRUST
these over your own memory when they conflict).

Reason about what the game requires here (e.g. fetch an item from a shop and deliver it, talk to
a specific person, enter a building), then output an ORDERED list of steps. Each step is a MAP to
go to, optionally talking to an NPC once there. The LAST step should continue toward the gym once
unblocked.

For EACH step give an ACCEPTANCE CRITERION (done_when) — the checkable condition that PROVES the
step is complete (like a quest objective), so a step can't be marked done prematurely. Choose:
  "on_map"            — arrived on that map (default for pure travel).
  "has_item:<name>"   — that item is now in the bag (collect/buy it -> has_item:<item>).
  "no_item:<name>"    — that item is gone (delivered/used: hand it over -> no_item:<item>).
  "level>=<N>"        — party reached level N.   "badges>=<N>" — earned N badges.
  "hp_frac>=<F>"      — party healed to fraction F of max HP (talk to a Poké Center nurse -> hp_frac>=0.95).
  "talked"            — had a conversation on that map (only when nothing more specific fits).
  "verify:<yes/no question>" — a verifier judges it from game state, when none of the above fit.

When "talk" is true, set "who" to the NPC you must talk to (e.g. "the Mart clerk", a named person) so the
agent approaches the RIGHT person, not the nearest one.

Return ONLY JSON:
{"plan": "one-line summary",
 "steps": [{"map": <int map id>, "talk": <true|false>, "who": "<npc name to talk to>",
            "done_when": "<criterion>", "why": "<short>"}]}"""


L1_SYSTEM = """You are the L1 STRATEGIST for an agent playing Pokémon Red, working through the
game's gyms and story. This is a PERIODIC strategic REVIEW, not a rescue: look
at the whole situation — the current MISSION/MILESTONE, the standing PLAN (its steps with their
status), where the agent is, and the SIGNALS (e.g. how long it's been blocked) — and decide
whether the plan still makes sense. Usually it does; say so and change nothing.

THE PLAN is an ordered list of quest STEPS. Each step is:
  {"map": <int map id>, "talk": <true|false>, "who": "<npc name, if talk>",
   "done_when": "<criterion>", "why": "<short>"}
The done_when is the checkable ACCEPTANCE CRITERION that proves a step is complete. It MUST be
one of:
  "on_map"            — arrived on that map (default for pure travel).
  "talked"            — had a conversation on that map (only when nothing more specific fits).
  "has_item:<name>"   — that item is now in the bag (collect/buy it -> has_item:<item>).
  "no_item:<name>"    — that item is gone (delivered/used -> no_item:<item>).
  "level>=<N>"        — party reached level N.   "badges>=<N>" — earned N badges.
  "hp_frac>=<F>"      — party healed to fraction F of max HP (Poké Center nurse -> hp_frac>=0.95).
  "verify:<yes/no question>" — a verifier judges it from game state, when none of the above fit.

TOOL — knowledge base: you MAY look things up in a Pokémon Red guide before revising. To search,
reply with ONLY {"search": ["query1", "query2"]} (1-3 queries); results come back in
KNOWLEDGE_GATHERED and you can search again or finalize. SEARCH_ROUNDS_LEFT limits searches.
SEARCH THE KB before adding any story-gate or errand step (WHERE an item/objective is, what a
gate requires) — ground it in the guide rather than guessing.

WHAT TO RETURN — ONLY a JSON object:
  {"assessment": "<one line: how the plan is doing>",
   "change": <true|false>,
   "mission": "<the overall mission, e.g. 'earn the next badge'>",
   "milestone": "<the current concrete sub-goal>",
   "add": [ <new step objects, each with a VALID done_when> ],
   "remove": [ <step ids to drop from the current plan> ]}

When the plan is fine (the COMMON case) reply {"change": false, "add": [], "remove": []} — do
NOT churn a working plan. Only when something is wrong (missing a required errand, stuck on a
story gate, a step that can never complete) propose `add` steps to fix it and `remove` the ids of
steps that should go. Every added step MUST have a done_when from the list above."""


TRIAGE_SYSTEM = """You are the L1 TRIAGE gate for an agent playing Pokémon Red. This is a CHEAP,
FAST check that runs every periodic review, before any expensive reasoning: look at the standing
PLAN (steps + statuses), the GOALS (primary / secondary / tertiary, with GOAL_STATUS and any
INTERRUPTED focus), and the SIGNALS (blocked duration, low HP, emergency_heal, etc.) and decide ONLY whether the plan needs to change at all. Do NOT propose
what to change — that is a separate, more expensive step. You have NO knowledge-base access here;
answer from what's given, do not search.

Usually the plan is fine — say so. Say change=true only when something is clearly wrong: a step
that can't complete, a stuck/blocked signal, an emergency (e.g. low HP with no heal step in the
plan). A goal that already shows met is not by itself a reason to change; say change=true for goals
only if a goal is clearly wrong or impossible. A battle LOST recently (SIGNALS.recent_battles) is a reason to change — the
plan needs to adapt, not repeat. A standing catch goal that SIGNALS.catch says is not ready
(e.g. no Poké Balls) IS a reason to change — the plan needs a step to fix it or the goal dropped.
MEMORY: SINCE_LAST_REVIEW is what happened since the last review (maps entered, plan changes and why,
what people said). ATTEMPTS lists steps that failed before, with counts. STALL.steps_without_progress
counts steps with nothing new; STALL.critique is a reviewer's diagnosis. The same step failing again
(ATTEMPTS), maps entered over and over, or a critique ARE reasons to change. An ACTIVE step whose
"doing" is "travelling to <its map>" is working normally — NOT a reason to change; an edit listed as
IGNORED in SINCE_LAST_REVIEW.your_last_edits is not a reason to try it again.

Return ONLY JSON: {"change": <true|false>, "why": "<one short sentence>"}"""


BRAINSTORM_SYSTEM = """You are the L1 BRAINSTORM step for an agent playing Pokémon Red, working
toward its own GOALS (primary = the long-term aim, e.g. the next badge). TRIAGE has flagged that the plan may need to
change. Your job here is OPEN-ENDED assessment, not a final plan: think through the situation —
current MAP, PARTY, ITEMS, BADGES, SIGNALS, GOALS, NOTEPAD — and what the game actually
requires next (a story gate, an errand, healing, grinding, the next town). A later DECIDE step
will turn your assessment into concrete quest steps, so be concrete and specific (name the map /
item / NPC where you can), but do NOT emit step objects or JSON steps yourself here.
GOALS are the agent's own horizons (primary = the long-term why, secondary = the current chapter,
tertiary = the immediate focus, possibly a diversion; GOAL_STATUS says whether a criterion holds now;
INTERRUPTED is a paused focus); NOTEPAD is the agent's own notes. Say if a goal should change.
ITEMS and SIGNALS.money are ground truth — where the NOTEPAD disagrees, trust ITEMS.

MEMORY — what you have already tried and been told (harness-written ground truth):
  - SINCE_LAST_REVIEW: what happened since you last reviewed — maps entered (entered_repeatedly = a
    bounce), plan changes with the REAL reason each step failed, and HEARD (what people, signs and
    objects said, with who and where; "heard Nx" = said more than once — hints often point the way).
  - ATTEMPTS: every step that failed before and how often. Do NOT retry the same thing the same way;
    if it failed repeatedly, something you believe is wrong — change the approach.
  - HEARD: digest = long-term notes of what you've been told; here = what was said on this map.
  - UNEXPLORED_HERE: on this map, doors to places you've never been, people you haven't talked to,
    objects you haven't checked.
  - STALL: steps without progress, plus a reviewer's CRITIQUE when you've been stuck a while.
  - INTERRUPTING the ACTIVE step: a plain "remove" of the active step is ignored. When it truly can't
    wait or has become wrong (e.g. HP critically low before walking through grass, the step became
    impossible), put its id in "remove", add the replacement, AND set "interrupt_active": {"why":
    "<the concrete reason it can't wait>"}. A reviewer checks the reason first; reordering for its
    own sake, or an active step that is merely still travelling, will be rejected.
  - SINCE_LAST_REVIEW.your_last_edits: what happened to YOUR previous edits (e.g. a remove that was
    ignored because the step is active, an add that duplicated an existing step). Don't repeat an edit
    that was ignored — it will be ignored again.
PLAN status "active" = in progress. An action step at another map is active WHILE travelling there:
"doing" says what it's doing now (e.g. "travelling to <its map>"). That is normal, not a bug —
the step can't be removed and doesn't need reordering.
When you're stuck and don't know the way, do what a player does: explore — go through the doors you
haven't tried, talk to the people you haven't talked to (an explore step, see DECIDE).

TEAM — the agent owns a TEAM, not just its starter. A single Pokémon is fragile: when the lead
faints the run is over for that fight, and later gyms punish a one-type team (a gym leader's type
can wall a team that's weak to it). Growing and balancing the team is part of the long-term plan, and YOU decide when
it's worth it: catching wild Pokémon is how the team grows. Catching needs Poké Balls in ITEMS (bought
at a Poké Mart) and a free party slot; set a standing CATCH goal to have the battle layer catch the
species you want. SIGNALS.catch shows your current catch goal and whether it can fire right now (ready,
and why not — e.g. no Poké Balls).

TOOL — knowledge base: you SHOULD look things up in a Pokémon Red guide before concluding —
especially WHERE things are (which map has the item / NPC / Poké Center) and what a story gate
requires. To search, reply with ONLY {"search": ["query1", "query2"]} (1-3 queries); results come
back in KNOWLEDGE_GATHERED and you can search again or finalize. SEARCH_ROUNDS_LEFT limits
searches. Trust KNOWLEDGE_GATHERED over your own memory when they conflict.

Return ONLY JSON (when ready): {"assessment": "<a few sentences: what's going on, what's needed
next, and why>"}"""


DECIDE_SYSTEM = """You are the L1 DECIDE step for an agent playing Pokémon Red, working toward its
own GOALS (primary = the long-term aim, e.g. the next badge). BRAINSTORM has already assessed the situation (see
BRAINSTORM below); your job now is to turn that into a MINIMAL, concrete set of quest-step edits
anchored to the EXISTING plan — do NOT redesign the whole plan from scratch, only add what's
missing and remove what's broken.

CHANGING NOTHING IS THE COMMON, PREFERRED OUTCOME. You are often called just because the agent is
still travelling or momentarily blocked — that does NOT mean the plan is wrong. If the standing
plan already covers the situation and the active step is still valid (its destination is reachable,
its goal not yet met), return EMPTY "add" and EMPTY "remove" — that means "keep going, continue the
active step". Do NOT re-add or restate a step that already exists in the PLAN and is in progress
(e.g. do not add another heal step when one is already active, nor another "go to X" when that is
already the active step) — repeating a step never helps and just thrashes the plan. ONLY add a step
that is genuinely MISSING, and ONLY remove one that is truly impossible or already obsolete.

GOALS — your own horizons, all held at once: PRIMARY = the long-term why (e.g. earn the next
badge); SECONDARY = the current chapter (e.g. reach the next gym with a team that can win); TERTIARY =
the immediate focus, which may be a diversion (heal, shop, grind, a story errand). GOALS and NOTEPAD
follow the same rule as steps: omit "goals", "notepad" and "catch" unless something actually changed
— a goal was achieved, you are diverting, or you learned something worth keeping. Rewording is not a
change. When you do edit, include only the tier(s) that changed. GOAL_STATUS "met" means that
criterion holds right now; you decide when to move on. A brief diversion can be just a step (e.g. a
heal step) — you don't have to rewrite a goal for it.
  - Diverting / returning: rewrite the tertiary (or the secondary if the chapter itself changed).
    INTERRUPTED shows the focus you paused — return to it, or add "interrupted": "" to drop it (a
    paused focus without a checkable criterion is only dropped when you say so).
  - Plan concrete steps for your current focus and chapter; keep later chapters as goals / notepad
    rather than queued steps.
  - NOTEPAD is your own short notepad — intentions for later, lessons, things not to retry. Send the
    full rewritten text only when it changes; keep it short (NOTEPAD_TRUNCATED = it was cut).
  - A goal's optional done_when uses ONLY has_item: (incl. has_item:<name>>=N)/no_item:/level>=/badges>=/hp_frac>=/verify:
    (never on_map or talked — goals have no map).
  - ITEMS and SIGNALS.money are GROUND TRUTH. If your NOTEPAD disagrees with ITEMS (e.g. it says you
    have Poké Balls and ITEMS has none), trust ITEMS and correct the notepad. Catching needs Poké
    Balls in ITEMS; buying needs enough money.

You are given: CURRENT_MAP, PARTY, ITEMS, BADGES, PLAN (existing steps), SIGNALS, GOALS,
GOAL_STATUS, INTERRUPTED, NOTEPAD, SINCE_LAST_REVIEW, ATTEMPTS, HEARD, UNEXPLORED_HERE, STALL (see
BRAINSTORM's MEMORY notes: what happened, what failed, what you were told, what you haven't tried),
BRAINSTORM (the prior assessment), and MAPS (an id->name table —
you MUST use these exact ids for any "map" field).

EVERY step you add MUST include an explicit "kind" — this is a HARD requirement; a step with no
kind silently breaks execution downstream:
  {"kind": "travel", "map": <int>, "talk": false, "who": null,
   "done_when": "on_map", "why": "<short>", "after": null}
  {"kind": "action", "map": <int>, "talk": <true|false>, "who": "<npc name, if talk, else null>",
   "done_when": "<criterion>", "why": "<short>", "after": null}
  {"kind": "explore", "map": <int>, "who": "<optional: an UNEXPLORED_HERE entry to try first>", "why": "<short>", "after": null}

The kind rule:
  - "travel" is ONLY for moving to a map with no other objective on arrival; its done_when is
    ALWAYS "on_map" and it must NEVER talk to anyone (talk must be false).
  - "action" is for anything that must happen there (talk to an NPC, heal, grind, wait on a story
    flag). An action step's done_when must be a REAL, non-"on_map" criterion — merely arriving on
    the map is never enough to call an action step done.

done_when MUST be exactly one of (this is the full grammar — nothing else parses):
  "on_map"                    — arrived on the map (travel steps only).
  "has_item:<name>"           — that item is now in the bag.
  "has_item:<name>>=<N>"      — at least N of that item (buying several: has_item:Poke Ball>=5).
  "level:<nickname>>=<N>"     — ONE party member reached level N (training: level:Sophie>=16).
  "no_item:<name>"            — that item is gone (used/delivered).
  "level>=<N>"                — party reached level N.
  "badges>=<N>"               — earned N badges.
  "hp_frac>=<F>"              — party healed to fraction F of max HP.
  "talked"                    — had a conversation (only when nothing more specific fits).
  "verify:<yes/no question>"  — judged by a verifier from game state; LAST RESORT ONLY, when the
                                 objective genuinely isn't RAM-checkable. Prefer any RAM-checkable
                                 form above over verify: whenever one applies — verify: is
                                 expensive and fuzzy.

WORKED EXAMPLES (one per objective class — copy the SHAPE, adapt the specifics):
(<angle brackets> are placeholders: put REAL map ids from MAPS and real names there.)
  pickup an item  -> {"kind":"action","map":<map where it is>,"talk":true,"who":"<who hands it over>",
                       "done_when":"has_item:<item>","why":"collect the item"}
  buy at a Mart   -> {"kind":"action","map":<a Mart's map>,"talk":true,"who":"the Mart clerk",
                       "done_when":"has_item:Potion","why":"buy Potions before the gym"}
                      Talking to a Mart clerk opens the shop and the buy runs automatically. Add this
                      when you have money and want consumables (e.g. Potions before a gym). Only items
                      on that Mart's shelf are bought — each Mart stocks different items (search the KB).
                      A wedged step's "why_wedged" says why it failed (e.g. the shelf it saw): don't
                      re-add the same step where it already failed.
  deliver an item -> {"kind":"action","map":<recipient's map>,"talk":true,"who":"<recipient>",
                       "done_when":"no_item:<item>","why":"hand the item over"}
                      CANONICAL: deliver -> no_item:<item>. The item LEAVING the bag proves
                      delivery. Do NOT model a delivery as has_item:<something else>.
  use an object   -> {"kind":"action","map":<its map>,"talk":true,"who":"<the object, e.g. a PC>",
                       "done_when":"verify:<did it do what you needed?>","why":"use the machine"}
                      "who" can name a THING you press A on (a PC, machine, trash can, statue) —
                      current_map.objects lists them; current_map.people lists who is here now.
  heal            -> {"kind":"action","map":<a Poké Center's map>,"talk":true,"who":"the Nurse",
                       "done_when":"hp_frac>=1.0","why":"heal the party at the Poké Center"}
                      Add a step like this ONLY when SIGNALS shows low HP / emergency_heal — don't
                      invent healing from generic caution.
  grind           -> {"kind":"action","map":<a map with wild grass>,"talk":false,"who":null,
                       "done_when":"level>=<N>","why":"grind in the grass toward the goal"}
  earn a badge    -> {"kind":"action","map":<the gym's map>,"talk":true,"who":"<the gym leader>",
                       "done_when":"badges>=<current badges + 1>","why":"beat the gym leader"}
  reach a place   -> {"kind":"travel","map":<destination map>,"talk":false,"who":null,
                       "done_when":"on_map","why":"head to <place>"}
                      Routing knows the real paths — through caves, gates and maps that are split into
                      separate parts (one route can have halves joined only THROUGH a cave). To cross a
                      dungeon, add ONE travel step to the place beyond it, not a step per map you pass
                      through: "go to <the route>" is already true at the cave entrance.
  story beat not  -> {"kind":"action","map":<its map>,"talk":true,"who":"the guard",
  RAM-trackable        "done_when":"verify:did the guard let us pass?","why":"..."}
  explore         -> {"kind":"explore","map":<this map>,"who":"<an entry copied from UNEXPLORED_HERE>",
                       "why":"find the way forward"}
                      "who" (optional) = what to try first: COPY one entry from UNEXPLORED_HERE (or give
                      its coordinates, e.g. "(12,7)").
                      Visits what's unexplored on that map — doors to places never visited, people not
                      talked to, objects not checked (UNEXPLORED_HERE), your "who" first if given — until
                      something NEW turns up (a new place or something new heard); then you review again.
                      Use it when stuck, when you don't know the way, or when ATTEMPTS show the same step
                      failing — instead of re-adding the failed step. No done_when needed.

PLACEMENT — "after" says where a new step goes. Leave it null (the default) for something to do
NEXT: right AFTER the ACTIVE step finishes, before the rest of the plan (replacements for a wedged step
go here). NEXT never stops the active step — if something can't wait for the active step to finish
(e.g. an emergency heal while the active step walks through grass), INTERRUPT it (see INTERRUPTING
under MEMORY). Set
"after": "<id>" only when the step must come AFTER an existing step that hasn't happened yet; the id
must be from PLAN with status active or pending. "after" is NOT inherited from the step listed before
it — put it on EVERY step that must wait. Several steps with the same "after" run in the order you list
them. "end" appends after everything. Example: PLAN [q4 travel-><town> (active), q5 buy Potions
at <the town's Mart> (pending)]; grinding on <the next route> and then entering <the next area>
should both wait until after shopping ->
{"kind":"action","map":<route id>,"talk":false,"who":null,"done_when":"level>=10","why":"grind","after":"q5"}
{"kind":"travel","map":<area id>,"talk":false,"who":null,"done_when":"on_map","why":"next area","after":"q5"}

RULES:
  - Emit MINIMAL steps: only what's missing from the existing PLAN, anchored to it — don't repeat
    steps already present and on track.
  - EVERY step MUST include "kind" ("travel", "action" or "explore"); never omit it.
  - Never re-add a step that ATTEMPTS shows failing the same way; change the approach (often: explore).
  - Put lessons in the NOTEPAD (what failed and why, hints you heard) so later reviews keep them.
  - Use the correct map id from MAPS for every "map" field.
  - Prefer a RAM-checkable done_when (has_item/no_item/level/badges/hp_frac/on_map) over
    "verify:" whenever one applies.

TEAM — the agent owns a TEAM, not just its starter. A single Pokémon is fragile: when the lead
faints the run is over for that fight, and later gyms punish a one-type team (a gym leader's type
can wall a team that's weak to it). Growing and balancing the team is part of the long-term plan, and YOU decide when
it's worth it: catching wild Pokémon is how the team grows. Catching needs Poké Balls in ITEMS (bought
at a Poké Mart) and a free party slot; set a standing CATCH goal to have the battle layer catch the
species you want. SIGNALS.catch shows your current catch goal and whether it can fire right now (ready,
and why not — e.g. no Poké Balls).
LEARN FROM LOSSES — SIGNALS.recent_battles lists your last fights and their results. If you LOST (e.g.
to a gym leader), don't simply heal and retry the same way: change something first — train team members,
buy Potions, pick a better lead or a counter type — and say in the notepad what you're changing and why.
TRAINING — only Pokémon that take part in a battle earn EXP, and the LEAD (first in the party) starts
every battle, so a bench that never leads stays weak. Training a member is your call: set "lead":
"<nickname>" (the harness moves it to the front of the party), then grind with a step whose done_when is
"level:<nickname>>=N" — somewhere its level can handle (weak wild Pokémon, heal often). Set the lead
back (e.g. "lead": "Dylan") before a gym battle. PARTY shows each member's nickname, level and HP.
To set or change the CATCH goal add "catch": ["<species>", ...] (or ["any"]); the battle layer then
catches a matching wild Pokémon when it can (weakened into the catch band, then a ball). A goal that
SIGNALS.catch says isn't ready does nothing until you fix the reason (e.g. add a step to buy Poké Balls:
{"kind":"action","map":<mart>,"talk":true,"who":"the Mart clerk","done_when":"has_item:Poke Ball>=5"}).
Omit "catch" when it isn't changing; send "catch": "clear" ONLY to remove a goal that SIGNALS.catch
shows is set.

Return ONLY JSON:
{"assessment": "<one line: what changed and why>",
 "add": [ <new step objects as above, each with its "after" (null unless it must follow a PLAN step)> ],
 "remove": [ <ids of existing plan steps to drop> ],
 "goals": { <ONLY the tier(s) that changed, e.g. "tertiary": {"text": "<short>", "done_when": "<criterion or null>"}> },
 "notepad": "<full rewritten notepad>",
 "catch": [ <species or "any"> ],
 "lead": "<party nickname to put first>",
 "interrupt_active": {"why": "<only when replacing the ACTIVE step — the concrete reason it can't wait>"}}
Omit "goals", "notepad", "catch", "lead" and "interrupt_active" entirely when not needed (the common case). To drop the
paused focus, add "interrupted": "" (see GOALS above); otherwise never include "interrupted"."""


REPAIR_SYSTEM = """You are the L1 REPAIR step for an agent playing Pokémon Red. ONE quest step
failed deterministic validation. You are given the done_when GRAMMAR (below), the BAD_STEP exactly
as emitted, and the exact ERROR from the validator. Re-emit ONLY that one step, same shape, with a
corrected "done_when" and/or "kind" so it validates. Do not change anything else about the step
(map/who/why) unless it is the cause of the error.

done_when MUST be exactly one of:
  "on_map" | "has_item:<name>" | "has_item:<name>>=<N>" | "no_item:<name>" | "level>=<N>" | "badges>=<N>" |
  "hp_frac>=<F>" | "talked" | "verify:<yes/no question>"

kind is "travel" (done_when must be "on_map", and it must never talk) or "action" (done_when must
be a real, non-"on_map" criterion).

Return ONLY the corrected step as JSON, same shape as BAD_STEP:
{"kind": "<travel|action>", "map": <int>, "talk": <true|false>, "who": "<name or null>",
 "done_when": "<corrected criterion>", "why": "<short>"}"""


class Planner:
    def __init__(self, *, goal_map: int | None = None, level_target: int = 0,
                 heal_hp: float = 0.80, reflector=None, provider=None, strategist=None, knowledge=None):
        self.goal_map = goal_map
        self.level_target = level_target
        self.heal_hp = heal_hp
        self.reflector = reflector      # optional generative strategist (maintains AgentPlan)
        self.provider = provider        # tier-1 chat_json provider (LunaRoute-fast) for targets
        self.strategist = strategist    # tier-2 chat_json provider (LunaRoute-strong) for quests
        self.knowledge = knowledge      # optional Orrery KnowledgeBase for retrieval-grounded quests
        self.on_search = None           # optional callback(query, n_results) for logging KB tool-calls
        self.capture = None             # optional distillation Capture (mounted by the loop; §3)

    def _capture_decision(self, layer, provider, state, content, parsed, latency_ms, usage) -> None:
        """Best-effort per-decision capture (§3): recover the tokens/latency the call site
        discarded (`chat_json` returns (raw, latency_ms, usage); tokens live inside usage).
        Never raises / never changes behavior — guarded on the optional mounted Capture."""
        cap = getattr(self, "capture", None)
        if cap is None:
            return
        tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
        cap.record(layer, model=getattr(provider, "model", None),
                   input=state, output_raw=content, parsed=parsed,
                   tokens=tokens, latency_ms=latency_ms, extra={"usage": usage})

    def _cap_target(self, state, content, target, latency_ms, usage):
        """Record a proposer decision (§3) and return the target UNCHANGED (zero-behavior) — used
        at `propose_target`'s several return points so exactly one record lands per call."""
        self._capture_decision("l2_propose_target", self.provider, state, content, target, latency_ms, usage)
        return target

    def _llm_with_search(self, provider, system: str, state: dict, *, final_key: str,
                         max_rounds: int = 3, capture_layer: str | None = None) -> dict:
        """Run an LLM call where the model MAY use the knowledge base as a tool: each round it
        returns either {"search": [...]} (we run the queries, feed results back as
        KNOWLEDGE_GATHERED) or its final JSON object (which contains ``final_key``). The model
        decides when and what to look up. Returns the final parsed dict.

        Distillation capture (§3, best-effort): when ``capture_layer`` is set, each intermediate
        KB-search round is emitted as a ``model_round`` sharing a ``group_id``, and the final
        parse is emitted as ONE decision record whose ``tokens`` is the sum across rounds and
        whose ``extra.search_queries`` lists what was looked up. Never changes behavior."""
        cap = getattr(self, "capture", None) if capture_layer else None
        gid = f"{capture_layer}-{id(state)}-{time.time_ns()}" if cap is not None else None
        tok_sum = 0
        search_queries: list[str] = []

        def _tok(usage):
            return usage.get("total_tokens", 0) if isinstance(usage, dict) else 0

        gathered: list[dict] = []
        data: dict = {}
        for round_i in range(max_rounds + 1):
            s = {**state, "knowledge_gathered": gathered, "search_rounds_left": max_rounds - round_i}
            try:
                content, _lat, _usage = provider.chat_json(system, s)
                data = json.loads(strip_fences(content))
            except Exception:
                return data
            tok_sum += _tok(_usage)
            queries = data.get("search")
            is_search = (isinstance(queries, list) and queries and self.knowledge is not None
                         and final_key not in data and round_i < max_rounds)
            if not is_search:
                if cap is not None:
                    cap.record(capture_layer, group_id=gid, model=getattr(provider, "model", None),
                               input=s, output_raw=content, parsed=data, tokens=tok_sum,
                               latency_ms=_lat, extra={"search_queries": search_queries,
                                                       "rounds": round_i, "usage": _usage})
                return data
            if cap is not None:
                cap.record_round(capture_layer, group_id=gid, model=getattr(provider, "model", None),
                                 input=s, output_raw=content, parsed=data, tokens=_tok(_usage),
                                 latency_ms=_lat)
            for q in [str(x) for x in queries][:3]:
                search_queries.append(q)
                res = self.knowledge.query_texts(q, top_k=4)
                gathered.append({"query": q, "results": res})
                if self.on_search:
                    self.on_search(q, len(res))
        return data

    def strategize(self, emu, memory=None, *, why: str = "") -> list[Directive]:
        """Tier-2 problem solving: given a blocked situation, return an ORDERED quest of
        directives that unblocks progress (travel/enter to a map, optionally talk there). Empty
        list if no strategist is wired or the plan can't be parsed."""
        prov = self.strategist or self.provider
        if prov is None:
            return []
        cur = emu.read_memory(WCURMAP)
        try:
            state = {
                "why_blocked": why,
                "current_map": {"id": cur, "name": map_name(cur)},
                "goal": f"reach map {self.goal_map} ({map_name(self.goal_map)})" if self.goal_map is not None else "progress",
                "party": [f"{p['species']} L{p['level']}" for p in read_party(emu)],
                "items": [it["item"] for it in read_items(emu)],
                "badges": read_badges(emu)["count"],
                "maps": {str(mid): name for mid, name in MAP_NAMES_RAW.items()},
            }
            # the strategist MAY search the knowledge base as a tool before finalizing the quest
            data = self._llm_with_search(prov, STRATEGIST_SYSTEM, state, final_key="steps",
                                         max_rounds=3, capture_layer="l1_strategize")
            plan_note = str(data.get("plan") or "quest").strip()
            quest: list[Directive] = []
            for step in data.get("steps", []):
                mp = int(step["map"])
                why_s = str(step.get("why") or plan_note)[:80]
                done = self._parse_done_when(step.get("done_when"), mp)
                quest.append(Directive(intent=Intent.TRAVEL, target={"kind": "map", "map": mp},
                                       success={"on_map": mp}, reason=f"quest: go to {map_name(mp)} — {why_s}"))
                if step.get("talk"):
                    # the talk step's acceptance is the MODEL-authored criterion (e.g. has_item:parcel)
                    # so it can't complete on a random dialog; default to talked_on_map only if unset.
                    who = str(step.get("who") or "").strip() or None
                    tgt = {"kind": "npc", "map": mp}
                    if who:
                        tgt["sprite"] = who      # so approach_npc reaches the NAMED person, not the nearest
                    quest.append(Directive(intent=Intent.TALK_TO, target=tgt,
                                           success=(done or {"talked_on_map": mp}),
                                           reason=f"quest: talk to {who or 'someone'} in {map_name(mp)} "
                                                  f"[{step.get('done_when') or 'talked'}] — {why_s}"))
            return quest
        except Exception:
            return []

    def revise_quests(self, emu, context: dict) -> dict:
        """L1 periodic strategic review: look at the whole situation (mission/milestone, the
        standing plan with step statuses, current map, signals) and decide whether the plan needs
        to change. Returns {"assessment","change","mission","milestone","add":[...],"remove":[...]}.

        Uses the strategist provider and MAY ground itself in the Orrery KB via `_llm_with_search`.
        NEVER wipes the plan on a model hiccup: any parse/validation failure — or no provider —
        returns {"change": False, "add": [], "remove": []}. Added steps whose done_when is
        unparseable (or whose map is invalid) are dropped so a bad criterion can't enter the plan."""
        prov = self.strategist or self.provider
        if prov is None:
            return {"change": False, "add": [], "remove": []}
        try:
            state = {
                "current_map": context.get("current_map"),
                "party": context.get("party"),
                "items": context.get("items"),
                "badges": context.get("badges"),
                "plan": context.get("plan"),
                "signals": context.get("signals"),
                "mission": context.get("mission"),
                "milestone": context.get("milestone"),
                "maps": {str(mid): name for mid, name in MAP_NAMES_RAW.items()},
            }
            # L1 MAY search the KB before revising (story gates / where things are)
            data = self._llm_with_search(prov, L1_SYSTEM, state, final_key="assessment")
            if not isinstance(data, dict):
                return {"change": False, "add": [], "remove": []}
            add = []
            for s in (data.get("add") or []):
                if not isinstance(s, dict):
                    continue
                try:
                    mp = int(s.get("map", -1))
                except (TypeError, ValueError):
                    continue
                if Planner._parse_done_when(s.get("done_when"), mp) is not None:
                    add.append(s)
            remove = [str(x) for x in (data.get("remove") or [])]
            return {"assessment": str(data.get("assessment") or ""),
                    "change": bool(data.get("change", bool(add or remove))),
                    "mission": str(data.get("mission") or ""),
                    "milestone": str(data.get("milestone") or ""),
                    "add": add, "remove": remove}
        except Exception:
            return {"change": False, "add": [], "remove": []}

    @staticmethod
    def _goal_fields(context: dict, *, notepad: bool = True) -> dict:
        """The goal view for an L1 state dict. A legacy context (captured inputs, old evals) has
        only mission/milestone -> derive primary/secondary from them so the new prompts still see
        sensible goals."""
        goals = context.get("goals")
        if not isinstance(goals, dict):
            goals = {"primary": {"text": str(context.get("mission") or ""), "done_when": None},
                     "secondary": {"text": str(context.get("milestone") or ""), "done_when": None},
                     "tertiary": {"text": "", "done_when": None}}
        out = {"goals": goals,
               "goal_status": context.get("goal_status") or {t: "none" for t in goals},
               "interrupted": context.get("interrupted")}
        if notepad:
            out["notepad"] = context.get("notepad") or ""
            out["notepad_truncated"] = bool(context.get("notepad_truncated"))
        return out

    @staticmethod
    def _memory_fields(context: dict, *, full: bool = True) -> dict:
        """What happened since the last review, what was tried / heard / not explored, and stall state
        (spec 2026-09-24). Absent keys are simply omitted (old captured contexts, evals)."""
        keys = ("since_last_review", "attempts", "stall") + (("heard", "unexplored_here") if full else ())
        return {k: context[k] for k in keys if context.get(k) not in (None, [], {}, "")}

    def l1_triage(self, context: dict) -> dict:
        """L1 pipeline step 1 (TRIAGE): a cheap, fast, NO-search check of whether the standing
        plan needs to change at all, given the plan + signals. Gates the expensive
        brainstorm/decide calls — most reviews should say "no change" and stop here. Uses
        ``self.provider`` (the fast tier) with plain ``chat_json`` (no KB tool-loop).

        NEVER raises: no provider / parse failure -> {"change": False, "why": ""}."""
        if self.provider is None:
            return {"change": False, "why": ""}
        try:
            state = {
                "plan": context.get("plan"),
                "signals": context.get("signals"),
                **self._goal_fields(context, notepad=False),
                "current_map": context.get("current_map"),
                **self._memory_fields(context, full=False),
            }
            content, _lat, _usage = self.provider.chat_json(TRIAGE_SYSTEM, state)
            data = json.loads(strip_fences(content))
            self._capture_decision("l1_triage", self.provider, state, content, data, _lat, _usage)
            if not isinstance(data, dict):
                return {"change": False, "why": ""}
            return {"change": bool(data.get("change", False)), "why": str(data.get("why") or "")}
        except Exception:
            return {"change": False, "why": ""}

    def l1_brainstorm(self, emu, context: dict) -> dict:
        """L1 pipeline step 2 (BRAINSTORM): open-ended assessment of the situation, letting the
        model drive its own KB searches via ``_llm_with_search`` before concluding. Returns
        {"assessment": str}; NEVER emits step objects itself (that is DECIDE's job).

        NEVER raises: no provider / parse failure -> {"assessment": ""}."""
        prov = self.strategist or self.provider
        if prov is None:
            return {"assessment": ""}
        try:
            state = {
                "current_map": context.get("current_map"),
                "party": context.get("party"),
                "items": context.get("items"),
                "badges": context.get("badges"),
                "signals": context.get("signals"),
                "plan": context.get("plan"),
                **self._goal_fields(context),
                **self._memory_fields(context),
            }
            data = self._llm_with_search(prov, BRAINSTORM_SYSTEM, state, final_key="assessment",
                                         capture_layer="l1_brainstorm")
            if not isinstance(data, dict):
                return {"assessment": ""}
            return {"assessment": str(data.get("assessment") or "")}
        except Exception:
            return {"assessment": ""}

    def l1_decide(self, context: dict, brainstorm: dict) -> dict:
        """L1 pipeline step 3 (DECIDE): turn the BRAINSTORM assessment into a MINIMAL set of quest
        step edits anchored to the existing plan. Plain ``chat_json`` (single call, no KB
        tool-loop — BRAINSTORM already did the searching), robust parse via strip_fences+
        json.loads. Does NOT validate criteria (that is the pipeline's job, task 7) — steps are
        passed through as emitted so a bad one can be routed to ``l1_repair``.

        NEVER raises: no provider / parse failure -> {"add": [], "remove": []}."""
        prov = self.strategist or self.provider
        if prov is None:
            return {"add": [], "remove": []}
        try:
            state = {
                "current_map": context.get("current_map"),
                "party": context.get("party"),
                "items": context.get("items"),
                "badges": context.get("badges"),
                "plan": context.get("plan"),
                "signals": context.get("signals"),
                **self._goal_fields(context),
                **self._memory_fields(context),
                "brainstorm": brainstorm.get("assessment", ""),
                "maps": {str(mid): name for mid, name in MAP_NAMES_RAW.items()},
            }
            content, _lat, _usage = prov.chat_json(DECIDE_SYSTEM, state)
            data = json.loads(strip_fences(content))
            self._capture_decision("l1_decide", prov, state, content, data, _lat, _usage)
            if not isinstance(data, dict):
                return {"add": [], "remove": []}
            add = [s for s in (data.get("add") or []) if isinstance(s, dict)]
            remove = [str(x) for x in (data.get("remove") or [])]
            # optional standing catch goal (§7.1): a species list (or ["any"]) sets it, "clear" clears
            # it; anything else (absent / [] — an echo) = unchanged (tiered-goals spec §3.3).
            raw_catch = data.get("catch")
            if isinstance(raw_catch, list):
                catch = [str(x) for x in raw_catch]
            elif isinstance(raw_catch, str) and raw_catch.strip().lower() == "clear":
                catch = "clear"
            else:
                catch = None
            out = {"assessment": str(data.get("assessment") or ""), "add": add, "remove": remove,
                   "catch": catch,
                   # tiered goals (§3.3): passed through raw; goals.detect_change validates +
                   # decides whether anything really changed. No back-fill from the context.
                   "goals": data.get("goals") if isinstance(data.get("goals"), dict) else None,
                   "notepad": data.get("notepad") if isinstance(data.get("notepad"), str) else None}
            if "interrupted" in data:
                out["interrupted"] = data["interrupted"]
            if isinstance(data.get("lead"), str) and data["lead"].strip():
                out["lead"] = data["lead"].strip()
            ia = data.get("interrupt_active")
            if isinstance(ia, dict) and str(ia.get("why") or "").strip():
                out["interrupt_active"] = {"why": str(ia["why"]).strip()[:300]}
            # legacy keys only if the model still emits them (honored alongside a step edit only)
            for k in ("mission", "milestone"):
                if isinstance(data.get(k), str) and data[k].strip():
                    out[k] = data[k]
            return out
        except Exception:
            return {"add": [], "remove": []}

    def l1_repair(self, context: dict, bad_step: dict, error: str) -> dict:
        """L1 pipeline step (REPAIR): given the DSL grammar, the offending step, and the exact
        validator error, re-emit ONLY that one step (same shape) with a corrected done_when/kind.
        Uses ``self.strategist or self.provider`` with plain ``chat_json``.

        NEVER raises: no provider / parse failure / empty response -> ``bad_step`` unchanged."""
        prov = self.strategist or self.provider
        if prov is None:
            return bad_step
        try:
            state = {
                "bad_step": bad_step,
                "error": error,
                "current_map": context.get("current_map"),
                "maps": {str(mid): name for mid, name in MAP_NAMES_RAW.items()},
            }
            content, _lat, _usage = prov.chat_json(REPAIR_SYSTEM, state)
            data = json.loads(strip_fences(content))
            self._capture_decision("l1_repair", prov, state, content, data, _lat, _usage)
            if not isinstance(data, dict) or not data:
                return bad_step
            return data
        except Exception:
            return bad_step

    def plan(self, intent: Intent, emu, memory=None, *, why: str = "", context: dict | None = None) -> Directive:
        """Compute the concrete directive for ``intent``. ``why`` explains the replan
        (fed to the LLM as Inner Monologue) so it never re-issues a failed directive.

        When the replan is because we're STUCK/impossible on a target-bearing intent, and a
        map view + provider are available, ask LunaRoute for a concrete WAYPOINT tile on the
        current map and route there (the LLM's spatial reasoning gets teeth — BFS alone can't
        un-stick itself)."""
        stuck = context is not None and ("stuck" in why or "impossible" in why)
        if stuck and intent in (Intent.TRAVEL, Intent.GRIND):
            wp = self._waypoint(emu, context)
            if wp is not None:
                cur = emu.read_memory(WCURMAP)
                x, y, reason = wp
                return Directive(intent=intent, target={"kind": "waypoint", "map": cur, "x": x, "y": y},
                                 success={"at_xy": [cur, x, y]}, reason=f"unstuck waypoint: {reason}")
        if intent == Intent.TRAVEL:
            return self._travel(emu, memory, why)
        if intent == Intent.GRIND:
            # grinding needs a destination: grass is en route to the goal, so head that way
            # and level up along the corridor. Success is the level, not arrival.
            cur = emu.read_memory(WCURMAP)
            graph = getattr(memory, "graph", None)
            target = self._goal_travel_target(cur, self.goal_map, graph)
            return Directive(
                intent=Intent.GRIND, target=target,
                success={"level": f">={self.level_target}"} if self.level_target else {"level": ">=999"},
                reason=f"grind party to level {self.level_target} (heading toward grass en route to the goal)",
            )
        if intent == Intent.HEAL:
            return Directive(intent=Intent.HEAL, target=None, success={"hp_frac": f">={self.heal_hp}"},
                             reason="restore party HP")
        if intent == Intent.BATTLE:
            return Directive(intent=Intent.BATTLE, target=None, success={"in_battle": 0},
                             reason="win/flee the current battle")
        return self._travel(emu, memory, why)

    @staticmethod
    def _parse_done_when(done_when, map_id: int) -> dict | None:
        """Turn a model-authored acceptance criterion string into a checkable predicate dict.
        Returns None for an unrecognized/empty criterion (caller supplies a default)."""
        if not done_when or not isinstance(done_when, str):
            return None
        s = done_when.strip()
        low = s.lower()
        if low == "on_map":
            return {"on_map": map_id}
        if low == "talked":
            return {"talked_on_map": map_id}
        if low == "explored":                              # an explore step: done on a discovery
            return {"explored": map_id}
        if low.startswith("level:") and ">=" in s:        # one party member: level:Sophie>=16
            name, _, num = s[len("level:"):].rpartition(">=")
            num = num.strip()
            return {"member_level": [name.strip(), int(num)]} if name.strip() and num.isdigit() else None
        if low.startswith("has_item:") and ">=" in s:     # a COUNT: has_item:Poke Ball>=5
            name, _, num = s[len("has_item:"):].rpartition(">=")
            iid = resolve_item_id(name.strip())
            num = num.strip()
            return {"item_count": [iid, int(num)]} if iid is not None and num.isdigit() and int(num) > 0 else None
        for pre, key in (("has_item:", "has_item"), ("no_item:", "no_item")):
            if low.startswith(pre):
                iid = resolve_item_id(s[len(pre):])
                return {key: iid} if iid is not None else None
        for pre, key in (("level>=", "level"), ("badges>=", "badges")):
            if low.startswith(pre):
                num = s[len(pre):].strip()
                return {key: f">={num}"} if num.isdigit() else None
        if low.startswith("hp_frac>="):  # healed to a fraction of max HP (e.g. after a Poké Center)
            num = s[len("hp_frac>="):].strip()
            try:
                return {"hp_frac": f">={float(num)}"}
            except ValueError:
                return None
        if low.startswith("verify:"):
            return {"verify": s[len("verify:"):].strip()}  # judged by the verifier (not RAM)
        return None

    def next_waypoint(self, emu, context: dict) -> tuple[int, int, str] | None:
        """L2 (tactical navigator): ask LunaRoute for the next grid tile on the CURRENT map to
        head toward, given the ASCII map view + the goal direction / exit tile. BFS then routes
        to it; this is called again on arrival or when BFS can't get closer. VALIDATES the pick
        against the deterministic reachable set (rejecting walls / unreachable picks) and retries
        once. Returns (x, y, reason) or None (caller falls back to the exit tile)."""
        if self.provider is None:
            return None
        player = context.get("player") or {}
        reachable = context.get("reachable")  # set[(x,y)] BFS-reachable, avoiding NPCs
        exit_tile = context.get("exit_tile")  # (x,y) door/edge toward the goal, if known
        state = {
            "objective": context.get("objective"),
            "destination": context.get("destination"),
            "goal_dir": context.get("goal_dir"),
            "player": player,
            "map_view": context.get("map_view"),
            "exit_tile": list(exit_tile) if exit_tile else None,
            "recent_trail": context.get("recent_trail"),
            "recent_waypoints": context.get("recent_waypoints"),
            "why": context.get("why", "pick the next tile toward the goal"),
        }
        for _ in range(2):  # one retry if the model picks an invalid tile
            try:
                content, _lat, _usage = self.provider.chat_json(WAYPOINT_SYSTEM, state)
                data = json.loads(strip_fences(content))
                x, y = int(data["x"]), int(data["y"])
            except Exception:
                continue
            reason = str(data.get("reason") or "").strip() or "head toward the goal corridor"
            px, py = player.get("x"), player.get("y")
            # the pick must be reachable and not the tile we're already on; the exit tile is
            # always a legal pick (heading onto the door is how you leave), even if it's close.
            is_exit = exit_tile is not None and (x, y) == tuple(exit_tile)
            progresses = (px is None) or (x, y) != (px, py)
            ok = progresses and (is_exit or reachable is None or (x, y) in reachable)
            if ok:
                self._capture_decision("l2_next_waypoint", self.provider, state, content,
                                       {"x": x, "y": y, "reason": reason}, _lat, _usage)
                return x, y, reason
        return None

    # backward-compat alias (was the stuck-only deadlock breaker)
    _waypoint = next_waypoint

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
            "focus": context.get("focus"),   # L1's tertiary goal (the immediate focus), if any
            "destination": context.get("destination"),
            "goal_dir": context.get("goal_dir"),
            "player": context.get("player"),
            "map_view": context.get("map_view"),
            "exit_tile": list(exit_tile) if exit_tile else None,
            "candidate_exits": context.get("candidate_exits"),
            "npcs": context.get("npcs"),
            "recent_trail": context.get("recent_trail"),
            "recent_targets": context.get("recent_targets"),
            "default": default,
            "why": context.get("why", "pick the next target toward the goal"),
        }
        reason = "no response"
        last_raw = None
        content, _lat, _usage = None, 0, {}   # bound for the unresolved-after-failure record
        for _ in range(2):  # one retry on an invalid pick
            try:
                content, _lat, _usage = self.provider.chat_json(PROPOSER_SYSTEM, state)
                last_raw = content
                data = json.loads(strip_fences(content))
                kind = data.get("kind")
            except Exception as e:
                reason = f"call/parse failed: {e}"
                continue
            note = str(data.get("why") or data.get("note") or "").strip()
            if kind == "tile" or (kind is None and "x" in data and "y" in data):
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
                    return self._cap_target(state, content, {"kind": "tile", "x": x, "y": y, "note": note}, _lat, _usage)
                reason = f"tile ({x},{y}) unreachable / no-op"
                continue
            if kind == "enter":
                try:
                    return self._cap_target(state, content, {"kind": "enter", "map": int(data["map"]), "note": note}, _lat, _usage)
                except (KeyError, TypeError, ValueError):
                    reason = "enter missing map"
                    continue
            if kind == "approach_npc":
                sprite = (data.get("sprite") or None)
                if not sprite:
                    reason = "approach_npc missing sprite"
                    continue
                return self._cap_target(state, content, {"kind": "approach_npc", "sprite": sprite, "note": note}, _lat, _usage)
            if kind == "exit":
                return self._cap_target(state, content, {"kind": "exit", "note": note}, _lat, _usage)
            reason = f"unknown kind {kind!r}"
        # a wired provider failed to produce a usable target -> surface it; do NOT guess deterministically
        return self._cap_target(state, content, {"kind": "unresolved", "reason": reason,
                "raw": (str(last_raw)[:300] if last_raw else None),
                "note": f"proposer failed: {reason}"}, _lat, _usage)

    # --- travel: LLM picks the target map; graph does the geometry + fallback --
    def _travel(self, emu, memory, why: str) -> Directive:
        cur = emu.read_memory(WCURMAP)
        goal = self.goal_map
        if goal is None or cur == goal:
            return Directive(intent=Intent.TRAVEL, target=None,
                             success={"on_map": goal if goal is not None else -1},
                             reason="hold position / explore (no distinct goal map)")
        graph = getattr(memory, "graph", None)
        target_map, reason = self._choose_target_map(cur, goal, graph, memory, why)
        return self._travel_directive(cur, target_map, goal, graph, reason)

    def _choose_target_map(self, cur, goal, graph, memory, why) -> tuple[int, str]:
        """The default target is the FINAL goal (bold commitment — one persistent travel
        directive across the corridor; the servo walks the route each step). The LLM is asked
        to reroute to an INTERMEDIATE reachable map only when a replan needs it; any problem
        falls back to the goal."""
        default = goal
        default_reason = f"travel toward map {goal} ({map_name(goal)})"
        if self.provider is None or graph is None:
            return default, default_reason
        try:
            route = graph.route(cur, goal) or []
            neighbors = [{"map": m, "name": map_name(m), "dir": graph.direction_between(cur, m)}
                         for m in graph.neighbors(cur)]
            state = {
                "mission": f"reach map {goal} ({map_name(goal)}) — the next gym town",
                "goal_map": goal,
                "current_map": {"id": cur, "name": map_name(cur)},
                "route": [{"map": m, "name": map_name(m)} for m in route],
                "neighbors": neighbors,
                "map_history": list(getattr(memory, "map_history", []) or [])[-10:],
                "tried_failed": list(getattr(memory, "tried_failed", []) or []),
                "why_replan": why or "starting a new travel directive",
            }
            # the tier-1 planner MAY search the KB as a tool before choosing the reroute target
            data = self._llm_with_search(self.provider, PLANNER_SYSTEM, state, final_key="target_map",
                                         capture_layer="l2_travel_reroute")
            tm = int(data.get("target_map"))
            reason = str(data.get("reason") or "").strip() or default_reason
            # validate: must be reachable and not the current map
            if tm != cur and graph.route(cur, tm):
                return tm, reason
        except Exception:
            pass
        return default, default_reason

    def _goal_travel_target(self, cur, target_map, graph) -> dict | None:
        """A target dict toward ``target_map``, with the exact exit tile on the CURRENT map
        attached when the graph knows it (so the servo BFS walks straight to it)."""
        if target_map is None:
            return None
        target = {"kind": "map", "map": target_map}
        if graph is not None:
            hop = graph.next_hop(cur, target_map)
            if hop is not None:
                next_map, tile = hop
                target = {"kind": "warp", "map": target_map, "next_map": next_map}
                if tile is not None:
                    target["x"], target["y"] = int(tile[0]), int(tile[1])
        return target

    def _travel_directive(self, cur, target_map, goal, graph, reason) -> Directive:
        """Build the travel directive toward ``target_map``."""
        target = self._goal_travel_target(cur, target_map, graph)
        # success keys off the FINAL goal when heading straight there, else the chosen subgoal
        success_map = goal if target_map == goal else target_map
        return Directive(intent=Intent.TRAVEL, target=target, success={"on_map": success_map},
                         reason=reason)
