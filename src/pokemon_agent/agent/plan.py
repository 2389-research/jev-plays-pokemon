"""The typed Plan + Directive contract (planner/executor architecture).

Two levels, one file:

* ``AgentPlan`` — the *strategic* wrapper the planner (LunaRoute) maintains:
  mission / milestone / hypotheses / tried_failed. Long-lived, rarely rewritten.
* ``Directive`` — the *option with teeth* the executor obeys. Exactly ONE is
  active at a time (near-bold commitment); a small suspension ``stack`` holds
  directives a higher-priority need preempted. A directive names a closed
  ``intent``, a concrete ``target``, and a **machine-checkable ``success``**
  predicate (the option's termination β). The executor's action repertoire is
  masked/biased by the directive — it is a bounded chooser, not a free agent.

The reflection fields (progress / plan / next_objective / explore_note) stay so
the existing LunaRoute reflect prompt keeps working; ``ReflectionPlan`` remains an
alias of ``AgentPlan``.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """The closed set of intents. Selects the executor's action repertoire."""
    TRAVEL = "travel"        # go to a map/tile
    TALK_TO = "talk_to"      # reach an NPC and interact
    GRAB_ITEM = "grab_item"  # reach an item/ball and pick it up
    ENTER = "enter"          # step onto a specific warp/door
    HEAL = "heal"            # restore party HP (Poké Center)
    GRIND = "grind"          # raise party level (no fixed tile target)
    SHOP = "shop"            # buy at a Mart
    BATTLE = "battle"        # in-battle; the battle controller owns the turn


# Intents whose directive carries a concrete `target` the deterministic servo (BFS/graph)
# routes to. GRIND is target-bearing when the planner attaches a travel-toward-goal target
# (grass is en route, so grinding happens as you walk the route); BATTLE is owned by the
# mode controller and never target-bearing (spec §3/§5).
TARGET_BEARING: frozenset[Intent] = frozenset(
    {Intent.TRAVEL, Intent.TALK_TO, Intent.GRAB_ITEM, Intent.ENTER, Intent.HEAL, Intent.SHOP, Intent.GRIND}
)


class Directive(BaseModel):
    """One active option the executor is mechanically bound to.

    ``target`` is a typed dict ``{kind, map, x, y, id}`` (any subset). ``success`` is
    a machine-checkable predicate dict evaluated against RAM every step
    (see ``games/pokemon_red/predicates.py``) — the single most important field, because
    a directive without a checkable termination never *commits*.
    """
    intent: Intent
    target: dict | None = Field(default=None, description="{kind, map, x, y, id} — concrete + typed")
    success: dict = Field(default_factory=dict,
                          description="machine-checkable predicate, e.g. {'on_map': 2} | {'hp_frac': '>=0.8'}")
    allowed_options: list[str] | None = Field(default=None,
                                              description="optional explicit whitelist for the executor's Choice")
    option_bias: list[str] = Field(default_factory=list, description="options to prefer (SayCan prior)")
    reason: str = Field(default="", description="provenance / why — for logs + replan feedback")
    quest_id: str | None = Field(default=None, description="id of the QuestStep this directive was compiled from")

    @property
    def target_bearing(self) -> bool:
        return self.intent in TARGET_BEARING and self.target is not None

    @property
    def target_map(self) -> int | None:
        return int(self.target["map"]) if self.target and self.target.get("map") is not None else None

    @property
    def target_xy(self) -> tuple[int, int] | None:
        if self.target and self.target.get("x") is not None and self.target.get("y") is not None:
            return int(self.target["x"]), int(self.target["y"])
        return None


class AgentPlan(BaseModel):
    # --- mission / milestone framing ---
    mission: str = Field(default="", description="the overall mission")
    milestone: str = Field(default="", description="the current milestone being pursued")

    # --- reflection fields (kept for backward-compat with the LunaRoute reflect prompt) ---
    progress: str = Field(default="", description="what has been accomplished / found so far")
    plan: str = Field(default="", description="the strategy to reach the goal from here")
    next_objective: str = Field(default="", description="the concrete objective for the next steps")
    explore_note: str = Field(default="", description="which unexplored area to check, if searching")

    # --- executor-steering contract (legacy levers, kept for the generative reasoner) ---
    mode_hint: str = Field(default="overworld", description="overworld|battle|menu|shop|route")
    option_bias: list[str] = Field(default_factory=list,
                                   description="option keys to add/prefer in the executor's Choice set")
    target: dict | None = Field(default=None, description="a concrete target: {kind, map, x, y, item, ...}")

    # --- the active option (planner/executor contract) ---
    directive: Directive | None = Field(default=None, description="the ONE active directive the executor obeys")
    stack: list[Directive] = Field(default_factory=list,
                                   description="suspended directives (a higher need preempted them)")

    # --- strategy / anti-stuck (option generation + tried-and-failed ledger) ---
    hypotheses: list[str] = Field(default_factory=list,
                                  description="alternative approaches to try if the current one fails")
    tried_failed: list[str] = Field(default_factory=list,
                                    description="approaches already shown not to work; do not retry")

    @property
    def objective(self) -> str:
        """Canonical objective string (aliases the reflection field)."""
        return self.next_objective or self.milestone or self.plan


# Backward-compatible alias: the reason loop / reasoner refer to `ReflectionPlan`.
ReflectionPlan = AgentPlan
