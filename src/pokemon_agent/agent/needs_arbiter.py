"""Arbitration over the RAM-verified needs stack (the 'hierarchy of needs' controller).

Design (from the research synthesis, docs/agent-to-brock-spec.md §12b):
- **Bucketed priority stack** (subsumption for the tiers): SURVIVE > BATTLE > READINESS >
  PROGRESS. A higher active need suppresses lower ones each step.
- **Hysteresis** so it doesn't thrash at a threshold: once SURVIVE latches (HP < low_hp
  or a mon fainted) it stays latched until HP recovers to heal_hp — "once you decide to
  heal, heal up," not flip the instant you cross the line. (READINESS has no chatter: its
  exit — level ≥ target — is a hard ceiling.)
- **Step budget** on READINESS so grinding can't starve PROGRESS forever.
- **RAM owns activation/termination** (needs.py predicates); this class only arbitrates.
  The strategist owns the *parameters* (goal_map, level_target, thresholds), not the truth.
"""
from __future__ import annotations

from ..emulator.interface import Emulator
from ..games.pokemon_red import needs


class NeedsArbiter:
    def __init__(self, *, goal_map: int | None = None, level_target: int = 0,
                 low_hp: float = 0.30, heal_hp: float = 0.80, grind_budget: int = 250):
        self.goal_map = goal_map
        self.level_target = level_target
        self.low_hp = low_hp
        self.heal_hp = heal_hp
        self.grind_budget = grind_budget
        self._survive_latched = False
        self._grind_steps = 0
        self._current = "idle"

    def current(self, emu: Emulator) -> needs.Need:
        """The need the agent should pursue right now (highest active, with hysteresis
        and the grind budget applied). Call once per step."""
        hp = needs.party_hp_fraction(emu)
        fainted = needs.any_fainted(emu)
        has_party = needs.has_party(emu)

        # SURVIVE hysteresis latch (dual threshold)
        if self._survive_latched:
            if hp >= self.heal_hp and not fainted:
                self._survive_latched = False
        elif has_party and (hp < self.low_hp or fainted):
            self._survive_latched = True

        stack = needs.assess_needs(emu, goal_map=self.goal_map,
                                   level_target=self.level_target, low_hp=self.low_hp)
        for n in stack:
            if n.name == "survive":
                n.active = self._survive_latched  # latch overrides the raw predicate

        top = None
        for n in stack:  # highest priority first
            if not n.active:
                continue
            if n.name == "readiness" and self._grind_steps >= self.grind_budget:
                continue  # grind budget spent -> let PROGRESS run (anti-starvation)
            top = n
            break
        top = top or needs.Need("idle", False, 0, "nothing to do")

        self._grind_steps = self._grind_steps + 1 if top.name == "readiness" else 0
        self._current = top.name
        return top

    def intent(self, emu: Emulator):
        """The executor INTENT for the current need (spec §4). This is the arbiter's ONLY
        steering output now — it no longer picks a goal map or route; the planner computes
        the concrete target from this intent. SURVIVE→heal, BATTLE→battle, READINESS→grind,
        PROGRESS→travel."""
        from .planner_llm import intent_for_need
        return intent_for_need(self.current(emu).name)

    @property
    def needs_heal(self) -> bool:
        return self._current == "survive"
