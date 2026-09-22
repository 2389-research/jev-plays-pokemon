"""Unit test (no ROM) for the L1 -> battle_L2 goal plumbing (design §7.1, Phase 4).

L1 sets a compact `battle_goals` field on the plan (e.g. `{"catch": ["Pidgey"]}`). The loop
derives the battle layer's compact `{"catch": [...], "level_target": int}` from that plan field
plus its level target, so `battle_L2` sees the current goals. Here we exercise the derivation and
its effect on the objective without touching the emulator.
"""
from __future__ import annotations

from pokemon_agent.agent.plan import AgentPlan
from pokemon_agent.games.pokemon_red.battle_l2 import (
    CAPTURE,
    GRIND_EXP,
    battle_goals_from_plan,
    choose_objective,
)


def _state(enemy_species="Pidgey"):
    return {
        "enemy": {"species": enemy_species, "hp": 20, "max_hp": 20},
        "active": {"species": "Squirtle", "hp": 22, "max_hp": 22},
        "party": [{"species": "Squirtle", "hp": 22, "max_hp": 22}],
        "items": [{"item": "Poke Ball", "qty": 5}],
        "is_trainer": False,
    }


def test_plan_default_has_no_catch_goal():
    plan = AgentPlan()
    goals = battle_goals_from_plan(plan.battle_goals, level_target=0)
    assert goals == {"catch": [], "level_target": 0}
    # Empty goals -> the default GRIND objective (nothing regresses).
    assert choose_objective(_state(), goals) == GRIND_EXP


def test_plan_catch_goal_derives_capture():
    plan = AgentPlan(battle_goals={"catch": ["Pidgey"]})
    goals = battle_goals_from_plan(plan.battle_goals, level_target=12)
    assert goals == {"catch": ["Pidgey"], "level_target": 12}
    # The matching wild encounter becomes a CAPTURE objective.
    assert choose_objective(_state("Pidgey"), goals) == CAPTURE
    # A non-matching species stays GRIND.
    assert choose_objective(_state("Rattata"), goals) == GRIND_EXP


def test_plan_catch_any_matches_anything():
    plan = AgentPlan(battle_goals={"catch": ["any"]})
    goals = battle_goals_from_plan(plan.battle_goals)
    assert choose_objective(_state("Weedle"), goals) == CAPTURE
