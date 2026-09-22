"""Unit tests (no ROM) for the battle_L2 objective layer + Jev choose_action (design §2.2/§2.3).

`choose_objective` and `choose_action` are pure functions over a plain state dict, so
crafted encounters pin the deterministic policy without touching the emulator.
"""
from __future__ import annotations

from pokemon_agent.games.pokemon_red import battle_agent
from pokemon_agent.games.pokemon_red.battle_l2 import (
    CAPTURE,
    ESCAPE,
    GRIND_EXP,
    SURVIVE,
    choose_objective,
)

BALLS_AND_POTIONS = [
    {"item": "Poke Ball", "qty": 5},
    {"item": "Potion", "qty": 3},
]


def _state(*, is_trainer=False, enemy=None, active=None, party=None, items=None):
    enemy = enemy or {"species": "Pidgey", "level": 5, "hp": 20, "max_hp": 20}
    active = active or {"species": "Squirtle", "level": 6, "hp": 22, "max_hp": 22}
    party = party if party is not None else [active]
    items = items if items is not None else list(BALLS_AND_POTIONS)
    return {"enemy": enemy, "active": active, "party": party,
            "items": items, "is_trainer": is_trainer}


# ---------------------------------------------------------------- choose_objective

def test_trainer_battle_is_grind():
    # A healthy trainer battle grinds it down (the current default path).
    assert choose_objective(_state(is_trainer=True)) == GRIND_EXP


def test_trainer_battle_endangered_is_survive():
    hurt = {"species": "Squirtle", "level": 6, "hp": 5, "max_hp": 22}  # ~23%
    assert choose_objective(_state(is_trainer=True, active=hurt, party=[hurt])) == SURVIVE


def test_wild_with_catch_goal_slot_and_ball_is_capture():
    obj = choose_objective(_state(), {"catch": ["Pidgey"]})
    assert obj == CAPTURE


def test_wild_catch_goal_wildcard_matches_any_species():
    assert choose_objective(_state(), {"catch": ["any"]}) == CAPTURE


def test_wild_catch_goal_wrong_species_is_grind():
    assert choose_objective(_state(), {"catch": ["Rattata"]}) == GRIND_EXP


def test_wild_no_ball_falls_back_to_grind():
    obj = choose_objective(_state(items=[{"item": "Potion", "qty": 1}]), {"catch": ["Pidgey"]})
    assert obj == GRIND_EXP


def test_wild_full_party_cannot_capture():
    full = [{"species": f"m{i}", "hp": 10, "max_hp": 10} for i in range(6)]
    st = _state(party=full)
    assert choose_objective(st, {"catch": ["Pidgey"]}) == GRIND_EXP


def test_wild_critical_hp_is_escape():
    dying = {"species": "Squirtle", "level": 6, "hp": 2, "max_hp": 22}  # ~9%
    # ESCAPE takes priority over a catch goal when our mon is about to faint.
    assert choose_objective(_state(active=dying, party=[dying]), {"catch": ["Pidgey"]}) == ESCAPE


def test_wild_no_goal_is_grind():
    assert choose_objective(_state()) == GRIND_EXP


# ---------------------------------------------------------------- choose_action

def test_action_grind_is_a_move():
    assert battle_agent.choose_action(GRIND_EXP, _state())["kind"] == "move"


def test_action_escape_is_run():
    assert battle_agent.choose_action(ESCAPE, _state())["kind"] == "run"


def test_action_survive_low_hp_uses_potion():
    hurt = {"species": "Squirtle", "hp": 4, "max_hp": 22}
    act = battle_agent.choose_action(SURVIVE, _state(active=hurt))
    assert act["kind"] == "item"
    assert "potion" in act["item"].lower()


def test_action_survive_no_potion_falls_back_to_move():
    hurt = {"species": "Squirtle", "hp": 4, "max_hp": 22}
    act = battle_agent.choose_action(SURVIVE, _state(active=hurt, items=[{"item": "Poke Ball", "qty": 1}]))
    assert act["kind"] == "move"


def test_action_capture_full_hp_weakens_with_move():
    # A wild big enough to survive a hit (max_hp > CATCH_SMALL_MAX_HP): weaken with a move first.
    full = {"species": "Pidgeotto", "hp": 60, "max_hp": 60}
    assert battle_agent.choose_action(CAPTURE, _state(enemy=full))["kind"] == "move"


def test_action_capture_small_full_hp_throws_directly():
    # A small wild (max_hp <= CATCH_SMALL_MAX_HP): the max-damage move would KO it, losing the catch,
    # so throw immediately instead of weakening (finding #1).
    small = {"species": "Caterpie", "hp": 20, "max_hp": 20}
    act = battle_agent.choose_action(CAPTURE, _state(enemy=small))
    assert act["kind"] == "ball"
    assert "ball" in act["item"].lower()


def test_action_capture_low_hp_throws_ball():
    weak = {"species": "Pidgey", "hp": 3, "max_hp": 20}  # ~15%, below the catch band
    act = battle_agent.choose_action(CAPTURE, _state(enemy=weak))
    assert act["kind"] == "ball"
    assert "ball" in act["item"].lower()


def test_action_capture_no_ball_falls_back_to_move():
    weak = {"species": "Pidgey", "hp": 3, "max_hp": 20}
    act = battle_agent.choose_action(CAPTURE, _state(enemy=weak, items=[{"item": "Potion", "qty": 1}]))
    assert act["kind"] == "move"


# ---------------------------------------------------------------- safety_override (§2.1)

from pokemon_agent.games.pokemon_red.battle_l2 import (  # noqa: E402
    battle_goals_from_plan,
    safety_override,
)

CRITICAL = {"species": "Squirtle", "level": 6, "hp": 2, "max_hp": 22}  # ~9%, below critical band


def test_safety_override_capture_wild_flips_to_escape():
    # CAPTURE + our mon at critical HP in a WILD battle -> ESCAPE (cut losses, don't faint).
    assert safety_override(CAPTURE, _state(active=CRITICAL)) == ESCAPE


def test_safety_override_grind_trainer_flips_to_survive():
    # GRIND-EXP + critical HP in a TRAINER battle (can't run) -> SURVIVE (heal).
    assert safety_override(GRIND_EXP, _state(is_trainer=True, active=CRITICAL)) == SURVIVE


def test_safety_override_healthy_mon_is_unchanged():
    assert safety_override(CAPTURE, _state()) == CAPTURE
    assert safety_override(GRIND_EXP, _state()) == GRIND_EXP


def test_safety_override_leaves_survive_and_escape_alone():
    # Already-defensive objectives are never overridden, even at critical HP.
    assert safety_override(SURVIVE, _state(active=CRITICAL)) == SURVIVE
    assert safety_override(ESCAPE, _state(active=CRITICAL)) == ESCAPE


# ---------------------------------------------------------------- battle_goals_from_plan (§7.1)

def test_battle_goals_from_plan_carries_catch_and_level_target():
    goals = battle_goals_from_plan({"catch": ["Pidgey"]}, level_target=12)
    assert goals == {"catch": ["Pidgey"], "level_target": 12}


def test_battle_goals_from_plan_defaults_empty():
    assert battle_goals_from_plan(None) == {"catch": [], "level_target": 0}
    assert battle_goals_from_plan({}) == {"catch": [], "level_target": 0}


def test_derived_goals_drive_capture_objective():
    # End-to-end of the pure layer: a plan catch goal -> derived goals -> CAPTURE objective.
    goals = battle_goals_from_plan({"catch": ["Kakuna"]}, level_target=10)
    state = _state(enemy={"species": "Kakuna", "hp": 20, "max_hp": 20})
    assert choose_objective(state, goals) == CAPTURE
