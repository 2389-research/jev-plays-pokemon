"""Interaction reliability (spec docs/superpowers/specs/2026-09-22-interaction-reliability-design.md).

Regression tests for the Oak's Parcel delivery incident, each reproduced from a recorded run:

* F1 — `interact` must register: a 1-frame A tap aliases with Gen 1's joypad sampling and can
  miss every time (`oak_tap_alias`: at (5,3) facing Oak, parcel in bag — the 1839 run pressed A
  107x here with zero dialog).
* F2 — `read_npcs` must not report sprites the game has hidden (missable-object flags).
* F3.1 — a move must not return in the middle of a warp (map id flipped, coords/sprites stale).

Fixtures are local save states (states/*.state is gitignored); the tests skip when absent, like
the other ROM-backed tests. See the spec §4 for where each fixture was captured.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROM_PATH = ROOT / "roms" / "pokemon_red.gb"
STATES = ROOT / "states"


def _need(*names: str) -> None:
    if not ROM_PATH.exists() or not all((STATES / n).exists() for n in names):
        pytest.skip("ROM/fixture not present")


@pytest.fixture(scope="module")
def emu():
    if not ROM_PATH.exists():
        pytest.skip("ROM not present")
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator

    e = PyBoyEmulator(str(ROM_PATH), window="null")
    yield e
    e.close()


def _pos(emu):
    return emu.read_memory(0xD35E), emu.read_memory(0xD362), emu.read_memory(0xD361)


# ---------------------------------------------------------------- F1: held press registers
def test_interact_opens_oak_dialog_at_every_frame_offset(emu):
    """One `interact` in front of Oak opens his text regardless of frame phase."""
    _need("oak_tap_alias.state")
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.core.models import InteractAction
    from pokemon_agent.games.pokemon_red.game_state import read_screen_text

    missed = []
    for pre in range(8):
        emu.load_state(STATES / "oak_tap_alias.state")
        emu.tick(pre)
        ActionController(emu).execute(InteractAction())
        if not read_screen_text(emu)[1]:
            missed.append(pre)
    assert missed == [], f"A press not registered at frame offsets {missed}"


def test_held_advance_reaches_battle_root_menu_without_auto_selecting(emu):
    """Regression: battle text goes through the same press path. A held A must advance it and
    land on the ROOT battle menu (FIGHT/ITEM/RUN) — never auto-select into a submenu."""
    _need("battle.state")
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.core.models import AdvanceDialogAction
    from pokemon_agent.games.pokemon_red import menus
    from pokemon_agent.games.pokemon_red.game_state import read_screen_text

    emu.load_state(STATES / "battle.state")
    ctl = ActionController(emu)
    for _ in range(20):
        ctl.execute(AdvanceDialogAction())
        if menus.read_menu(emu).get("open"):
            break
    text = read_screen_text(emu)[0]
    assert menus.read_menu(emu).get("open"), "battle text never reached the battle menu"
    assert "FIGHT" in text and "RUN" in text, f"not the root battle menu: {text!r}"


# ---------------------------------------------------------------- F2: hidden sprites
def test_read_npcs_skips_hidden_sprites_in_oaks_lab(emu):
    _need("lab_deliver.state")
    from pokemon_agent.games.pokemon_red.game_state import read_npcs

    emu.load_state(STATES / "lab_deliver.state")
    seen = {(n["sprite"], n["x"], n["y"]) for n in read_npcs(emu)}
    for phantom in [("Oak", 5, 10), ("Blue", 4, 3), ("Poke Ball", 7, 3), ("Poke Ball", 8, 3)]:
        assert phantom not in seen, f"hidden sprite reported: {phantom}"
    assert ("Oak", 5, 2) in seen            # the real, visible Oak
    assert ("Poke Ball", 6, 3) in seen       # the one ball still on the table


def test_read_npcs_skips_hidden_intro_oak_in_pallet(emu):
    _need("pallet_ready.state")
    from pokemon_agent.games.pokemon_red.game_state import read_npcs

    emu.load_state(STATES / "pallet_ready.state")
    assert "Oak" not in {n["sprite"] for n in read_npcs(emu)}


# ---------------------------------------------------------------- F3.1: never return mid-warp
def test_entering_lab_door_returns_settled_inside(emu):
    _need("lab_door_warp.state")
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.core.models import Direction, MoveAction
    from pokemon_agent.games.pokemon_red.game_state import read_npcs

    emu.load_state(STATES / "lab_door_warp.state")
    ActionController(emu).execute(MoveAction(direction=Direction.NORTH))
    assert _pos(emu) == (40, 5, 11), f"returned mid-warp: {_pos(emu)}"
    assert "Scientist" in {n["sprite"] for n in read_npcs(emu)}   # lab sprites loaded


def test_leaving_lab_returns_settled_in_pallet(emu):
    _need("lab_exit_mat.state")
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.core.models import Direction, MoveAction
    from pokemon_agent.games.pokemon_red.game_state import read_npcs

    emu.load_state(STATES / "lab_exit_mat.state")
    ActionController(emu).execute(MoveAction(direction=Direction.SOUTH))
    m, x, y = _pos(emu)
    assert m == 0 and (x, y) in {(12, 11), (12, 12)}, f"returned mid-warp: {(m, x, y)}"
    assert "Scientist" not in {n["sprite"] for n in read_npcs(emu)}   # Pallet sprites, not the lab's


def test_map_edge_connection_does_not_wait_for_a_warp(emu):
    _need("pallet_route1_edge.state")
    from pokemon_agent.actions.controller import ActionController
    from pokemon_agent.core.models import Direction, MoveAction

    emu.load_state(STATES / "pallet_route1_edge.state")
    res = ActionController(emu).execute(MoveAction(direction=Direction.NORTH))
    assert _pos(emu)[0] == 12, f"did not cross to Route 1: {_pos(emu)}"
    assert res.frames_elapsed < 40, f"edge crossing waited {res.frames_elapsed} frames"
