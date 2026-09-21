"""Offline unit tests for the flow-gate ROUTER (scripts/probe_flow_gate.py::route).

Pure logic over canned Jev answers — no network. The live probe (run_eval) is manual/gated.
"""
from scripts.probe_flow_gate import route


def test_confident_dialogue_routes_to_dialogue():
    assert route({"dialogue": ("yes", 0.9), "menu": ("no", 0.8)}, ram_menu_open=False) == "dialogue"


def test_no_text_routes_to_navigate():
    assert route({"dialogue": ("no", 0.9), "menu": ("no", 0.9)}, ram_menu_open=False) == "navigate"


def test_ram_menu_open_wins_even_if_jev_unsure():
    # A on a menu SELECTS, so the RAM menu signal must route to menu regardless of the dialogue answer.
    assert route({"dialogue": ("yes", 0.9), "menu": ("no", 0.3)}, ram_menu_open=True) == "menu"


def test_confident_jev_menu_cross_checks_to_menu():
    assert route({"dialogue": ("yes", 0.9), "menu": ("yes", 0.8)}, ram_menu_open=False) == "menu"


def test_hedged_dialogue_falls_through_to_navigate():
    # below FLOW_MIN_CONF -> don't trust the dialogue pick
    assert route({"dialogue": ("yes", 0.4), "menu": ("no", 0.9)}, ram_menu_open=False) == "navigate"
