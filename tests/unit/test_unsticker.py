"""The screen unsticker: when the agent loops on a text/menu screen it doesn't understand, a model takes
temporary control for a multi-step EPISODE (each turn sees the screen + its own history), then hands back
when it says done, when play looks normal again, or when its press budget runs out."""
from __future__ import annotations

import json

from pokemon_agent.agent.unsticker import Unsticker, looks_normal


def _obs(u, screen, hp=50, text_screen=True):
    u.observe((65, 4, 3, 1, 30, hp, 3, 1193), screen, text_screen=text_screen)


class Script:
    """A provider that replays JSON answers and records what it was shown."""
    def __init__(self, answers):
        self.answers, self.seen = list(answers), []

    def chat_json(self, system, state, image=None):
        self.seen.append(state)
        return json.dumps(self.answers.pop(0) if self.answers else {"button": "A", "done": False}), 0, {}


CYCLE = ["Dylan is trying to learn BITE!", "But, Dylan can't learn more", "than 4 moves!", "Delet",
         "Delete an older move to", "make room for BITE?", "Which", "Which move should be forgotten?"]


def test_triggers_on_a_repeating_cycle_with_scrolling_text():
    u = Unsticker(Script([]))
    for i in range(40):
        _obs(u, CYCLE[i % len(CYCLE)])          # a long cycle of scrolling frames, nothing progressing
    assert u.should_start(step=100)


def test_no_trigger_for_a_long_one_off_dialogue_a_progressing_battle_or_the_overworld():
    u = Unsticker(Script([]))
    for i in range(40):
        _obs(u, f"line {i} of Oak's long speech")          # never repeats
    assert not u.should_start(step=100)
    u2 = Unsticker(Script([]))
    for i in range(40):
        _obs(u2, CYCLE[i % 3], hp=50 - i)                   # HP changes: the battle is moving
    assert not u2.should_start(step=100)
    u3 = Unsticker(Script([]))
    for _ in range(40):
        _obs(u3, "", text_screen=False)
    assert not u3.should_start(step=100)


def test_an_episode_is_multi_step_with_history_and_ends_when_the_model_says_done():
    prov = Script([{"button": "A", "done": False, "reason": "yes, learn it"},
                   {"button": "DOWN", "done": False, "reason": "cursor to Tail Whip"},
                   {"button": "A", "done": False, "reason": "forget it"},
                   {"button": "A", "done": True, "reason": "learned; back to the fight"}])
    events, pressed = [], []
    u = Unsticker(prov, on_event=lambda k, p: events.append(k), press=lambda b: pressed.append(b))
    u.start(step=10, screen="Delete an older move?")
    for _ in range(6):
        if not u.active:
            break
        u.turn({"screen": ["Delete an older move?"], "menu": {"open": True}}, normal=False, step=11)
    assert pressed == ["A", "DOWN", "A", "A"] and not u.active
    assert len(prov.seen[2]["history"]) == 2 and prov.seen[2]["history"][1]["pressed"] == "DOWN"
    assert events[0] == "unstick_start" and events[-1] == "unstick_end"


def test_ends_on_normal_play_or_budget_then_cools_down():
    u = Unsticker(Script([{"button": "B", "done": False}] * 20), budget=3, press=lambda b: None)
    u.start(step=10, screen="?")
    for _ in range(5):
        if u.active:
            u.turn({"screen": ["?"]}, normal=False, step=12)
    assert not u.active and u.last_end == "budget" and not u.should_start(step=20)
    u2 = Unsticker(Script([{"button": "B", "done": False}]), press=lambda b: None)
    u2.start(step=10, screen="?")
    u2.turn({"screen": []}, normal=True, step=11)
    assert not u2.active and u2.last_end == "normal"


def test_looks_normal():
    assert looks_normal(in_battle=False, text="", menu_open=False, fight_menu=False)
    assert looks_normal(in_battle=True, text="FIGHT PKMN", menu_open=True, fight_menu=True)
    assert not looks_normal(in_battle=True, text="Which move should be forgotten?", menu_open=True, fight_menu=False)
