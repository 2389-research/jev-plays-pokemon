"""The screen unsticker — a model takes over when the agent loops on a screen it doesn't understand.

Deterministic handlers cover the screens we know (nickname, learn-move, switch, No PP ...). When a new
one shows up, the loop typically presses the same button against it forever. The unsticker watches for
that (a TEXT/MENU screen cycling through <= a few states with nothing else changing), then hands control
to a fast model for an EPISODE: each turn it sees the current screen, the menu/cursor, the objective and
its own history this episode, and presses ONE button. The episode ends when the model says it's done,
when play looks normal again (walking the map, or the battle's FIGHT menu), or when its press budget runs
out — then a cooldown. Every episode is logged so recurring screens can become deterministic handlers.
"""
from __future__ import annotations

import json
from collections import deque

from ..providers.parsing import strip_fences

BUTTONS = ("A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT")

UNSTICK_SYSTEM = """You are taking temporary control of an agent playing Pokémon Red because it is STUCK: the same
screen keeps coming back and its normal logic isn't getting past it. Each turn you get the current SCREEN (the
game's text tiles, row by row), MENU (whether a menu is open, cursor index, number of options), IN_BATTLE, the
agent's OBJECTIVE, the PARTY, and HISTORY — every screen you've seen and button you've pressed in THIS episode,
oldest first — plus PRESSES_LEFT.

Press ONE button per turn and work through the situation the way a player would: read the text, answer prompts
sensibly, choose menu options by moving the cursor (UP/DOWN) and confirming with A; B backs out / cancels / says
NO. Look at HISTORY: if a button didn't change anything, try something else. Prefer choices that keep the game
moving and don't hurt the team (never release Pokémon, decline nicknames). Learning a new move with 4 known:
usually learn it and forget the WEAKEST move (a status move like Tail Whip / Growl first), unless the new move
is clearly worse than everything you know.

When the situation is resolved — normal play (walking the map) or the battle's FIGHT / PKMN / ITEM / RUN menu is
ready — set "done": true to hand control back. If you truly can't make progress, set "done": true and say why.

Return ONLY JSON: {"button": "A|B|UP|DOWN|LEFT|RIGHT|START|SELECT", "done": <true|false>, "reason": "<short>"}"""


def looks_normal(*, in_battle: bool, text: str, menu_open: bool, fight_menu: bool) -> bool:
    """Normal play the regular loop handles: the battle's root FIGHT menu, or the overworld with no text / menu."""
    if in_battle:
        return fight_menu
    return not (text or "").strip() and not menu_open


class Unsticker:
    def __init__(self, provider, *, window: int = 40, repeats: int = 3, budget: int = 15,
                 cooldown: int = 40, on_event=None, press=None, capture=None):
        self.provider = provider
        self.window, self.repeats, self.budget, self.cooldown = window, repeats, budget, cooldown
        self.on_event = on_event or (lambda k, p: None)
        self.press = press
        self.capture = capture
        self._sigs: deque = deque(maxlen=window)
        self.active = False
        self.history: list[dict] = []
        self.presses = 0
        self.started_at = 0
        self.cool_until = 0
        self.last_end: str | None = None

    # ---- detection ----------------------------------------------------------------------------------
    def observe(self, progress: tuple, screen: str, *, text_screen: bool) -> None:
        """One step: ``progress`` = what should change when play advances (map/position, HP, bag, money,
        moves); ``screen`` = the text/menu screen. Non-text screens reset the window (walking isn't
        'stuck' here)."""
        if not text_screen:
            self._sigs.clear()
            return
        self._sigs.append((progress, screen))

    def should_start(self, *, step: int) -> bool:
        """Stuck = a full window of text/menu steps where nothing progressed AND some screen keeps coming
        back (a loop of any length — scrolling text makes each cycle many distinct frames); a long
        one-off dialogue doesn't repeat its screens."""
        if self.provider is None or self.active or step < self.cool_until or len(self._sigs) < self.window:
            return False
        if len({p for p, _ in self._sigs}) != 1:
            return False
        counts: dict[str, int] = {}
        for _, s in self._sigs:
            counts[s] = counts.get(s, 0) + 1
        return max(counts.values()) >= self.repeats

    # ---- the episode --------------------------------------------------------------------------------
    def start(self, *, step: int, screen: str) -> None:
        self.active, self.history, self.presses, self.started_at = True, [], 0, step
        self.on_event("unstick_start", {"step": step, "screen": (screen or "")[:200]})

    def _end(self, why: str, step: int) -> None:
        self.active = False
        self.last_end = why
        self.cool_until = step + self.cooldown
        self._sigs.clear()
        self.on_event("unstick_end", {"step": step, "why": why, "presses": self.presses,
                                      "history": self.history[-self.budget:]})

    def turn(self, state: dict, *, normal: bool, step: int) -> str | None:
        """One decision + press. Returns the button pressed (None when the episode ended instead)."""
        if normal:
            self._end("normal", step)
            return None
        if self.presses >= self.budget:
            self._end("budget", step)
            return None
        s = {**state, "history": list(self.history), "presses_left": self.budget - self.presses}
        try:
            raw, _lat, _usage = self.provider.chat_json(UNSTICK_SYSTEM, s)
            data = json.loads(strip_fences(raw))
        except Exception:
            data = {}
        button = str(data.get("button") or "").upper()
        if self.capture is not None:
            try:
                self.capture.record("unstick", model=getattr(self.provider, "model", None), input=s,
                                    output_raw=json.dumps(data), parsed=data)
            except Exception:
                pass
        if button not in BUTTONS:
            button = "B"                                # a safe default: back out / decline
        if self.press is not None:
            self.press(button)
        self.presses += 1
        screen_now = state.get("screen") or []
        self.history.append({"screen": " / ".join(screen_now)[-160:], "pressed": button,
                             "reason": str(data.get("reason") or "")[:120]})
        self.on_event("unstick_press", {"step": step, "button": button, "reason": data.get("reason")})
        if data.get("done"):
            self._end("done", step)
        return button
