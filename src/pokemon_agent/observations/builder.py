"""Build a provider-independent Observation from the emulator + game state.

Walkability is best-effort: if the emulator can supply a walkable grid (the
FakeEmulator, or later the PyBoy gen1 collision matrix) we render a small
window around the player with '@'. Otherwise it's None and the agent leans on
coordinates / screenshot.
"""
from __future__ import annotations

from ..core.models import GameMode, Observation, PlayerState, StuckInfo
from ..emulator.interface import Emulator, ImageObservation
from ..games.pokemon_red.game_state import read_game_state
from ..games.pokemon_red.state import detect_mode, read_exits, read_map_dims, read_player

ACTIONS_BY_MODE: dict[GameMode, list[str]] = {
    GameMode.OVERWORLD: ["move_north", "move_south", "move_east", "move_west", "interact", "press_a", "press_b"],
    GameMode.BATTLE: ["press_a", "press_b", "move_north", "move_south", "move_east", "move_west"],
    GameMode.DIALOG: ["advance_dialog", "press_a", "press_b"],
    GameMode.MENU: ["move_north", "move_south", "press_a", "press_b"],
    GameMode.UNKNOWN: ["press_a", "press_b", "wait"],
}


def _walkability_window(emu: Emulator, player: PlayerState | None) -> list[str] | None:
    fn = getattr(emu, "local_map_ascii", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


class ObservationBuilder:
    def __init__(self, emu: Emulator):
        self.emu = emu

    def build(
        self,
        *,
        recent_events: list[str] | None = None,
        stuck: StuckInfo | None = None,
        capture_screenshot: bool = False,
    ) -> tuple[Observation, ImageObservation | None]:
        mode = detect_mode(self.emu)
        player = read_player(self.emu)
        shot = self.emu.screenshot() if capture_screenshot else None
        obs = Observation(
            mode=mode,
            player=player,
            walkability=_walkability_window(self.emu, player),
            exits=read_exits(self.emu) if mode == GameMode.OVERWORLD else [],
            map_dims=read_map_dims(self.emu) if mode == GameMode.OVERWORLD else None,
            game_state=read_game_state(self.emu),
            recent_events=recent_events or [],
            available_actions=ACTIONS_BY_MODE.get(mode, ACTIONS_BY_MODE[GameMode.UNKNOWN]),
            stuck=stuck or StuckInfo(),
            has_screenshot=shot is not None,
        )
        return obs, shot
