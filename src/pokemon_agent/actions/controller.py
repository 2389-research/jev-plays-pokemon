"""Turn semantic actions into button sequences and OBSERVE the result.

The controller never assumes a fixed frame count means success: for moves it
watches the player's coordinates and reports `blocked` when they never change.
"""
from __future__ import annotations

from ..core.models import (
    ActionResult,
    AdvanceDialogAction,
    AgentAction,
    Direction,
    DIRECTION_BUTTON,
    GameButton,
    InteractAction,
    MoveAction,
    PressAction,
    WaitAction,
)
from ..emulator.interface import Emulator
from ..games.pokemon_red.state import detect_mode, read_player

# Per-tile budget: a Red overworld step animates over ~16 frames.
FRAMES_PER_TILE_BUDGET = 24
POLL = 2  # check position every N frames


def _pos(emu: Emulator) -> tuple[int, int, int] | None:
    p = read_player(emu)
    return (p.x, p.y, p.map_id) if p else None


class ActionController:
    def __init__(self, emu: Emulator):
        self.emu = emu

    def execute(self, action: AgentAction) -> ActionResult:
        mode_before = detect_mode(self.emu)
        if isinstance(action, MoveAction):
            res = self._move(action, mode_before)
        elif isinstance(action, (PressAction, InteractAction, AdvanceDialogAction)):
            res = self._press(action, mode_before)
        elif isinstance(action, WaitAction):
            res = self._wait(action, mode_before)
        else:  # pragma: no cover - discriminated union is exhaustive
            return ActionResult(
                success=False, result="invalid", mode_before=mode_before,
                mode_after=mode_before, detail=f"unknown action {action!r}",
            )
        return res

    # --- move -------------------------------------------------------------
    def _move(self, action: MoveAction, mode_before) -> ActionResult:
        button = DIRECTION_BUTTON[action.direction]
        moved_tiles = 0
        total_frames = 0
        events: list[str] = []
        for _ in range(action.tiles):
            before = _pos(self.emu)
            self.emu.hold(button)
            changed = False
            elapsed = 0
            while elapsed < FRAMES_PER_TILE_BUDGET:
                self.emu.tick(POLL)
                elapsed += POLL
                if _pos(self.emu) != before:
                    changed = True
                    break
            self.emu.release(button)
            self.emu.tick(1)  # settle
            total_frames += elapsed + 1
            if changed:
                moved_tiles += 1
                events.append("player_moved")
            else:
                events.append(f"movement_blocked:{action.direction.value}")
                break  # stop the multi-tile move at the wall

        mode_after = detect_mode(self.emu)
        if moved_tiles == 0:
            return ActionResult(
                success=False, result="blocked", mode_before=mode_before, mode_after=mode_after,
                player_moved=False, frames_elapsed=total_frames, events=events,
                detail=f"blocked moving {action.direction.value}",
            )
        return ActionResult(
            success=True, result="completed", mode_before=mode_before, mode_after=mode_after,
            player_moved=True, frames_elapsed=total_frames, events=events,
            detail=f"moved {moved_tiles}/{action.tiles} tiles {action.direction.value}",
        )

    # --- discrete button presses -----------------------------------------
    def _press(self, action, mode_before) -> ActionResult:
        if isinstance(action, PressAction):
            button = action.button
        else:  # interact / advance_dialog both map to A
            button = GameButton.A
        self.emu.press(button)
        # dialog/menu text needs time to scroll+react; 8 frames was too few (the
        # post-pick starter cutscene stalled), so give a real settle window.
        self.emu.tick(24)
        mode_after = detect_mode(self.emu)
        return ActionResult(
            success=True, result="completed", mode_before=mode_before, mode_after=mode_after,
            frames_elapsed=25, events=[f"pressed:{button.value}"],
            detail=f"pressed {button.value}",
        )

    def _wait(self, action: WaitAction, mode_before) -> ActionResult:
        self.emu.tick(action.frames)
        mode_after = detect_mode(self.emu)
        return ActionResult(
            success=True, result="completed", mode_before=mode_before, mode_after=mode_after,
            frames_elapsed=action.frames, events=["waited"], detail=f"waited {action.frames} frames",
        )
