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
# A/B/START/SELECT are held this long so the game's once-per-loop joypad sampling always sees
# them (8 frames is the measured minimum in the worst recorded state; +2 margin).
PRESS_HOLD_FRAMES = 10
PRESS_SETTLE_FRAMES = 24
DIRECTION_BUTTONS = frozenset(DIRECTION_BUTTON.values())
# A warp commits its destination coords + sprites ~35 frames after the map id flips; cap the wait.
WARP_SETTLE_MAX = 120
# Some warps land on the SAME (x, y) on the new map (e.g. Red's House 1F/2F stairs at (7,1)), so the
# coords never change; past the observed commit window (34-36 frames) treat the warp as settled.
WARP_SAMEXY_SETTLE = 60


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
                after = _pos(self.emu)
                if before is not None and after is not None and after[2] != before[2]:
                    # the map changed: never return (or keep walking) mid-transition
                    total_frames += self._settle_map_change(before, after, events)
                    break
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

    def _settle_map_change(self, before, flip, events: list[str]) -> int:
        """After the map id flips, wait out a WARP until its coords + sprites commit; return the
        frames spent. A warp (door/mat/stairs) flips the map on or next to the pre-step tile and
        holds the coords there (with the OLD map's sprites) for ~35 frames before jumping to the
        destination; observing in that window gave the agent a torn frame (map = new, coords and
        NPCs = old). A map-EDGE connection flips map + coords + sprites on one frame, far from the
        pre-step tile, and is already settled."""
        if abs(flip[0] - before[0]) + abs(flip[1] - before[1]) > 1:
            events.append("map_connection")
            return 0
        waited = 0
        while waited < WARP_SETTLE_MAX:
            self.emu.tick(POLL)
            waited += POLL
            now = _pos(self.emu)
            if now is not None and (now[0], now[1]) != (flip[0], flip[1]):
                events.append("warp_settled")
                return waited
            if waited >= WARP_SAMEXY_SETTLE and now is not None and now[2] == flip[2]:
                events.append("warp_settled_samexy")   # destination tile == source tile
                return waited
        events.append("warp_settle_timeout")
        return waited

    # --- discrete button presses -----------------------------------------
    def _press(self, action, mode_before) -> ActionResult:
        if isinstance(action, PressAction):
            button = action.button
        else:  # interact / advance_dialog both map to A
            button = GameButton.A
        if button in DIRECTION_BUTTONS:
            # a direction TAP only turns the player; holding it would start a walk
            self.emu.press(button)
            held = 1
        else:
            # A/B/START/SELECT are HELD: Gen 1 samples the joypad once per overworld-loop
            # iteration, so a 1-frame tap can alias with the sampling and miss every time (the
            # 1839 run pressed A 107x facing Oak with no dialog). Text/menus advance on a NEW
            # press edge, so holding never skips a box or double-selects.
            self.emu.hold(button)
            self.emu.tick(PRESS_HOLD_FRAMES)
            self.emu.release(button)
            held = PRESS_HOLD_FRAMES
        # dialog/menu text needs time to scroll+react; 8 frames was too few (the
        # post-pick starter cutscene stalled), so give a real settle window.
        self.emu.tick(PRESS_SETTLE_FRAMES)
        mode_after = detect_mode(self.emu)
        return ActionResult(
            success=True, result="completed", mode_before=mode_before, mode_after=mode_after,
            frames_elapsed=held + PRESS_SETTLE_FRAMES, events=[f"pressed:{button.value}"],
            detail=f"pressed {button.value}",
        )

    def _wait(self, action: WaitAction, mode_before) -> ActionResult:
        self.emu.tick(action.frames)
        mode_after = detect_mode(self.emu)
        return ActionResult(
            success=True, result="completed", mode_before=mode_before, mode_after=mode_after,
            frames_elapsed=action.frames, events=["waited"], detail=f"waited {action.frames} frames",
        )
