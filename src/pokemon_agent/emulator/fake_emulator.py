"""Deterministic in-memory emulator for tests and offline agent-loop runs.

Models a tiny walkable grid so the action controller and agent loop can be
exercised end-to-end without PyBoy or a ROM. Player lives at (x, y); walls
block movement, mirroring real "blocked move" semantics.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from .interface import GameButton, ImageObservation

# Memory addresses this fake honors, so extractor code can read it like the real thing.
ADDR_PLAYER_X = 0xD362
ADDR_PLAYER_Y = 0xD361
ADDR_MAP_ID = 0xD35E


class FakeEmulator:
    """A 2D grid world. `#` = wall, `.` = floor. Player moves one tile per press."""

    def __init__(self, grid: list[str] | None = None, start: tuple[int, int] = (2, 2), map_id: int = 40):
        self.grid = grid or [
            "########",
            "#......#",
            "#..##..#",
            "#......#",
            "#..##..#",
            "#......#",
            "###..###",  # opening at bottom -> "doorway"
        ]
        self.x, self.y = start
        self.map_id = map_id
        self.frame = 0
        self._held: set[GameButton] = set()
        self.saved_states: dict[str, tuple[int, int, int]] = {}

    # --- movement ---------------------------------------------------------
    def _walkable(self, x: int, y: int) -> bool:
        if y < 0 or y >= len(self.grid):
            return False
        row = self.grid[y]
        if x < 0 or x >= len(row):
            return False
        return row[x] != "#"

    def _apply_button(self, b: GameButton) -> None:
        dx, dy = {
            GameButton.UP: (0, -1),
            GameButton.DOWN: (0, 1),
            GameButton.LEFT: (-1, 0),
            GameButton.RIGHT: (1, 0),
        }.get(b, (0, 0))
        if (dx or dy) and self._walkable(self.x + dx, self.y + dy):
            self.x += dx
            self.y += dy

    # --- Emulator protocol ------------------------------------------------
    def tick(self, frames: int = 1, render: bool = True) -> None:
        for _ in range(frames):
            self.frame += 1
            for b in list(self._held):
                self._apply_button(b)

    def press(self, button: GameButton) -> None:
        self._apply_button(button)
        self.frame += 1

    def hold(self, button: GameButton) -> None:
        self._held.add(button)

    def release(self, button: GameButton) -> None:
        self._held.discard(button)

    def screenshot(self) -> ImageObservation:
        # Render the grid as a tiny image so screenshot hashing/vision paths work.
        cell = 8
        w, h = len(self.grid[0]) * cell, len(self.grid) * cell
        img = Image.new("RGB", (w, h), (0, 0, 0))
        px = img.load()
        for gy, row in enumerate(self.grid):
            for gx, ch in enumerate(row):
                color = (40, 40, 40) if ch == "#" else (200, 200, 200)
                if (gx, gy) == (self.x, self.y):
                    color = (220, 40, 40)
                for i in range(cell):
                    for j in range(cell):
                        px[gx * cell + i, gy * cell + j] = color
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return ImageObservation(mime_type="image/png", data=buf.getvalue(), width=w, height=h)

    def local_map_ascii(self, radius: int = 3) -> list[str] | None:
        rows: list[str] = []
        for gy in range(self.y - radius, self.y + radius + 1):
            line = ""
            for gx in range(self.x - radius, self.x + radius + 1):
                if (gx, gy) == (self.x, self.y):
                    line += "@"
                elif 0 <= gy < len(self.grid) and 0 <= gx < len(self.grid[gy]):
                    line += "." if self.grid[gy][gx] != "#" else "#"
                else:
                    line += "#"
            rows.append(line)
        return rows

    def ledge_dirs(self) -> set:
        return set()  # the fake grid world has no one-way ledges

    def read_memory(self, address: int, bank: int | None = None) -> int:
        return {
            ADDR_PLAYER_X: self.x,
            ADDR_PLAYER_Y: self.y,
            ADDR_MAP_ID: self.map_id,
        }.get(address, 0)

    def save_state(self, path: Path) -> None:
        self.saved_states[str(path)] = (self.x, self.y, self.map_id)

    def load_state(self, path: Path) -> None:
        self.x, self.y, self.map_id = self.saved_states[str(path)]

    def close(self) -> None:
        self._held.clear()
