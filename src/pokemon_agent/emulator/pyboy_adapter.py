"""PyBoy-backed emulator. The ONLY module allowed to import pyboy."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from .interface import GameButton, ImageObservation


class PyBoyEmulator:
    """Wraps a live PyBoy instance behind the Emulator protocol.

    PyBoy only advances when tick() is called; the SDL window updates on tick.
    """

    def __init__(self, rom_path: str | Path, *, window: str = "SDL2", speed: int = 1, sound: bool = False):
        from pyboy import PyBoy  # local import keeps the dependency behind this adapter

        self._pyboy = PyBoy(str(rom_path), window=window, sound=sound)
        self._pyboy.set_emulation_speed(speed)

    def tick(self, frames: int = 1, render: bool = True) -> None:
        self._pyboy.tick(frames, render)

    def press(self, button: GameButton) -> None:
        # PyBoy auto-releases the button on the following tick.
        self._pyboy.button(button.value)

    def hold(self, button: GameButton) -> None:
        self._pyboy.button_press(button.value)

    def release(self, button: GameButton) -> None:
        self._pyboy.button_release(button.value)

    def screenshot(self) -> ImageObservation:
        img: Image.Image = self._pyboy.screen.image.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return ImageObservation(mime_type="image/png", data=buf.getvalue(), width=img.width, height=img.height)

    def local_map_ascii(self) -> list[str] | None:
        """Player-centered walkability from the gen1 collision matrix.

        '@' = player (fixed screen center in Red), '.' = walkable, '#' = blocked.
        Returns None if the wrapper can't produce a matrix (menus/battle/etc.).
        """
        try:
            import numpy as np

            col = np.asarray(self._pyboy.game_area_collision())
            if col.ndim != 2 or col.size == 0:
                return None
            down = col[::2, ::2]  # collapse the 2x2-doubled cells to per-tile
            h, w = down.shape
            pr, pc = 4, 4  # Red centers the player at this downsampled cell
            rows: list[str] = []
            for r in range(h):
                line = ""
                for c in range(w):
                    line += "@" if (r, c) == (pr, pc) else ("." if down[r, c] else "#")
                rows.append(line)
            return rows
        except Exception:
            return None

    # Ledge tiles (pokered data/tilesets/ledge_tiles.asm): the tile id alone gives the
    # one-way hop direction. Overworld tileset only. game_area() tile ids carry a +0x100
    # VRAM offset. There are NO up-ledges.
    _LEDGE_DIR = {0x36: "south", 0x37: "south", 0x27: "west", 0x0D: "east", 0x1D: "east"}
    _LEDGE_NEIGHBOR = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}

    def ledge_dirs(self) -> set[str]:
        """Directions from the player that would trigger a one-way LEDGE HOP (so the
        navigator can treat them as non-traversable for routing and never try to climb
        back). Empty off the overworld tileset."""
        try:
            if self._pyboy.memory[0xD367] != 0:  # wCurMapTileset: 0 = OVERWORLD
                return set()
            import numpy as np

            ga = np.asarray(self._pyboy.game_area()) - 0x100  # 18x20 screen tiles, VRAM offset
            out: set[str] = set()
            pr, pc = 4, 4  # player metatile in the 9x10 grid (Red centers the player here)
            for d, (dr, dc) in self._LEDGE_NEIGHBOR.items():
                mr, mc = pr + dr, pc + dc
                block = ga[2 * mr:2 * mr + 2, 2 * mc:2 * mc + 2].flatten()
                if any(self._LEDGE_DIR.get(int(t)) == d for t in block):
                    out.add(d)
            return out
        except Exception:
            return set()

    def read_memory(self, address: int, bank: int | None = None) -> int:
        if bank is None:
            return self._pyboy.memory[address]
        return self._pyboy.memory[bank, address]

    def save_state(self, path: Path) -> None:
        with open(path, "wb") as f:
            self._pyboy.save_state(f)

    def load_state(self, path: Path) -> None:
        with open(path, "rb") as f:
            self._pyboy.load_state(f)
        # After a load the map buffers (wOverworldMap, tileset, dims, coords) are still
        # mid-transition for ~1s of frames — reading them immediately yields the PREVIOUS
        # map's data. Settle before returning so the first observation is ground truth.
        self._pyboy.tick(60, False)

    def close(self) -> None:
        self._pyboy.stop(save=False)
