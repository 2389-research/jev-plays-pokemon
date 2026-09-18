"""Emulator abstraction. Nothing outside this package should import PyBoy directly."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class GameButton(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    A = "a"
    B = "b"
    START = "start"
    SELECT = "select"


class ImageObservation(BaseModel):
    """Provider-independent screenshot. Providers serialize this themselves."""

    mime_type: str
    data: bytes
    width: int
    height: int


@runtime_checkable
class Emulator(Protocol):
    """The only surface the rest of the app is allowed to know about."""

    def tick(self, frames: int = 1, render: bool = True) -> None: ...

    def press(self, button: GameButton) -> None:
        """Press-and-release across the next tick (semantic tap)."""

    def hold(self, button: GameButton) -> None: ...

    def release(self, button: GameButton) -> None: ...

    def screenshot(self) -> ImageObservation: ...

    def local_map_ascii(self) -> list[str] | None:
        """Player-centered walkability grid ('@'/'.'/'#'), or None if unavailable."""

    def read_memory(self, address: int, bank: int | None = None) -> int: ...

    def save_state(self, path: Path) -> None: ...

    def load_state(self, path: Path) -> None: ...

    def close(self) -> None: ...
