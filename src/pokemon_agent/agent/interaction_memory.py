"""Persistent bookkeeping of interactions and conversations.

Grounded in the ACTION, not fuzzy NPC detection: when the agent presses A /
interacts while facing a tile, that target tile (map, x, y) is marked
interacted-with — whatever is there (person, sign, object). Any NPC standing on
a marked tile is then flagged 'interacted' so the agent knows it already dealt
with them. Also accumulates the dialog it has read.
"""
from __future__ import annotations

from ..core.models import PlayerState

_FACING_DELTA = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
_INTERACT_TYPES = {"interact", "advance_dialog"}


def _faced_tile(player: PlayerState) -> tuple[int, int, int] | None:
    if player.facing not in _FACING_DELTA:
        return None
    dx, dy = _FACING_DELTA[player.facing]
    return (player.map_id, player.x + dx, player.y + dy)


class InteractionMemory:
    def __init__(self, dialog_max: int = 30):
        self.talked: set[tuple[int, int, int]] = set()      # tiles where A -> a real dialog
        self.empty_tiles: set[tuple[int, int, int]] = set()  # tiles where A -> nothing happened
        self.interaction_log: list[dict] = []               # [{step, faced, result}]
        self.dialog_log: list[dict] = []                    # [{step, map, text}]
        self._last_text = ""
        self._dialog_max = dialog_max

    def record_action(self, step: int, player: PlayerState | None, action, caused_dialog: bool) -> None:
        """Call AFTER executing. For an interact-like action, record the faced tile
        as a real conversation (caused_dialog) or as empty (nothing was there)."""
        if player is None or action is None:
            return
        a = action.model_dump() if hasattr(action, "model_dump") else dict(action)
        is_interact = a.get("type") in _INTERACT_TYPES or (a.get("type") == "press" and str(a.get("button")).lower() == "a")
        if not is_interact:
            return
        tile = _faced_tile(player)
        if tile is None:
            return
        if caused_dialog:
            self.talked.add(tile)
            self.empty_tiles.discard(tile)
        elif tile not in self.talked:
            self.empty_tiles.add(tile)  # pressed A here and got no response
        self.interaction_log.append({"step": step, "faced": [tile[1], tile[2]],
                                     "result": "talked" if caused_dialog else "nothing"})

    def record_dialog(self, step: int, player: PlayerState | None, game_state: dict | None) -> None:
        gs = game_state or {}
        text = gs.get("screen_text") or ""
        if gs.get("dialog_active") and text and text != self._last_text:
            self.dialog_log.append({"step": step, "map": player.map_id if player else None, "text": text})
            self.dialog_log = self.dialog_log[-self._dialog_max:]
            self._last_text = text
        elif not gs.get("dialog_active"):
            self._last_text = ""

    def annotate_npcs(self, player: PlayerState | None, npcs: list[dict]) -> list[dict]:
        if player is None:
            return npcs
        out = []
        for n in npcs:
            key = (player.map_id, n["x"], n["y"])
            out.append({**n, "talked_to": key in self.talked,
                        "interact_did_nothing": key in self.empty_tiles})
        return out

    def summary(self) -> dict:
        return {
            "talked_to_count": len(self.talked),
            "tiles_where_A_did_nothing": [[t[1], t[2]] for t in list(self.empty_tiles)[-8:]],
            "recent_interactions": self.interaction_log[-6:],
            "recent_dialog": [d["text"] for d in self.dialog_log[-8:]],
        }

    # --- serialization (for checkpointing) --------------------------------
    def to_dict(self) -> dict:
        return {
            "talked": [list(t) for t in self.talked],
            "empty_tiles": [list(t) for t in self.empty_tiles],
            "interaction_log": self.interaction_log,
            "dialog_log": self.dialog_log,
            "last_text": self._last_text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InteractionMemory":
        im = cls()
        im.talked = {tuple(t) for t in d.get("talked", [])}
        im.empty_tiles = {tuple(t) for t in d.get("empty_tiles", [])}
        im.interaction_log = d.get("interaction_log", [])
        im.dialog_log = d.get("dialog_log", [])
        im._last_text = d.get("last_text", "")
        return im
