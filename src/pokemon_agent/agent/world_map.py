"""Persistent, coordinate-grounded exploration map.

Unlike the per-frame local walkability window, this accumulates what the agent
has learned across steps, keyed by real map-local coordinates (map_id, x, y):

  * a tile the player has STOOD ON is floor (ground truth);
  * a tile a move was BLOCKED toward is a wall (ground truth);
  * `.` cells seen in the local collision window are marked floor (safe: we never
    infer walls from the noisy collision matrix, only from real blocked moves);
  * everything else is unknown -> a frontier to explore.

This gives the agent memory ("I already tried south here") and lets it explore
systematically toward unknown tiles instead of oscillating.
"""
from __future__ import annotations

from collections import defaultdict, deque

from ..core.models import Direction, PlayerState

# screen/world convention: +x = east, +y = south (row down)
DELTA: dict[Direction, tuple[int, int]] = {
    Direction.NORTH: (0, -1),
    Direction.SOUTH: (0, 1),
    Direction.EAST: (1, 0),
    Direction.WEST: (-1, 0),
}

FLOOR, WALL = "floor", "wall"


class WorldMap:
    def __init__(self) -> None:
        self.tiles: dict[int, dict[tuple[int, int], str]] = defaultdict(dict)
        self.visits: dict[int, dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))

    # --- updates ----------------------------------------------------------
    def ingest_collision(self, map_id: int, width: int, height: int,
                         walkable: set[tuple[int, int]]) -> None:
        """Load a full-map collision grid (from RAM's wOverworldMap) as ground truth: every
        cell in bounds becomes FLOOR or WALL. This gives the navigator the whole map up front
        so it can route around buildings instead of guessing over unseen tiles."""
        m = self.tiles[map_id]
        for y in range(height):
            for x in range(width):
                m[(x, y)] = FLOOR if (x, y) in walkable else WALL

    def observe(self, player: PlayerState | None, local_ascii: list[str] | None) -> None:
        if player is None:
            return
        m = self.tiles[player.map_id]
        m[(player.x, player.y)] = FLOOR
        self.visits[player.map_id][(player.x, player.y)] += 1
        if not local_ascii:
            return
        # locate '@' in the local window to anchor the projection
        anchor = None
        for r, row in enumerate(local_ascii):
            c = row.find("@")
            if c != -1:
                anchor = (r, c)
                break
        if anchor is None:
            return
        pr, pc = anchor
        for r, row in enumerate(local_ascii):
            for c, ch in enumerate(row):
                if ch != ".":
                    continue  # only trust open-floor cells; walls come from real blocks
                gx, gy = player.x + (c - pc), player.y + (r - pr)
                m.setdefault((gx, gy), FLOOR)  # don't overwrite a known wall

    def mark_blocked(self, player: PlayerState | None, direction: Direction) -> None:
        if player is None or direction not in DELTA:
            return
        dx, dy = DELTA[direction]
        self.tiles[player.map_id][(player.x + dx, player.y + dy)] = WALL

    # --- queries ----------------------------------------------------------
    def unexplored_directions(self, player: PlayerState | None) -> list[str]:
        """Directions whose adjacent tile is not a known wall and not yet floor."""
        if player is None:
            return []
        m = self.tiles[player.map_id]
        out = []
        for d, (dx, dy) in DELTA.items():
            t = m.get((player.x + dx, player.y + dy))
            if t != WALL and t != FLOOR:
                out.append(d.value)
        return out

    def explore_step(self, player: PlayerState | None) -> str | None:
        """BFS over known floor to the nearest tile adjacent to an unknown cell;
        return the first move's direction (a Direction value)."""
        if player is None:
            return None
        m = self.tiles[player.map_id]
        start = (player.x, player.y)
        seen = {start}
        # queue holds (tile, first_direction_taken)
        q: deque[tuple[tuple[int, int], str | None]] = deque([(start, None)])
        while q:
            (x, y), first = q.popleft()
            for d, (dx, dy) in DELTA.items():
                nxt = (x + dx, y + dy)
                t = m.get(nxt)
                if t is None:  # unknown & reachable adjacent to a floor tile -> frontier
                    return first or d.value
                if t == FLOOR and nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, first or d.value))
        return None

    def render(
        self,
        player: PlayerState | None,
        exits: list[dict] | None = None,
        dims: tuple[int, int] | None = None,
        max_span: int = 24,
    ) -> list[str] | None:
        """Unified map view in map coordinates: '@'=you, 'E'=exit, '.'=floor,
        '#'=wall, '?'=unexplored. Bounded to the real map size when known, else a
        window around the player. Exits (from RAM) are plotted even if unexplored."""
        if player is None:
            return None
        m = self.tiles[player.map_id]
        exit_cells = {(e["x"], e["y"]): e for e in (exits or [])}
        if dims and dims[0] <= max_span and dims[1] <= max_span:
            x0, x1, y0, y1 = 0, dims[0] - 1, 0, dims[1] - 1  # whole map fits
        else:
            r = max_span // 2
            x0, x1, y0, y1 = player.x - r, player.x + r, player.y - r, player.y + r
        rows = []
        header = "   " + "".join(str(x % 10) for x in range(x0, x1 + 1))
        rows.append(header)
        for y in range(y0, y1 + 1):
            line = ""
            for x in range(x0, x1 + 1):
                if (x, y) == (player.x, player.y):
                    line += "@"
                elif (x, y) in exit_cells:
                    line += "E"
                else:
                    line += {FLOOR: ".", WALL: "#"}.get(m.get((x, y)), "?")
            rows.append(f"{y:2d}|{line}")
        return rows

    # --- serialization (for checkpointing) --------------------------------
    def to_dict(self) -> dict:
        return {
            "tiles": {str(m): [[x, y, v] for (x, y), v in cells.items()]
                      for m, cells in self.tiles.items()},
            "visits": {str(m): [[x, y, n] for (x, y), n in cells.items()]
                       for m, cells in self.visits.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorldMap":
        wm = cls()
        for m, cells in (d.get("tiles") or {}).items():
            for x, y, v in cells:
                wm.tiles[int(m)][(x, y)] = v
        for m, cells in (d.get("visits") or {}).items():
            for x, y, n in cells:
                wm.visits[int(m)][(x, y)] = n
        return wm

    def ascii(self, player: PlayerState | None, radius: int = 5) -> list[str] | None:
        if player is None:
            return None
        m = self.tiles[player.map_id]
        rows: list[str] = []
        for gy in range(player.y - radius, player.y + radius + 1):
            line = ""
            for gx in range(player.x - radius, player.x + radius + 1):
                if (gx, gy) == (player.x, player.y):
                    line += "@"
                else:
                    t = m.get((gx, gy))
                    line += {FLOOR: ".", WALL: "#"}.get(t, "?")
            rows.append(line)
        return rows
