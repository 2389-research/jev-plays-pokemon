"""Provider-independent domain models. No OpenAI/PyBoy types leak in here."""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from ..emulator.interface import GameButton


class GameMode(str, Enum):
    OVERWORLD = "overworld"
    BATTLE = "battle"
    DIALOG = "dialog"
    MENU = "menu"
    CUTSCENE = "cutscene"
    UNKNOWN = "unknown"


class Direction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


# Direction -> button mapping (north = up, screen-relative).
DIRECTION_BUTTON: dict[Direction, GameButton] = {
    Direction.NORTH: GameButton.UP,
    Direction.SOUTH: GameButton.DOWN,
    Direction.EAST: GameButton.RIGHT,
    Direction.WEST: GameButton.LEFT,
}


# --- Actions --------------------------------------------------------------
class MoveAction(BaseModel):
    type: Literal["move"] = "move"
    direction: Direction
    tiles: int = Field(default=1, ge=1, le=10)


class PressAction(BaseModel):
    type: Literal["press"] = "press"
    button: GameButton


class InteractAction(BaseModel):
    type: Literal["interact"] = "interact"


class AdvanceDialogAction(BaseModel):
    type: Literal["advance_dialog"] = "advance_dialog"


class WaitAction(BaseModel):
    type: Literal["wait"] = "wait"
    frames: int = Field(default=30, ge=1, le=600)


class MenuSelectAction(BaseModel):
    """Select option `index` (0-based, top = 0) in an open list/yes-no menu."""
    type: Literal["menu_select"] = "menu_select"
    index: int = Field(default=0, ge=0)
    label: str | None = None


class GoToAction(BaseModel):
    """Navigate to a target tile (an object/NPC/exit), deterministically pathed.

    The decider names WHERE to go; the navigator handles the routing. For an
    object/NPC (`interact=True`) the agent walks to a tile adjacent to (x,y), faces
    it, and presses A. For an exit (`interact=False`) it steps onto (x,y) itself.
    """
    type: Literal["goto"] = "goto"
    x: int
    y: int
    interact: bool = True
    label: str | None = None  # human tag for logs, e.g. "Poke Ball (11,7)"


AgentAction = Annotated[
    Union[MoveAction, PressAction, InteractAction, AdvanceDialogAction, WaitAction, GoToAction, MenuSelectAction],
    Field(discriminator="type"),
]


# --- State snapshots (minimal for the walking slice) ----------------------
class PlayerState(BaseModel):
    x: int
    y: int
    map_id: int
    facing: str | None = None  # north/south/east/west the player is currently facing
    is_outdoor: bool | None = None  # derived from tileset (real state, not a label)
    map_name: str | None = None  # optional human annotation for logs; not load-bearing


class StuckInfo(BaseModel):
    stuck: bool = False
    reason: str | None = None
    repeat_count: int = 0
    kind: str | None = None       # local_loop | no_objective_progress | forced_movement_suppressed
    setback: bool = False         # a real regression (e.g. whiteout / money drop) — not "stuck", but notable


# --- Observation ----------------------------------------------------------
class Observation(BaseModel):
    observation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = Field(default_factory=time.time)
    mode: GameMode = GameMode.UNKNOWN
    player: PlayerState | None = None
    walkability: list[str] | None = None  # screen-relative ascii grid, '@' = player
    explored_map: list[str] | None = None  # persistent stitched map ('.'=floor '#'=wall '?'=unknown)
    map_view: list[str] | None = None  # unified map-coord view ('@'/'E'/'.'/'#'/'?'), bounded to real size
    map_dims: tuple[int, int] | None = None  # (width, height) of current map in tiles
    exits: list[dict] = Field(default_factory=list)  # known exit tiles {x,y,dest_map} from RAM
    game_state: dict | None = None  # rich RAM state: party, battle, badges, money, items, text
    unexplored_directions: list[str] = Field(default_factory=list)
    suggested_explore: str | None = None  # BFS hint toward nearest unexplored tile
    recent_events: list[str] = Field(default_factory=list)
    available_actions: list[str] = Field(default_factory=list)
    stuck: StuckInfo = Field(default_factory=StuckInfo)
    has_screenshot: bool = False

    def to_model_json(self) -> dict:
        """The subset actually sent to the model (never raw screenshot bytes)."""
        return self.model_dump(exclude={"has_screenshot", "timestamp"})


# --- Action result --------------------------------------------------------
class ActionResult(BaseModel):
    success: bool
    result: Literal["completed", "blocked", "timeout", "interrupted", "invalid"]
    mode_before: GameMode
    mode_after: GameMode
    player_moved: bool | None = None
    frames_elapsed: int = 0
    events: list[str] = Field(default_factory=list)
    detail: str | None = None


# --- Agent decision -------------------------------------------------------
class GoalState(BaseModel):
    primary: str = "Progress through Pokémon Red"
    current: str = "Leave the current area"
    status: str = "in_progress"


class AgentDecision(BaseModel):
    action: AgentAction
    decision_note: str
    current_goal: str | None = None
    goal_status: str | None = None
