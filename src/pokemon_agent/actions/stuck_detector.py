"""Detect the agent wedging — two distinct signals, plus setback detection.

1. LOCAL LOOP (fast): same action + same resting position, no progress, repeated.
   Catches wall-bumping / oscillation in place.
2. OBJECTIVE WEDGE (slow): no progress TOWARD the goal over many steps. This is the
   real long-horizon killer, and it is NOT "no state change" — a whiteout loop,
   cross-map wandering, and slow HP/PP decay all keep state *changing* while making no
   progress. So the wedge signal is measured against a progress signature (badges,
   party level, maps discovered) or, when known, graph-distance-to-objective decreasing.

Also flags SETBACKS (e.g. a whiteout drops money + teleports to a Center) — not
"stuck", but a real regression the caller should treat as negative progress.

FORCED MOVEMENT (scripted cutscene / warp animation) SUPPRESSES stuck: input is
ignored there, so "no progress" is expected and must never trigger a reload.

Backward-compatible: `update(action, result, player, screenshot)` behaves exactly as
before; the new signals activate only when `progress`/`forced_movement`/
`objective_distance` are passed.
"""
from __future__ import annotations

import hashlib
from collections import deque

from ..core.models import ActionResult, AgentAction, StuckInfo


class StuckDetector:
    def __init__(self, threshold: int = 3, wedge_threshold: int = 40):
        self.threshold = threshold            # local-loop repeats
        self.wedge_threshold = wedge_threshold  # steps of no objective progress
        self._last_key: str | None = None
        self._count = 0
        # objective-wedge tracking
        self._best_sig: tuple | None = None
        self._best_dist: int | None = None
        self._wedge = 0
        self._maps_seen: set[int] = set()
        self._last_money: int | None = None
        self._recent_pos: deque[tuple] = deque(maxlen=6)  # for A-B oscillation detection

    def reset_objective(self) -> None:
        """Start a fresh objective budget for a NEW plan step. Without this, `_best_dist` is the
        closest map-hop distance EVER seen across all steps, so a later step whose target is farther
        can never register "getting closer" and wedges from its first update (the Oak's Parcel
        incident: each fresh step wedged after exactly BLOCK_TRIGGER steps, one tile from Oak).
        Cumulative exploration progress (`_best_sig`) and money/setback tracking are kept."""
        self._best_dist = None
        self._wedge = 0
        self._recent_pos.clear()
        self._count = 0
        self._last_key = None

    @staticmethod
    def _key(action: AgentAction, player, screenshot_bytes: bytes | None) -> str:
        parts = [action.model_dump_json()]
        parts.append(f"{player.x},{player.y},{player.map_id}" if player else "no-player")
        if screenshot_bytes is not None:
            parts.append(hashlib.sha1(screenshot_bytes).hexdigest())
        return "|".join(parts)

    def _progress_sig(self, progress: dict) -> tuple:
        """Higher is better. New badges, levels, maps, AND newly-discovered tiles all
        count — the last is what makes traversing a LONG single map (e.g. Route 1)
        register as progress, so it isn't falsely flagged as an objective wedge."""
        self._maps_seen.add(progress.get("map_id"))
        return (progress.get("badges", 0), progress.get("max_party_level", 0),
                len(self._maps_seen), progress.get("tiles_known", 0))

    def update(
        self,
        action: AgentAction,
        result: ActionResult,
        player,
        screenshot_bytes: bytes | None = None,
        *,
        progress: dict | None = None,
        forced_movement: bool = False,
        objective_distance: int | None = None,
    ) -> StuckInfo:
        # --- setback: money dropped (whiteout proxy) ---
        setback = False
        if progress is not None:
            money = progress.get("money")
            if self._last_money is not None and money is not None and money < self._last_money:
                setback = True
            if money is not None:
                self._last_money = money

        # --- forced movement suppresses stuck entirely ---
        if forced_movement:
            self._count = 0
            return StuckInfo(stuck=False, kind="forced_movement_suppressed", setback=setback)

        # --- local loop (unchanged behavior) ---
        key = self._key(action, player, screenshot_bytes)
        no_progress = result.result in ("blocked", "timeout") or result.player_moved is False
        if no_progress and key == self._last_key:
            self._count += 1
        elif no_progress:
            self._count = 1
        else:
            self._count = 0
        self._last_key = key

        # --- objective wedge: progress is EITHER getting closer to the goal map OR
        # discovering new ground (badges/levels/maps/tiles). Reset the wedge on either;
        # only stagnation on BOTH counts as wedged. (Map-hop distance alone is too
        # coarse: it's constant while you cross a long map, which isn't being stuck.) ---
        improved = False
        if objective_distance is not None:
            if self._best_dist is None or objective_distance < self._best_dist:
                self._best_dist = objective_distance
                improved = True
        if progress is not None:
            sig = self._progress_sig(progress)
            if self._best_sig is None or sig > self._best_sig:
                self._best_sig = sig
                improved = True
        if objective_distance is not None or progress is not None:
            self._wedge = 0 if improved else self._wedge + 1

        # position stall / oscillation: the last 6 positions occupy <=2 tiles — either
        # stationary (input eaten, e.g. bumping a sign that opens a dialog) or bouncing
        # A<->B in a corridor. The same-action counter misses these (action/pos alternate).
        if player is not None:
            self._recent_pos.append((player.map_id, player.x, player.y))
        oscillating = len(self._recent_pos) >= 6 and len(set(self._recent_pos)) <= 2

        if no_progress and self._count >= self.threshold:
            return StuckInfo(stuck=True, reason="same_action_no_progress_repeated",
                             kind="local_loop", repeat_count=self._count, setback=setback)
        if oscillating:
            return StuckInfo(stuck=True, reason="position_oscillation",
                             kind="local_loop", repeat_count=len(self._recent_pos), setback=setback)
        if (objective_distance is not None or progress is not None) and self._wedge >= self.wedge_threshold:
            return StuckInfo(stuck=True, reason="no_objective_progress",
                             kind="no_objective_progress", repeat_count=self._wedge, setback=setback)
        return StuckInfo(stuck=False, repeat_count=self._count, setback=setback)
