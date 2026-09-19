"""Full-fidelity run recorder — so we never have to hand-reproduce a state again.

Writes, into ``runs/<run_id>/``:
  * ``log.jsonl``  — one line per step with EVERYTHING the agent saw (player, npcs+kind,
    facing/interaction, context, exits, the active directive/quest, action, result, events).
  * ``shots/NNNNNN.png`` — the screen each step (or every ``shot_every`` steps).
  * ``states/map<M>_step<N>.state`` — a reloadable emulator save state on ENTERING A NEW MAP
    (and periodically), so any newly-seen area can be reloaded and inspected offline.
  * ``viewer.html`` — a self-contained local viewer to scrub THIS run (screen + state panel).

It also drops ``runs/_viewer.html`` (one level up) — the multi-run PLAYER with a run-picker
dropdown, play/pause, speed, and live-follow — so serving ``runs/`` and opening ``_viewer.html``
lets you browse and play back every run. Serve with ``python -m http.server`` from ``runs/``.

Best-effort: any failure is swallowed so recording never breaks a run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

_VIEWER_SRC = Path(__file__).with_name("viewer.html")   # per-run viewer (this run only)
_PLAYER_SRC = Path(__file__).with_name("player.html")   # multi-run player w/ run dropdown (runs/ root)


class RunRecorder:
    def __init__(self, emu, run_dir: str | Path, *, shot_every: int = 1,
                 state_every: int = 0):
        self.emu = emu
        self.dir = Path(run_dir)
        self.shots = self.dir / "shots"
        self.states = self.dir / "states"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shots.mkdir(exist_ok=True)
        self.states.mkdir(exist_ok=True)
        self.shot_every = max(1, shot_every)
        self.state_every = state_every  # 0 = only on new map
        self._fh = (self.dir / "log.jsonl").open("a", encoding="utf-8")
        self._seen_maps: set[int] = set()
        self._events: list[dict] = []
        try:
            if _VIEWER_SRC.exists():
                (self.dir / "viewer.html").write_bytes(_VIEWER_SRC.read_bytes())
            # drop the multi-run PLAYER at the runs/ root so a single URL browses every run
            if _PLAYER_SRC.exists():
                (self.dir.parent / "_viewer.html").write_bytes(_PLAYER_SRC.read_bytes())
        except Exception:
            pass

    def on_event(self, kind: str, payload: dict) -> None:
        """Collect the step's events (directive/quest/kb_search/escalate/stuck/...) to attach
        to the next record. Wrap an existing on_event by calling this alongside it."""
        self._events.append({"kind": kind, **{k: v for k, v in payload.items() if k != "step"}})

    def record(self, *, step: int, obs, action, result, extra: dict | None = None) -> None:
        try:
            gs = obs.game_state or {}
            player = obs.player.model_dump() if obs.player else None
            mid = obs.player.map_id if obs.player else None
            shot_rel = self._maybe_shot(step)
            state_rel = self._maybe_state(step, mid)
            mv = getattr(obs, "map_view", None)
            rec = {
                "step": step, "t": round(time.time(), 2),
                "map_id": mid, "map_name": getattr(obs.player, "map_name", None) if obs.player else None,
                "player": player,
                # the coordinate-labeled ASCII map the agent navigates on + what it's tracking now
                "map_view": ("\n".join(mv) if isinstance(mv, list) else mv),
                **(extra or {}),
                "context": {k: gs.get("context", {}).get(k) for k in ("kind", "in_battle", "screen_text")},
                "npcs": [{"x": n.get("x"), "y": n.get("y"), "sprite": n.get("sprite"),
                          "kind": n.get("kind"), "facing": n.get("facing")} for n in (gs.get("npcs") or [])],
                "facing": gs.get("facing"),
                "exits": obs.exits,
                "party": [f"{p.get('species')} L{p.get('level')} {p.get('hp')}/{p.get('max_hp')}"
                          for p in (gs.get("party") or [])],
                "items": [it.get("item") for it in (gs.get("items") or [])],
                "action": action.model_dump() if action is not None else None,
                "result": getattr(result, "result", None),
                "detail": getattr(result, "detail", None),
                "events": self._events,
                "shot": shot_rel,
                "state": state_rel,
            }
            self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()
        except Exception:
            pass
        finally:
            self._events = []

    def _maybe_shot(self, step: int) -> str | None:
        if step % self.shot_every != 0:
            return None
        try:
            png = self.emu.screenshot().data
            name = f"{step:06d}.png"
            (self.shots / name).write_bytes(png)
            return f"shots/{name}"
        except Exception:
            return None

    def _maybe_state(self, step: int, mid: int | None) -> str | None:
        # save a reloadable state when we ENTER A NEW MAP (a new area to inspect), + optionally
        # every state_every steps.
        new_map = mid is not None and mid not in self._seen_maps
        periodic = self.state_every and step % self.state_every == 0
        if not (new_map or periodic):
            return None
        if mid is not None:
            self._seen_maps.add(mid)
        try:
            name = f"map{mid}_step{step}.state"
            self.emu.save_state(self.states / name)
            return f"states/{name}"
        except Exception:
            return None

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
