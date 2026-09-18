"""Persistent agent memory + checkpoint format (P0).

One container for everything the agent learns, serialized to JSON alongside the
emulator save state so a checkpoint restores BOTH (reloading emu state without
memory desyncs the world model — see the spec's recovery section). Holds:

  * world        — per-map tile occupancy (WorldMap)
  * graph        — cross-map warp graph (WorldGraph), seeded with the Kanto corridor
  * interactions — talked/empty tiles + dialog log (InteractionMemory)
  * plan         — the current AgentPlan
  * map_history  — ordered distinct maps visited
  * tried_failed — approaches shown not to work (executor/recovery consult this)
  * notes        — freeform memory entries, each tagged observed (RAM truth) vs inferred
"""
from __future__ import annotations

import json
from pathlib import Path

from .interaction_memory import InteractionMemory
from .plan import AgentPlan
from .world_graph import WorldGraph, full_kanto_graph
from .world_map import WorldMap

SCHEMA_VERSION = 1


class AgentMemory:
    def __init__(
        self,
        *,
        world: WorldMap | None = None,
        interactions: InteractionMemory | None = None,
        graph: WorldGraph | None = None,
        plan: AgentPlan | None = None,
    ):
        self.world = world or WorldMap()
        self.interactions = interactions or InteractionMemory()
        self.graph = graph or full_kanto_graph()
        self.plan = plan
        self.map_history: list[int] = []
        self.tried_failed: list[str] = []
        self.notes: list[dict] = []  # {text, source: observed|inferred, step}
        # dead-end ledger: (map_id, x, y, direction) proven BLOCKED by a real move whose
        # coords didn't change. Append-only, RAM-keyed, model-immutable — the mask source.
        self.blocked_edges: set[tuple[int, int, int, str]] = set()

    # --- updates ----------------------------------------------------------
    def observe_map(self, map_id: int | None) -> None:
        if map_id is not None and (not self.map_history or self.map_history[-1] != map_id):
            self.map_history.append(map_id)

    def note(self, text: str, *, source: str = "inferred", step: int | None = None) -> None:
        """Record a memory entry. `observed` = grounded in RAM; `inferred` = an LLM guess
        (the executor should trust observed over inferred)."""
        self.notes.append({"text": text, "source": source, "step": step})

    def mark_tried_failed(self, text: str) -> None:
        if text and text not in self.tried_failed:
            self.tried_failed.append(text)

    def mark_blocked_edge(self, map_id: int, x: int, y: int, direction: str) -> None:
        self.blocked_edges.add((int(map_id), int(x), int(y), direction))

    def is_blocked_edge(self, map_id: int, x: int, y: int, direction: str) -> bool:
        return (int(map_id), int(x), int(y), direction) in self.blocked_edges

    def summary(self) -> dict:
        """Bounded view for the strategist / logs — never the whole memory."""
        return {
            "plan": self.plan.model_dump() if self.plan else None,
            "map_history": self.map_history[-12:],
            "tried_failed": self.tried_failed,
            "notes": self.notes[-10:],
            "social": self.interactions.summary(),
        }

    # --- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "world": self.world.to_dict(),
            "graph": self.graph.to_dict(),
            "interactions": self.interactions.to_dict(),
            "plan": self.plan.model_dump() if self.plan else None,
            "map_history": self.map_history,
            "tried_failed": self.tried_failed,
            "notes": self.notes,
            "blocked_edges": [list(e) for e in self.blocked_edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentMemory":
        m = cls(
            world=WorldMap.from_dict(d.get("world") or {}),
            interactions=InteractionMemory.from_dict(d.get("interactions") or {}),
            graph=WorldGraph.from_dict(d.get("graph") or {}),
            plan=AgentPlan.model_validate(d["plan"]) if d.get("plan") else None,
        )
        m.map_history = list(d.get("map_history") or [])
        m.tried_failed = list(d.get("tried_failed") or [])
        m.notes = list(d.get("notes") or [])
        m.blocked_edges = {tuple(e) for e in d.get("blocked_edges") or []}
        return m

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "AgentMemory":
        return cls.from_dict(json.loads(Path(path).read_text()))
