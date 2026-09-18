"""Orrery knowledge-base client — the agent's durable, queryable game knowledge.

Separation of concerns (per the design): STATIC reference knowledge (walkthroughs,
mechanics, quest requirements, gym/type info) lives in an Orrery noosphere and is
retrieved by semantic search + graph queries when the planner needs to reason ("how do
I get past the old man?"). MUTABLE execution state (the todo list, RAM-verified progress)
stays local. This grounds planning in retrieved guides instead of model priors.

Best-effort and optional: any failure (Orrery down, empty KB) returns nothing, so the
agent degrades to model-knowledge planning rather than breaking. No new dependencies —
stdlib urllib only.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


class KnowledgeBase:
    def __init__(self, base_url: str, workspace_id: str, *, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.workspace_id = workspace_id
        self.timeout = timeout

    # --- construction from env/flags ------------------------------------
    @classmethod
    def from_env(cls, base_url: str | None = None, workspace_id: str | None = None):
        """Build from args or env (ORRERY_BASE_URL / ORRERY_WORKSPACE_ID). Returns None when
        no workspace is configured (KB simply disabled)."""
        ws = workspace_id or os.environ.get("ORRERY_WORKSPACE_ID")
        if not ws:
            return None
        url = base_url or os.environ.get("ORRERY_BASE_URL", "http://localhost:8100")
        return cls(url, ws)

    # --- semantic search -------------------------------------------------
    def query(self, text: str, *, top_k: int = 5) -> list[dict]:
        """Semantic search -> a list of {title, text} passages, most relevant first. Empty on
        any error (the KB is an optional aid, never a hard dependency)."""
        qs = urllib.parse.urlencode({"q": text, "top_k": top_k})
        try:
            req = urllib.request.Request(f"{self.base_url}/search?{qs}",
                                         headers={"X-Workspace-Id": self.workspace_id})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except Exception:
            return []
        out: list[dict] = []
        for c in data.get("chunks", []) or []:
            txt = c.get("text") or c.get("content") or ""
            if txt:
                out.append({"title": c.get("document_title") or "", "text": txt})
        return out

    def query_texts(self, text: str, *, top_k: int = 5, max_chars: int = 1600) -> list[str]:
        """Retrieved passages as ``"title: text"`` strings, trimmed to a total budget — ready
        to drop into a planner prompt."""
        chunks = self.query(text, top_k=top_k)
        out: list[str] = []
        used = 0
        for c in chunks:
            s = (f"{c['title']}: " if c["title"] else "") + c["text"]
            if used + len(s) > max_chars:
                s = s[: max(0, max_chars - used)]
            if s:
                out.append(s)
                used += len(s)
            if used >= max_chars:
                break
        return out
