"""Orrery KnowledgeBase client (pure — no live server)."""
from pokemon_agent.agent.knowledge import KnowledgeBase


def test_from_env_disabled_without_workspace(monkeypatch):
    monkeypatch.delenv("ORRERY_WORKSPACE_ID", raising=False)
    assert KnowledgeBase.from_env() is None
    kb = KnowledgeBase.from_env(workspace_id="ws1")
    assert kb is not None and kb.workspace_id == "ws1"


def test_query_texts_formats_title_and_body():
    kb = KnowledgeBase("http://x", "w")
    kb.query = lambda text, top_k=5: [{"title": "Parcel", "text": "get it from the clerk"},
                                      {"title": "", "text": "no title"}]
    assert kb.query_texts("q") == ["Parcel: get it from the clerk", "no title"]


def test_query_texts_respects_char_budget():
    kb = KnowledgeBase("http://x", "w")
    kb.query = lambda text, top_k=5: [{"title": "", "text": "x" * 100}, {"title": "", "text": "y" * 100}]
    out = kb.query_texts("q", max_chars=50)
    assert sum(len(s) for s in out) <= 50


def test_query_swallows_errors():
    kb = KnowledgeBase("http://127.0.0.1:9", "w", timeout=0.2)  # nothing listening
    assert kb.query("anything") == []  # best-effort: never raises
