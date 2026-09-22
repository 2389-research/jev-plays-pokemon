"""Test: can a LunaRoute model QUERY the Orrery KG and decide a navigation step from what it gets back?

A small manual tool-loop: the model is given a goal and a `search(query)` tool backed by Orrery. It
issues a query, we run it against the live KG (workspace 6d677a16) and feed the results back, then it
answers with the next area to head to. This tests the intended architecture — Orrery as the queryable
knowledge backend the model actually uses — as opposed to handing it a pre-rendered view.

Live (spends LunaRoute credits + hits Orrery): uv run python scripts/eval_kg_query.py
"""
import json
import re
import urllib.parse
import urllib.request

ORRERY = "http://localhost:8100"
WS = "6d677a16"


def _get(path):
    req = urllib.request.Request(f"{ORRERY}{path}", headers={"X-Workspace-Id": WS})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def kg_search(query: str) -> dict:
    """What the model's tool returns: top entities for the query, plus the 1-hop neighbours (connected
    places) of the best-matching location — i.e. Orrery's knowledge, not our deterministic graph."""
    q = urllib.parse.urlencode({"q": query})
    ents = _get(f"/search?{q}").get("entities", [])[:6]
    out = {"entities": [{"name": e["name"], "type": e.get("type")} for e in ents]}
    loc = next((e for e in ents if e.get("type") == "location"), None)
    if loc:
        nb = _get(f"/graph/neighborhood/{urllib.parse.quote(loc['name'])}")
        out["neighbours_of"] = loc["name"]
        out["connected_to"] = [n["name"] for n in nb.get("nodes", []) if n.get("depth", 0) > 0][:10]
    return out


TOOL_SYS = (
    "You are navigating Pokémon Red. You can consult a knowledge base with a search query. "
    "Given the goal and any knowledge so far, reply ONLY as JSON, EITHER:\n"
    '  {"search": "<a query to look up map/route knowledge>"}  to consult the KB, OR\n'
    '  {"answer": "<the single next area to travel to>"}  once you know enough.\n'
    "Prefer to search first. Then answer with the immediate next place to head toward the goal.")

SCENARIOS = [
    ("In Viridian Forest, goal reach Pewter City", "Viridian Forest", "Pewter City",
     {"pewter", "north gate", "route 2"}),
    ("In Viridian City, goal reach Pewter City", "Viridian City", "Pewter City",
     {"route 2", "pewter"}),
    ("In Pallet Town, goal reach Viridian City", "Pallet Town", "Viridian City",
     {"route 1", "viridian"}),
]


def run(model):
    from pokemon_agent.providers.lunaroute import LunaRouteProvider
    prov = LunaRouteProvider(model=model)
    print(f"\n################ MODEL: {model} ################")
    passed = 0
    for title, loc, goal, accept in SCENARIOS:
        convo = {"you_are": loc, "goal": f"reach {goal}", "knowledge_so_far": None}
        answer = None
        queries = []
        MAX = 3
        for i in range(MAX):  # up to MAX tool turns; the last one MUST answer
            sysp = TOOL_SYS
            if i == MAX - 1:
                sysp += "\nThis is your LAST turn: you MUST reply with {\"answer\": ...} now."
                convo["must_answer_now"] = True
            content, _, _ = prov.chat_json(sysp, convo)
            try:
                data = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
            except Exception:
                break
            if "answer" in data:
                answer = str(data["answer"]); break
            q = str(data.get("search", "")).strip()
            queries.append(q)
            try:
                convo["knowledge_so_far"] = kg_search(q)
            except Exception as e:
                convo["knowledge_so_far"] = {"error": str(e)}
        ok = answer is not None and any(a in answer.lower() for a in accept)
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {title}")
        print(f"        queries: {queries}")
        print(f"        answer:  {answer!r}  (accept any of {accept})")
    print(f"  SCORE: {passed}/{len(SCENARIOS)}")


if __name__ == "__main__":
    print("=== sample KG query result: 'route from Viridian Forest to Pewter' ===")
    print(json.dumps(kg_search("route from Viridian Forest to Pewter City"), indent=1))
    for m in ["deepseek-4.1-flash", "glm-5.3"]:
        try:
            run(m)
        except Exception as e:
            print(f"model {m}: {type(e).__name__}: {e}")
