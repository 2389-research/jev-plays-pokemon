"""Can a LunaRoute model QUERY Orrery and figure out WHERE to buy an item — with no curated shop doc,
just the KG's existing entities/relations? Tests the claim that a curated 'shops & items' doc is
needed. A tool-loop: the model issues search() against the live Orrery KG (workspace 6d677a16), we
feed back the entities + their 1-hop neighbours, and it answers where to buy the item.

Live (spends LunaRoute credits + hits Orrery): uv run python scripts/eval_shop_kg_query.py
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
    q = urllib.parse.urlencode({"q": query})
    ents = _get(f"/search?{q}").get("entities", [])[:8]
    out = {"entities": [{"name": e["name"], "type": e.get("type")} for e in ents]}
    # add neighbours for the top couple of entities so relations (e.g. sold-at) surface
    out["connections"] = {}
    for e in ents[:3]:
        try:
            nb = _get(f"/graph/neighborhood/{urllib.parse.quote(e['name'])}")
            out["connections"][e["name"]] = [n["name"] for n in nb.get("nodes", []) if n.get("depth", 0) > 0][:8]
        except Exception:
            pass
    return out


SYS = (
    "You are playing Pokémon Red and need to know WHERE to buy an item. You can consult a knowledge "
    "base with a search query. Reply ONLY as JSON, EITHER:\n"
    '  {"search": "<query>"}   to look it up, OR\n'
    '  {"answer": "<the place (town/Mart) to buy it), and how you know>"}  once you can tell.\n'
    "Search first; then answer with the specific place to buy it.")

QUESTIONS = [
    ("Where do I buy Potions before the Pewter Gym?", {"pewter"}),
    ("Where can I buy Poké Balls early in the game?", {"viridian", "pewter", "mart"}),
]


def run(model):
    from pokemon_agent.providers.lunaroute import LunaRouteProvider
    prov = LunaRouteProvider(model=model)
    print(f"\n################ MODEL: {model} ################")
    for q, accept in QUESTIONS:
        convo = {"question": q, "knowledge_so_far": None}
        answer, queries = None, []
        MAX = 4
        for i in range(MAX):
            sysp = SYS + ("\nThis is your LAST turn: you MUST answer now." if i == MAX - 1 else "")
            content, _, _ = prov.chat_json(sysp, convo)
            try:
                data = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
            except Exception:
                break
            if "answer" in data:
                answer = str(data["answer"]); break
            qq = str(data.get("search", "")).strip(); queries.append(qq)
            try:
                convo["knowledge_so_far"] = kg_search(qq)
            except Exception as e:
                convo["knowledge_so_far"] = {"error": str(e)}
        ok = answer is not None and any(a in answer.lower() for a in accept)
        print(f"  [{'PASS' if ok else 'FAIL'}] {q}")
        print(f"        queries: {queries}")
        print(f"        answer:  {answer!r}")


if __name__ == "__main__":
    print("=== sample KG result for 'buy Potions Poke Mart' ===")
    print(json.dumps(kg_search("buy Potions Poke Mart Pewter"), indent=1)[:900])
    for m in ["glm-5.3", "deepseek-4.1-flash"]:
        try:
            run(m)
        except Exception as e:
            print(f"model {m}: {type(e).__name__}: {e}")
