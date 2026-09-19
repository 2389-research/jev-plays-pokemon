"""Weighted routing under policies + Jev policy selection."""
from types import SimpleNamespace

from pokemon_agent.agent.routing import policy_first_step, tile_cost, grass_nearby
from pokemon_agent.agent.world_map import WorldMap
from pokemon_agent.core.models import Direction


def _world():
    # a 5x1 corridor: start at (0,0), goal (4,0); (2,0) is GRASS, rest floor. Going straight is
    # shortest; there is no non-grass detour here (1-wide), so all policies must cross the grass.
    w = WorldMap()
    walk = {(x, 0) for x in range(5)}
    terr = {(x, 0): ("grass" if x == 2 else "floor") for x in range(5)}
    w.ingest_collision(0, 5, 1, walk, terrain=terr)
    return w


def test_tile_cost_by_policy():
    assert tile_cost("grass", "dodge-grass") == 12.0 and tile_cost("floor", "dodge-grass") == 1.0
    assert tile_cost("grass", "farm-exp") == 1.0 and tile_cost("floor", "farm-exp") == 4.0
    assert tile_cost("grass", "shortest") == 1.0


def test_policy_route_first_step_toward_goal():
    w = _world()
    d = policy_first_step(w, 0, (0, 0), (4, 0), "shortest", set())
    assert d == Direction.EAST


def test_dodge_prefers_grass_free_detour():
    # 3x3: straight east row y=1 has grass at (1,1); y=0 is a clear detour. dodge-grass should
    # step up (north) to avoid grass; shortest goes straight east.
    w = WorldMap()
    walk = {(x, y) for x in range(3) for y in range(2)}
    terr = {c: "floor" for c in walk}
    terr[(1, 1)] = "grass"
    w.ingest_collision(0, 3, 2, walk, terrain=terr)
    assert policy_first_step(w, 0, (0, 1), (2, 1), "shortest", set()) == Direction.EAST
    assert policy_first_step(w, 0, (0, 1), (2, 1), "dodge-grass", set()) == Direction.NORTH


def test_grass_nearby():
    w = _world()
    assert grass_nearby(w, 0, (0, 0), radius=4) is True
    assert grass_nearby(w, 0, (0, 0), radius=1) is False


class PolClient:
    def __init__(self, choice): self.choice = choice
    def system_one(self, *, state, questions):
        self.state = state; self.criteria = questions["pol"].criteria
        return SimpleNamespace(answers={"pol": SimpleNamespace(choice=self.choice, confidence=0.9)}, usage=None)


def test_jev_choose_policy():
    from pokemon_agent.agent.typesafe_reasoner import TypeSafeReasoner
    c = PolClient("dodge-grass")
    pol, conf = TypeSafeReasoner(client=c).choose_policy(hp_frac=0.15, level=7, level_target=10,
                                                         objective="deliver parcel", area="Route 1")
    assert pol == "dodge-grass" and conf == 0.9
    assert set(c.criteria) == {"shortest", "dodge-grass", "farm-exp"}
    assert c.state["party_hp_frac"] == 0.15
