from pokemon_agent.agent.world_graph import WorldGraph, seeded_kanto_graph
from pokemon_agent.games.pokemon_red.constants import MAP_NAMES_RAW


# --- basic graph construction / queries -------------------------------------
def test_add_warp_and_neighbors():
    g = WorldGraph()
    g.add_warp(0, 3, 4, 12)
    g.add_warp(0, 1, 1, 99)
    assert g.neighbors(0) == {12, 99}
    assert g.neighbors(12) == set()  # node exists but no outgoing edges


def test_route_simple_chain():
    g = WorldGraph()
    g.add_warp(0, 3, 4, 12)
    g.add_warp(12, 5, 6, 1)
    g.add_warp(1, 7, 8, 2)
    assert g.route(0, 2) == [0, 12, 1, 2]


def test_route_same_endpoint_is_inclusive_singleton():
    g = WorldGraph()
    assert g.route(5, 5) == [5]


def test_route_unreachable_is_none():
    g = WorldGraph()
    g.add_warp(0, 3, 4, 12)
    assert g.route(0, 99) is None
    assert g.route(50, 51) is None  # nodes not even present


def test_route_picks_shortest():
    g = WorldGraph()
    # long way: 0->1->2->3 ; short way: 0->3
    g.add_warp(0, 1, 1, 1)
    g.add_warp(1, 1, 1, 2)
    g.add_warp(2, 1, 1, 3)
    g.add_warp(0, 9, 9, 3)
    assert g.route(0, 3) == [0, 3]


# --- next_hop ---------------------------------------------------------------
def test_next_hop_returns_map_and_tile():
    g = WorldGraph()
    g.add_warp(0, 3, 4, 12)
    g.add_warp(12, 5, 6, 1)
    assert g.next_hop(0, 1) == (12, (3, 4))
    assert g.next_hop(12, 1) == (1, (5, 6))


def test_next_hop_none_when_same_map():
    g = WorldGraph()
    assert g.next_hop(5, 5) is None


def test_next_hop_none_when_no_route():
    g = WorldGraph()
    g.add_warp(0, 3, 4, 12)
    assert g.next_hop(0, 99) is None


def test_next_hop_unknown_tile_still_returns_hop():
    g = WorldGraph()
    g.add_adjacency(0, 12)  # adjacency only, tile unknown
    hop = g.next_hop(0, 12)
    assert hop == (12, None)


# --- observe_exits ----------------------------------------------------------
def test_observe_exits_adds_edges():
    g = WorldGraph()
    exits = [
        {"x": 3, "y": 4, "dest_map": 12, "dest_name": "Route 1"},
        {"x": 8, "y": 0, "dest_map": 40, "dest_name": "Oaks Lab"},
    ]
    g.observe_exits(0, exits)
    assert g.neighbors(0) == {12, 40}
    assert g.next_hop(0, 12) == (12, (3, 4))


def test_observe_exits_refines_seeded_adjacency():
    g = WorldGraph()
    g.add_adjacency(0, 12)
    assert g.next_hop(0, 12) == (12, None)
    g.observe_exits(0, [{"x": 3, "y": 4, "dest_map": 12}])
    assert g.next_hop(0, 12) == (12, (3, 4))  # tile filled in


def test_adjacency_does_not_clobber_known_tile():
    g = WorldGraph()
    g.add_warp(0, 3, 4, 12)
    g.add_adjacency(0, 12)  # must NOT wipe the known tile
    assert g.edges[0][12] == (3, 4)


def test_observe_exits_skips_missing_dest():
    g = WorldGraph()
    g.observe_exits(0, [{"x": 1, "y": 1, "dest_map": None}])
    assert g.neighbors(0) == set()


# --- serialization ----------------------------------------------------------
def test_to_dict_from_dict_roundtrip():
    import json

    g = WorldGraph()
    g.add_warp(0, 3, 4, 12)
    g.add_adjacency(12, 1)
    g.unresolved = ["Somewhere"]
    d = g.to_dict()
    # must be JSON-serializable
    d2 = json.loads(json.dumps(d))
    g2 = WorldGraph.from_dict(d2)
    assert g2.edges[0][12] == (3, 4)
    assert g2.neighbors(12) == g.neighbors(12)
    assert g2.route(0, 1) == [0, 12, 1]
    assert g2.unresolved == ["Somewhere"]


# --- seed -------------------------------------------------------------------
def test_seeded_kanto_graph_connected_pallet_to_pewter():
    g = seeded_kanto_graph()
    name_to_id = {n: i for i, n in MAP_NAMES_RAW.items()}
    pallet = name_to_id["Pallet Town"]
    pewter = name_to_id["Pewter City"]
    route = g.route(pallet, pewter)
    assert route is not None, "Pallet -> Pewter must be connected in the seed"
    assert route[0] == pallet and route[-1] == pewter
    # nothing left unresolved for the core corridor names
    assert g.unresolved == []


def test_seeded_corridor_resolved_ids():
    g = seeded_kanto_graph()
    name_to_id = {n: i for i, n in MAP_NAMES_RAW.items()}
    # sanity: the specific ids the seed relies on
    assert name_to_id["Pallet Town"] == 0
    assert name_to_id["Route 1"] == 12
    assert name_to_id["Viridian City"] == 1
    assert name_to_id["Route 2"] == 13
    assert name_to_id["Viridian Forest"] == 51
    assert name_to_id["Pewter City"] == 2
    # gate buildings present too
    assert name_to_id["Viridian Forest South Gate"] == 50
    assert name_to_id["Viridian Forest North Gate"] == 47
    # each resolved corridor node has at least one edge
    for mid in (0, 12, 1, 13, 51, 47, 50, 2):
        assert g.neighbors(mid), f"map {mid} should have neighbors in seed"


def test_seeded_next_hop_walks_toward_pewter():
    g = seeded_kanto_graph()
    # from Pallet(0) the first hop is Route 1(12), tile unknown until observed
    hop = g.next_hop(0, 2)
    assert hop is not None
    assert hop[0] == 12
    assert hop[1] is None


def test_full_kanto_graph_from_pokered_routes_pallet_to_pewter():
    from pokemon_agent.agent.world_graph import full_kanto_graph
    g = full_kanto_graph()
    route = g.route(0, 2)                       # Pallet Town -> Pewter City
    assert route and route[0] == 0 and route[-1] == 2
    # every hop toward Pewter is "north" (the ripped connection directions)
    assert g.route_direction(0, 2) == "north"
    assert g.route_direction(12, 2) == "north"  # from Route 1
    # ripped, not hand-authored: a decent chunk of the region is present
    assert len(g.edges) >= 20
