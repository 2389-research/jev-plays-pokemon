"""PortalGraph interface: routing, live component lookup, and the L2 view render."""
from pokemon_agent.agent.portal_graph import PortalGraph

FOREST, ROUTE2, PEWTER = 51, 13, 2
NORTH_GATE, SOUTH_GATE = 47, 50
ROUTE2_SOUTH, ROUTE2_NORTH = 10, 0   # walkable components (from the rip)


def _pg():
    return PortalGraph.load()


# --- L2: routing over the portal graph --------------------------------------
def test_route_forest_to_pewter_goes_via_north_gate_not_south():
    pg = _pg()
    dests = [p["dest_map"] for p in pg.route(FOREST, 0, PEWTER)]
    assert NORTH_GATE in dests
    assert SOUTH_GATE not in dests          # the fix: cross via the north gate, not back south
    assert dests[-1] == PEWTER


def test_next_portal_from_forest_is_the_north_gate():
    pg = _pg()
    nxt = pg.next_portal(FOREST, 0, PEWTER)
    assert nxt is not None and nxt["dest_map"] == NORTH_GATE


def test_route2_south_component_must_detour_through_the_forest():
    pg = _pg()
    dests = [p["dest_map"] for p in pg.route(ROUTE2, ROUTE2_SOUTH, PEWTER)]
    # from the SOUTH side of Route 2 you cannot step straight to Pewter — you go through the forest
    assert SOUTH_GATE in dests and FOREST in dests
    assert dests.index(SOUTH_GATE) < dests.index(PEWTER)


def test_route2_north_component_reaches_pewter_directly():
    pg = _pg()
    r = pg.route(ROUTE2, ROUTE2_NORTH, PEWTER)
    assert [p["dest_map"] for p in r] == [PEWTER]   # one hop: the north edge


def test_reachable_maps_from_forest_includes_pewter():
    pg = _pg()
    assert PEWTER in pg.reachable_maps(FOREST, 0)


def test_route_to_same_map_is_empty():
    assert _pg().route(FOREST, 0, FOREST) == []


# --- live: locate the player's component from the collision map --------------
def test_component_at_returns_portal_component_on_portal_tile():
    pg = _pg()
    # standing on the forest's north-gate warp tile (1,0) -> its component (0)
    assert pg.component_at(FOREST, 1, 0, {(1, 0)}) == 0


def test_component_at_flood_fills_to_a_portal():
    pg = _pg()
    # a little walkable corridor from (3,0) back to the warp tile (1,0)
    walk = {(3, 0), (2, 0), (1, 0)}
    assert pg.component_at(FOREST, 3, 0, walk) == 0


def test_component_at_none_when_isolated():
    pg = _pg()
    assert pg.component_at(FOREST, 99, 99, {(99, 99)}) is None


# --- L2 model input: the directed natural-language view ----------------------
def test_render_view_forest_names_the_gates_and_directions():
    view = _pg().render_view(FOREST, 0, PEWTER)
    assert "Viridian Forest North Gate" in view
    assert "Viridian Forest South Gate" in view
    assert "GOAL: reach Pewter City" in view
    assert "NORTH ->" in view                # directed connectivity present
    # no raw coordinates leak into the model-facing view
    assert "x=" not in view and "coord" not in view
