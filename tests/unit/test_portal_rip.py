"""Unit tests for the pure logic of the static portal-graph ripper
(``scripts/rip_portals.py``): connected-component splitting and BFS routing.

These use synthetic in-memory data only — no pokered files, no emulator — so they
run fast and deterministically via ``uv run python -m pytest``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "rip_portals", Path(__file__).resolve().parents[2] / "scripts" / "rip_portals.py"
)
rip = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rip)


# --------------------------------------------------------------------------- #
# connected_components
# --------------------------------------------------------------------------- #
def test_components_single_blob():
    walk = {(0, 0), (1, 0), (0, 1), (1, 1)}
    comp = rip.connected_components(walk)
    assert len(set(comp.values())) == 1


def test_components_two_disjoint_blobs():
    left = {(0, 0), (0, 1), (1, 0)}
    right = {(5, 5), (5, 6), (6, 5)}
    comp = rip.connected_components(left | right)
    assert len(set(comp.values())) == 2
    # each blob internally shares one id; the two blobs differ
    assert len({comp[c] for c in left}) == 1
    assert len({comp[c] for c in right}) == 1
    assert next(iter(left)) != next(iter(right))
    assert comp[(0, 0)] != comp[(5, 5)]


def test_components_diagonal_is_not_connected():
    # 4-connectivity: diagonal touch does NOT join.
    walk = {(0, 0), (1, 1)}
    comp = rip.connected_components(walk)
    assert len(set(comp.values())) == 2


def test_components_deterministic_ordering():
    walk = {(9, 9), (0, 0)}
    comp = rip.connected_components(walk)
    # numbered in ascending (y, x); (0,0) seen first -> id 0
    assert comp[(0, 0)] == 0
    assert comp[(9, 9)] == 1


# --------------------------------------------------------------------------- #
# route  — the Route-2-split / forest-crossing shape, in miniature
# --------------------------------------------------------------------------- #
def _corridor_portals():
    """A minimal portal graph reproducing the crux geometry:

      Route2 (map 13) has TWO components:
        - north comp 0: north edge -> Pewter (2), and warp -> North Gate (47)
        - south comp 1: warp -> South Gate (50)
      North Gate (47): warp back to Route2 north comp, warp to Forest
      South Gate (50): warp to Route2 south comp, warp to Forest
      Forest (51): single comp; warps to both gates
      Pewter (2): south edge back to Route2 north.
    """
    P = {
        # Route2 north component (0)
        "route2:edge_north_c0": {"id": "route2:edge_north_c0", "map": 13, "kind": "edge",
                                 "component": 0, "dest_map": 2, "dest_portal": "pewter:edge_south_c0"},
        "route2:warp_ng": {"id": "route2:warp_ng", "map": 13, "kind": "warp",
                           "component": 0, "dest_map": 47, "dest_portal": "northgate:warp_r2"},
        # Route2 south component (1)
        "route2:warp_sg": {"id": "route2:warp_sg", "map": 13, "kind": "warp",
                           "component": 1, "dest_map": 50, "dest_portal": "southgate:warp_r2"},
        "route2:edge_south_c1": {"id": "route2:edge_south_c1", "map": 13, "kind": "edge",
                                 "component": 1, "dest_map": 1, "dest_portal": None},
        # Pewter
        "pewter:edge_south_c0": {"id": "pewter:edge_south_c0", "map": 2, "kind": "edge",
                                 "component": 0, "dest_map": 13, "dest_portal": "route2:edge_north_c0"},
        # North Gate
        "northgate:warp_r2": {"id": "northgate:warp_r2", "map": 47, "kind": "warp",
                              "component": 0, "dest_map": 13, "dest_portal": "route2:warp_ng"},
        "northgate:warp_forest": {"id": "northgate:warp_forest", "map": 47, "kind": "warp",
                                  "component": 0, "dest_map": 51, "dest_portal": "viridianforest:warp_ng"},
        # South Gate
        "southgate:warp_r2": {"id": "southgate:warp_r2", "map": 50, "kind": "warp",
                              "component": 0, "dest_map": 13, "dest_portal": "route2:warp_sg"},
        "southgate:warp_forest": {"id": "southgate:warp_forest", "map": 50, "kind": "warp",
                                  "component": 0, "dest_map": 51, "dest_portal": "viridianforest:warp_sg"},
        # Forest (single component)
        "viridianforest:warp_ng": {"id": "viridianforest:warp_ng", "map": 51, "kind": "warp",
                                   "component": 0, "dest_map": 47, "dest_portal": "northgate:warp_forest"},
        "viridianforest:warp_sg": {"id": "viridianforest:warp_sg", "map": 51, "kind": "warp",
                                   "component": 0, "dest_map": 50, "dest_portal": "southgate:warp_forest"},
    }
    return P


def test_route_forest_to_pewter_via_north_gate():
    P = _corridor_portals()
    path = rip.route(P, from_map=51, from_component=0, to_map=2)
    assert path is not None
    assert any("northgate" in pid for pid in path), path
    assert not any("southgate" in pid for pid in path), path
    # ends by stepping through the Route2 north edge onto Pewter
    assert P[path[-1]]["dest_map"] == 2


def test_route_route2_south_must_detour_through_forest():
    P = _corridor_portals()
    path = rip.route(P, from_map=13, from_component=1, to_map=2)
    assert path is not None
    # cannot reach Pewter from south component without crossing the forest
    assert any(pid.startswith("viridianforest:") for pid in path), path
    assert any("southgate" in pid for pid in path), path


def test_route_same_component_only():
    # From Route2 NORTH component, Pewter is reachable directly via the north edge.
    P = _corridor_portals()
    path = rip.route(P, from_map=13, from_component=0, to_map=2)
    assert path == ["route2:edge_north_c0"] or path[-1] == "route2:edge_north_c0"


def test_route_same_map_is_empty():
    P = _corridor_portals()
    assert rip.route(P, from_map=13, from_component=0, to_map=13) == []


def test_route_unreachable_returns_none():
    P = _corridor_portals()
    # map 999 does not exist in the graph
    assert rip.route(P, from_map=51, from_component=0, to_map=999) is None
