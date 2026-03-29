#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the hypergraph-native Walker module.
"""

import unittest
from math import pi, tau

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole
from leggedsnake.walker import Walker, walker_from_legacy


def _make_fourbar_walker() -> Walker:
    """Create a simple 4-bar walker for testing."""
    hg = HypergraphLinkage(name="fourbar")
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("follower", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("edge_0", "frame", "crank"))
    hg.add_edge(Edge("edge_1", "frame", "follower"))
    hg.add_edge(Edge("edge_2", "crank", "follower"))
    dims = Dimensions(
        node_positions={"frame": (0, 0), "crank": (1, 0), "follower": (0, 2)},
        driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={
            "edge_0": 1.0,
            "edge_1": 2.0,
            "edge_2": 1.5,
        },
    )
    return Walker(hg, dims, name="fourbar")


def _make_walker_with_foot() -> Walker:
    """Create a walker with a terminal 'foot' node.

    Topology (6 nodes):
        frame --e0-- crank --e2-- upper --e4-- foot
        frame2 ------e1------^     ^----e3---- coupler
                                   coupler --e5-- foot

    'foot' has degree 2 (edges e4, e5) but is the deepest driven node,
    making it a terminal joint in the kinematic chain sense. However,
    for get_feet() we only check graph degree == 1, so we also add a
    truly terminal 'toe' node hanging off 'foot' alone — but that would
    break to_mechanism(). Instead we keep foot with degree 2 and test
    get_feet() with the simpler _make_foot_topology_only() helper.

    This linkage is fully solvable by to_mechanism().
    """
    hg = HypergraphLinkage(name="with_foot")
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("frame2", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("upper", role=NodeRole.DRIVEN))
    hg.add_node(Node("foot", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("e0", "frame", "crank"))
    hg.add_edge(Edge("e1", "frame2", "upper"))
    hg.add_edge(Edge("e2", "crank", "upper"))
    hg.add_edge(Edge("e3", "crank", "foot"))
    hg.add_edge(Edge("e4", "upper", "foot"))
    dims = Dimensions(
        node_positions={
            "frame": (0, 0),
            "frame2": (2, 0),
            "crank": (1, 0),
            "upper": (1, 2),
            "foot": (1, 3),
        },
        driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={
            "e0": 1.0,
            "e1": 2.24,
            "e2": 2.0,
            "e3": 3.16,
            "e4": 1.0,
        },
    )
    return Walker(hg, dims, name="with_foot")


def _make_foot_topology_only() -> Walker:
    """Create a walker with a degree-1 'toe' node for get_feet() testing.

    This walker is for topology queries only (get_feet). It may NOT be
    convertible to a Mechanism since 'toe' has only one neighbor.
    """
    hg = HypergraphLinkage(name="foot_topo")
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("follower", role=NodeRole.DRIVEN))
    hg.add_node(Node("toe", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("e0", "frame", "crank"))
    hg.add_edge(Edge("e1", "frame", "follower"))
    hg.add_edge(Edge("e2", "crank", "follower"))
    hg.add_edge(Edge("e3", "follower", "toe"))
    dims = Dimensions(
        node_positions={
            "frame": (0, 0),
            "crank": (1, 0),
            "follower": (0, 2),
            "toe": (0, 3),
        },
        driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={"e0": 1.0, "e1": 2.0, "e2": 1.5, "e3": 1.0},
    )
    return Walker(hg, dims, name="foot_topo")


class TestWalkerCreation(unittest.TestCase):
    """Test basic Walker creation and attributes."""

    def test_creation(self):
        """Walker can be created with topology and dimensions."""
        walker = _make_fourbar_walker()
        self.assertIsInstance(walker, Walker)

    def test_name(self):
        """Walker stores the name supplied at construction."""
        walker = _make_fourbar_walker()
        self.assertEqual(walker.name, "fourbar")

    def test_topology(self):
        """Walker exposes the HypergraphLinkage topology."""
        walker = _make_fourbar_walker()
        self.assertIsInstance(walker.topology, HypergraphLinkage)
        self.assertEqual(len(walker.topology.nodes), 3)
        self.assertEqual(len(walker.topology.edges), 3)

    def test_dimensions(self):
        """Walker exposes the Dimensions object."""
        walker = _make_fourbar_walker()
        self.assertIsInstance(walker.dimensions, Dimensions)
        self.assertIn("frame", walker.dimensions.node_positions)
        self.assertIn("crank", walker.dimensions.driver_angles)

    def test_name_defaults_to_topology_name(self):
        """When name is empty, Walker uses the topology name."""
        hg = HypergraphLinkage(name="auto_name")
        hg.add_node(Node("a", role=NodeRole.GROUND))
        hg.add_node(Node("b", role=NodeRole.DRIVER))
        hg.add_edge(Edge("ab", "a", "b"))
        dims = Dimensions(
            node_positions={"a": (0, 0), "b": (1, 0)},
            driver_angles={"b": DriverAngle(angular_velocity=0.1)},
            edge_distances={"ab": 1.0},
        )
        walker = Walker(hg, dims)
        self.assertEqual(walker.name, "auto_name")

    def test_default_motor_rates(self):
        """Default motor_rates is -4.0."""
        walker = _make_fourbar_walker()
        self.assertEqual(walker.motor_rates, -4.0)


class TestWalkerStep(unittest.TestCase):
    """Test kinematic simulation via step()."""

    def test_step_yields_positions(self):
        """step() is a generator that yields position tuples."""
        walker = _make_fourbar_walker()
        positions = list(walker.step())
        self.assertGreater(len(positions), 0)
        # Each yielded item is a tuple of (x, y) coordinate tuples
        first = positions[0]
        self.assertIsInstance(first, tuple)
        # Should have one position per joint (3 nodes)
        self.assertEqual(len(first), 3)

    def test_step_with_iterations(self):
        """step(iterations=N) yields exactly N position tuples."""
        walker = _make_fourbar_walker()
        n = 5
        positions = list(walker.step(iterations=n))
        self.assertEqual(len(positions), n)

    def test_to_mechanism(self):
        """to_mechanism() returns a Mechanism object."""
        walker = _make_fourbar_walker()
        mechanism = walker.to_mechanism()
        # Mechanism should have joints matching the topology nodes
        self.assertEqual(len(mechanism.joints), 3)

    def test_to_mechanism_cached(self):
        """Repeated to_mechanism() calls return the same cached object."""
        walker = _make_fourbar_walker()
        m1 = walker.to_mechanism()
        m2 = walker.to_mechanism()
        self.assertIs(m1, m2)


class TestWalkerFeet(unittest.TestCase):
    """Test get_feet() — terminal node identification."""

    def test_fourbar_has_no_feet(self):
        """In the 4-bar, all non-ground/non-driver nodes have degree > 1."""
        walker = _make_fourbar_walker()
        feet = walker.get_feet()
        # 'follower' participates in 2 edges (frame_follower, crank_follower)
        # so it is NOT a foot (degree 2)
        self.assertEqual(feet, [])

    def test_walker_with_foot_finds_foot(self):
        """A linkage with a degree-1 DRIVEN node reports it as a foot."""
        walker = _make_foot_topology_only()
        feet = walker.get_feet()
        self.assertIn("toe", feet)

    def test_ground_nodes_not_feet(self):
        """Ground nodes are never returned as feet."""
        walker = _make_foot_topology_only()
        feet = walker.get_feet()
        self.assertNotIn("frame", feet)

    def test_driver_nodes_not_feet(self):
        """Driver nodes are never returned as feet even if degree 1."""
        walker = _make_foot_topology_only()
        feet = walker.get_feet()
        self.assertNotIn("crank", feet)

    def test_solvable_walker_no_degree1_feet(self):
        """In the solvable 5-node walker, 'foot' has degree 2 (not a foot)."""
        walker = _make_walker_with_foot()
        feet = walker.get_feet()
        # 'foot' connects to both 'crank' and 'upper', so degree == 2
        self.assertNotIn("foot", feet)


class TestAddLegs(unittest.TestCase):
    """Test add_legs() — adding phase-offset leg copies."""

    def test_add_legs_increases_node_count(self):
        """add_legs(1) adds new non-ground nodes to the topology."""
        walker = _make_fourbar_walker()
        initial_count = len(walker.topology.nodes)
        walker.add_legs(1)
        self.assertGreater(len(walker.topology.nodes), initial_count)

    def test_add_legs_preserves_ground_nodes(self):
        """Ground nodes are shared, not duplicated."""
        walker = _make_fourbar_walker()
        ground_before = {n.id for n in walker.topology.ground_nodes()}
        walker.add_legs(1)
        ground_after = {n.id for n in walker.topology.ground_nodes()}
        # Original ground nodes should still be present
        self.assertTrue(ground_before.issubset(ground_after))

    def test_add_legs_invalidates_cache(self):
        """After add_legs, the cached Mechanism is cleared."""
        walker = _make_fourbar_walker()
        _ = walker.to_mechanism()
        self.assertIsNotNone(walker._mechanism)
        walker.add_legs(1)
        # Cache should have been invalidated
        self.assertIsNone(walker._mechanism)

    def test_add_zero_legs_no_change(self):
        """add_legs(0) is a no-op."""
        walker = _make_fourbar_walker()
        initial_count = len(walker.topology.nodes)
        walker.add_legs(0)
        self.assertEqual(len(walker.topology.nodes), initial_count)

    def test_add_legs_increases_edge_count(self):
        """add_legs(1) adds new edges for the cloned leg."""
        walker = _make_fourbar_walker()
        initial_edges = len(walker.topology.edges)
        walker.add_legs(1)
        self.assertGreater(len(walker.topology.edges), initial_edges)


class TestAddOppositeLeg(unittest.TestCase):
    """Test add_opposite_leg() — mirroring the mechanism."""

    def test_increases_node_count(self):
        """add_opposite_leg() adds mirrored nodes."""
        walker = _make_walker_with_foot()
        initial_count = len(walker.topology.nodes)
        walker.add_opposite_leg()
        self.assertGreater(len(walker.topology.nodes), initial_count)

    def test_mirrors_positions(self):
        """Opposite nodes have X-coordinates reflected across the axis."""
        walker = _make_walker_with_foot()
        walker.add_opposite_leg(axis_x=0.0)
        # 'crank' is at (1, 0), so opposite should be at (-1, 0)
        opp_pos = walker.dimensions.get_node_position("crank (opposite)")
        self.assertIsNotNone(opp_pos)
        self.assertAlmostEqual(opp_pos[0], -1.0)
        self.assertAlmostEqual(opp_pos[1], 0.0)

    def test_mirrors_positions_custom_axis(self):
        """Mirroring across axis_x=2 reflects correctly."""
        walker = _make_walker_with_foot()
        walker.add_opposite_leg(axis_x=2.0)
        # 'crank' at x=1 mirrored across x=2 → x=3
        opp_pos = walker.dimensions.get_node_position("crank (opposite)")
        self.assertIsNotNone(opp_pos)
        self.assertAlmostEqual(opp_pos[0], 3.0)

    def test_on_axis_ground_shared(self):
        """Ground nodes on the mirror axis are not duplicated."""
        walker = _make_walker_with_foot()
        # 'frame' is at (0, 0), which is on axis_x=0
        walker.add_opposite_leg(axis_x=0.0)
        # 'frame' should not have an opposite copy
        self.assertNotIn("frame (opposite)", walker.topology.nodes)
        # 'frame' itself should still be there
        self.assertIn("frame", walker.topology.nodes)

    def test_off_axis_ground_cloned(self):
        """Ground nodes off the mirror axis are cloned."""
        walker = _make_walker_with_foot()
        # 'frame2' is at (2, 0), off axis_x=0
        walker.add_opposite_leg(axis_x=0.0)
        self.assertIn("frame2 (opposite)", walker.topology.nodes)

    def test_invalidates_cache(self):
        """After mirroring, the cached Mechanism is cleared."""
        walker = _make_walker_with_foot()
        _ = walker.to_mechanism()
        walker.add_opposite_leg()
        # Cache should have been invalidated
        self.assertIsNone(walker._mechanism)

    def test_mirror_leg_alias(self):
        """mirror_leg is an alias for add_opposite_leg."""
        walker = _make_walker_with_foot()
        initial_count = len(walker.topology.nodes)
        walker.mirror_leg()
        self.assertGreater(len(walker.topology.nodes), initial_count)

    def test_opposite_names_have_suffix(self):
        """Opposite nodes include '(opposite)' in their IDs."""
        walker = _make_walker_with_foot()
        walker.add_opposite_leg()
        opposite_ids = [
            nid for nid in walker.topology.nodes if "(opposite)" in nid
        ]
        self.assertGreater(len(opposite_ids), 0)

    def test_add_opposite_leg_increases_edge_count(self):
        """add_opposite_leg() adds new edges for the mirrored leg."""
        walker = _make_walker_with_foot()
        initial_edges = len(walker.topology.edges)
        walker.add_opposite_leg()
        self.assertGreater(len(walker.topology.edges), initial_edges)


class TestOptimizationInterface(unittest.TestCase):
    """Test get/set_num_constraints and get/set_coords."""

    def test_get_num_constraints_returns_list(self):
        """get_num_constraints() returns a list of floats."""
        walker = _make_fourbar_walker()
        constraints = walker.get_num_constraints()
        self.assertIsInstance(constraints, list)
        self.assertGreater(len(constraints), 0)
        for c in constraints:
            self.assertIsInstance(c, float)

    def test_set_num_constraints_roundtrip(self):
        """set_num_constraints(get_num_constraints()) preserves values."""
        walker = _make_fourbar_walker()
        original = walker.get_num_constraints()
        walker.set_num_constraints(original)
        after = walker.get_num_constraints()
        for a, b in zip(original, after):
            self.assertAlmostEqual(a, b)

    def test_set_num_constraints_accepts_list(self):
        """set_num_constraints accepts a list without raising."""
        walker = _make_fourbar_walker()
        original = walker.get_num_constraints()
        # Modify and set — should not raise
        modified = [c * 1.1 for c in original]
        walker.set_num_constraints(modified)
        # At minimum the constraint count should be unchanged
        after = walker.get_num_constraints()
        self.assertEqual(len(after), len(original))

    def test_get_coords_returns_positions(self):
        """get_coords() returns a list of (x, y) tuples."""
        walker = _make_fourbar_walker()
        coords = walker.get_coords()
        self.assertIsInstance(coords, list)
        self.assertEqual(len(coords), 3)
        for coord in coords:
            self.assertEqual(len(coord), 2)

    def test_set_coords_roundtrip(self):
        """set_coords(get_coords()) preserves values."""
        walker = _make_fourbar_walker()
        original = walker.get_coords()
        walker.set_coords(original)
        after = walker.get_coords()
        for (ox, oy), (ax, ay) in zip(original, after):
            self.assertAlmostEqual(ox, ax)
            self.assertAlmostEqual(oy, ay)

    def test_set_coords_updates_dimensions(self):
        """set_coords syncs back to the Dimensions object."""
        walker = _make_fourbar_walker()
        coords = walker.get_coords()
        # Shift all positions by (10, 10)
        shifted = [(x + 10.0, y + 10.0) for x, y in coords]
        walker.set_coords(shifted)
        # Check that Dimensions were updated
        frame_pos = walker.dimensions.get_node_position("frame")
        self.assertIsNotNone(frame_pos)
        self.assertAlmostEqual(frame_pos[0], 10.0)
        self.assertAlmostEqual(frame_pos[1], 10.0)


class TestWalkerFromLegacy(unittest.TestCase):
    """Test walker_from_legacy() — conversion from legacy Linkage."""

    def _make_legacy_linkage(self):
        """Create a legacy pylinkage Linkage for conversion."""
        from pylinkage import Static, Crank, Pivot

        base = Static(0, 0, name="base")
        crank = Crank(
            1, 0, joint0=base, distance=1, angle=0.1, name="crank"
        )
        follower = Pivot(
            0, 2, joint0=base, joint1=crank,
            distance0=2, distance1=1.5, name="follower"
        )
        from pylinkage import Linkage

        linkage = Linkage(
            joints=(base, crank, follower),
            order=(base, crank, follower),
            name="legacy_fourbar",
        )
        return linkage

    def test_returns_walker(self):
        """walker_from_legacy returns a Walker instance."""
        linkage = self._make_legacy_linkage()
        walker = walker_from_legacy(linkage)
        self.assertIsInstance(walker, Walker)

    def test_preserves_name(self):
        """The converted Walker keeps the legacy linkage's name."""
        linkage = self._make_legacy_linkage()
        walker = walker_from_legacy(linkage)
        self.assertEqual(walker.name, "legacy_fourbar")

    def test_has_topology(self):
        """The converted Walker has a valid HypergraphLinkage topology."""
        linkage = self._make_legacy_linkage()
        walker = walker_from_legacy(linkage)
        self.assertIsInstance(walker.topology, HypergraphLinkage)
        self.assertGreater(len(walker.topology.nodes), 0)

    def test_has_dimensions(self):
        """The converted Walker has Dimensions with positions."""
        linkage = self._make_legacy_linkage()
        walker = walker_from_legacy(linkage)
        self.assertIsInstance(walker.dimensions, Dimensions)
        self.assertGreater(len(walker.dimensions.node_positions), 0)

    def test_can_step(self):
        """The converted Walker can run kinematic simulation."""
        linkage = self._make_legacy_linkage()
        walker = walker_from_legacy(linkage)
        positions = list(walker.step())
        self.assertGreater(len(positions), 0)


if __name__ == "__main__":
    unittest.main()
