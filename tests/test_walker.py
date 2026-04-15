#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the hypergraph-native Walker module.
"""

import unittest
from math import tau

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole
from leggedsnake.walker import Walker


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

    def test_step_skip_unbuildable_accepts_kwarg(self):
        """skip_unbuildable=True is accepted and returns the requested count."""
        walker = _make_fourbar_walker()
        positions = list(walker.step(iterations=8, skip_unbuildable=True))
        self.assertEqual(len(positions), 8)

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


class TestWalkerStepWithDerivatives(unittest.TestCase):
    """Test step_with_derivatives — temporary finite-difference shim."""

    def test_yields_triples(self):
        """Each yielded item is a (positions, velocities, accelerations) triple."""
        walker = _make_fourbar_walker()
        frames = list(walker.step_with_derivatives(iterations=6))
        self.assertEqual(len(frames), 6)
        for pos, vel, acc in frames:
            # All three tuples have one entry per joint (3 here).
            self.assertEqual(len(pos), 3)
            self.assertEqual(len(vel), 3)
            self.assertEqual(len(acc), 3)

    def test_crank_tip_speed_matches_omega_radius(self):
        """|v| at the crank joint should approximate |omega| * radius.

        Walker with unit-length crank spinning at tau/12 rad/step: the
        crank tip moves on the unit circle, so |v| ≈ tau/12 per step
        (central-difference estimate against dt=1).
        """
        from math import hypot
        walker = _make_fourbar_walker()
        # crank joint is index 1 in _make_fourbar_walker (frame=0, crank=1, follower=2)
        frames = list(walker.step_with_derivatives(iterations=24, dt=1.0))
        # Skip the first and last frame (forward/backward diff, less accurate).
        middle = frames[5:20]
        speeds = [hypot(vel[1][0], vel[1][1]) for _, vel, _ in middle]
        expected = abs(-tau / 12) * 1.0  # |omega| * radius
        for s in speeds:
            self.assertAlmostEqual(s, expected, delta=expected * 0.1)

    def test_accelerations_finite(self):
        """Accelerations are computable from velocities (no Nones on buildable runs)."""
        walker = _make_fourbar_walker()
        frames = list(walker.step_with_derivatives(iterations=10))
        for _, _, acc in frames[1:-1]:
            for ax, ay in acc:
                self.assertIsNotNone(ax)
                self.assertIsNotNone(ay)


class TestWalkerMobility(unittest.TestCase):
    """Test pylinkage 0.9 topology-analysis adoption (compute_dof)."""

    def _make_real_fourbar(self) -> Walker:
        """Build a canonical 4-bar (4 nodes, 4 edges: two ground pins + coupler)."""
        hg = HypergraphLinkage(name="real_fourbar")
        hg.add_node(Node("O1", role=NodeRole.GROUND))
        hg.add_node(Node("O2", role=NodeRole.GROUND))
        hg.add_node(Node("A", role=NodeRole.DRIVER))
        hg.add_node(Node("B", role=NodeRole.DRIVEN))
        hg.add_edge(Edge("crank", "O1", "A"))
        hg.add_edge(Edge("coupler", "A", "B"))
        hg.add_edge(Edge("rocker", "O2", "B"))
        dims = Dimensions(
            node_positions={"O1": (0, 0), "O2": (3, 0), "A": (1, 0), "B": (3, 2)},
            driver_angles={"A": DriverAngle(angular_velocity=-tau / 12)},
            edge_distances={"crank": 1.0, "coupler": 2.5, "rocker": 2.0},
        )
        return Walker(hg, dims, name="real_fourbar")

    def test_real_fourbar_dof_is_one(self):
        """A canonical 4-bar (4 nodes, 3 edges + ground) is 1-DOF."""
        walker = self._make_real_fourbar()
        self.assertEqual(walker.dof, 1)

    def test_triangle_dof_is_three(self):
        """A 3-node / 3-edge triangle is under-constrained (DOF=3).

        Sanity-check against Grübler: 4 links − 2·3 joints = 9 − 6 = 3.
        The triangle walker used elsewhere in the test suite is really a
        crank-follower pair without a second ground pin.
        """
        walker = _make_fourbar_walker()  # misnomer: actually a triangle
        self.assertEqual(walker.dof, 3)

    def test_mobility_reports_links_and_joints(self):
        """``mobility`` surfaces the full MobilityInfo."""
        walker = self._make_real_fourbar()
        info = walker.mobility
        self.assertEqual(info.dof, 1)
        self.assertEqual(info.num_full_joints, 4)
        # ground + 3 driven edges = 4 links
        self.assertEqual(info.num_links, 4)

    def test_dof_is_integer(self):
        """DOF is always an integer (Grübler returns an int)."""
        walker = self._make_real_fourbar()
        self.assertIsInstance(walker.dof, int)


class TestWalkerFeet(unittest.TestCase):
    """Test get_feet() — terminal node identification."""

    def test_fourbar_follower_is_foot(self):
        """In the 4-bar, the follower is the outermost driven node."""
        walker = _make_fourbar_walker()
        feet = walker.get_feet()
        # 'follower' neighbours are frame (GROUND) and crank (DRIVER),
        # making it the outermost driven node — effectively the foot.
        self.assertEqual(feet, ["follower"])

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

    def test_solvable_walker_finds_feet(self):
        """In the solvable 5-node walker, 'foot' and 'upper' are outermost driven."""
        walker = _make_walker_with_foot()
        feet = walker.get_feet()
        # Both 'foot' and 'upper' are DRIVEN with only ground/driver/driven
        # neighbours, so the heuristic identifies them.
        self.assertIn("foot", feet)


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


class TestWalkerSixBarFactories(unittest.TestCase):
    """Test Walker.from_watt / from_stephenson (six-bar factories)."""

    def test_watt_builds_six_node_walker(self):
        walker = Walker.from_watt(
            crank=1.5, coupler1=4.0, rocker1=3.5,
            link4=3.0, link5=2.5, rocker2=3.0,
            ground_length=6.0,
        )
        self.assertEqual(walker.name, "watt")
        # Watt six-bar: 2 grounds + 4 dependent joints = 6 nodes.
        self.assertEqual(len(walker.topology.nodes), 6)
        roles = {nid: n.role for nid, n in walker.topology.nodes.items()}
        self.assertEqual(roles["A"], NodeRole.GROUND)
        self.assertEqual(roles["D"], NodeRole.GROUND)
        self.assertEqual(roles["B"], NodeRole.DRIVER)

    def test_watt_is_steppable(self):
        walker = Walker.from_watt(
            crank=1.5, coupler1=4.0, rocker1=3.5,
            link4=3.0, link5=2.5, rocker2=3.0,
            ground_length=6.0,
        )
        positions = list(walker.step(iterations=8, skip_unbuildable=True))
        self.assertEqual(len(positions), 8)

    def test_watt_accepts_custom_motor_rates(self):
        walker = Walker.from_watt(
            crank=1.5, coupler1=4.0, rocker1=3.5,
            link4=3.0, link5=2.5, rocker2=3.0,
            ground_length=6.0,
            motor_rates=-2.5,
        )
        self.assertEqual(walker.motor_rates, -2.5)

    def test_stephenson_builds_six_node_walker(self):
        walker = Walker.from_stephenson(
            crank=1.0, coupler=3.0, rocker=2.5,
            link4=2.0, link5=3.5, link6=2.0,
            ground_length=5.0,
        )
        self.assertEqual(walker.name, "stephenson")
        self.assertEqual(len(walker.topology.nodes), 6)

    def test_stephenson_is_steppable(self):
        walker = Walker.from_stephenson(
            crank=1.0, coupler=3.0, rocker=2.5,
            link4=2.0, link5=3.5, link6=2.0,
            ground_length=5.0,
        )
        positions = list(walker.step(iterations=8, skip_unbuildable=True))
        self.assertEqual(len(positions), 8)


class TestWalkerClassicalFactories(unittest.TestCase):
    """Test Walker.from_jansen / from_klann / from_chebyshev factories."""

    def test_jansen_topology(self):
        walker = Walker.from_jansen(scale=0.04)
        self.assertEqual(walker.name, "jansen")
        # 2 ground + 1 driver + 5 driven = 8 nodes, 11 edges, no hyperedges.
        self.assertEqual(len(walker.topology.nodes), 8)
        self.assertEqual(len(walker.topology.edges), 11)
        self.assertEqual(len(walker.topology.hyperedges), 0)
        roles = {nid: n.role for nid, n in walker.topology.nodes.items()}
        self.assertEqual(roles["O"], NodeRole.GROUND)
        self.assertEqual(roles["B"], NodeRole.GROUND)
        self.assertEqual(roles["A"], NodeRole.DRIVER)
        self.assertEqual(roles["G"], NodeRole.DRIVEN)

    def test_jansen_matches_holy_numbers(self):
        from leggedsnake._classical import JANSEN_HOLY_NUMBERS
        walker = Walker.from_jansen(scale=1.0)
        # Crank radius = m, unscaled.
        self.assertAlmostEqual(
            walker.dimensions.edge_distances["O_A"],
            JANSEN_HOLY_NUMBERS["m"],
        )
        self.assertAlmostEqual(
            walker.dimensions.edge_distances["F_G"],
            JANSEN_HOLY_NUMBERS["h"],
        )

    def test_jansen_is_steppable(self):
        walker = Walker.from_jansen(scale=0.04)
        positions = list(walker.step(iterations=24, skip_unbuildable=True))
        self.assertEqual(len(positions), 24)

    def test_jansen_scale_multiplies_lengths(self):
        w1 = Walker.from_jansen(scale=1.0)
        w2 = Walker.from_jansen(scale=2.0)
        for eid, d1 in w1.dimensions.edge_distances.items():
            self.assertAlmostEqual(w2.dimensions.edge_distances[eid], 2.0 * d1)

    def test_klann_topology(self):
        walker = Walker.from_klann()
        self.assertEqual(walker.name, "klann")
        # 3 ground + 1 driver + 4 driven = 8 nodes, 7 edges, 2 rigid triangles.
        self.assertEqual(len(walker.topology.nodes), 8)
        self.assertEqual(len(walker.topology.edges), 7)
        self.assertEqual(len(walker.topology.hyperedges), 2)
        self.assertIn("ternary_coupler", walker.topology.hyperedges)
        self.assertIn("ternary_leg", walker.topology.hyperedges)

    def test_klann_foot_detected(self):
        walker = Walker.from_klann()
        self.assertIn("foot", walker.get_feet())

    def test_klann_is_steppable(self):
        walker = Walker.from_klann()
        positions = list(walker.step(iterations=24, skip_unbuildable=True))
        self.assertEqual(len(positions), 24)

    def test_chebyshev_topology(self):
        walker = Walker.from_chebyshev()
        self.assertEqual(walker.name, "chebyshev")
        # 2 ground + 1 driver + 2 driven = 5 nodes, 4 edges, 1 triangle.
        self.assertEqual(len(walker.topology.nodes), 5)
        self.assertEqual(len(walker.topology.edges), 4)
        self.assertEqual(len(walker.topology.hyperedges), 1)

    def test_chebyshev_is_steppable(self):
        walker = Walker.from_chebyshev()
        positions = list(walker.step(iterations=24, skip_unbuildable=True))
        self.assertEqual(len(positions), 24)

    def test_chebyshev_foot_ratio_affects_p(self):
        w_mid = Walker.from_chebyshev(foot_ratio=0.5)
        w_end = Walker.from_chebyshev(foot_ratio=1.0)
        self.assertAlmostEqual(
            w_mid.dimensions.edge_distances["A_P"],
            0.5 * w_mid.dimensions.edge_distances["A_B"],
        )
        self.assertAlmostEqual(
            w_end.dimensions.edge_distances["A_P"],
            w_end.dimensions.edge_distances["A_B"],
        )

    def test_chebyshev_raises_on_unreachable_geometry(self):
        # Coupler + rocker smaller than the ground/crank separation → no solution.
        with self.assertRaises(ValueError):
            Walker.from_chebyshev(
                crank=0.1, coupler=0.1, rocker=0.1, ground_length=10.0,
            )

    def test_factories_accept_custom_motor_rates(self):
        w = Walker.from_jansen(motor_rates=-2.0)
        self.assertEqual(w.motor_rates, -2.0)
        w = Walker.from_klann(motor_rates=3.0)
        self.assertEqual(w.motor_rates, 3.0)
        w = Walker.from_chebyshev(motor_rates={"A": -1.5})
        self.assertEqual(w.motor_rates, {"A": -1.5})

    def test_factories_pair_with_add_legs(self):
        walker = Walker.from_jansen(scale=0.04)
        walker.add_opposite_leg()
        walker.add_legs(2)
        # Driver count should be 2 (original + opposite) * 3 phase copies = 6.
        self.assertEqual(len(walker.topology.driver_nodes()), 6)


class TestWalkerFromSimLinkage(unittest.TestCase):
    """Test the temporary SimLinkage → Walker shim."""

    def _make_sim_linkage(self):
        from pylinkage.actuators import Crank
        from pylinkage.components import Ground
        from pylinkage.dyads import RRRDyad
        from pylinkage.simulation import Linkage as SimLinkage

        g_a = Ground(0, 0, name="A")
        g_d = Ground(3, 0, name="D")
        crank = Crank(
            g_a, radius=1.0, angular_velocity=0.1, name="B",
        )
        follower = RRRDyad(
            anchor1=crank, anchor2=g_d,
            distance1=2.5, distance2=2.0, x=3, y=2, name="C",
        )
        return SimLinkage((g_a, g_d, crank, follower), name="fourbar")

    def test_roundtrip_four_bar_produces_walker(self):
        from leggedsnake.walker import _walker_from_sim_linkage

        walker = _walker_from_sim_linkage(self._make_sim_linkage())
        self.assertIsInstance(walker, Walker)
        self.assertEqual(walker.name, "fourbar")
        # 4 components → 4 nodes (2 ground + 1 driver + 1 driven)
        self.assertEqual(len(walker.topology.nodes), 4)

    def test_node_roles_mapped(self):
        from leggedsnake.walker import _walker_from_sim_linkage

        walker = _walker_from_sim_linkage(self._make_sim_linkage())
        roles = {nid: n.role for nid, n in walker.topology.nodes.items()}
        self.assertEqual(roles["A"], NodeRole.GROUND)
        self.assertEqual(roles["D"], NodeRole.GROUND)
        self.assertEqual(roles["B"], NodeRole.DRIVER)
        self.assertEqual(roles["C"], NodeRole.DRIVEN)

    def test_walker_is_steppable(self):
        from leggedsnake.walker import _walker_from_sim_linkage

        walker = _walker_from_sim_linkage(self._make_sim_linkage())
        positions = list(walker.step(iterations=6, skip_unbuildable=True))
        self.assertEqual(len(positions), 6)

    def test_unknown_component_raises(self):
        from leggedsnake.walker import _walker_from_sim_linkage

        class Foo:
            x, y, name = 0, 0, "foo"

        class FakeSim:
            components = (Foo(),)
            name = "fake"

        with self.assertRaises(NotImplementedError):
            _walker_from_sim_linkage(FakeSim())


if __name__ == "__main__":
    unittest.main()
