#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the hypergraph_physics module.
"""

import unittest
from math import pi

import pymunk as pm
from pylinkage import Static, Crank, Fixed, Pivot, Linkage
from pylinkage.hypergraph import from_linkage

from leggedsnake.hypergraph_physics import (
    PhysicsMapping,
    create_bodies_from_hypergraph,
    get_node_world_position,
)


class TestPhysicsMapping(unittest.TestCase):
    """Test PhysicsMapping dataclass."""

    def test_physics_mapping_creation(self):
        """Test creating an empty PhysicsMapping."""
        mapping = PhysicsMapping()
        self.assertEqual(mapping.edge_to_body, {})
        self.assertEqual(mapping.node_to_bodies, {})
        self.assertEqual(mapping.constraints, [])
        self.assertEqual(mapping.motors, [])


class TestCreateBodiesFromHypergraph(unittest.TestCase):
    """Test create_bodies_from_hypergraph function."""

    def setUp(self):
        """Create a basic linkage setup for testing."""
        self.space = pm.Space()
        self.space.gravity = (0, -9.8)
        self.load_body = pm.Body(1, 1)
        self.load_body.position = (0, 0)
        self.space.add(self.load_body)
        self.filter = pm.ShapeFilter(group=1)

    def test_simple_crank_linkage(self):
        """Test body creation for a simple crank."""
        # Create a simple crank: Static -> Crank
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")

        linkage = Linkage(joints=(base, crank), name="simple_crank")
        hg, dims = from_linkage(linkage)

        mapping = create_bodies_from_hypergraph(
            hg, dims, self.space, self.load_body, density=1, thickness=0.1, shape_filter=self.filter
        )

        # Should have one edge (base -> crank) which creates one body
        self.assertEqual(len(mapping.edge_to_body), 1)

        # Base (ground) should be on load_body
        self.assertIn("base", mapping.node_to_bodies)
        self.assertIn(self.load_body, mapping.node_to_bodies["base"])

        # Crank (driver) should have a motor
        self.assertGreater(len(mapping.motors), 0)

    def test_four_bar_linkage(self):
        """Test body creation for a four-bar linkage."""
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        follower = Pivot(
            0, 2, joint0=base, joint1=crank,
            distance0=2, distance1=1.5, name="follower"
        )
        output = Fixed(
            joint0=crank, joint1=follower,
            distance=1, angle=-pi/2, name="output"
        )

        linkage = Linkage(joints=(base, crank, follower, output), name="fourbar")
        hg, dims = from_linkage(linkage)

        mapping = create_bodies_from_hypergraph(
            hg, dims, self.space, self.load_body, density=1, thickness=0.1, shape_filter=self.filter
        )

        # Should have edges for each bar connection
        self.assertGreater(len(mapping.edge_to_body), 0)

        # All nodes should have at least one body
        for node_id in hg.nodes.keys():
            self.assertIn(node_id, mapping.node_to_bodies)
            self.assertGreater(len(mapping.node_to_bodies[node_id]), 0)

    def test_pivot_constraints_created(self):
        """Test that pivot constraints are created at shared nodes."""
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        follower = Pivot(
            0, 2, joint0=base, joint1=crank,
            distance0=2, distance1=1.5, name="follower"
        )

        linkage = Linkage(joints=(base, crank, follower), name="test")
        hg, dims = from_linkage(linkage)

        mapping = create_bodies_from_hypergraph(
            hg, dims, self.space, self.load_body, density=1, thickness=0.1, shape_filter=self.filter
        )

        # Should have pivot constraints where multiple bodies meet
        # (at least at the base and crank positions)
        self.assertGreater(len(mapping.constraints), 0)

    def test_motor_constraints_for_drivers(self):
        """Test that motor constraints are created for driver nodes."""
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0.5, name="crank")

        linkage = Linkage(joints=(base, crank), name="test")
        hg, dims = from_linkage(linkage)

        mapping = create_bodies_from_hypergraph(
            hg, dims, self.space, self.load_body, density=1, thickness=0.1, shape_filter=self.filter
        )

        # Should have a motor for the crank
        self.assertEqual(len(mapping.motors), 1)
        # Motor pivot is now created by _create_pivot_constraints (not separately)
        # The pivot at the ground connection is included in mapping.constraints


class TestGetNodeWorldPosition(unittest.TestCase):
    """Test get_node_world_position function."""

    def test_get_position_from_mapping(self):
        """Test getting world position from physics mapping."""
        space = pm.Space()
        load_body = pm.Body(1, 1)
        load_body.position = (0, 0)
        space.add(load_body)

        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")

        linkage = Linkage(joints=(base, crank), name="test")
        hg, dims = from_linkage(linkage)

        mapping = create_bodies_from_hypergraph(
            hg, dims, space, load_body, density=1, thickness=0.1, shape_filter=None
        )

        # Get position of base node
        pos = get_node_world_position("base", mapping)
        self.assertAlmostEqual(pos.x, 0, places=5)
        self.assertAlmostEqual(pos.y, 0, places=5)

    def test_get_position_unknown_node(self):
        """Test getting position for unknown node returns zero."""
        mapping = PhysicsMapping()
        pos = get_node_world_position("unknown", mapping)
        self.assertEqual(pos.x, 0)
        self.assertEqual(pos.y, 0)


if __name__ == "__main__":
    unittest.main()
