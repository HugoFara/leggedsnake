#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the hypergraph_physics module."""

import unittest
from math import pi, tau

import pymunk as pm
from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, Hyperedge, NodeRole

from leggedsnake.hypergraph_physics import (
    PhysicsMapping,
    create_bodies_from_hypergraph,
    get_node_world_position,
)


def _make_simple_crank() -> tuple[HypergraphLinkage, Dimensions]:
    """Create a simple crank: ground -> driver."""
    hg = HypergraphLinkage(name="simple_crank")
    hg.add_node(Node("base", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_edge(Edge("base_crank", "base", "crank"))

    dims = Dimensions(
        node_positions={"base": (0, 0), "crank": (1, 0)},
        driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={"base_crank": 1.0},
    )
    return hg, dims


def _make_four_bar() -> tuple[HypergraphLinkage, Dimensions]:
    """Create a four-bar linkage: base-crank-follower-output."""
    hg = HypergraphLinkage(name="fourbar")
    hg.add_node(Node("base", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("follower", role=NodeRole.DRIVEN))

    hg.add_edge(Edge("base_crank", "base", "crank"))
    hg.add_edge(Edge("base_follower", "base", "follower"))
    hg.add_edge(Edge("crank_follower", "crank", "follower"))

    dims = Dimensions(
        node_positions={"base": (0, 0), "crank": (1, 0), "follower": (0, 2)},
        driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={
            "base_crank": 1.0,
            "base_follower": 2.0,
            "crank_follower": 1.5,
        },
    )
    return hg, dims


class TestPhysicsMapping(unittest.TestCase):
    def test_physics_mapping_creation(self):
        mapping = PhysicsMapping()
        self.assertEqual(mapping.edge_to_body, {})
        self.assertEqual(mapping.node_to_bodies, {})
        self.assertEqual(mapping.constraints, [])
        self.assertEqual(mapping.motors, [])
        self.assertEqual(mapping.motor_node_ids, [])


class TestCreateBodiesFromHypergraph(unittest.TestCase):
    def setUp(self):
        self.space = pm.Space()
        self.space.gravity = (0, -9.8)
        self.load_body = pm.Body(1, 1)
        self.load_body.position = (0, 0)
        self.space.add(self.load_body)
        self.filter = pm.ShapeFilter(group=1)

    def test_simple_crank_linkage(self):
        hg, dims = _make_simple_crank()
        mapping = create_bodies_from_hypergraph(
            hg, dims, self.space, self.load_body,
            density=1, thickness=0.1, shape_filter=self.filter,
        )

        self.assertEqual(len(mapping.edge_to_body), 1)
        self.assertIn("base", mapping.node_to_bodies)
        self.assertIn(self.load_body, mapping.node_to_bodies["base"])
        self.assertGreater(len(mapping.motors), 0)

    def test_four_bar_linkage(self):
        hg, dims = _make_four_bar()
        mapping = create_bodies_from_hypergraph(
            hg, dims, self.space, self.load_body,
            density=1, thickness=0.1, shape_filter=self.filter,
        )

        self.assertGreater(len(mapping.edge_to_body), 0)
        for node_id in hg.nodes:
            self.assertIn(node_id, mapping.node_to_bodies)
            self.assertGreater(len(mapping.node_to_bodies[node_id]), 0)

    def test_pivot_constraints_created(self):
        hg, dims = _make_four_bar()
        mapping = create_bodies_from_hypergraph(
            hg, dims, self.space, self.load_body,
            density=1, thickness=0.1, shape_filter=self.filter,
        )
        self.assertGreater(len(mapping.constraints), 0)

    def test_motor_constraints_for_drivers(self):
        hg, dims = _make_simple_crank()
        mapping = create_bodies_from_hypergraph(
            hg, dims, self.space, self.load_body,
            density=1, thickness=0.1, shape_filter=self.filter,
        )
        self.assertEqual(len(mapping.motors), 1)
        self.assertEqual(mapping.motor_node_ids, ["crank"])

    def test_multi_driver_rates(self):
        """Two independent drivers with different motor rates."""
        hg = HypergraphLinkage(name="multi_dof")
        hg.add_node(Node("ground", role=NodeRole.GROUND))
        hg.add_node(Node("driver_a", role=NodeRole.DRIVER))
        hg.add_node(Node("driver_b", role=NodeRole.DRIVER))
        hg.add_edge(Edge("g_a", "ground", "driver_a"))
        hg.add_edge(Edge("g_b", "ground", "driver_b"))

        dims = Dimensions(
            node_positions={"ground": (0, 0), "driver_a": (1, 0), "driver_b": (-1, 0)},
            driver_angles={
                "driver_a": DriverAngle(angular_velocity=0.1),
                "driver_b": DriverAngle(angular_velocity=0.2),
            },
            edge_distances={"g_a": 1.0, "g_b": 1.0},
        )

        mapping = create_bodies_from_hypergraph(
            hg, dims, self.space, self.load_body,
            density=1, thickness=0.1, shape_filter=self.filter,
            motor_rates={"driver_a": -3.0, "driver_b": -5.0},
        )

        self.assertEqual(len(mapping.motors), 2)
        self.assertEqual(len(mapping.motor_node_ids), 2)
        self.assertIn("driver_a", mapping.motor_node_ids)
        self.assertIn("driver_b", mapping.motor_node_ids)

        # Check rates are different
        rates = set()
        for motor in mapping.motors:
            rates.add(motor.rate)
        self.assertEqual(len(rates), 2)

    def test_single_float_motor_rate(self):
        """Single float motor_rates applies to all drivers."""
        hg = HypergraphLinkage(name="multi_dof")
        hg.add_node(Node("ground", role=NodeRole.GROUND))
        hg.add_node(Node("driver_a", role=NodeRole.DRIVER))
        hg.add_node(Node("driver_b", role=NodeRole.DRIVER))
        hg.add_edge(Edge("g_a", "ground", "driver_a"))
        hg.add_edge(Edge("g_b", "ground", "driver_b"))

        dims = Dimensions(
            node_positions={"ground": (0, 0), "driver_a": (1, 0), "driver_b": (-1, 0)},
            driver_angles={
                "driver_a": DriverAngle(angular_velocity=0.1),
                "driver_b": DriverAngle(angular_velocity=0.2),
            },
            edge_distances={"g_a": 1.0, "g_b": 1.0},
        )

        mapping = create_bodies_from_hypergraph(
            hg, dims, self.space, self.load_body,
            density=1, thickness=0.1, shape_filter=self.filter,
            motor_rates=-4.0,
        )

        self.assertEqual(len(mapping.motors), 2)
        # Both should have the same rate
        for motor in mapping.motors:
            self.assertAlmostEqual(motor.rate, -4.0)

    def test_rigid_group_from_hyperedge(self):
        """Hyperedge creates a single rigid body for grouped edges."""
        hg = HypergraphLinkage(name="triangle")
        hg.add_node(Node("A", role=NodeRole.GROUND))
        hg.add_node(Node("B", role=NodeRole.DRIVER))
        hg.add_node(Node("C", role=NodeRole.DRIVEN))
        hg.add_edge(Edge("AB", "A", "B"))
        hg.add_edge(Edge("AC", "A", "C"))
        hg.add_edge(Edge("BC", "B", "C"))
        # This hyperedge marks ABC as a rigid body
        hg.add_hyperedge(Hyperedge("ABC", nodes=("A", "B", "C")))

        dims = Dimensions(
            node_positions={"A": (0, 0), "B": (1, 0), "C": (0.5, 1)},
            driver_angles={"B": DriverAngle(angular_velocity=0.1)},
            edge_distances={"AB": 1.0, "AC": 1.12, "BC": 1.12},
        )

        mapping = create_bodies_from_hypergraph(
            hg, dims, self.space, self.load_body,
            density=1, thickness=0.1, shape_filter=self.filter,
        )

        # Edges AC and BC should share a body (rigid group)
        # AB has ground on one end so may be on load_body
        non_load_bodies = {
            eid: body for eid, body in mapping.edge_to_body.items()
            if body is not self.load_body
        }
        # The non-ground edges should share a body
        bodies = list(non_load_bodies.values())
        if len(bodies) >= 2:
            self.assertIs(bodies[0], bodies[1])


class TestGetNodeWorldPosition(unittest.TestCase):
    def test_get_position_from_mapping(self):
        space = pm.Space()
        load_body = pm.Body(1, 1)
        load_body.position = (0, 0)
        space.add(load_body)

        hg, dims = _make_simple_crank()
        mapping = create_bodies_from_hypergraph(
            hg, dims, space, load_body,
            density=1, thickness=0.1, shape_filter=None,
        )

        pos = get_node_world_position("base", mapping)
        self.assertAlmostEqual(pos.x, 0, places=5)
        self.assertAlmostEqual(pos.y, 0, places=5)

    def test_get_position_unknown_node(self):
        mapping = PhysicsMapping()
        pos = get_node_world_position("unknown", mapping)
        self.assertEqual(pos.x, 0)
        self.assertEqual(pos.y, 0)


if __name__ == "__main__":
    unittest.main()
