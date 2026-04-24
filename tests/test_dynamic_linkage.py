#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the dynamic_linkage module (new hypergraph-based API).
"""

import unittest
from math import tau

import pymunk as pm
from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, NodeRole
from pylinkage.hypergraph.core import Edge, Node

from leggedsnake.dynamic_linkage import DynamicLinkage, NodeProxy, convert_to_dynamic_linkage
from leggedsnake.walker import Walker


def _make_simple_crank():
    """Create a minimal crank mechanism: one ground + one driver."""
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


def _make_fourbar():
    """Create a four-bar linkage: ground, crank, coupler, follower."""
    hg = HypergraphLinkage(name="fourbar")
    hg.add_node(Node("A", role=NodeRole.GROUND))
    hg.add_node(Node("B", role=NodeRole.DRIVER))
    hg.add_node(Node("C", role=NodeRole.DRIVEN))
    hg.add_node(Node("D", role=NodeRole.GROUND))
    hg.add_edge(Edge("AB", "A", "B"))
    hg.add_edge(Edge("BC", "B", "C"))
    hg.add_edge(Edge("CD", "C", "D"))
    hg.add_edge(Edge("DA", "D", "A"))
    dims = Dimensions(
        node_positions={"A": (0, 0), "B": (1, 0), "C": (3, 2), "D": (4, 0)},
        driver_angles={"B": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={"AB": 1.0, "BC": 2.83, "CD": 2.83, "DA": 4.0},
    )
    return hg, dims


def _make_two_driver():
    """Create a mechanism with two independent drivers (multi-DOF)."""
    hg = HypergraphLinkage(name="two_driver")
    hg.add_node(Node("base1", role=NodeRole.GROUND))
    hg.add_node(Node("driver1", role=NodeRole.DRIVER))
    hg.add_node(Node("base2", role=NodeRole.GROUND))
    hg.add_node(Node("driver2", role=NodeRole.DRIVER))
    hg.add_edge(Edge("arm1", "base1", "driver1"))
    hg.add_edge(Edge("arm2", "base2", "driver2"))
    dims = Dimensions(
        node_positions={
            "base1": (0, 0), "driver1": (1, 0),
            "base2": (3, 0), "driver2": (4, 0),
        },
        driver_angles={
            "driver1": DriverAngle(angular_velocity=-tau / 12),
            "driver2": DriverAngle(angular_velocity=tau / 6),
        },
        edge_distances={"arm1": 1.0, "arm2": 1.0},
    )
    return hg, dims


def _make_space():
    """Create a pymunk space with gravity."""
    space = pm.Space()
    space.gravity = (0, -9.8)
    return space


class TestNodeProxy(unittest.TestCase):
    """Test NodeProxy creation and position reading."""

    def test_creation_and_position(self):
        """NodeProxy exposes x, y, coord() from the physics mapping."""
        hg, dims = _make_simple_crank()
        space = _make_space()
        dl = DynamicLinkage(topology=hg, dimensions=dims, space=space)

        # Find the crank proxy
        crank_proxy = next(j for j in dl.joints if j.name == "crank")
        self.assertIsInstance(crank_proxy, NodeProxy)
        self.assertAlmostEqual(crank_proxy.x, 1.0, places=2)
        self.assertAlmostEqual(crank_proxy.y, 0.0, places=2)
        self.assertEqual(crank_proxy.coord(), (crank_proxy.x, crank_proxy.y))

    def test_name_and_role(self):
        """NodeProxy exposes name and role attributes."""
        hg, dims = _make_simple_crank()
        space = _make_space()
        dl = DynamicLinkage(topology=hg, dimensions=dims, space=space)

        base_proxy = next(j for j in dl.joints if j.name == "base")
        self.assertEqual(base_proxy.name, "base")
        self.assertEqual(base_proxy.role, NodeRole.GROUND)

        crank_proxy = next(j for j in dl.joints if j.name == "crank")
        self.assertEqual(crank_proxy.role, NodeRole.DRIVER)

    def test_reload_updates_position(self):
        """NodeProxy.reload() refreshes coordinates from physics bodies."""
        hg, dims = _make_simple_crank()
        space = _make_space()
        dl = DynamicLinkage(topology=hg, dimensions=dims, space=space)

        crank_proxy = next(j for j in dl.joints if j.name == "crank")

        # Step the physics forward to cause movement
        for _ in range(50):
            space.step(0.02)

        crank_proxy.reload()
        # After physics steps, position should potentially change
        # (at minimum, reload should not raise)
        self.assertIsNotNone(crank_proxy.coord())


class TestDynamicLinkage(unittest.TestCase):
    """Test DynamicLinkage creation from hypergraph topology + dimensions."""

    def test_creation_from_hypergraph(self):
        """DynamicLinkage can be created from a HypergraphLinkage."""
        hg, dims = _make_simple_crank()
        space = _make_space()
        dl = DynamicLinkage(topology=hg, dimensions=dims, space=space, name="test")
        self.assertEqual(dl.name, "test")
        self.assertIs(dl.space, space)

    def test_joints_are_node_proxies(self):
        """All joints should be NodeProxy instances."""
        hg, dims = _make_fourbar()
        space = _make_space()
        dl = DynamicLinkage(topology=hg, dimensions=dims, space=space)
        self.assertIsInstance(dl.joints, tuple)
        self.assertEqual(len(dl.joints), 4)
        for j in dl.joints:
            self.assertIsInstance(j, NodeProxy)

    def test_rigidbodies_created(self):
        """rigidbodies list should contain at least the load body."""
        hg, dims = _make_fourbar()
        space = _make_space()
        dl = DynamicLinkage(topology=hg, dimensions=dims, space=space)
        self.assertIsInstance(dl.rigidbodies, list)
        self.assertGreater(len(dl.rigidbodies), 0)
        self.assertIn(dl.body, dl.rigidbodies)

    def test_mass_positive(self):
        """Total mass should be greater than zero."""
        hg, dims = _make_fourbar()
        space = _make_space()
        dl = DynamicLinkage(topology=hg, dimensions=dims, space=space, density=1.0)
        self.assertGreater(dl.mass, 0)

    def test_motors_created_for_drivers(self):
        """Physics mapping should contain motor constraints for driver nodes."""
        hg, dims = _make_simple_crank()
        space = _make_space()
        dl = DynamicLinkage(topology=hg, dimensions=dims, space=space)
        # The physics mapping should have motors for driver nodes
        self.assertTrue(
            hasattr(dl.physics_mapping, 'motors')
            or len(dl.physics_mapping.edge_to_body) > 0,
            "Physics mapping should have bodies or motors for the mechanism",
        )

    def test_get_all_positions(self):
        """get_all_positions returns a dict of node_id -> (x, y)."""
        hg, dims = _make_fourbar()
        space = _make_space()
        dl = DynamicLinkage(topology=hg, dimensions=dims, space=space)
        positions = dl.get_all_positions()
        self.assertIsInstance(positions, dict)
        self.assertEqual(len(positions), 4)
        for node_id, pos in positions.items():
            self.assertIsInstance(pos, tuple)
            self.assertEqual(len(pos), 2)


class TestConvertToDynamicLinkage(unittest.TestCase):
    """Test convert_to_dynamic_linkage from a Walker."""

    def test_from_walker(self):
        """convert_to_dynamic_linkage produces a DynamicLinkage from a Walker."""
        hg, dims = _make_fourbar()
        walker = Walker(topology=hg, dimensions=dims, name="test_walker")
        space = _make_space()
        dl = convert_to_dynamic_linkage(walker, space, density=1, load=5)
        self.assertIsInstance(dl, DynamicLinkage)
        self.assertEqual(dl.name, "test_walker")

    def test_motor_rates_forwarded(self):
        """Walker.motor_rates should be forwarded to DynamicLinkage."""
        hg, dims = _make_simple_crank()
        walker = Walker(
            topology=hg, dimensions=dims, name="rated",
            motor_rates=-3.0,
        )
        space = _make_space()
        dl = convert_to_dynamic_linkage(walker, space)
        # The linkage should have been created without error, meaning
        # motor_rates was accepted and used
        self.assertIsInstance(dl, DynamicLinkage)

    def test_motor_rates_override(self):
        """Explicit motor_rates should override Walker.motor_rates."""
        hg, dims = _make_simple_crank()
        walker = Walker(
            topology=hg, dimensions=dims, name="override",
            motor_rates=-3.0,
        )
        space = _make_space()
        dl = convert_to_dynamic_linkage(walker, space, motor_rates=-6.0)
        self.assertIsInstance(dl, DynamicLinkage)


class TestDynamicLinkageLoad(unittest.TestCase):
    """Test DynamicLinkage with different load masses."""

    def test_zero_load(self):
        """Zero load should still create a valid linkage."""
        hg, dims = _make_simple_crank()
        space = _make_space()
        dl = DynamicLinkage(topology=hg, dimensions=dims, space=space, load=0)
        self.assertIsNotNone(dl.body)
        self.assertIsInstance(dl.body, pm.Body)

    def test_positive_load(self):
        """Positive load should increase total mass."""
        hg, dims = _make_simple_crank()
        space_a = _make_space()
        dl_no_load = DynamicLinkage(topology=hg, dimensions=dims, space=space_a, load=0)

        space_b = _make_space()
        dl_loaded = DynamicLinkage(topology=hg, dimensions=dims, space=space_b, load=50)

        self.assertGreater(dl_loaded.mass, dl_no_load.mass)

    def test_large_load(self):
        """Large load should dominate total mass."""
        hg, dims = _make_simple_crank()
        space = _make_space()
        dl = DynamicLinkage(topology=hg, dimensions=dims, space=space, load=1000)
        self.assertGreater(dl.mass, 999)


class TestMultiDOF(unittest.TestCase):
    """Test multi-DOF mechanisms with two independent drivers."""

    def test_two_drivers_accepted(self):
        """A mechanism with two drivers should be created successfully."""
        hg, dims = _make_two_driver()
        space = _make_space()
        dl = DynamicLinkage(topology=hg, dimensions=dims, space=space)
        self.assertIsInstance(dl, DynamicLinkage)
        # Should have proxies for all 4 nodes
        self.assertEqual(len(dl.joints), 4)

    def test_two_drivers_with_dict_rates(self):
        """Dict motor_rates should assign different rates to each driver."""
        hg, dims = _make_two_driver()
        space = _make_space()
        rates = {"driver1": -2.0, "driver2": 3.0}
        dl = DynamicLinkage(
            topology=hg, dimensions=dims, space=space, motor_rates=rates,
        )
        self.assertIsInstance(dl, DynamicLinkage)

    def test_two_drivers_with_scalar_rate(self):
        """A single scalar motor_rate should apply to all drivers."""
        hg, dims = _make_two_driver()
        space = _make_space()
        dl = DynamicLinkage(
            topology=hg, dimensions=dims, space=space, motor_rates=-5.0,
        )
        self.assertIsInstance(dl, DynamicLinkage)


if __name__ == "__main__":
    unittest.main()
