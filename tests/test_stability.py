#!/usr/bin/env python3
"""Tests for the stability metrics module."""

import unittest
from math import tau

import pymunk as pm
from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole

from leggedsnake.dynamiclinkage import DynamicLinkage
from leggedsnake.physicsengine import World, params
from leggedsnake.stability import (
    StabilitySnapshot,
    StabilityTimeSeries,
    approximate_zmp,
    compute_com,
    compute_com_velocity,
    compute_stability_snapshot,
    compute_tip_over_margin,
    get_support_polygon,
)
from leggedsnake.walker import Walker


def _make_fourbar():
    hg = HypergraphLinkage(name="fourbar")
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("follower", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("frame_crank", "frame", "crank"))
    hg.add_edge(Edge("frame_follower", "frame", "follower"))
    hg.add_edge(Edge("crank_follower", "crank", "follower"))
    dims = Dimensions(
        node_positions={"frame": (0, 0), "crank": (1, 0), "follower": (0, 2)},
        driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={
            "frame_crank": 1.0, "frame_follower": 2.0, "crank_follower": 1.5,
        },
    )
    return hg, dims


class TestComputeCom(unittest.TestCase):
    """Center-of-mass computation."""

    def test_single_body(self):
        """CoM of a single body equals its position."""
        space = pm.Space()
        space.gravity = params["physics"]["gravity"]
        hg, dims = _make_fourbar()
        dl = DynamicLinkage(hg, dims, space, name="test")
        com = compute_com(dl)
        self.assertEqual(len(com), 2)
        self.assertIsInstance(com[0], float)
        self.assertIsInstance(com[1], float)

    def test_com_finite(self):
        """CoM coordinates should be finite numbers."""
        space = pm.Space()
        space.gravity = params["physics"]["gravity"]
        hg, dims = _make_fourbar()
        dl = DynamicLinkage(hg, dims, space, load=5, name="test")
        com = compute_com(dl)
        self.assertTrue(all(abs(c) < 1e6 for c in com))


class TestComputeComVelocity(unittest.TestCase):

    def test_initial_velocity_near_zero(self):
        """A freshly created linkage should have near-zero CoM velocity."""
        space = pm.Space()
        space.gravity = (0, 0)  # No gravity to avoid initial acceleration
        hg, dims = _make_fourbar()
        dl = DynamicLinkage(hg, dims, space, name="test")
        vel = compute_com_velocity(dl)
        self.assertAlmostEqual(vel[0], 0.0, places=1)
        self.assertAlmostEqual(vel[1], 0.0, places=1)


class TestApproximateZmp(unittest.TestCase):

    def test_stationary(self):
        """ZMP equals CoM x-position when stationary (no acceleration)."""
        com = (5.0, 2.0)
        vel = (1.0, 0.0)
        prev_vel = (1.0, 0.0)  # Same velocity → zero acceleration
        zmp = approximate_zmp(com, vel, prev_vel, dt=0.02, gravity=9.81)
        self.assertAlmostEqual(zmp, 5.0)

    def test_accelerating_shifts_zmp(self):
        """ZMP shifts opposite to the direction of horizontal acceleration."""
        com = (5.0, 2.0)
        vel = (2.0, 0.0)
        prev_vel = (0.0, 0.0)  # Accelerating rightward
        zmp = approximate_zmp(com, vel, prev_vel, dt=0.02, gravity=9.81)
        # x_zmp = 5.0 - (2.0 * 100.0) / 9.81 < 5.0
        self.assertLess(zmp, 5.0)

    def test_zero_dt_returns_com_x(self):
        """With dt=0, ZMP defaults to CoM x."""
        zmp = approximate_zmp((3.0, 1.0), (1.0, 0.0), (0.0, 0.0), dt=0, gravity=9.81)
        self.assertAlmostEqual(zmp, 3.0)


class TestSupportPolygon(unittest.TestCase):

    def test_no_contacts(self):
        """No feet on ground → empty polygon."""
        space = pm.Space()
        space.gravity = params["physics"]["gravity"]
        hg, dims = _make_fourbar()
        dl = DynamicLinkage(hg, dims, space, name="test")
        # Set threshold very low so nothing qualifies
        poly = get_support_polygon(dl, ground_threshold=-100)
        self.assertEqual(len(poly), 0)

    def test_with_contacts(self):
        """Joints near y=0 should appear in support polygon."""
        space = pm.Space()
        space.gravity = params["physics"]["gravity"]
        hg, dims = _make_fourbar()
        dl = DynamicLinkage(hg, dims, space, name="test")
        # Use a generous threshold
        poly = get_support_polygon(dl, ground_threshold=10)
        self.assertGreater(len(poly), 0)


class TestTipOverMargin(unittest.TestCase):

    def test_centered_positive(self):
        """CoM centered between supports has positive margin."""
        com = (5.0, 2.0)
        support = [(3.0, 0.0), (7.0, 0.0)]
        margin = compute_tip_over_margin(com, support)
        self.assertGreater(margin, 0)
        self.assertAlmostEqual(margin, 2.0)

    def test_outside_negative(self):
        """CoM outside support has negative margin."""
        com = (10.0, 2.0)
        support = [(3.0, 0.0), (7.0, 0.0)]
        margin = compute_tip_over_margin(com, support)
        self.assertLess(margin, 0)

    def test_empty_support(self):
        """No support → negative margin."""
        margin = compute_tip_over_margin((5.0, 2.0), [])
        self.assertEqual(margin, -1.0)

    def test_single_point(self):
        """Single contact point: margin = 0 when CoM above it."""
        margin = compute_tip_over_margin((5.0, 2.0), [(5.0, 0.0)])
        self.assertAlmostEqual(margin, 0.0)

    def test_at_edge(self):
        """CoM exactly at support edge gives margin 0."""
        margin = compute_tip_over_margin((3.0, 2.0), [(3.0, 0.0), (7.0, 0.0)])
        self.assertAlmostEqual(margin, 0.0)


class TestStabilityTimeSeries(unittest.TestCase):

    def _make_series(self, margins):
        snaps = [
            StabilitySnapshot(
                time=i * 0.02, com=(0, 1), com_velocity=(0, 0),
                zmp_x=0.0, support_polygon=[], tip_over_margin=m,
                body_angle=0.01 * i,
            )
            for i, m in enumerate(margins)
        ]
        return StabilityTimeSeries(snaps)

    def test_mean_margin(self):
        series = self._make_series([1.0, 2.0, 3.0])
        self.assertAlmostEqual(series.mean_tip_over_margin, 2.0)

    def test_min_margin(self):
        series = self._make_series([1.0, -0.5, 3.0])
        self.assertAlmostEqual(series.min_tip_over_margin, -0.5)

    def test_empty_series(self):
        series = StabilityTimeSeries()
        self.assertEqual(series.mean_tip_over_margin, 0.0)
        self.assertEqual(series.min_tip_over_margin, 0.0)
        self.assertEqual(series.zmp_excursion, 0.0)
        self.assertEqual(series.angular_stability, 0.0)

    def test_summary_metrics(self):
        series = self._make_series([1.0, 2.0])
        m = series.summary_metrics()
        self.assertIn("mean_tip_over_margin", m)
        self.assertIn("min_tip_over_margin", m)
        self.assertIn("zmp_excursion", m)
        self.assertIn("angular_stability", m)

    def test_com_trajectory(self):
        series = self._make_series([1.0, 2.0])
        self.assertEqual(len(series.com_trajectory), 2)


class TestStabilitySnapshotIntegration(unittest.TestCase):
    """Integration test: collect snapshots during a short simulation."""

    def test_collect_snapshots(self):
        hg, dims = _make_fourbar()
        walker = Walker(hg, dims)

        world = World()
        world.add_linkage(walker, load=5)
        dl = world.linkages[0]

        gravity = abs(world.space.gravity[1])
        dt = 0.02
        prev = None
        snapshots = []

        for step_i in range(50):
            world.update()
            snap = compute_stability_snapshot(
                dl, prev, step_i * dt, dt, gravity,
            )
            snapshots.append(snap)
            prev = snap

        series = StabilityTimeSeries(snapshots)
        self.assertEqual(len(series.snapshots), 50)
        self.assertIsInstance(series.mean_tip_over_margin, float)
        self.assertIsInstance(series.angular_stability, float)


if __name__ == "__main__":
    unittest.main()
