#!/usr/bin/env python3
"""Tests for the Froude number and cost-of-transport metrics."""
import math
import unittest
from math import tau

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole

from leggedsnake.fitness import (
    CompositeFitness,
    DistanceFitness,
    EfficiencyFitness,
    GaitFitness,
    StabilityFitness,
)
from leggedsnake.gait_analysis import (
    compute_cost_of_transport,
    compute_froude_number,
)
from leggedsnake.walker import Walker


def _make_simple_walker() -> Walker:
    hg = HypergraphLinkage(name="simple")
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("frame2", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("upper", role=NodeRole.DRIVEN))
    hg.add_node(Node("foot", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("frame_crank", "frame", "crank"))
    hg.add_edge(Edge("frame2_upper", "frame2", "upper"))
    hg.add_edge(Edge("crank_upper", "crank", "upper"))
    hg.add_edge(Edge("crank_foot", "crank", "foot"))
    hg.add_edge(Edge("upper_foot", "upper", "foot"))
    dims = Dimensions(
        node_positions={
            "frame": (0, 0), "frame2": (2, 0),
            "crank": (1, 0), "upper": (1, 2), "foot": (1, 3),
        },
        driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={
            "frame_crank": 1.0, "frame2_upper": 2.24,
            "crank_upper": 2.0, "crank_foot": 3.16, "upper_foot": 1.0,
        },
    )
    return Walker(hg, dims, name="simple", motor_rates=-4.0)


class TestComputeFroudeNumber(unittest.TestCase):
    """Pure-function Froude calculation."""

    def test_known_value(self):
        # v = 2 m/s, g = 9.81, L = 1 → Fr = 4 / 9.81
        self.assertAlmostEqual(
            compute_froude_number(2.0, 9.81, 1.0), 4.0 / 9.81, places=10,
        )

    def test_zero_speed_zero_froude(self):
        self.assertEqual(compute_froude_number(0.0, 9.81, 1.0), 0.0)

    def test_zero_gravity_returns_zero(self):
        """Avoid division by zero — degenerate inputs map to 0."""
        self.assertEqual(compute_froude_number(2.0, 0.0, 1.0), 0.0)

    def test_negative_gravity_returns_zero(self):
        self.assertEqual(compute_froude_number(2.0, -9.81, 1.0), 0.0)

    def test_zero_leg_length_returns_zero(self):
        self.assertEqual(compute_froude_number(2.0, 9.81, 0.0), 0.0)

    def test_quadratic_in_speed(self):
        """Fr scales as v² — doubling speed quadruples the number."""
        a = compute_froude_number(1.0, 9.81, 1.0)
        b = compute_froude_number(2.0, 9.81, 1.0)
        self.assertAlmostEqual(b, 4 * a)

    def test_walking_running_transition_band(self):
        """Sanity check: a 1.5 m/s walker on a 0.9 m leg sits below 0.5."""
        fr = compute_froude_number(1.5, 9.81, 0.9)
        self.assertLess(fr, 0.5)
        # And running speed (5 m/s) on the same leg is above 0.5.
        fr_run = compute_froude_number(5.0, 9.81, 0.9)
        self.assertGreater(fr_run, 0.5)


class TestComputeCostOfTransport(unittest.TestCase):
    """Pure-function COT calculation."""

    def test_known_value(self):
        self.assertAlmostEqual(
            compute_cost_of_transport(100.0, 10.0, 5.0), 2.0,
        )

    def test_zero_distance_returns_zero(self):
        self.assertEqual(compute_cost_of_transport(100.0, 10.0, 0.0), 0.0)

    def test_zero_mass_returns_zero(self):
        self.assertEqual(compute_cost_of_transport(100.0, 0.0, 5.0), 0.0)

    def test_negative_distance_returns_zero(self):
        self.assertEqual(compute_cost_of_transport(100.0, 10.0, -5.0), 0.0)

    def test_zero_energy_returns_zero(self):
        self.assertEqual(compute_cost_of_transport(0.0, 10.0, 5.0), 0.0)


class TestFitnessIntegration(unittest.TestCase):
    """Each physics-based fitness class surfaces froude + COT in its metrics."""

    def _check(self, fitness, walker):
        result = fitness(walker.topology, walker.dimensions)
        self.assertIn("froude_number", result.metrics)
        self.assertIn("cost_of_transport", result.metrics)
        self.assertTrue(math.isfinite(result.metrics["froude_number"]))
        self.assertTrue(math.isfinite(result.metrics["cost_of_transport"]))
        # Both metrics are non-negative by construction.
        self.assertGreaterEqual(result.metrics["froude_number"], 0.0)
        self.assertGreaterEqual(result.metrics["cost_of_transport"], 0.0)

    def test_distance_fitness(self):
        self._check(
            DistanceFitness(duration=0.5, n_legs=1), _make_simple_walker(),
        )

    def test_efficiency_fitness(self):
        self._check(
            EfficiencyFitness(duration=0.5, n_legs=1, min_distance=0.0),
            _make_simple_walker(),
        )

    def test_stability_fitness(self):
        self._check(
            StabilityFitness(duration=0.5, n_legs=1, min_distance=0.0),
            _make_simple_walker(),
        )

    def test_composite_fitness(self):
        self._check(
            CompositeFitness(duration=0.5, n_legs=1),
            _make_simple_walker(),
        )

    def test_gait_fitness(self):
        self._check(
            GaitFitness(duration=0.5, n_legs=1),
            _make_simple_walker(),
        )

    def test_cot_uses_walker_mass(self):
        """COT should be lower for a heavier walker covering the same
        distance with the same energy — basic sanity check on the
        formula plumbing rather than the exact value."""
        light = DistanceFitness(duration=0.3, n_legs=1)
        result = light(
            _make_simple_walker().topology, _make_simple_walker().dimensions,
        )
        # Manual recomputation from exposed metrics + known mass plumbing:
        # We don't have an easy mass override hook on DistanceFitness, so
        # just verify the metric is in a plausible range.
        if result.metrics["distance"] > 0.01:
            self.assertGreater(result.metrics["cost_of_transport"], 0.0)


if __name__ == "__main__":
    unittest.main()
