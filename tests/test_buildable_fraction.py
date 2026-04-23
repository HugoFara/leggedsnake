#!/usr/bin/env python3
"""Tests for the structured ``buildable_fraction`` failure metric.

The metric replaces the binary ``UnbuildableError`` / ``valid=False``
path so optimizers see a smooth signal toward the buildable region of
design space instead of a flat ``score=0`` cliff.
"""
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
    StrideFitness,
    _compute_buildable_fraction,
    _SimulationResult,
    _scale_invariant_metrics,
)
from leggedsnake.walker import Walker


def _make_buildable_walker() -> Walker:
    """A trivially buildable four-bar — every crank angle assembles."""
    hg = HypergraphLinkage(name="ok")
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
    return Walker(hg, dims, name="ok", motor_rates=-4.0)


def _make_near_miss_walker() -> Walker:
    """An RRR dyad where the foot can almost never reach both anchors.

    A and B are 100 units apart, but the dyad legs are length 1 — only
    a tiny window of crank angles produces a buildable assembly. This
    is the realistic optimizer near-miss case: a smooth signal of
    *how close* the design got, rather than a binary fail.
    """
    hg = HypergraphLinkage(name="near-miss")
    hg.add_node(Node("A", role=NodeRole.GROUND))
    hg.add_node(Node("B", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("C", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("A_crank", "A", "crank"))
    hg.add_edge(Edge("crank_C", "crank", "C"))
    hg.add_edge(Edge("B_C", "B", "C"))
    dims = Dimensions(
        node_positions={
            "A": (0, 0), "B": (100, 0),
            "crank": (1, 0), "C": (50, 0),
        },
        driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={"A_crank": 1.0, "crank_C": 1.0, "B_C": 1.0},
    )
    return Walker(hg, dims, name="near-miss", motor_rates=-4.0)


class TestComputeBuildableFraction(unittest.TestCase):
    """Direct exercise of the kinematic-preview helper."""

    def test_buildable_walker_returns_one(self):
        fr = _compute_buildable_fraction(_make_buildable_walker())
        self.assertEqual(fr, 1.0)

    def test_near_miss_returns_partial(self):
        """A mechanism that builds only at a few crank angles — exactly
        the case where the optimizer needs a gradient instead of a
        binary fail."""
        fr = _compute_buildable_fraction(_make_near_miss_walker())
        self.assertGreater(fr, 0.0)
        self.assertLess(fr, 1.0)

    def test_zero_iterations_returns_zero(self):
        """Empty preview → zero fraction (degenerate input boundary)."""
        fr = _compute_buildable_fraction(
            _make_buildable_walker(), iterations=0,
        )
        self.assertEqual(fr, 0.0)

    def test_iterations_override_caps_preview(self):
        """Caller can shorten the preview; ratio is preserved."""
        walker = _make_buildable_walker()
        self.assertEqual(_compute_buildable_fraction(walker, iterations=4), 1.0)


class TestScaleInvariantMetricsExposesFraction(unittest.TestCase):
    """``_scale_invariant_metrics`` carries the fraction through to
    every fitness class. This is the pivot point — adding the metric
    here populates it everywhere downstream."""

    def test_fraction_propagates_from_simulation_result(self):
        result = _SimulationResult(
            distance=1.0, mass=1.0, characteristic_length=1.0,
            buildable_fraction=0.42,
        )
        metrics = _scale_invariant_metrics(result, duration=1.0)
        self.assertEqual(metrics["buildable_fraction"], 0.42)

    def test_default_fraction_is_one(self):
        """Backwards compat: an unset ``buildable_fraction`` defaults to
        1.0 so existing callers keep their meaning."""
        result = _SimulationResult(
            distance=1.0, mass=1.0, characteristic_length=1.0,
        )
        metrics = _scale_invariant_metrics(result, duration=1.0)
        self.assertEqual(metrics["buildable_fraction"], 1.0)


class TestFitnessSurfacesMetric(unittest.TestCase):
    """Every physics fitness exposes ``buildable_fraction`` in metrics."""

    def _check(self, fitness, walker):
        result = fitness(walker.topology, walker.dimensions)
        self.assertIn("buildable_fraction", result.metrics)
        self.assertEqual(result.metrics["buildable_fraction"], 1.0)

    def test_distance_fitness(self):
        self._check(
            DistanceFitness(duration=0.3, n_legs=1),
            _make_buildable_walker(),
        )

    def test_efficiency_fitness(self):
        self._check(
            EfficiencyFitness(duration=0.3, n_legs=1, min_distance=0.0),
            _make_buildable_walker(),
        )

    def test_stability_fitness(self):
        self._check(
            StabilityFitness(duration=0.3, n_legs=1, min_distance=0.0),
            _make_buildable_walker(),
        )

    def test_composite_fitness(self):
        self._check(
            CompositeFitness(duration=0.3, n_legs=1),
            _make_buildable_walker(),
        )

    def test_gait_fitness(self):
        self._check(
            GaitFitness(duration=0.3, n_legs=1),
            _make_buildable_walker(),
        )

    def test_stride_fitness(self):
        result = StrideFitness(lap_points=12)(
            _make_buildable_walker().topology,
            _make_buildable_walker().dimensions,
        )
        self.assertIn("buildable_fraction", result.metrics)
        self.assertEqual(result.metrics["buildable_fraction"], 1.0)


class TestNearMissPreservesGradient(unittest.TestCase):
    """The whole point: a partial-buildability walker still surfaces
    its fraction so the optimizer has a non-zero gradient toward the
    buildable region — instead of a flat ``score=0, valid=False`` cliff."""

    def test_stride_fitness_partial_fraction(self):
        walker = _make_near_miss_walker()
        result = StrideFitness(lap_points=12)(
            walker.topology, walker.dimensions,
        )
        # The fraction is the gradient signal — it must be in (0, 1)
        # for the near-miss case, not collapsed to 0 or 1.
        fr = result.metrics["buildable_fraction"]
        self.assertGreater(fr, 0.0)
        self.assertLess(fr, 1.0)


if __name__ == "__main__":
    unittest.main()
