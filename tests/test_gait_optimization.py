#!/usr/bin/env python3
"""Tests for phase-offset multi-gait support."""
import unittest
from math import pi, tau

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole

from leggedsnake.fitness import DistanceFitness, FitnessResult
from leggedsnake.gait_optimization import (
    GaitOptimizationConfig,
    GaitOptimizationResult,
    optimize_gait,
)
from leggedsnake.walker import Walker


def _make_simple_walker() -> Walker:
    """Single-leg template (matches the other test fixtures)."""
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


def _driver_initial_angles(walker: Walker) -> list[float]:
    return [
        walker.dimensions.driver_angles[nid].initial_angle
        for nid in sorted(walker.dimensions.driver_angles)
    ]


class TestAddLegsBackwardCompat(unittest.TestCase):
    """add_legs(int) preserves the evenly-spaced behaviour."""

    def test_three_legs_evenly_spaced(self):
        walker = _make_simple_walker()
        walker.add_legs(3)
        angles = _driver_initial_angles(walker)
        # Template at 0 + 3 new legs at tau/4, tau/2, 3tau/4
        self.assertEqual(len(angles), 4)
        expected = sorted([0.0, tau / 4, tau / 2, 3 * tau / 4])
        self.assertEqual(sorted(a % tau for a in angles), expected)

    def test_zero_or_negative_is_noop(self):
        walker = _make_simple_walker()
        walker.add_legs(0)
        self.assertEqual(len(_driver_initial_angles(walker)), 1)
        walker.add_legs(-5)
        self.assertEqual(len(_driver_initial_angles(walker)), 1)


class TestAddLegsWithOffsets(unittest.TestCase):
    """add_legs with an explicit offsets sequence."""

    def test_explicit_offsets_produce_expected_angles(self):
        walker = _make_simple_walker()
        walker.add_legs([pi / 2, pi, 3 * pi / 2])
        angles = _driver_initial_angles(walker)
        self.assertEqual(len(angles), 4)
        expected = sorted([0.0, pi / 2, pi, 3 * pi / 2])
        self.assertEqual(sorted(a % tau for a in angles), expected)

    def test_non_uniform_offsets(self):
        """Asymmetric offsets like (0.3, 1.7) produce those angles, not even-spaced."""
        walker = _make_simple_walker()
        walker.add_legs([0.3, 1.7])
        angles = sorted(a % tau for a in _driver_initial_angles(walker))
        self.assertEqual(len(angles), 3)
        self.assertAlmostEqual(angles[0], 0.0)
        self.assertAlmostEqual(angles[1], 0.3)
        self.assertAlmostEqual(angles[2], 1.7)

    def test_offsets_normalized_modulo_tau(self):
        """Offsets outside [0, tau) are wrapped."""
        walker = _make_simple_walker()
        walker.add_legs([tau + 0.5, -0.3])
        angles = sorted(a % tau for a in _driver_initial_angles(walker))
        self.assertEqual(len(angles), 3)
        self.assertAlmostEqual(angles[0], 0.0, places=10)
        self.assertAlmostEqual(angles[1], 0.5, places=10)
        self.assertAlmostEqual(angles[2], (tau - 0.3) % tau, places=10)

    def test_empty_sequence_is_noop(self):
        walker = _make_simple_walker()
        walker.add_legs([])
        self.assertEqual(len(_driver_initial_angles(walker)), 1)

    def test_trot_pattern(self):
        """[π] produces a two-leg walker in trot-like opposition."""
        walker = _make_simple_walker()
        walker.add_legs([pi])
        angles = sorted(a % tau for a in _driver_initial_angles(walker))
        self.assertEqual(len(angles), 2)
        self.assertAlmostEqual(angles[0], 0.0)
        self.assertAlmostEqual(angles[1], pi)


class TestGaitOptimizationConfig(unittest.TestCase):
    """GaitOptimizationConfig validation."""

    def test_rejects_n_legs_less_than_2(self):
        with self.assertRaises(ValueError):
            GaitOptimizationConfig(
                walker_factory=_make_simple_walker,
                n_legs=1,
                fitness=DistanceFitness(duration=0.1),
            )

    def test_rejects_mismatched_initial_offsets(self):
        with self.assertRaises(ValueError):
            GaitOptimizationConfig(
                walker_factory=_make_simple_walker,
                n_legs=4,
                fitness=DistanceFitness(duration=0.1),
                initial_offsets=[pi, 2 * pi],  # 2 elements, needs 3
            )

    def test_accepts_matching_initial_offsets(self):
        cfg = GaitOptimizationConfig(
            walker_factory=_make_simple_walker,
            n_legs=4,
            fitness=DistanceFitness(duration=0.1),
            initial_offsets=[pi / 2, pi, 3 * pi / 2],
        )
        self.assertEqual(len(cfg.initial_offsets or []), 3)


class _StubFitness:
    """Deterministic fitness: rewards offsets close to a target pattern.

    Avoids running physics so we can test the optimizer loop fast.
    """

    def __init__(self, target: list[float]) -> None:
        self.target = target

    def __call__(
        self,
        topology,  # noqa: ARG002
        dimensions,
        config=None,  # noqa: ARG002
    ) -> FitnessResult:
        # Extract phase offsets from cloned drivers' initial_angle.
        angles = sorted(
            da.initial_angle % tau
            for da in dimensions.driver_angles.values()
        )
        # Skip the template (at 0) — compare the rest to the target.
        if len(angles) - 1 != len(self.target):
            return FitnessResult(score=0.0, valid=True)
        added = sorted(angles[1:])
        target_sorted = sorted(o % tau for o in self.target)
        # Circular distance per-angle, sum negated → higher score = closer.
        total = 0.0
        for a, t in zip(added, target_sorted):
            d = abs(a - t)
            total += min(d, tau - d)
        return FitnessResult(score=-total, valid=True)


class TestOptimizeGait(unittest.TestCase):
    """End-to-end phase-offset optimization."""

    def test_runs_and_returns_valid_result(self):
        cfg = GaitOptimizationConfig(
            walker_factory=_make_simple_walker,
            n_legs=3,
            fitness=_StubFitness(target=[pi / 2, pi]),
            popsize=5,
            maxiter=8,
            seed=1,
            workers=1,
        )
        result = optimize_gait(cfg)
        self.assertIsInstance(result, GaitOptimizationResult)
        self.assertEqual(len(result.best_offsets), 2)
        self.assertGreater(result.n_evaluations, 0)
        for o in result.best_offsets:
            self.assertGreaterEqual(o, 0.0)
            self.assertLess(o, tau + 1e-9)

    def test_recovers_target_pattern(self):
        """With a fitness that rewards proximity to a known pattern, DE
        should land close to that pattern."""
        target = [pi / 2, pi, 3 * pi / 2]
        cfg = GaitOptimizationConfig(
            walker_factory=_make_simple_walker,
            n_legs=4,
            fitness=_StubFitness(target=target),
            popsize=10,
            maxiter=25,
            seed=42,
            workers=1,
        )
        result = optimize_gait(cfg)
        found = sorted(result.best_offsets)
        expected = sorted(target)
        for f, e in zip(found, expected):
            d = abs(f - e)
            circular = min(d, tau - d)
            self.assertLess(circular, 0.4, f"offset {f} far from target {e}")

    def test_seeded_runs_reproducible(self):
        def run() -> list[float]:
            cfg = GaitOptimizationConfig(
                walker_factory=_make_simple_walker,
                n_legs=3,
                fitness=_StubFitness(target=[pi / 2, pi]),
                popsize=5,
                maxiter=5,
                seed=123,
                workers=1,
            )
            return optimize_gait(cfg).best_offsets

        self.assertEqual(run(), run())

    def test_initial_offsets_warm_start(self):
        """Warm-starting with the exact target should converge very quickly."""
        target = [pi / 2, pi]
        cfg = GaitOptimizationConfig(
            walker_factory=_make_simple_walker,
            n_legs=3,
            fitness=_StubFitness(target=target),
            popsize=5,
            maxiter=3,
            seed=7,
            workers=1,
            initial_offsets=list(target),
        )
        result = optimize_gait(cfg)
        # Seeded with the optimum → best score should be very close to 0
        # (the stub's maximum).
        self.assertGreater(result.best_score, -0.5)


if __name__ == "__main__":
    unittest.main()
