#!/usr/bin/env python3
"""Tests for Phase 8.3 gait-and-speed metrics.

Covers:
- ``GaitAnalysisResult.gait_asymmetry`` and ``energy_per_cycle``
- ``StabilityTimeSeries.mean_speed`` and ``speed_variance``
- ``GaitFitness`` end-to-end on a physics-simulated walker
- ``CompositeFitness`` with ``"gait"`` in ``objectives``
"""
import math
import unittest
from math import tau

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole

from leggedsnake.fitness import CompositeFitness, GaitFitness, DynamicFitness
from leggedsnake.gait_analysis import GaitAnalysisResult, GaitCycle
from leggedsnake.stability import StabilitySnapshot, StabilityTimeSeries
from leggedsnake.walker import Walker


def _make_simple_walker() -> Walker:
    """5-node walker used by test_fitness.py."""
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


def _cycle(foot_id: str, start: float, stance: float, swing: float) -> GaitCycle:
    return GaitCycle(
        foot_id=foot_id,
        stance_start=start,
        stance_end=start + stance,
        swing_start=start + stance,
        swing_end=start + stance + swing,
    )


class TestGaitAsymmetry(unittest.TestCase):
    """Duty-factor dispersion across feet."""

    def test_zero_for_single_foot(self):
        """One foot → asymmetry is 0 by construction (no comparand)."""
        result = GaitAnalysisResult(
            gait_cycles={"a": [_cycle("a", 0.0, 0.4, 0.6)]},
        )
        self.assertEqual(result.gait_asymmetry, 0.0)

    def test_zero_for_identical_feet(self):
        """Symmetric walker → 0 asymmetry."""
        result = GaitAnalysisResult(gait_cycles={
            "a": [_cycle("a", 0.0, 0.4, 0.6), _cycle("a", 1.0, 0.4, 0.6)],
            "b": [_cycle("b", 0.5, 0.4, 0.6), _cycle("b", 1.5, 0.4, 0.6)],
        })
        self.assertAlmostEqual(result.gait_asymmetry, 0.0)

    def test_positive_for_mismatched_feet(self):
        """Different duty factors across feet → positive asymmetry."""
        # foot_a: duty = 0.4 / 1.0 = 0.4
        # foot_b: duty = 0.7 / 1.0 = 0.7
        result = GaitAnalysisResult(gait_cycles={
            "a": [_cycle("a", 0.0, 0.4, 0.6)],
            "b": [_cycle("b", 0.0, 0.7, 0.3)],
        })
        # Population std of [0.4, 0.7] = 0.15
        self.assertAlmostEqual(result.gait_asymmetry, 0.15, places=6)

    def test_empty(self):
        """No cycles → 0."""
        self.assertEqual(GaitAnalysisResult().gait_asymmetry, 0.0)

    def test_in_summary_metrics(self):
        """gait_asymmetry is surfaced in summary_metrics."""
        result = GaitAnalysisResult(gait_cycles={
            "a": [_cycle("a", 0.0, 0.4, 0.6)],
            "b": [_cycle("b", 0.0, 0.7, 0.3)],
        })
        self.assertIn("gait_asymmetry", result.summary_metrics())


class TestEnergyPerCycle(unittest.TestCase):
    """Energy divided by the number of strides completed."""

    def test_zero_when_no_cycles(self):
        self.assertEqual(GaitAnalysisResult().energy_per_cycle(100.0), 0.0)

    def test_single_foot_single_cycle(self):
        """1 foot, 1 cycle → energy_per_cycle = total_energy."""
        result = GaitAnalysisResult(gait_cycles={
            "a": [_cycle("a", 0.0, 0.4, 0.6)],
        })
        self.assertAlmostEqual(result.energy_per_cycle(50.0), 50.0)

    def test_two_feet_equal_cycles(self):
        """2 feet × 3 cycles each = 3 strides of the walker."""
        result = GaitAnalysisResult(gait_cycles={
            "a": [_cycle("a", 0.0, 0.4, 0.6)] * 3,
            "b": [_cycle("b", 0.5, 0.4, 0.6)] * 3,
        })
        # total_cycles = 6, n_feet = 2 → energy/3 strides
        self.assertAlmostEqual(result.energy_per_cycle(120.0), 40.0)

    def test_four_feet(self):
        """4 feet × 5 cycles each → 5 walker strides."""
        cycles = {
            fid: [_cycle(fid, 0.0, 0.4, 0.6)] * 5
            for fid in ("a", "b", "c", "d")
        }
        result = GaitAnalysisResult(gait_cycles=cycles)
        # total_cycles = 20, n_feet = 4 → energy * 4 / 20 = energy / 5
        self.assertAlmostEqual(result.energy_per_cycle(100.0), 20.0)


class TestStabilitySpeedMetrics(unittest.TestCase):
    """mean_speed and speed_variance on StabilityTimeSeries."""

    def _snap(self, t: float, vx: float) -> StabilitySnapshot:
        return StabilitySnapshot(
            time=t, com=(0.0, 0.0),
            com_velocity=(vx, 0.0),
            zmp_x=0.0, support_polygon=[],
            tip_over_margin=0.0, body_angle=0.0,
        )

    def test_empty(self):
        ts = StabilityTimeSeries()
        self.assertEqual(ts.mean_speed, 0.0)
        self.assertEqual(ts.speed_variance, 0.0)

    def test_constant_speed(self):
        ts = StabilityTimeSeries(snapshots=[
            self._snap(i * 0.02, 2.5) for i in range(10)
        ])
        self.assertAlmostEqual(ts.mean_speed, 2.5)
        self.assertAlmostEqual(ts.speed_variance, 0.0)

    def test_variable_speed(self):
        vxs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ts = StabilityTimeSeries(snapshots=[
            self._snap(i * 0.02, v) for i, v in enumerate(vxs)
        ])
        # mean = 3.0, population variance = 2.0
        self.assertAlmostEqual(ts.mean_speed, 3.0)
        self.assertAlmostEqual(ts.speed_variance, 2.0)

    def test_single_snapshot(self):
        """variance needs ≥ 2 samples; returns 0 otherwise."""
        ts = StabilityTimeSeries(snapshots=[self._snap(0.0, 4.2)])
        self.assertEqual(ts.speed_variance, 0.0)
        self.assertAlmostEqual(ts.mean_speed, 4.2)

    def test_in_summary_metrics(self):
        ts = StabilityTimeSeries(snapshots=[
            self._snap(0.0, 1.0), self._snap(0.02, 3.0),
        ])
        summary = ts.summary_metrics()
        self.assertIn("mean_speed", summary)
        self.assertIn("speed_variance", summary)


class TestGaitFitness(unittest.TestCase):
    """GaitFitness runs physics and returns gait metrics."""

    def test_protocol_conformance(self):
        self.assertIsInstance(GaitFitness(), DynamicFitness)

    def test_returns_fitness_result(self):
        """Short simulation produces a FitnessResult with gait metrics."""
        walker = _make_simple_walker()
        fitness = GaitFitness(duration=0.5, n_legs=1)
        result = fitness(walker.topology, walker.dimensions)
        self.assertTrue(math.isfinite(result.score))
        # Gait metrics must be present regardless of whether cycles were found
        for key in (
            "mean_duty_factor",
            "mean_stride_frequency",
            "mean_stride_length",
            "gait_asymmetry",
            "energy_per_cycle",
            "distance",
            "total_energy",
        ):
            self.assertIn(key, result.metrics)

    def test_loci_opt_in(self):
        """record_loci=False (default) produces empty loci dict."""
        walker = _make_simple_walker()
        fitness = GaitFitness(duration=0.3, n_legs=1)
        result = fitness(walker.topology, walker.dimensions)
        self.assertEqual(result.loci, {})

        fitness_loci = GaitFitness(duration=0.3, n_legs=1, record_loci=True)
        result_loci = fitness_loci(walker.topology, walker.dimensions)
        # Simple walker has at least one joint trajectory recorded
        self.assertGreater(len(result_loci.loci), 0)


class TestCompositeFitnessGaitObjective(unittest.TestCase):
    """CompositeFitness exposes gait metrics when "gait" is in objectives."""

    def test_gait_objective_adds_metrics(self):
        walker = _make_simple_walker()
        fitness = CompositeFitness(
            duration=0.5, n_legs=1,
            objectives=("distance", "gait"),
        )
        result = fitness(walker.topology, walker.dimensions)
        self.assertIn("distance", result.metrics)
        self.assertIn("mean_duty_factor", result.metrics)
        self.assertIn("gait_asymmetry", result.metrics)
        self.assertIn("energy_per_cycle", result.metrics)

    def test_gait_objective_absent_by_default(self):
        """Default objectives don't include gait → metrics omitted."""
        walker = _make_simple_walker()
        fitness = CompositeFitness(duration=0.5, n_legs=1)
        result = fitness(walker.topology, walker.dimensions)
        self.assertNotIn("gait_asymmetry", result.metrics)
        self.assertNotIn("energy_per_cycle", result.metrics)


if __name__ == "__main__":
    unittest.main()
